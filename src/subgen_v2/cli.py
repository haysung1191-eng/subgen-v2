from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

from .config import AlignmentConfig, ASRConfig, AudioConfig, DebugConfig, PipelineConfig, SubtitleConfig, VADConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="subgen-v2", description="Minimal timing-first Korean subtitle pipeline (v2).")
    parser.add_argument("input", type=Path, help="Input media path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output SRT path")
    parser.add_argument("--temp-wav", type=Path, default=None, help="Intermediate WAV path")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=250)
    parser.add_argument("--vad-min-silence-ms", type=int, default=150)
    parser.add_argument("--vad-pre-roll-ms", type=int, default=100)
    parser.add_argument("--vad-post-roll-ms", type=int, default=100)
    parser.add_argument("--vad-merge-gap-ms", type=int, default=150)
    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--align-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--align-model", default=None)
    parser.add_argument("--align-min-conf", type=float, default=0.35)
    parser.add_argument("--align-utterance-padding-ms", type=int, default=180)
    parser.add_argument("--subtitle-hold-ms", type=int, default=180)
    parser.add_argument("--min-gap-to-next-ms", type=int, default=50)
    parser.add_argument("--min-duration-ms", type=int, default=220)
    parser.add_argument("--tiny-overlap-fix-ms", type=int, default=20)
    parser.add_argument("--end-fallback-threshold-ms", type=int, default=320)
    parser.add_argument("--debug-dir", type=Path, default=None, help="Optional per-stage JSON dump directory")
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        run_doctor()
        return
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path.with_suffix(".v2.srt")
    debug_enabled = args.debug_dir is not None
    debug_dir = args.debug_dir.resolve() if args.debug_dir else None
    temp_context = None
    try:
        temp_wav = args.temp_wav.resolve() if args.temp_wav else _unique_temp_wav(input_path)
        if args.temp_wav is None:
            temp_context = temp_wav.parent
        config = _build_config(args, input_path, output_path, temp_wav, debug_enabled, debug_dir)
        preflight(config)
        from .pipeline import run_pipeline

        result = run_pipeline(config, progress=lambda message: print(f"[subgen_v2] {message}", flush=True))
        _print_result(result)
    finally:
        if temp_context is not None:
            shutil.rmtree(temp_context, ignore_errors=True)


def _build_config(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    temp_wav: Path,
    debug_enabled: bool,
    debug_dir: Path | None,
) -> PipelineConfig:
    return PipelineConfig(
        input_path=input_path,
        output_path=output_path,
        temp_wav_path=temp_wav,
        audio=AudioConfig(sample_rate=args.sample_rate),
        vad=VADConfig(
            threshold=args.vad_threshold,
            min_speech_ms=args.vad_min_speech_ms,
            min_silence_ms=args.vad_min_silence_ms,
            pre_roll_ms=args.vad_pre_roll_ms,
            post_roll_ms=args.vad_post_roll_ms,
            merge_gap_ms=args.vad_merge_gap_ms,
        ),
        asr=ASRConfig(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            language=args.language,
        ),
        alignment=AlignmentConfig(
            device=args.align_device,
            model_name=args.align_model,
            min_word_confidence=args.align_min_conf,
            utterance_padding_ms=args.align_utterance_padding_ms,
        ),
        subtitle=SubtitleConfig(
            hold_ms=args.subtitle_hold_ms,
            min_gap_to_next_ms=args.min_gap_to_next_ms,
            min_duration_ms=args.min_duration_ms,
            tiny_overlap_fix_ms=args.tiny_overlap_fix_ms,
            end_fallback_threshold_ms=args.end_fallback_threshold_ms,
        ),
        debug=DebugConfig(enabled=debug_enabled, output_dir=debug_dir),
    )


def _unique_temp_wav(input_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="subgen_v2_"))
    return temp_dir / f"{input_path.stem}.16k.wav"


def preflight(config: PipelineConfig) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found on PATH. Install ffmpeg and verify with: ffmpeg -version")
    if not config.input_path.exists():
        raise SystemExit(f"Input file does not exist: {config.input_path}")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if not config.output_path.parent.exists():
        raise SystemExit(f"Output directory does not exist: {config.output_path.parent}")
    if config.asr.device == "cpu" and config.asr.compute_type == "float16":
        raise SystemExit("CPU mode cannot use --compute-type float16. Use --compute-type int8 or --compute-type float32.")
    if config.asr.device == "cuda" or config.alignment.device == "cuda":
        _require_cuda()
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise SystemExit("faster-whisper is not installed. Run: pip install -e .") from exc
    if config.alignment.backend == "whisperx":
        try:
            import whisperx  # noqa: F401
        except ImportError as exc:
            raise SystemExit('whisperx is not installed. Run: pip install -e ".[align-whisperx]"') from exc


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("CUDA was requested but torch is not installed.") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False. Use --device cpu --align-device cpu or install CUDA-enabled PyTorch.")


def run_doctor() -> None:
    checks = [
        ("ffmpeg", shutil.which("ffmpeg") is not None, "ffmpeg -version"),
        ("faster-whisper", _can_import("faster_whisper"), "pip install -e ."),
        ("whisperx", _can_import("whisperx"), 'pip install -e ".[align-whisperx]"'),
        ("torch", _can_import("torch"), "install PyTorch"),
    ]
    for name, ok, fix in checks:
        status = "OK" if ok else f"FAIL ({fix})"
        print(f"{name}: {status}")
    if _can_import("torch"):
        import torch

        print(f"torch_cuda_available: {torch.cuda.is_available()}")


def _can_import(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _print_result(result) -> None:
    print(f"timing_authority={result.timing_authority}")
    print(
        " ".join(
            [
                f"regions={result.region_count}",
                f"drafts={result.draft_count}",
                f"aligned_tokens={result.aligned_token_count}",
                f"subtitles={result.subtitle_count}",
            ]
        )
    )
    print(
        " ".join(
            [
                f"aligned_starts={result.aligned_start_count}",
                f"draft_start_fallbacks={result.draft_start_fallback_count}",
                f"draft_end_fallbacks={result.draft_end_fallback_count}",
                f"zero_aligned_subtitles={result.subtitles_with_zero_aligned_tokens}",
            ]
        )
    )
    print(f"end_gap_median_ms={result.median_end_gap_ms:.1f} end_gap_max_ms={result.max_end_gap_ms:.1f}")
    if result.timing_authority != "aligned_tokens":
        print("warning=some subtitles used draft timing fallback; inspect summary.json or 05_subtitles_final.json")
    print(f"output={result.output_path}")
    if result.debug_dir is not None:
        print(f"debug_dir={result.debug_dir}")


if __name__ == "__main__":
    main()
