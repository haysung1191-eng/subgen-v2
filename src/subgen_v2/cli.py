from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .config import AlignmentConfig, ASRConfig, AudioConfig, DebugConfig, PipelineConfig, SubtitleConfig, VADConfig
from .pipeline import run_pipeline


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
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path.with_suffix(".v2.srt")
    temp_wav = args.temp_wav.resolve() if args.temp_wav else Path(tempfile.gettempdir()) / f"{input_path.stem}.subgen_v2.16k.wav"
    debug_enabled = args.debug_dir is not None
    config = PipelineConfig(
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
        debug=DebugConfig(enabled=debug_enabled, output_dir=args.debug_dir.resolve() if args.debug_dir else None),
    )
    result = run_pipeline(config, progress=lambda message: print(f"[subgen_v2] {message}", flush=True))
    print(f"timing_authority={result.timing_authority}")
    print(f"regions={result.region_count} drafts={result.draft_count} aligned_tokens={result.aligned_token_count} subtitles={result.subtitle_count}")
    print(f"output={result.output_path}")
    if result.debug_dir is not None:
        print(f"debug_dir={result.debug_dir}")


if __name__ == "__main__":
    main()
