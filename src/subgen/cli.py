from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from rich.console import Console

from .config import AlignmentConfig, PipelineConfig, RuntimeConfig, TimingConfig, TranscriptionConfig, VADConfig
from .errors import InputMediaError, SubgenError
from .runner import run_batch_with_runtime, run_single_with_runtime


console = Console()


KO_SYNC_CONSERVATIVE_PRESET: dict[str, object] = {
    "vad_threshold": 0.48,
    "vad_min_speech_ms": 240,
    "vad_min_silence_ms": 180,
    "vad_pad_ms": 110,
    "vad_pre_roll_ms": 130,
    "vad_post_roll_ms": 120,
    "vad_merge_gap_ms": 140,
    "beam_size": 8,
    "language": "ko",
    "overlap_sec": 0.45,
    "alignment": "on",
    "alignment_normalization_mode": "conservative",
    "min_segment_sec": 0.24,
    "hard_gap_ms": 70,
    "grouping_mode": "one_unit_per_subtitle",
    "cleanup_mode": "conservative",
    "global_shift_ms": 0,
    "timing_correction_ms": 0,
    "end_tail_padding_ms": 100,
    "max_end_tail_padding_ms": 200,
    "min_gap_to_next_ms": 60,
    "model": "large-v2",
}

KO_SYNC_BALANCED_PRESET: dict[str, object] = {
    "vad_threshold": 0.52,
    "vad_min_speech_ms": 220,
    "vad_min_silence_ms": 170,
    "vad_pad_ms": 90,
    "vad_pre_roll_ms": 110,
    "vad_post_roll_ms": 95,
    "vad_merge_gap_ms": 120,
    "beam_size": 8,
    "language": "ko",
    "overlap_sec": 0.36,
    "alignment": "on",
    "alignment_normalization_mode": "conservative",
    "min_segment_sec": 0.22,
    "hard_gap_ms": 60,
    "grouping_mode": "one_unit_per_subtitle",
    "cleanup_mode": "conservative",
    "global_shift_ms": 0,
    "timing_correction_ms": 0,
    "end_tail_padding_ms": 95,
    "max_end_tail_padding_ms": 190,
    "min_gap_to_next_ms": 60,
    "model": "large-v2",
}

KO_SYNC_ALIGNMENT_HEAVY_PRESET: dict[str, object] = {
    "vad_threshold": 0.56,
    "vad_min_speech_ms": 260,
    "vad_min_silence_ms": 220,
    "vad_pad_ms": 70,
    "vad_pre_roll_ms": 95,
    "vad_post_roll_ms": 85,
    "vad_merge_gap_ms": 180,
    "beam_size": 7,
    "language": "ko",
    "overlap_sec": 0.30,
    "alignment": "on",
    "alignment_normalization_mode": "conservative",
    "min_segment_sec": 0.20,
    "hard_gap_ms": 55,
    "grouping_mode": "one_unit_per_subtitle",
    "cleanup_mode": "minimal",
    "global_shift_ms": 0,
    "timing_correction_ms": 0,
    "end_tail_padding_ms": 0,
    "max_end_tail_padding_ms": 0,
    "min_gap_to_next_ms": 60,
    "model": "large-v2",
}

KO_SYNC_FINAL_PRESET: dict[str, object] = {
    **KO_SYNC_ALIGNMENT_HEAVY_PRESET,
    "preset_family": "ko_sync_final",
}

KO_SYNC_FINAL_TAILPAD_PRESET: dict[str, object] = {
    **KO_SYNC_FINAL_PRESET,
    "end_tail_padding_ms": 150,
    "max_end_tail_padding_ms": 260,
    "min_gap_to_next_ms": 45,
    "preset_family": "ko_sync_final_tailpad",
}

KO_SYNC_FINAL_ACOUSTIC_TAIL_PRESET: dict[str, object] = {
    **KO_SYNC_FINAL_TAILPAD_PRESET,
    "enable_acoustic_tail_extension": True,
    "acoustic_tail_probe_ms": 220,
    "max_acoustic_tail_extension_ms": 180,
    "min_tail_energy_threshold": 0.010,
    "preset_family": "ko_sync_final_acoustic_tail",
}

PRESETS: dict[str, dict[str, object]] = {
    "ko_sync_conservative": {**KO_SYNC_CONSERVATIVE_PRESET, "preset_family": "ko_sync_conservative"},
    "ko_sync_balanced": {**KO_SYNC_BALANCED_PRESET, "preset_family": "ko_sync_balanced"},
    "ko_sync_alignment_heavy": {**KO_SYNC_ALIGNMENT_HEAVY_PRESET, "preset_family": "ko_sync_alignment_heavy"},
    "ko_sync_final": KO_SYNC_FINAL_PRESET,
    "ko_sync_final_tailpad": KO_SYNC_FINAL_TAILPAD_PRESET,
    "ko_sync_final_acoustic_tail": KO_SYNC_FINAL_ACOUSTIC_TAIL_PRESET,
    "ko-sync": {**KO_SYNC_BALANCED_PRESET, "preset_family": "ko-sync"},
    "ko-sync-tight": {**KO_SYNC_CONSERVATIVE_PRESET, "preset_family": "ko-sync-tight"},
    "ko-sync-final": {**KO_SYNC_FINAL_PRESET, "preset_family": "ko-sync-final"},
}

MEDIA_EXTS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".m4v",
    ".webm",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
}


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subgen",
        description="Stable Korean subtitle generator using Silero VAD + faster-whisper + optional alignment.",
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input video/audio file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .srt path (single-file mode)")
    parser.add_argument("--input-dir", type=Path, default=None, help="Batch input directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Batch output directory (default: input-dir)")
    parser.add_argument("--batch-glob", default="**/*", help="Glob pattern for batch file discovery")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file")

    parser.add_argument("--temp-wav", type=Path, default=None, help="Intermediate extracted WAV path")
    parser.add_argument("--keep-wav", action="store_true", help="Keep intermediate WAV file")

    parser.add_argument("--sample-rate", type=int, default=16000, help="Extraction sample rate (Hz)")
    parser.add_argument("--global-shift-ms", type=int, default=0, help="Shift all subtitle timestamps by this amount")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Apply a synchronization tuning preset",
    )

    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD threshold")
    parser.add_argument("--vad-min-speech-ms", type=int, default=250, help="Minimum speech duration (ms)")
    parser.add_argument("--vad-min-silence-ms", type=int, default=120, help="Minimum silence duration (ms)")
    parser.add_argument("--vad-pad-ms", type=int, default=80, help="Speech padding (ms)")
    parser.add_argument("--vad-pre-roll-ms", type=int, default=80, help="Conservative VAD pre-roll padding")
    parser.add_argument("--vad-post-roll-ms", type=int, default=80, help="Conservative VAD post-roll padding")
    parser.add_argument("--vad-merge-gap-ms", type=int, default=140, help="Merge adjacent segments if gap <= this (ms)")

    parser.add_argument("--model", default="large-v3-turbo", help="faster-whisper model size")
    parser.add_argument("--device", default=_default_device(), choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--compute-type", default="float16", help="faster-whisper compute_type")
    parser.add_argument("--beam-size", type=int, default=7, help="Decoding beam size")
    parser.add_argument("--language", default=None, help="Language code (e.g. ko, en). Auto if omitted")
    parser.add_argument("--overlap-sec", type=float, default=0.35, help="Overlap around each VAD segment")

    parser.add_argument("--alignment", choices=["on", "off"], default="on", help="Enable forced alignment stage")
    parser.add_argument("--alignment-backend", choices=["whisperx"], default="whisperx", help="Timing alignment backend")
    parser.add_argument("--align-model", default=None, help="Alignment model name override")
    parser.add_argument("--align-device", choices=["cuda", "cpu"], default="cuda", help="Alignment inference device")
    parser.add_argument("--align-min-conf", type=float, default=0.35, help="Minimum aligned word confidence")
    parser.add_argument("--align-boundary-slack-ms", type=int, default=80, help="Boundary slack for aligned timestamps")
    parser.add_argument("--alignment-normalization-mode", choices=["none", "conservative"], default="conservative", help="Alignment text normalization")

    parser.add_argument("--min-segment-sec", type=float, default=0.20, help="Minimum subtitle duration after cleanup")
    parser.add_argument("--max-segment-sec", type=float, default=4.0, help="Maximum subtitle duration before split")
    parser.add_argument("--hard-gap-ms", type=int, default=60, help="Minimum enforced gap between subtitle segments")
    parser.add_argument("--timing-correction-ms", type=int, default=0, help="Conservative onset pull-up amount")
    parser.add_argument("--end-tail-padding-ms", type=int, default=90, help="Readable padding after last aligned token")
    parser.add_argument("--max-end-tail-padding-ms", type=int, default=180, help="Maximum allowed tail padding")
    parser.add_argument("--enable-acoustic-tail-extension", choices=["on", "off"], default="off", help="Extend subtitle tail when trailing speech energy remains")
    parser.add_argument("--acoustic-tail-probe-ms", type=int, default=180, help="Probe window after last aligned token")
    parser.add_argument("--max-acoustic-tail-extension-ms", type=int, default=120, help="Maximum dynamic tail hold from acoustic evidence")
    parser.add_argument("--min-tail-energy-threshold", type=float, default=0.012, help="Minimum RMS energy for acoustic tail hold")
    parser.add_argument("--min-gap-to-next-ms", type=int, default=60, help="Minimum preserved gap before next subtitle")

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Log verbosity")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for per-run log files")
    parser.add_argument("--debug-export-dir", type=Path, default=None, help="Directory for debug JSON exports")
    parser.add_argument("--batch-concurrency", type=int, default=1, help="Batch worker count (CPU mode only)")
    return parser


def _flag_provided(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def _load_json_config(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file root must be an object.")
    return data


def _apply_config(args: argparse.Namespace, argv: list[str]) -> None:
    if args.config is None:
        return

    values = _load_json_config(args.config)
    flag_map = {
        "preset": "--preset",
        "sample_rate": "--sample-rate",
        "global_shift_ms": "--global-shift-ms",
        "vad_threshold": "--vad-threshold",
        "vad_min_speech_ms": "--vad-min-speech-ms",
        "vad_min_silence_ms": "--vad-min-silence-ms",
        "vad_pad_ms": "--vad-pad-ms",
        "vad_pre_roll_ms": "--vad-pre-roll-ms",
        "vad_post_roll_ms": "--vad-post-roll-ms",
        "vad_merge_gap_ms": "--vad-merge-gap-ms",
        "model": "--model",
        "device": "--device",
        "compute_type": "--compute-type",
        "beam_size": "--beam-size",
        "language": "--language",
        "overlap_sec": "--overlap-sec",
        "alignment": "--alignment",
        "alignment_backend": "--alignment-backend",
        "align_model": "--align-model",
        "align_device": "--align-device",
        "align_min_conf": "--align-min-conf",
        "align_boundary_slack_ms": "--align-boundary-slack-ms",
        "alignment_normalization_mode": "--alignment-normalization-mode",
        "min_segment_sec": "--min-segment-sec",
        "max_segment_sec": "--max-segment-sec",
        "hard_gap_ms": "--hard-gap-ms",
        "timing_correction_ms": "--timing-correction-ms",
        "end_tail_padding_ms": "--end-tail-padding-ms",
        "max_end_tail_padding_ms": "--max-end-tail-padding-ms",
        "enable_acoustic_tail_extension": "--enable-acoustic-tail-extension",
        "acoustic_tail_probe_ms": "--acoustic-tail-probe-ms",
        "max_acoustic_tail_extension_ms": "--max-acoustic-tail-extension-ms",
        "min_tail_energy_threshold": "--min-tail-energy-threshold",
        "min_gap_to_next_ms": "--min-gap-to-next-ms",
        "log_level": "--log-level",
        "debug_export_dir": "--debug-export-dir",
        "batch_concurrency": "--batch-concurrency",
    }
    for field, flag in flag_map.items():
        if field in values and not _flag_provided(argv, flag):
            setattr(args, field, values[field])


def _apply_preset(args: argparse.Namespace, argv: list[str]) -> None:
    preset_values: dict[str, object] | None = None
    if args.preset is None:
        return
    preset_values = PRESETS.get(args.preset)
    if preset_values is None:
        return

    flag_map = {
        "vad_threshold": "--vad-threshold",
        "vad_min_speech_ms": "--vad-min-speech-ms",
        "vad_min_silence_ms": "--vad-min-silence-ms",
        "vad_pad_ms": "--vad-pad-ms",
        "vad_pre_roll_ms": "--vad-pre-roll-ms",
        "vad_post_roll_ms": "--vad-post-roll-ms",
        "vad_merge_gap_ms": "--vad-merge-gap-ms",
        "beam_size": "--beam-size",
        "language": "--language",
        "overlap_sec": "--overlap-sec",
        "alignment": "--alignment",
        "alignment_normalization_mode": "--alignment-normalization-mode",
        "min_segment_sec": "--min-segment-sec",
        "hard_gap_ms": "--hard-gap-ms",
        "global_shift_ms": "--global-shift-ms",
        "timing_correction_ms": "--timing-correction-ms",
        "end_tail_padding_ms": "--end-tail-padding-ms",
        "max_end_tail_padding_ms": "--max-end-tail-padding-ms",
        "enable_acoustic_tail_extension": "--enable-acoustic-tail-extension",
        "acoustic_tail_probe_ms": "--acoustic-tail-probe-ms",
        "max_acoustic_tail_extension_ms": "--max-acoustic-tail-extension-ms",
        "min_tail_energy_threshold": "--min-tail-energy-threshold",
        "min_gap_to_next_ms": "--min-gap-to-next-ms",
        "model": "--model",
    }
    for field, value in preset_values.items():
        if field not in flag_map:
            continue
        if _flag_provided(argv, flag_map[field]):
            continue
        setattr(args, field, value)


def _build_runtime(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        log_level=args.log_level,
        log_dir=args.log_dir.resolve() if args.log_dir else None,
        batch_concurrency=max(1, int(args.batch_concurrency)),
        debug_export_dir=args.debug_export_dir.resolve() if args.debug_export_dir else None,
    )


def _build_config(args: argparse.Namespace, input_path: Path, output_path: Path, temp_wav: Path) -> PipelineConfig:
    return PipelineConfig(
        input_path=input_path,
        output_path=output_path,
        temp_wav_path=temp_wav,
        sample_rate=args.sample_rate,
        global_shift_ms=args.global_shift_ms,
        vad=VADConfig(
            threshold=args.vad_threshold,
            min_speech_ms=args.vad_min_speech_ms,
            min_silence_ms=args.vad_min_silence_ms,
            pad_ms=args.vad_pad_ms,
            pre_roll_ms=args.vad_pre_roll_ms,
            post_roll_ms=args.vad_post_roll_ms,
            merge_gap_ms=args.vad_merge_gap_ms,
        ),
        transcription=TranscriptionConfig(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            language=args.language,
            overlap_sec=args.overlap_sec,
        ),
        alignment=AlignmentConfig(
            enabled=(args.alignment == "on"),
            backend=args.alignment_backend,
            device=args.align_device,
            model_name=args.align_model,
            min_word_confidence=args.align_min_conf,
            boundary_slack_ms=args.align_boundary_slack_ms,
            normalization_mode=args.alignment_normalization_mode,
            fallback_on_failure=True,
        ),
        timing=TimingConfig(
            min_duration_sec=args.min_segment_sec,
            hard_gap_sec=args.hard_gap_ms / 1000.0,
            max_duration_sec=args.max_segment_sec,
            onset_nudge_ms=args.timing_correction_ms,
            end_tail_padding_ms=args.end_tail_padding_ms,
            max_end_tail_padding_ms=args.max_end_tail_padding_ms,
            enable_acoustic_tail_extension=(args.enable_acoustic_tail_extension == "on"),
            acoustic_tail_probe_ms=args.acoustic_tail_probe_ms,
            max_acoustic_tail_extension_ms=args.max_acoustic_tail_extension_ms,
            min_tail_energy_threshold=args.min_tail_energy_threshold,
            min_gap_to_next_ms=args.min_gap_to_next_ms,
        ),
    )


def _collect_batch_inputs(input_dir: Path, pattern: str) -> list[Path]:
    files = [p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in MEDIA_EXTS]
    return sorted(files)


def _validate_single_input(args: argparse.Namespace) -> Path:
    if args.input is None:
        raise InputMediaError("입력 파일이 필요합니다.", detail="Input file is required unless --input-dir is used.")
    input_path = args.input.resolve()
    if not input_path.exists():
        raise InputMediaError("입력 파일을 찾을 수 없습니다.", detail=f"Input file not found: {input_path}")
    return input_path


def _validate_batch_input(args: argparse.Namespace) -> Path:
    if args.input_dir is None:
        raise InputMediaError("입력 폴더가 필요합니다.", detail="Batch input directory is required.")
    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise InputMediaError("입력 폴더를 찾을 수 없습니다.", detail=f"Batch input directory not found: {input_dir}")
    return input_dir


def _run_single(args: argparse.Namespace) -> None:
    input_path = _validate_single_input(args)
    output_path = args.output.resolve() if args.output else input_path.with_suffix(".srt")
    temp_wav = args.temp_wav.resolve() if args.temp_wav else Path(tempfile.gettempdir()) / f"{input_path.stem}.subgen.16k.wav"
    config = _build_config(args, input_path=input_path, output_path=output_path, temp_wav=temp_wav)
    runtime = _build_runtime(args)
    report = run_single_with_runtime(config, runtime, keep_wav=args.keep_wav)
    console.print(f"[green]Done[/green]: {report.output_path}")
    console.print(
        f"duration={report.audio_duration_sec:.1f}s speech_regions={report.speech_segment_count} "
        f"avg_speech={report.avg_speech_segment_sec:.2f}s short_speech={report.short_speech_segment_count} "
        f"pre_roll={report.avg_pre_roll_ms:.1f}ms post_roll={report.avg_post_roll_ms:.1f}ms"
    )
    console.print(
        f"rough_segments={report.rough_segment_count} "
        f"final_segments={report.segment_count} "
        f"avg_final={report.avg_final_segment_sec:.2f}s short_final={report.short_final_segment_ratio:.2%} "
        f"alignment={'applied' if report.alignment_applied else 'skipped'} "
        f"transcription_device={report.transcription_device}"
    )
    console.print(
        f"alignment_avg_shift={report.alignment_avg_abs_shift_ms:.1f}ms "
        f"alignment_stddev={report.alignment_onset_stddev_ms:.1f}ms "
        f"materially_changed={report.materially_changed_count} "
        f"alignment_skipped_units={report.alignment_skipped_unit_count} "
        f"base_tail={report.avg_base_end_extension_ms:.1f}ms "
        f"acoustic_tail={report.avg_acoustic_tail_extension_ms:.1f}ms "
        f"tail_extension={report.avg_end_extension_ms:.1f}ms "
        f"end_trim={report.avg_end_trim_ms:.1f}ms "
        f"end_shortened={report.materially_shortened_end_count} "
        f"tail_segments={report.acoustic_tail_extended_count} "
        f"tail_clamped={report.next_subtitle_clamp_count} "
        f"cleanup_clipped={report.clipped_segment_count} "
        f"cleanup_overlap_repairs={report.overlap_repair_count}"
    )
    if report.log_path:
        console.print(f"log={report.log_path}")
    if report.debug_export_path:
        console.print(f"debug_export={report.debug_export_path}")


def _run_batch(args: argparse.Namespace) -> None:
    input_dir = _validate_batch_input(args)
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_batch_inputs(input_dir, args.batch_glob)
    if not files:
        raise InputMediaError("처리할 미디어 파일이 없습니다.", detail="No media files found for batch mode.")

    configs: list[PipelineConfig] = []
    for src in files:
        rel = src.relative_to(input_dir)
        out_srt = (output_dir / rel).with_suffix(".srt")
        out_srt.parent.mkdir(parents=True, exist_ok=True)
        temp_wav = args.temp_wav.resolve() if args.temp_wav else Path(tempfile.gettempdir()) / f"{src.stem}.subgen.16k.wav"
        configs.append(_build_config(args, input_path=src, output_path=out_srt, temp_wav=temp_wav))

    runtime = _build_runtime(args)
    summary = run_batch_with_runtime(configs, runtime, keep_wav=args.keep_wav)
    console.print(
        f"[bold green]Batch Summary[/bold green]: total={len(summary.items)} ok={summary.succeeded} fail={summary.failed} "
        f"requested_workers={summary.requested_concurrency} effective_workers={summary.effective_concurrency}"
    )
    for item in summary.items:
        if item.success and item.report is not None:
            console.print(
                f"[green]OK[/green] {item.input_path.name} -> {item.output_path.name} "
                f"segments={item.report.segment_count} avg_shift={item.report.alignment_avg_abs_shift_ms:.1f}ms "
                f"stddev={item.report.alignment_onset_stddev_ms:.1f}ms"
            )
        else:
            console.print(
                f"[red]FAIL[/red] {item.input_path.name} "
                f"category={item.error_category} "
                f"message={item.user_message}"
            )
            if item.log_path is not None:
                console.print(f"  log={item.log_path}")


def main() -> None:
    argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_config(args, argv)
    _apply_preset(args, argv)

    try:
        if args.input_dir is not None:
            _run_batch(args)
        else:
            _run_single(args)
    except SubgenError as exc:
        console.print(f"[red]Failed[/red]: {exc.user_message}")
        console.print(f"[yellow]Detail[/yellow]: {exc.detail}")
        raise SystemExit(1) from exc
    except Exception as exc:
        console.print(f"[red]Failed[/red]: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
