from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from subgen.cli import PRESETS, _default_device
from subgen.config import AlignmentConfig, PipelineConfig, RuntimeConfig, TimingConfig, TranscriptionConfig, VADConfig
from subgen.runner import run_single_with_runtime


@dataclass(slots=True)
class ComparisonRow:
    preset: str
    alignment: str
    output_srt: str
    review_json: str | None
    rough_segments: int
    final_segments: int
    avg_segment_sec: float
    avg_gap_sec: float
    short_segment_ratio: float
    short_segment_count: int
    timing_correction_ms: int
    global_shift_ms: int
    alignment_applied: bool
    avg_alignment_signed_shift_ms: float
    avg_alignment_abs_shift_ms: float
    median_alignment_signed_shift_ms: float
    avg_final_signed_shift_ms: float
    median_final_signed_shift_ms: float
    final_shift_stddev_ms: float
    materially_changed_count: int
    materially_changed_ratio: float
    overlap_repairs: int
    clamp_operations: int
    avg_base_end_extension_ms: float
    avg_acoustic_tail_extension_ms: float
    avg_end_extension_ms: float
    avg_end_trim_ms: float
    materially_shortened_end_count: int
    overlap_end_trim_count: int
    acoustic_tail_extended_count: int
    next_subtitle_clamp_count: int
    alignment_skipped_units: int
    vad_tight_regions: int
    onset_distribution_p10_ms: float
    onset_distribution_p50_ms: float
    onset_distribution_p90_ms: float
    processing_sec: float
    log_path: str | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run timing-first preset comparisons on one Korean media file.")
    parser.add_argument("input", type=Path, help="Input media path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write comparison SRTs and reports")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["ko_sync_conservative", "ko_sync_balanced", "ko_sync_alignment_heavy", "ko_sync_final"],
        choices=sorted(PRESETS.keys()),
    )
    parser.add_argument("--alignment-modes", nargs="+", default=["on", "off"], choices=["on", "off"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default=_default_device(), help="ASR device")
    parser.add_argument("--align-device", choices=["cuda", "cpu"], default=_default_device(), help="Alignment device")
    parser.add_argument("--model", default=None, help="Override ASR model for all runs")
    parser.add_argument("--language", default="ko", help="Language code")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def _preset_value(preset_name: str, key: str, default: object) -> object:
    return PRESETS[preset_name].get(key, default)


def _make_config(input_path: Path, output_path: Path, preset_name: str, alignment_mode: str, device: str, align_device: str, model: str | None, language: str) -> PipelineConfig:
    temp_wav = Path(tempfile.gettempdir()) / f"{input_path.stem}.{preset_name}.{alignment_mode}.16k.wav"
    return PipelineConfig(
        input_path=input_path,
        output_path=output_path,
        temp_wav_path=temp_wav,
        sample_rate=16000,
        global_shift_ms=int(_preset_value(preset_name, "global_shift_ms", 0)),
        vad=VADConfig(
            threshold=float(_preset_value(preset_name, "vad_threshold", 0.5)),
            min_speech_ms=int(_preset_value(preset_name, "vad_min_speech_ms", 250)),
            min_silence_ms=int(_preset_value(preset_name, "vad_min_silence_ms", 120)),
            pad_ms=int(_preset_value(preset_name, "vad_pad_ms", 80)),
            pre_roll_ms=int(_preset_value(preset_name, "vad_pre_roll_ms", _preset_value(preset_name, "vad_pad_ms", 80))),
            post_roll_ms=int(_preset_value(preset_name, "vad_post_roll_ms", _preset_value(preset_name, "vad_pad_ms", 80))),
            merge_gap_ms=int(_preset_value(preset_name, "vad_merge_gap_ms", 140)),
        ),
        transcription=TranscriptionConfig(
            model_size=model or str(_preset_value(preset_name, "model", "large-v2")),
            device=device,
            compute_type="float16",
            beam_size=int(_preset_value(preset_name, "beam_size", 7)),
            language=language,
            overlap_sec=float(_preset_value(preset_name, "overlap_sec", 0.35)),
        ),
        alignment=AlignmentConfig(
            enabled=(alignment_mode == "on"),
            backend="whisperx",
            device=align_device,
            normalization_mode=str(_preset_value(preset_name, "alignment_normalization_mode", "conservative")),
            fallback_on_failure=True,
        ),
        timing=TimingConfig(
            min_duration_sec=float(_preset_value(preset_name, "min_segment_sec", 0.2)),
            hard_gap_sec=float(_preset_value(preset_name, "hard_gap_ms", 60)) / 1000.0,
            max_duration_sec=float(_preset_value(preset_name, "max_segment_sec", 4.0)),
            onset_nudge_ms=int(_preset_value(preset_name, "timing_correction_ms", 0)),
            end_tail_padding_ms=int(_preset_value(preset_name, "end_tail_padding_ms", 90)),
            max_end_tail_padding_ms=int(_preset_value(preset_name, "max_end_tail_padding_ms", 180)),
            enable_acoustic_tail_extension=bool(_preset_value(preset_name, "enable_acoustic_tail_extension", False)),
            acoustic_tail_probe_ms=int(_preset_value(preset_name, "acoustic_tail_probe_ms", 180)),
            max_acoustic_tail_extension_ms=int(_preset_value(preset_name, "max_acoustic_tail_extension_ms", 120)),
            min_tail_energy_threshold=float(_preset_value(preset_name, "min_tail_energy_threshold", 0.012)),
            min_gap_to_next_ms=int(_preset_value(preset_name, "min_gap_to_next_ms", 60)),
        ),
    )


def _output_name(input_path: Path, preset_name: str, alignment_mode: str) -> str:
    return f"{input_path.stem}.{preset_name}.align-{alignment_mode}.srt"


def _review_stem(input_path: Path, preset_name: str, alignment_mode: str) -> str:
    return f"{input_path.stem}.{preset_name}.align-{alignment_mode}"


def _row_from_report(preset_name: str, alignment_mode: str, report, config: PipelineConfig) -> ComparisonRow:
    distribution = report.onset_change_distribution_ms or {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    review_json = None
    if report.debug_export_path is not None:
        review_json = str(report.debug_export_path)
    return ComparisonRow(
        preset=preset_name,
        alignment=alignment_mode,
        output_srt=str(report.output_path),
        review_json=review_json,
        rough_segments=report.rough_segment_count,
        final_segments=report.segment_count,
        avg_segment_sec=report.avg_final_segment_sec,
        avg_gap_sec=report.avg_gap_sec,
        short_segment_ratio=report.short_final_segment_ratio,
        short_segment_count=report.short_final_segment_count,
        timing_correction_ms=config.timing.onset_nudge_ms,
        global_shift_ms=config.global_shift_ms,
        alignment_applied=report.alignment_applied,
        avg_alignment_signed_shift_ms=report.alignment_avg_signed_shift_ms,
        avg_alignment_abs_shift_ms=report.alignment_avg_abs_shift_ms,
        median_alignment_signed_shift_ms=report.alignment_median_signed_shift_ms,
        avg_final_signed_shift_ms=report.final_avg_signed_shift_ms,
        median_final_signed_shift_ms=report.final_median_signed_shift_ms,
        final_shift_stddev_ms=report.final_onset_stddev_ms,
        materially_changed_count=report.materially_changed_count,
        materially_changed_ratio=report.materially_changed_ratio,
        overlap_repairs=report.overlap_repair_count,
        clamp_operations=report.clamp_operation_count,
        avg_base_end_extension_ms=report.avg_base_end_extension_ms,
        avg_acoustic_tail_extension_ms=report.avg_acoustic_tail_extension_ms,
        avg_end_extension_ms=report.avg_end_extension_ms,
        avg_end_trim_ms=report.avg_end_trim_ms,
        materially_shortened_end_count=report.materially_shortened_end_count,
        overlap_end_trim_count=report.overlap_end_trim_count,
        acoustic_tail_extended_count=report.acoustic_tail_extended_count,
        next_subtitle_clamp_count=report.next_subtitle_clamp_count,
        alignment_skipped_units=report.alignment_skipped_unit_count,
        vad_tight_regions=report.vad_tight_region_count,
        onset_distribution_p10_ms=distribution["p10"],
        onset_distribution_p50_ms=distribution["p50"],
        onset_distribution_p90_ms=distribution["p90"],
        processing_sec=report.processing_sec,
        log_path=str(report.log_path) if report.log_path else None,
    )


def _write_per_run_review(output_dir: Path, stem: str, row: ComparisonRow) -> tuple[Path, Path]:
    json_path = output_dir / f"{stem}.review.json"
    md_path = output_dir / f"{stem}.review.md"
    json_path.write_text(json.dumps(asdict(row), ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = [
        f"# {stem}",
        "",
        f"- preset: `{row.preset}`",
        f"- alignment: `{row.alignment}`",
        f"- output_srt: `{Path(row.output_srt).name}`",
        f"- rough/final: `{row.rough_segments} -> {row.final_segments}`",
        f"- avg segment: `{row.avg_segment_sec:.2f}s`",
        f"- avg gap: `{row.avg_gap_sec:.2f}s`",
        f"- short segments: `{row.short_segment_count}` (`{row.short_segment_ratio:.2%}`)",
        f"- avg alignment signed shift: `{row.avg_alignment_signed_shift_ms:.1f}ms`",
        f"- median alignment signed shift: `{row.median_alignment_signed_shift_ms:.1f}ms`",
        f"- avg final signed shift: `{row.avg_final_signed_shift_ms:.1f}ms`",
        f"- final shift stddev: `{row.final_shift_stddev_ms:.1f}ms`",
        f"- materially changed: `{row.materially_changed_count}` (`{row.materially_changed_ratio:.2%}`)",
        f"- overlap repairs: `{row.overlap_repairs}`",
        f"- clamp operations: `{row.clamp_operations}`",
        f"- avg base end extension: `{row.avg_base_end_extension_ms:.1f}ms`",
        f"- avg acoustic tail extension: `{row.avg_acoustic_tail_extension_ms:.1f}ms`",
        f"- avg end extension: `{row.avg_end_extension_ms:.1f}ms`",
        f"- avg end trim: `{row.avg_end_trim_ms:.1f}ms`",
        f"- materially shortened ends: `{row.materially_shortened_end_count}`",
        f"- overlap end trims: `{row.overlap_end_trim_count}`",
        f"- acoustic tail extended segments: `{row.acoustic_tail_extended_count}`",
        f"- next subtitle clamps: `{row.next_subtitle_clamp_count}`",
        f"- alignment skipped units: `{row.alignment_skipped_units}`",
        f"- tight VAD regions: `{row.vad_tight_regions}`",
        f"- onset distribution: `p10={row.onset_distribution_p10_ms:.1f} p50={row.onset_distribution_p50_ms:.1f} p90={row.onset_distribution_p90_ms:.1f}`",
        f"- processing: `{row.processing_sec:.1f}s`",
    ]
    if row.log_path:
        md_lines.append(f"- log: `{Path(row.log_path).name}`")
    if row.review_json:
        md_lines.append(f"- debug export: `{Path(row.review_json).name}`")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def _write_reports(output_dir: Path, rows: list[ComparisonRow]) -> None:
    json_path = output_dir / "comparison-summary.json"
    md_path = output_dir / "comparison-summary.md"
    json_path.write_text(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Comparison Summary",
        "",
        "| preset | alignment | rough | final | avg seg (s) | avg gap (s) | short ratio | align avg signed (ms) | final avg signed (ms) | stddev (ms) | changed % | base tail (ms) | acoustic tail (ms) | end ext (ms) | end trim (ms) | shortened | tail segs | tail clamps | overlaps | clamps | skipped | tight vad | file |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.preset} | {row.alignment} | {row.rough_segments} | {row.final_segments} | "
            f"{row.avg_segment_sec:.2f} | {row.avg_gap_sec:.2f} | {row.short_segment_ratio:.2%} | "
            f"{row.avg_alignment_signed_shift_ms:.1f} | {row.avg_final_signed_shift_ms:.1f} | "
            f"{row.final_shift_stddev_ms:.1f} | {row.materially_changed_ratio:.2%} | {row.avg_base_end_extension_ms:.1f} | "
            f"{row.avg_acoustic_tail_extension_ms:.1f} | {row.avg_end_extension_ms:.1f} | {row.avg_end_trim_ms:.1f} | "
            f"{row.materially_shortened_end_count} | {row.acoustic_tail_extended_count} | {row.next_subtitle_clamp_count} | "
            f"{row.overlap_repairs} | {row.clamp_operations} | {row.alignment_skipped_units} | {row.vad_tight_regions} | `{Path(row.output_srt).name}` |"
        )
    lines.extend(
        [
            "",
            "## Review Guidance",
            "",
            "- Use `avg_final_signed_shift_ms` and `median_final_signed_shift_ms` as constant-offset indicators.",
            "- Use `final_shift_stddev_ms`, `materially_changed_ratio`, and onset distribution spread as sentence-level wobble indicators.",
            "- Review `alignment_skipped_units` first when sync feels unstable.",
            "- Review `vad_tight_regions` first when subtitles feel late at sentence starts.",
            "- Metrics are diagnostic only. Final judgment should still be by ear on the same player.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime = RuntimeConfig(
        log_level=args.log_level,
        log_dir=output_dir / "logs",
        batch_concurrency=1,
        debug_export_dir=output_dir / "debug",
    )

    rows: list[ComparisonRow] = []
    for preset_name in args.presets:
        for alignment_mode in args.alignment_modes:
            output_path = output_dir / _output_name(input_path, preset_name, alignment_mode)
            config = _make_config(
                input_path=input_path,
                output_path=output_path,
                preset_name=preset_name,
                alignment_mode=alignment_mode,
                device=args.device,
                align_device=args.align_device,
                model=args.model,
                language=args.language,
            )
            report = run_single_with_runtime(config, runtime, keep_wav=False)
            row = _row_from_report(preset_name, alignment_mode, report, config)
            rows.append(row)
            stem = _review_stem(input_path, preset_name, alignment_mode)
            _write_per_run_review(review_dir, stem, row)
            print(
                f"{preset_name} align={alignment_mode} final={row.final_segments} "
                f"final_shift={row.avg_final_signed_shift_ms:.1f}ms stddev={row.final_shift_stddev_ms:.1f}ms "
                f"changed={row.materially_changed_ratio:.2%} file={Path(row.output_srt).name}"
            )

    _write_reports(output_dir, rows)
    print(f"summary={output_dir / 'comparison-summary.md'}")


if __name__ == "__main__":
    main()
