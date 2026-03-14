from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from statistics import median, pstdev
from time import perf_counter

from rich.console import Console

from .alignment import align_units
from .audio import extract_audio_to_wav
from .calibration import TimingCalibrator, _apply_global_shift
from .config import PipelineConfig
from .grouper import SubtitleGrouper
from .segmentation import build_transcription_windows
from .srt_writer import write_srt
from .transcriber import transcribe_windows
from .types import DraftTranscriptUnit, SubtitleSegment
from .vad import detect_speech_regions
from .wav_io import get_audio_duration_sec, load_mono_wav_float32


console = Console()


@dataclass(slots=True)
class PipelineReport:
    output_path: Path
    audio_duration_sec: float
    speech_segment_count: int
    avg_speech_segment_sec: float
    short_speech_segment_count: int
    avg_pre_roll_ms: float
    avg_post_roll_ms: float
    rough_segment_count: int
    segment_count: int
    avg_final_segment_sec: float
    avg_gap_sec: float
    short_final_segment_ratio: float
    short_final_segment_count: int
    transcription_device: str
    alignment_applied: bool
    alignment_warning: str | None = None
    transcription_warning: str | None = None
    processing_sec: float = 0.0
    stage_seconds: dict[str, float] | None = None
    log_path: Path | None = None
    debug_export_path: Path | None = None
    alignment_avg_abs_shift_ms: float = 0.0
    alignment_avg_signed_shift_ms: float = 0.0
    alignment_median_signed_shift_ms: float = 0.0
    alignment_onset_stddev_ms: float = 0.0
    materially_changed_count: int = 0
    materially_changed_ratio: float = 0.0
    alignment_skipped_unit_count: int = 0
    vad_tight_region_count: int = 0
    clipped_segment_count: int = 0
    clamp_operation_count: int = 0
    overlap_repair_count: int = 0
    global_shift_ms: int = 0
    avg_base_end_extension_ms: float = 0.0
    avg_acoustic_tail_extension_ms: float = 0.0
    avg_end_extension_ms: float = 0.0
    avg_end_trim_ms: float = 0.0
    materially_shortened_end_count: int = 0
    overlap_end_trim_count: int = 0
    acoustic_tail_extended_count: int = 0
    next_subtitle_clamp_count: int = 0
    final_avg_signed_shift_ms: float = 0.0
    final_median_signed_shift_ms: float = 0.0
    final_onset_stddev_ms: float = 0.0
    onset_change_distribution_ms: dict[str, float] | None = None


def run_pipeline(config: PipelineConfig, keep_wav: bool = False) -> Path:
    return run_pipeline_with_report(config, keep_wav=keep_wav).output_path


def _emit(logger: logging.Logger | None, message: str, *, console_message: str | None = None) -> None:
    if console_message is not None:
        console.print(console_message)
    if logger is not None:
        logger.info(message)


def run_pipeline_with_report(
    config: PipelineConfig,
    keep_wav: bool = False,
    logger: logging.Logger | None = None,
    debug_export_path: Path | None = None,
) -> PipelineReport:
    run_started = perf_counter()
    stage_seconds: dict[str, float] = {}

    stage_started = perf_counter()
    audio_info = extract_audio_to_wav(config.input_path, config.temp_wav_path, sample_rate=config.sample_rate)
    stage_seconds["extract_audio"] = perf_counter() - stage_started
    _emit(
        logger,
        f"Extracting audio: {config.input_path} command={' '.join(audio_info.command)}",
        console_message=f"[cyan]Extracting audio[/cyan]: {config.input_path}",
    )
    _emit(
        logger,
        f"Canonical audio properties: sr={audio_info.sample_rate} channels={audio_info.channels} sample_width={audio_info.sample_width_bytes} frames={audio_info.frame_count}",
    )

    stage_started = perf_counter()
    audio, sample_rate = load_mono_wav_float32(config.temp_wav_path)
    duration = get_audio_duration_sec(len(audio), sample_rate)
    stage_seconds["load_audio"] = perf_counter() - stage_started
    _emit(logger, f"Audio duration: {duration:.2f}s")

    stage_started = perf_counter()
    speech_regions = detect_speech_regions(str(config.temp_wav_path), config.vad, sample_rate=sample_rate)
    stage_seconds["vad"] = perf_counter() - stage_started
    speech_lengths = [region.duration for region in speech_regions]
    avg_speech_segment_sec = sum(speech_lengths) / len(speech_lengths) if speech_lengths else 0.0
    short_speech_segment_count = sum(1 for length in speech_lengths if length < 0.35)
    vad_tight_region_count = sum(
        1 for region in speech_regions
        if region.pre_roll_applied < (config.vad.pre_roll_ms * 0.5 / 1000.0)
        or region.post_roll_applied < (config.vad.post_roll_ms * 0.5 / 1000.0)
    )
    avg_pre_roll_ms = _avg([region.pre_roll_applied * 1000.0 for region in speech_regions])
    avg_post_roll_ms = _avg([region.post_roll_applied * 1000.0 for region in speech_regions])
    _emit(
        logger,
        "VAD diagnostics: "
        f"regions={len(speech_regions)} avg_region={avg_speech_segment_sec:.2f}s short_regions={short_speech_segment_count} "
        f"avg_pre_roll_ms={avg_pre_roll_ms:.1f} avg_post_roll_ms={avg_post_roll_ms:.1f}",
        console_message="[cyan]Running Silero VAD[/cyan]",
    )

    if not speech_regions:
        _emit(logger, "No speech detected. Writing empty SRT.", console_message="[yellow]No speech detected. Writing empty SRT.[/yellow]")
        write_srt([], config.output_path)
        if not keep_wav:
            _safe_unlink(config.temp_wav_path)
        return PipelineReport(
            output_path=config.output_path,
            audio_duration_sec=duration,
            speech_segment_count=0,
            avg_speech_segment_sec=0.0,
            short_speech_segment_count=0,
            avg_pre_roll_ms=0.0,
            avg_post_roll_ms=0.0,
            rough_segment_count=0,
            segment_count=0,
            avg_final_segment_sec=0.0,
            avg_gap_sec=0.0,
            short_final_segment_ratio=0.0,
            short_final_segment_count=0,
            transcription_device=config.transcription.device,
            alignment_applied=False,
            processing_sec=perf_counter() - run_started,
            stage_seconds=stage_seconds,
            global_shift_ms=config.global_shift_ms,
            vad_tight_region_count=0,
            avg_base_end_extension_ms=0.0,
            avg_acoustic_tail_extension_ms=0.0,
            avg_end_extension_ms=0.0,
            avg_end_trim_ms=0.0,
            materially_shortened_end_count=0,
            overlap_end_trim_count=0,
            acoustic_tail_extended_count=0,
            next_subtitle_clamp_count=0,
            onset_change_distribution_ms={"p10": 0.0, "p50": 0.0, "p90": 0.0},
        )

    stage_started = perf_counter()
    windows = build_transcription_windows(
        speech_regions=speech_regions,
        overlap_sec=config.transcription.overlap_sec,
        audio_duration_sec=duration,
    )
    stage_seconds["build_windows"] = perf_counter() - stage_started
    _emit(logger, f"Built {len(windows)} transcription windows")

    stage_started = perf_counter()
    draft_result = transcribe_windows(
        audio=audio,
        sample_rate=sample_rate,
        windows=windows,
        config=config.transcription,
        normalization_mode=config.alignment.normalization_mode,
        show_progress=True,
    )
    stage_seconds["transcribe"] = perf_counter() - stage_started
    draft_units = draft_result.units
    if draft_result.warning:
        _emit(logger, draft_result.warning)
    _emit(logger, f"Draft ASR units: {len(draft_units)}", console_message=f"[cyan]Transcribing[/cyan]: {len(windows)} VAD regions")

    stage_started = perf_counter()
    alignment_result = align_units(
        audio=audio,
        sample_rate=sample_rate,
        units=draft_units,
        language=(config.transcription.language or "ko").lower(),
        config=config.alignment,
    )
    stage_seconds["alignment"] = perf_counter() - stage_started
    if config.alignment.enabled:
        _emit(logger, "Running timing alignment", console_message="[cyan]Running CTC alignment[/cyan]")
    if alignment_result.warning:
        _emit(logger, alignment_result.warning, console_message=f"[yellow]{alignment_result.warning}[/yellow]")

    stage_started = perf_counter()
    grouping_result = SubtitleGrouper().group_units(
        alignment_result.units,
        end_tail_padding_ms=config.timing.end_tail_padding_ms,
        max_end_tail_padding_ms=config.timing.max_end_tail_padding_ms,
        enable_acoustic_tail_extension=config.timing.enable_acoustic_tail_extension,
        acoustic_tail_probe_ms=config.timing.acoustic_tail_probe_ms,
        max_acoustic_tail_extension_ms=config.timing.max_acoustic_tail_extension_ms,
        min_tail_energy_threshold=config.timing.min_tail_energy_threshold,
        min_gap_to_next_ms=config.timing.min_gap_to_next_ms,
        audio=audio.tolist(),
        sample_rate=sample_rate,
    )
    stage_seconds["grouping"] = perf_counter() - stage_started

    stage_started = perf_counter()
    calibration_result = TimingCalibrator().cleanup(
        grouping_result.segments,
        min_duration_sec=config.timing.min_duration_sec,
        hard_gap_sec=config.timing.hard_gap_sec,
        global_shift_ms=config.global_shift_ms,
    )
    stage_seconds["calibration"] = perf_counter() - stage_started

    final_segments = calibration_result.segments
    overlap_count = _count_overlaps(final_segments)
    final_lengths = [segment.end - segment.start for segment in final_segments]
    avg_final_segment_sec = _avg(final_lengths)
    short_final_segment_count = sum(1 for length in final_lengths if length < 1.0)
    short_final_segment_ratio = (
        short_final_segment_count / len(final_lengths) if final_lengths else 0.0
    )
    avg_gap_sec = _avg(_subtitle_gaps(final_segments))
    final_signed_shifts_ms = _final_signed_shifts_ms(draft_units, final_segments)

    stage_started = perf_counter()
    write_srt(final_segments, config.output_path)
    stage_seconds["write_srt"] = perf_counter() - stage_started
    _emit(logger, f"Wrote SRT: {config.output_path}", console_message=f"[green]Wrote SRT[/green]: {config.output_path}")

    if debug_export_path is not None:
        _write_debug_export(
            debug_export_path,
            speech_regions=speech_regions,
            windows=windows,
            draft_units=draft_units,
            aligned_units=alignment_result.units,
            grouped_segments=grouping_result.segments,
            grouping_traces=grouping_result.traces,
            final_segments=final_segments,
            calibration_traces=calibration_result.traces,
            global_shift_ms=config.global_shift_ms,
            overlap_repair_count=calibration_result.overlap_repair_count,
            end_tail_padding_ms=config.timing.end_tail_padding_ms,
            max_end_tail_padding_ms=config.timing.max_end_tail_padding_ms,
            enable_acoustic_tail_extension=config.timing.enable_acoustic_tail_extension,
            acoustic_tail_probe_ms=config.timing.acoustic_tail_probe_ms,
            max_acoustic_tail_extension_ms=config.timing.max_acoustic_tail_extension_ms,
            min_tail_energy_threshold=config.timing.min_tail_energy_threshold,
            min_gap_to_next_ms=config.timing.min_gap_to_next_ms,
            avg_base_end_extension_ms=grouping_result.avg_base_end_extension_ms,
            avg_acoustic_tail_extension_ms=grouping_result.avg_acoustic_tail_extension_ms,
            avg_end_extension_ms=grouping_result.avg_end_extension_ms,
            avg_end_trim_ms=calibration_result.avg_end_trim_ms,
            materially_shortened_end_count=calibration_result.materially_shortened_end_count,
            overlap_end_trim_count=calibration_result.overlap_end_trim_count,
            acoustic_tail_extended_count=grouping_result.acoustic_tail_extended_count,
            next_subtitle_clamp_count=grouping_result.next_subtitle_clamp_count,
        )
        _emit(logger, f"Debug export: {debug_export_path}")

    if not keep_wav:
        _safe_unlink(config.temp_wav_path)

    return PipelineReport(
        output_path=config.output_path,
        audio_duration_sec=duration,
        speech_segment_count=len(speech_regions),
        avg_speech_segment_sec=avg_speech_segment_sec,
        short_speech_segment_count=short_speech_segment_count,
        avg_pre_roll_ms=avg_pre_roll_ms,
        avg_post_roll_ms=avg_post_roll_ms,
        rough_segment_count=len(draft_units),
        segment_count=len(final_segments),
        avg_final_segment_sec=avg_final_segment_sec,
        avg_gap_sec=avg_gap_sec,
        short_final_segment_ratio=short_final_segment_ratio,
        short_final_segment_count=short_final_segment_count,
        transcription_device=draft_result.device,
        alignment_applied=alignment_result.applied,
        alignment_warning=alignment_result.warning,
        transcription_warning=draft_result.warning,
        processing_sec=perf_counter() - run_started,
        stage_seconds=stage_seconds,
        debug_export_path=debug_export_path,
        alignment_avg_abs_shift_ms=alignment_result.avg_abs_shift_ms,
        alignment_avg_signed_shift_ms=alignment_result.avg_signed_shift_ms,
        alignment_median_signed_shift_ms=alignment_result.median_signed_shift_ms,
        alignment_onset_stddev_ms=alignment_result.onset_shift_stddev_ms,
        materially_changed_count=alignment_result.materially_changed_count,
        materially_changed_ratio=(alignment_result.materially_changed_count / len(draft_units)) if draft_units else 0.0,
        alignment_skipped_unit_count=alignment_result.skipped_unit_count,
        vad_tight_region_count=vad_tight_region_count,
        clipped_segment_count=calibration_result.clipped_segment_count,
        clamp_operation_count=calibration_result.clamp_operation_count,
        overlap_repair_count=calibration_result.overlap_repair_count,
        global_shift_ms=config.global_shift_ms,
        avg_base_end_extension_ms=grouping_result.avg_base_end_extension_ms,
        avg_acoustic_tail_extension_ms=grouping_result.avg_acoustic_tail_extension_ms,
        avg_end_extension_ms=grouping_result.avg_end_extension_ms,
        avg_end_trim_ms=calibration_result.avg_end_trim_ms,
        materially_shortened_end_count=calibration_result.materially_shortened_end_count,
        overlap_end_trim_count=calibration_result.overlap_end_trim_count,
        acoustic_tail_extended_count=grouping_result.acoustic_tail_extended_count,
        next_subtitle_clamp_count=grouping_result.next_subtitle_clamp_count,
        final_avg_signed_shift_ms=_avg(final_signed_shifts_ms),
        final_median_signed_shift_ms=(median(final_signed_shifts_ms) if final_signed_shifts_ms else 0.0),
        final_onset_stddev_ms=(pstdev(final_signed_shifts_ms) if len(final_signed_shifts_ms) > 1 else 0.0),
        onset_change_distribution_ms=_distribution(final_signed_shifts_ms),
    )


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _count_overlaps(segments: list[SubtitleSegment]) -> int:
    if len(segments) < 2:
        return 0
    overlaps = 0
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    for index in range(1, len(ordered)):
        if ordered[index].start < ordered[index - 1].end:
            overlaps += 1
    return overlaps


def _subtitle_gaps(segments: list[SubtitleSegment]) -> list[float]:
    if len(segments) < 2:
        return []
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    return [max(0.0, ordered[index].start - ordered[index - 1].end) for index in range(1, len(ordered))]


def _final_signed_shifts_ms(draft_units: list[DraftTranscriptUnit], final_segments: list[SubtitleSegment]) -> list[float]:
    draft_by_id = {unit.unit_id: unit for unit in draft_units}
    shifts: list[float] = []
    for segment in final_segments:
        if not segment.source_unit_ids:
            continue
        first = draft_by_id.get(segment.source_unit_ids[0])
        if first is None:
            continue
        shifts.append((segment.start - first.rough_start) * 1000.0)
    return shifts


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    ordered = sorted(values)
    return {
        "p10": _percentile(ordered, 0.10),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
    }


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * ratio))))
    return sorted_values[index]


def _compute_onset_error(rough: list[SubtitleSegment], aligned: list[SubtitleSegment]) -> tuple[float, int]:
    n = min(len(rough), len(aligned))
    if n == 0:
        return 0.0, 0
    diffs = [abs(aligned[index].start - rough[index].start) * 1000.0 for index in range(n)]
    return (sum(diffs) / len(diffs), len(diffs))


def _write_debug_export(
    output_path: Path,
    *,
    speech_regions,
    windows,
    draft_units: list[DraftTranscriptUnit],
    aligned_units,
    grouped_segments: list[SubtitleSegment],
    grouping_traces,
    final_segments: list[SubtitleSegment],
    calibration_traces,
    global_shift_ms: int,
    overlap_repair_count: int,
    end_tail_padding_ms: int,
    max_end_tail_padding_ms: int,
    enable_acoustic_tail_extension: bool,
    acoustic_tail_probe_ms: int,
    max_acoustic_tail_extension_ms: int,
    min_tail_energy_threshold: float,
    min_gap_to_next_ms: int,
    avg_base_end_extension_ms: float,
    avg_acoustic_tail_extension_ms: float,
    avg_end_extension_ms: float,
    avg_end_trim_ms: float,
    materially_shortened_end_count: int,
    overlap_end_trim_count: int,
    acoustic_tail_extended_count: int,
    next_subtitle_clamp_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "speech_regions": [region.to_dict() for region in speech_regions],
        "windows": [
            {
                "window_id": window.window_id,
                "region_id": window.region.region_id,
                "window": window.window.to_dict(),
            }
            for window in windows
        ],
        "draft_units": [unit.to_dict() for unit in draft_units],
        "aligned_units": [unit.to_dict() for unit in aligned_units],
        "grouped_segments": [segment.to_dict() for segment in grouped_segments],
        "final_segments": [segment.to_dict() for segment in final_segments],
        "global_shift_ms": global_shift_ms,
        "overlap_repair_count": overlap_repair_count,
        "tail_padding_policy": {
            "end_tail_padding_ms": end_tail_padding_ms,
            "max_end_tail_padding_ms": max_end_tail_padding_ms,
            "enable_acoustic_tail_extension": enable_acoustic_tail_extension,
            "acoustic_tail_probe_ms": acoustic_tail_probe_ms,
            "max_acoustic_tail_extension_ms": max_acoustic_tail_extension_ms,
            "min_tail_energy_threshold": min_tail_energy_threshold,
            "min_gap_to_next_ms": min_gap_to_next_ms,
        },
        "end_timing_diagnostics": {
            "avg_base_end_extension_ms": avg_base_end_extension_ms,
            "avg_acoustic_tail_extension_ms": avg_acoustic_tail_extension_ms,
            "avg_end_extension_ms": avg_end_extension_ms,
            "avg_end_trim_ms": avg_end_trim_ms,
            "materially_shortened_end_count": materially_shortened_end_count,
            "overlap_end_trim_count": overlap_end_trim_count,
            "acoustic_tail_extended_count": acoustic_tail_extended_count,
            "next_subtitle_clamp_count": next_subtitle_clamp_count,
        },
        "vad_tight_region_count": sum(
            1 for region in speech_regions
            if region.pre_roll_applied == 0.0 or region.post_roll_applied == 0.0
        ),
        "alignment_shift_stats": {
            "avg_abs_shift_ms": _avg([abs(unit.start - unit.rough_start) * 1000.0 for unit in aligned_units]),
            "avg_signed_shift_ms": _avg([(unit.start - unit.rough_start) * 1000.0 for unit in aligned_units]),
            "stddev_shift_ms": pstdev([abs(unit.start - unit.rough_start) * 1000.0 for unit in aligned_units]) if len(aligned_units) > 1 else 0.0,
            "distribution_ms": _distribution([(unit.start - unit.rough_start) * 1000.0 for unit in aligned_units]),
        },
        "segment_provenance": _build_segment_provenance(
            speech_regions=speech_regions,
            windows=windows,
            draft_units=draft_units,
            aligned_units=aligned_units,
            grouping_traces=grouping_traces,
            calibration_traces=calibration_traces,
        ),
        "timing_integrity": _build_timing_integrity_report(
            draft_units=draft_units,
            aligned_units=aligned_units,
            grouping_traces=grouping_traces,
            calibration_traces=calibration_traces,
        ),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_segment_provenance(
    *,
    speech_regions,
    windows,
    draft_units: list[DraftTranscriptUnit],
    aligned_units,
    grouping_traces,
    calibration_traces,
) -> list[dict[str, object]]:
    region_by_id = {region.region_id: region for region in speech_regions}
    window_by_id = {window.window_id: window for window in windows}
    draft_by_id = {unit.unit_id: unit for unit in draft_units}
    aligned_by_id = {unit.unit_id: unit for unit in aligned_units}
    calibration_by_unit = {
        trace.source_unit_ids[0]: trace for trace in calibration_traces if trace.source_unit_ids
    }
    rows: list[dict[str, object]] = []
    for trace in grouping_traces:
        unit_id = trace.unit_id
        draft = draft_by_id.get(unit_id)
        aligned = aligned_by_id.get(unit_id)
        calibration = calibration_by_unit.get(unit_id)
        window = window_by_id.get(trace.source_window_id)
        region = region_by_id.get(trace.source_region_id)
        if draft is None or aligned is None or window is None or region is None or calibration is None:
            continue
        last_aligned_end = aligned.end
        rows.append(
            {
                "unit_id": unit_id,
                "text": draft.display_text,
                "text_length": len(draft.display_text),
                "source_speech_region": {
                    "local_start": 0.0,
                    "local_end": region.duration,
                    "global_start": region.start,
                    "global_end": region.end,
                    "raw_global_start": region.raw_start,
                    "raw_global_end": region.raw_end,
                    "speech_region_offset_applied_ms": region.start * 1000.0,
                },
                "source_transcription_window": {
                    "local_start": 0.0,
                    "local_end": window.window.duration,
                    "global_start": window.window.start,
                    "global_end": window.window.end,
                    "window_offset_applied_ms": window.window.start * 1000.0,
                },
                "draft_unit": {
                    "local_start": draft.window_local_start,
                    "local_end": draft.window_local_end,
                    "global_start_unclamped": draft.window_global_start,
                    "global_end_unclamped": draft.window_global_end,
                    "global_start": draft.rough_start,
                    "global_end": draft.rough_end,
                    "draft_end_clipped_before_alignment": draft.rough_end < draft.window_global_end - 0.001,
                },
                "aligned_unit": {
                    "local_start_to_window": aligned.start - window.window.start,
                    "local_end_to_window": aligned.end - window.window.start,
                    "local_start_to_region": aligned.start - region.start,
                    "local_end_to_region": aligned.end - region.start,
                    "global_start": aligned.start,
                    "global_end": aligned.end,
                    "alignment_applied": aligned.alignment_applied,
                    "fallback_reason": aligned.fallback_reason,
                },
                "grouped_subtitle": {
                    "global_start": trace.grouped_start_global,
                    "global_end": trace.grouped_end_global,
                    "final_start_source": trace.final_start_source,
                    "final_end_source": trace.final_end_source,
                    "final_end_based_on_aligned_unit_id": trace.final_end_based_on_aligned_unit_id,
                    "base_tail_padding_ms": trace.base_tail_padding_ms,
                    "acoustic_tail_extension_ms": trace.acoustic_tail_extension_ms,
                    "total_end_extension_ms": trace.total_end_extension_ms,
                    "clamped_by_next_subtitle": trace.clamped_by_next,
                    "next_subtitle_start_global": trace.next_subtitle_start_global,
                },
                "final_cleaned_subtitle": {
                    "global_start": calibration.final_start,
                    "global_end": calibration.final_end,
                    "cleanup_start_override_applied": calibration.cleanup_start_override_applied,
                    "cleanup_end_override_applied": calibration.cleanup_end_override_applied,
                    "cleanup_end_trim_ms": calibration.cleanup_end_trim_ms,
                },
                "source_summary": {
                    "start_source_before_cleanup": trace.final_start_source,
                    "start_source_after_cleanup": "aligned_unit_start" if not calibration.cleanup_start_override_applied else "cleanup_override",
                    "end_source_before_cleanup": trace.final_end_source,
                    "end_source_after_cleanup": "cleanup_override" if calibration.cleanup_end_override_applied else trace.final_end_source,
                    "start_changed_by_grouping": abs(trace.grouped_start_global - aligned.start) > 0.001,
                    "end_changed_by_grouping": abs(trace.grouped_end_global - aligned.end) > 0.001,
                    "start_changed_by_cleanup": abs(calibration.final_start - trace.grouped_start_global) > 0.001,
                    "end_changed_by_cleanup": abs(calibration.final_end - trace.grouped_end_global) > 0.001,
                },
                "checks": {
                    "grouped_end_before_last_aligned_end": trace.grouped_end_global < last_aligned_end - 0.01,
                    "final_end_before_grouped_end": calibration.final_end < trace.grouped_end_global - 0.01,
                    "final_end_before_last_aligned_end": calibration.final_end < last_aligned_end - 0.01,
                    "long_utterance_candidate": (len(draft.display_text) >= 18) or ((draft.rough_end - draft.rough_start) >= 2.5),
                    "end_compression_ms": max(0.0, (trace.grouped_end_global - calibration.final_end) * 1000.0),
                },
            }
        )
    return rows


def _build_timing_integrity_report(
    *,
    draft_units: list[DraftTranscriptUnit],
    aligned_units,
    grouping_traces,
    calibration_traces,
) -> dict[str, object]:
    draft_by_id = {unit.unit_id: unit for unit in draft_units}
    aligned_by_id = {unit.unit_id: unit for unit in aligned_units}
    grouped_before_aligned = 0
    cleanup_end_shortened = 0
    clipped_before_alignment = 0
    long_utterance_compression = 0
    non_monotonic_global_conversion = 0

    for draft in draft_units:
        if draft.rough_end < draft.window_global_end - 0.001:
            clipped_before_alignment += 1
        if draft.window_global_end < draft.window_global_start or draft.rough_end < draft.rough_start:
            non_monotonic_global_conversion += 1

    for trace in grouping_traces:
        aligned = aligned_by_id.get(trace.unit_id)
        draft = draft_by_id.get(trace.unit_id)
        if aligned is None or draft is None:
            continue
        if trace.grouped_end_global < aligned.end - 0.01:
            grouped_before_aligned += 1
        if ((len(draft.display_text) >= 18) or ((draft.rough_end - draft.rough_start) >= 2.5)) and (trace.total_end_extension_ms < 60.0):
            long_utterance_compression += 1

    for trace in calibration_traces:
        if trace.cleanup_end_override_applied:
            cleanup_end_shortened += 1

    return {
        "aligned_end_before_aligned_start_count": sum(1 for unit in aligned_units if unit.end < unit.start),
        "draft_end_clipped_before_alignment_count": clipped_before_alignment,
        "grouped_end_before_last_aligned_end_count": grouped_before_aligned,
        "cleanup_end_override_count": cleanup_end_shortened,
        "cleanup_start_override_count": sum(1 for trace in calibration_traces if trace.cleanup_start_override_applied),
        "non_monotonic_local_to_global_count": non_monotonic_global_conversion,
        "long_utterance_low_tail_extension_count": long_utterance_compression,
    }
