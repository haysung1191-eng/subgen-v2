from __future__ import annotations

from dataclasses import dataclass

from .types import SubtitleSegment


@dataclass(slots=True)
class CalibrationTrace:
    source_unit_ids: tuple[int, ...]
    grouped_start: float
    grouped_end: float
    final_start: float
    final_end: float
    cleanup_start_override_applied: bool
    cleanup_end_override_applied: bool
    cleanup_end_trim_ms: float
    overlap_repair_applied: bool


@dataclass(slots=True)
class CalibrationResult:
    segments: list[SubtitleSegment]
    traces: list[CalibrationTrace]
    clipped_segment_count: int
    overlap_repair_count: int
    clamp_operation_count: int
    global_shift_ms: int
    avg_end_trim_ms: float
    materially_shortened_end_count: int
    overlap_end_trim_count: int


class TimingCalibrator:
    def cleanup(
        self,
        segments: list[SubtitleSegment],
        *,
        min_duration_sec: float,
        hard_gap_sec: float,
        global_shift_ms: int,
    ) -> CalibrationResult:
        if not segments:
            return CalibrationResult(
                segments=[],
                traces=[],
                clipped_segment_count=0,
                overlap_repair_count=0,
                clamp_operation_count=0,
                global_shift_ms=global_shift_ms,
                avg_end_trim_ms=0.0,
                materially_shortened_end_count=0,
                overlap_end_trim_count=0,
            )

        ordered = sorted(segments, key=lambda item: (item.start, item.end))
        clean: list[SubtitleSegment] = []
        traces: list[CalibrationTrace] = []
        clipped_count = 0
        overlap_repair_count = 0
        clamp_operation_count = 0
        overlap_end_trim_count = 0
        end_trim_ms: list[float] = []
        materially_shortened_end_count = 0

        for seg in ordered:
            original_start = seg.start
            original_end = seg.end
            start = seg.start
            end = seg.end
            if end <= start:
                clipped_count += 1
                continue

            if end - start < min_duration_sec:
                end = start + min_duration_sec
                clamp_operation_count += 1

            cleanup_end_override_applied = False
            cleanup_start_override_applied = False
            cleanup_end_trim_ms = 0.0
            if clean and start < clean[-1].end + hard_gap_sec:
                previous = clean[-1]
                allowed_previous_end = max(previous.start, start - hard_gap_sec)
                if allowed_previous_end < previous.end:
                    trim_ms = (previous.end - allowed_previous_end) * 1000.0
                    end_trim_ms.append(trim_ms)
                    if trim_ms >= 40.0:
                        materially_shortened_end_count += 1
                    overlap_end_trim_count += 1
                    clean[-1] = SubtitleSegment(
                        start=previous.start,
                        end=allowed_previous_end,
                        text=previous.text,
                        source_unit_ids=previous.source_unit_ids,
                    )
                    if traces:
                        traces[-1].final_end = allowed_previous_end
                        traces[-1].cleanup_end_override_applied = True
                        traces[-1].cleanup_end_trim_ms += trim_ms
                        traces[-1].overlap_repair_applied = True
                overlap_repair_count += 1

            if end <= start:
                end = start + min_duration_sec
                clipped_count += 1
                clamp_operation_count += 1
                cleanup_end_override_applied = True

            clean.append(SubtitleSegment(start=start, end=end, text=seg.text, source_unit_ids=seg.source_unit_ids))
            traces.append(
                CalibrationTrace(
                    source_unit_ids=seg.source_unit_ids,
                    grouped_start=original_start,
                    grouped_end=original_end,
                    final_start=start,
                    final_end=end,
                    cleanup_start_override_applied=cleanup_start_override_applied,
                    cleanup_end_override_applied=cleanup_end_override_applied,
                    cleanup_end_trim_ms=cleanup_end_trim_ms,
                    overlap_repair_applied=(cleanup_start_override_applied or cleanup_end_override_applied),
                )
            )

        shifted = _apply_global_shift(clean, global_shift_ms)
        shifted_traces: list[CalibrationTrace] = []
        shift_sec = global_shift_ms / 1000.0
        for trace, shifted_segment in zip(traces, shifted):
            shifted_traces.append(
                CalibrationTrace(
                    source_unit_ids=trace.source_unit_ids,
                    grouped_start=trace.grouped_start,
                    grouped_end=trace.grouped_end,
                    final_start=max(0.0, trace.final_start + shift_sec),
                    final_end=shifted_segment.end,
                    cleanup_start_override_applied=trace.cleanup_start_override_applied,
                    cleanup_end_override_applied=trace.cleanup_end_override_applied,
                    cleanup_end_trim_ms=trace.cleanup_end_trim_ms,
                    overlap_repair_applied=trace.overlap_repair_applied,
                )
            )
        return CalibrationResult(
            segments=shifted,
            traces=shifted_traces,
            clipped_segment_count=clipped_count,
            overlap_repair_count=overlap_repair_count,
            clamp_operation_count=clamp_operation_count,
            global_shift_ms=global_shift_ms,
            avg_end_trim_ms=(sum(end_trim_ms) / len(end_trim_ms) if end_trim_ms else 0.0),
            materially_shortened_end_count=materially_shortened_end_count,
            overlap_end_trim_count=overlap_end_trim_count,
        )


def _apply_global_shift(segments: list[SubtitleSegment], shift_ms: int) -> list[SubtitleSegment]:
    if shift_ms == 0:
        return segments
    shift_sec = shift_ms / 1000.0
    shifted: list[SubtitleSegment] = []
    for seg in segments:
        start = max(0.0, seg.start + shift_sec)
        end = max(start, seg.end + shift_sec)
        shifted.append(SubtitleSegment(start=start, end=end, text=seg.text, source_unit_ids=seg.source_unit_ids))
    return shifted
