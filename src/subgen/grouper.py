from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .types import AlignedTranscriptUnit, SubtitleSegment


@dataclass(slots=True)
class GroupingTrace:
    unit_id: int
    source_window_id: int
    source_region_id: int
    aligned_start_global: float
    aligned_end_global: float
    grouped_start_global: float
    grouped_end_global: float
    base_tail_padding_ms: float
    acoustic_tail_extension_ms: float
    total_end_extension_ms: float
    clamped_by_next: bool
    next_subtitle_start_global: float | None
    final_start_source: str
    final_end_source: str
    final_end_based_on_aligned_unit_id: int


@dataclass(slots=True)
class GroupingResult:
    segments: list[SubtitleSegment]
    traces: list[GroupingTrace]
    empty_unit_count: int
    avg_base_end_extension_ms: float
    avg_acoustic_tail_extension_ms: float
    avg_end_extension_ms: float
    materially_extended_count: int
    acoustic_tail_extended_count: int
    next_subtitle_clamp_count: int


class SubtitleGrouper:
    def group_units(
        self,
        units: list[AlignedTranscriptUnit],
        *,
        end_tail_padding_ms: int,
        max_end_tail_padding_ms: int,
        enable_acoustic_tail_extension: bool,
        acoustic_tail_probe_ms: int,
        max_acoustic_tail_extension_ms: int,
        min_tail_energy_threshold: float,
        min_gap_to_next_ms: int,
        audio: list[float] | None = None,
        sample_rate: int | None = None,
    ) -> GroupingResult:
        segments: list[SubtitleSegment] = []
        traces: list[GroupingTrace] = []
        empty_unit_count = 0
        base_end_extensions_ms: list[float] = []
        acoustic_end_extensions_ms: list[float] = []
        total_end_extensions_ms: list[float] = []
        materially_extended_count = 0
        acoustic_tail_extended_count = 0
        next_subtitle_clamp_count = 0
        padding_sec = max(0.0, end_tail_padding_ms / 1000.0)
        max_padding_sec = max(0.0, max_end_tail_padding_ms / 1000.0)
        acoustic_probe_sec = max(0.0, acoustic_tail_probe_ms / 1000.0)
        max_acoustic_sec = max(0.0, max_acoustic_tail_extension_ms / 1000.0)
        min_gap_sec = max(0.0, min_gap_to_next_ms / 1000.0)

        for index, unit in enumerate(units):
            if unit.end <= unit.start or not unit.display_text.strip():
                empty_unit_count += 1
                continue
            next_start = None
            for next_unit in units[index + 1:]:
                if next_unit.end > next_unit.start and next_unit.display_text.strip():
                    next_start = next_unit.start
                    break

            end = unit.end
            base_desired_end = min(end + padding_sec, end + max_padding_sec)
            acoustic_extension_sec = 0.0
            if enable_acoustic_tail_extension and audio is not None and sample_rate is not None:
                acoustic_extension_sec = self._detect_acoustic_tail_extension(
                    audio=audio,
                    sample_rate=sample_rate,
                    end_sec=end,
                    probe_sec=acoustic_probe_sec,
                    max_extension_sec=max_acoustic_sec,
                    min_tail_energy_threshold=min_tail_energy_threshold,
                )
            desired_end = min(base_desired_end + acoustic_extension_sec, end + max_padding_sec + max_acoustic_sec)
            if next_start is not None:
                clamped_end = min(desired_end, max(end, next_start - min_gap_sec))
                if clamped_end < desired_end:
                    next_subtitle_clamp_count += 1
                desired_end = clamped_end

            base_extension_ms = max(0.0, (min(base_desired_end, desired_end) - end) * 1000.0)
            acoustic_extension_ms = max(0.0, (desired_end - end) * 1000.0 - base_extension_ms)
            total_extension_ms = max(0.0, (desired_end - end) * 1000.0)
            base_end_extensions_ms.append(base_extension_ms)
            acoustic_end_extensions_ms.append(acoustic_extension_ms)
            total_end_extensions_ms.append(total_extension_ms)
            if acoustic_extension_ms >= 20.0:
                acoustic_tail_extended_count += 1
            if total_extension_ms >= 40.0:
                materially_extended_count += 1
            segments.append(
                SubtitleSegment(
                    start=unit.start,
                    end=desired_end,
                    text=unit.display_text,
                    source_unit_ids=(unit.unit_id,),
                )
            )
            traces.append(
                GroupingTrace(
                    unit_id=unit.unit_id,
                    source_window_id=unit.source_window_id,
                    source_region_id=unit.source_region_id,
                    aligned_start_global=unit.start,
                    aligned_end_global=end,
                    grouped_start_global=unit.start,
                    grouped_end_global=desired_end,
                    base_tail_padding_ms=base_extension_ms,
                    acoustic_tail_extension_ms=acoustic_extension_ms,
                    total_end_extension_ms=total_extension_ms,
                    clamped_by_next=(next_start is not None and desired_end < end + padding_sec + acoustic_extension_sec),
                    next_subtitle_start_global=next_start,
                    final_start_source="aligned_unit_start",
                    final_end_source="aligned_unit_end_plus_tail_policy",
                    final_end_based_on_aligned_unit_id=unit.unit_id,
                )
            )
        avg_base_end_extension_ms = sum(base_end_extensions_ms) / len(base_end_extensions_ms) if base_end_extensions_ms else 0.0
        avg_acoustic_tail_extension_ms = sum(acoustic_end_extensions_ms) / len(acoustic_end_extensions_ms) if acoustic_end_extensions_ms else 0.0
        avg_end_extension_ms = sum(total_end_extensions_ms) / len(total_end_extensions_ms) if total_end_extensions_ms else 0.0
        return GroupingResult(
            segments=segments,
            traces=traces,
            empty_unit_count=empty_unit_count,
            avg_base_end_extension_ms=avg_base_end_extension_ms,
            avg_acoustic_tail_extension_ms=avg_acoustic_tail_extension_ms,
            avg_end_extension_ms=avg_end_extension_ms,
            materially_extended_count=materially_extended_count,
            acoustic_tail_extended_count=acoustic_tail_extended_count,
            next_subtitle_clamp_count=next_subtitle_clamp_count,
        )

    def _detect_acoustic_tail_extension(
        self,
        *,
        audio: list[float],
        sample_rate: int,
        end_sec: float,
        probe_sec: float,
        max_extension_sec: float,
        min_tail_energy_threshold: float,
    ) -> float:
        if probe_sec <= 0.0 or max_extension_sec <= 0.0 or sample_rate <= 0:
            return 0.0
        start_index = max(0, min(len(audio), int(end_sec * sample_rate)))
        probe_end_index = max(start_index, min(len(audio), int((end_sec + probe_sec) * sample_rate)))
        if probe_end_index <= start_index:
            return 0.0

        frame_size = max(1, int(sample_rate * 0.02))
        max_extension_frames = max(1, int(max_extension_sec * sample_rate / frame_size))
        last_speech_end_index = start_index
        silence_run = 0
        speech_frames = 0

        for frame_index, frame_start in enumerate(range(start_index, probe_end_index, frame_size)):
            if frame_index >= max_extension_frames:
                break
            frame_end = min(probe_end_index, frame_start + frame_size)
            frame = audio[frame_start:frame_end]
            if not frame:
                break
            rms = sqrt(sum(sample * sample for sample in frame) / len(frame))
            if rms >= min_tail_energy_threshold:
                speech_frames += 1
                silence_run = 0
                last_speech_end_index = frame_end
            elif speech_frames == 0:
                continue
            else:
                silence_run += 1
                if silence_run >= 2:
                    break

        if speech_frames == 0 or last_speech_end_index <= start_index:
            return 0.0
        return max(0.0, (last_speech_end_index - start_index) / sample_rate)
