from __future__ import annotations

from dataclasses import dataclass

from .config import SubtitleConfig
from .align import tokens_by_utterance
from .types import AlignedToken, DraftUtterance, SubtitleSegment


@dataclass(slots=True)
class AssemblyResult:
    raw_segments: list[SubtitleSegment]
    final_segments: list[SubtitleSegment]
    summary: dict[str, int | float | str]


def build_subtitles(
    utterances: list[DraftUtterance],
    tokens: list[AlignedToken],
    config: SubtitleConfig,
) -> AssemblyResult:
    grouped_tokens = tokens_by_utterance(tokens)
    raw_segments: list[SubtitleSegment] = []
    hold_sec = config.hold_ms / 1000.0
    min_gap_sec = config.min_gap_to_next_ms / 1000.0

    fallback_threshold_sec = config.end_fallback_threshold_ms / 1000.0
    planned: list[SubtitleSegment] = []
    for utterance in utterances:
        utterance_tokens = grouped_tokens.get(utterance.utterance_id, [])
        if utterance_tokens:
            start = utterance_tokens[0].global_start
            token_end = utterance_tokens[-1].global_end
            end_anchor = token_end
            end_source = "aligned_tokens_plus_hold"
            end_fallback_applied = False
            timing_authority = "aligned_tokens"
            gap_to_draft_end = max(0.0, utterance.global_end - token_end)
            if gap_to_draft_end >= fallback_threshold_sec:
                end_anchor = max(end_anchor, utterance.global_end)
                end_source = "draft_end_fallback_plus_hold"
                end_fallback_applied = True
                timing_authority = "aligned_start_draft_end_fallback"
        else:
            start = utterance.global_start
            token_end = utterance.global_end
            end_anchor = utterance.global_end
            end_source = "draft_fallback_plus_hold"
            end_fallback_applied = True
            timing_authority = "draft_fallback"
            gap_to_draft_end = 0.0

        end = end_anchor + hold_sec
        planned.append(
            SubtitleSegment(
                segment_id=len(planned),
                utterance_id=utterance.utterance_id,
                region_id=utterance.region_id,
                text=utterance.display_text,
                start=start,
                end=end,
                token_start=start,
                token_end=token_end,
                draft_start=utterance.global_start,
                draft_end=utterance.global_end,
                aligned_token_count=len(utterance_tokens),
                start_source=("aligned_tokens" if utterance_tokens else "draft_fallback"),
                end_source=end_source,
                end_fallback_applied=end_fallback_applied,
                end_gap_ms=gap_to_draft_end * 1000.0,
                timing_authority=timing_authority,
            )
        )

    for index, segment in enumerate(planned):
        end = segment.end
        next_start = None
        if index + 1 < len(planned):
            next_start = planned[index + 1].start
            end = min(end, max(segment.token_end, next_start - min_gap_sec))
        raw_segments.append(
            SubtitleSegment(
                segment_id=segment.segment_id,
                utterance_id=segment.utterance_id,
                region_id=segment.region_id,
                text=segment.text,
                start=segment.start,
                end=end,
                token_start=segment.token_start,
                token_end=segment.token_end,
                draft_start=segment.draft_start,
                draft_end=segment.draft_end,
                aligned_token_count=segment.aligned_token_count,
                start_source=segment.start_source,
                end_source=segment.end_source,
                end_fallback_applied=segment.end_fallback_applied,
                end_gap_ms=segment.end_gap_ms,
                timing_authority=segment.timing_authority,
            )
        )
    final_segments = _cleanup(raw_segments, config)
    return AssemblyResult(
        raw_segments=raw_segments,
        final_segments=final_segments,
        summary=_summarize(final_segments),
    )


def _summarize(segments: list[SubtitleSegment]) -> dict[str, int | float | str]:
    aligned_start_count = sum(1 for item in segments if item.start_source == "aligned_tokens")
    draft_start_fallback_count = sum(1 for item in segments if item.start_source == "draft_fallback")
    draft_end_fallback_count = sum(1 for item in segments if item.end_fallback_applied)
    subtitles_with_zero_aligned_tokens = sum(1 for item in segments if item.aligned_token_count == 0)
    total = len(segments)
    if total == 0:
        authority = "none"
    elif draft_start_fallback_count == total:
        authority = "draft_fallback"
    elif draft_start_fallback_count > 0 or draft_end_fallback_count > 0:
        authority = "mixed"
    else:
        authority = "aligned_tokens"
    end_gaps = sorted(item.end_gap_ms for item in segments)
    median_end_gap = _median(end_gaps)
    max_end_gap = max(end_gaps) if end_gaps else 0.0
    return {
        "timing_authority_summary": authority,
        "subtitle_count": total,
        "aligned_start_count": aligned_start_count,
        "draft_start_fallback_count": draft_start_fallback_count,
        "draft_end_fallback_count": draft_end_fallback_count,
        "subtitles_with_zero_aligned_tokens": subtitles_with_zero_aligned_tokens,
        "median_end_gap_ms": median_end_gap,
        "max_end_gap_ms": max_end_gap,
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _cleanup(segments: list[SubtitleSegment], config: SubtitleConfig) -> list[SubtitleSegment]:
    if not segments:
        return []
    min_duration_sec = config.min_duration_ms / 1000.0
    tiny_overlap_fix_sec = config.tiny_overlap_fix_ms / 1000.0
    clean: list[SubtitleSegment] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        start = segment.start
        end = segment.end
        if end <= start:
            end = start + min_duration_sec
        if end - start < min_duration_sec:
            end = start + min_duration_sec
        if clean and start < clean[-1].end:
            prev = clean[-1]
            prev_end = max(prev.token_start, start - tiny_overlap_fix_sec)
            prev_end = max(prev.start + 0.001, prev_end)
            clean[-1] = SubtitleSegment(
                segment_id=prev.segment_id,
                utterance_id=prev.utterance_id,
                region_id=prev.region_id,
                text=prev.text,
                start=prev.start,
                end=prev_end,
                token_start=prev.token_start,
                token_end=prev.token_end,
                draft_start=prev.draft_start,
                draft_end=prev.draft_end,
                aligned_token_count=prev.aligned_token_count,
                start_source=prev.start_source,
                end_source=prev.end_source,
                end_fallback_applied=prev.end_fallback_applied,
                end_gap_ms=prev.end_gap_ms,
                timing_authority=prev.timing_authority,
            )
        clean.append(
            SubtitleSegment(
                segment_id=segment.segment_id,
                utterance_id=segment.utterance_id,
                region_id=segment.region_id,
                text=segment.text,
                start=start,
                end=end,
                token_start=segment.token_start,
                token_end=segment.token_end,
                draft_start=segment.draft_start,
                draft_end=segment.draft_end,
                aligned_token_count=segment.aligned_token_count,
                start_source=segment.start_source,
                end_source=segment.end_source,
                end_fallback_applied=segment.end_fallback_applied,
                end_gap_ms=segment.end_gap_ms,
                timing_authority=segment.timing_authority,
            )
        )
    return clean
