from subgen_v2.config import SubtitleConfig
from subgen_v2.subtitle import build_subtitles
from subgen_v2.types import AlignedToken, DraftUtterance


def test_build_subtitles_uses_aligned_tokens_as_timing_authority() -> None:
    utterances = [
        DraftUtterance(
            utterance_id=0,
            region_id=0,
            region_start=0.0,
            region_end=2.0,
            local_start=0.0,
            local_end=1.8,
            global_start=0.0,
            global_end=1.8,
            display_text="hello",
            alignment_text="hello",
        )
    ]
    tokens = [
        AlignedToken(utterance_id=0, region_id=0, text="he", global_start=0.3, global_end=0.5),
        AlignedToken(utterance_id=0, region_id=0, text="llo", global_start=0.55, global_end=1.0),
    ]
    result = build_subtitles(utterances, tokens, SubtitleConfig(hold_ms=80, end_fallback_threshold_ms=2000))
    assert result.raw_segments[0].start == 0.3
    assert round(result.raw_segments[0].end, 2) == 1.08
    assert result.final_segments[0].timing_authority == "aligned_tokens"
    assert result.final_segments[0].start_source == "aligned_tokens"


def test_end_policy_does_not_change_start_anchor() -> None:
    utterances = [
        DraftUtterance(
            utterance_id=0,
            region_id=0,
            region_start=0.0,
            region_end=1.0,
            local_start=0.0,
            local_end=0.8,
            global_start=0.0,
            global_end=0.8,
            display_text="a",
            alignment_text="a",
        )
    ]
    tokens = [AlignedToken(utterance_id=0, region_id=0, text="a", global_start=0.2, global_end=0.6)]
    baseline = build_subtitles(utterances, tokens, SubtitleConfig(hold_ms=0, end_fallback_threshold_ms=2000))
    variant = build_subtitles(utterances, tokens, SubtitleConfig(hold_ms=120, end_fallback_threshold_ms=2000))
    assert baseline.final_segments[0].start == variant.final_segments[0].start


def test_end_fallback_extends_when_aligned_tail_coverage_is_too_short() -> None:
    utterances = [
        DraftUtterance(
            utterance_id=0,
            region_id=0,
            region_start=0.0,
            region_end=2.0,
            local_start=0.0,
            local_end=1.8,
            global_start=0.0,
            global_end=1.8,
            display_text="hello there",
            alignment_text="hello there",
        )
    ]
    tokens = [AlignedToken(utterance_id=0, region_id=0, text="hello", global_start=0.3, global_end=1.0)]
    result = build_subtitles(utterances, tokens, SubtitleConfig(hold_ms=180, end_fallback_threshold_ms=320))
    segment = result.final_segments[0]
    assert segment.start == 0.3
    assert round(segment.end, 2) == 1.98
    assert segment.end_fallback_applied is True
    assert segment.end_source == "draft_end_fallback_plus_hold"


def test_missing_aligned_tokens_should_keep_utterance_via_draft_fallback() -> None:
    utterances = [
        DraftUtterance(
            utterance_id=0,
            region_id=0,
            region_start=0.0,
            region_end=1.2,
            local_start=0.0,
            local_end=0.8,
            global_start=0.1,
            global_end=0.9,
            display_text="fallback",
            alignment_text="fallback",
        )
    ]
    result = build_subtitles(utterances, [], SubtitleConfig(hold_ms=180))
    segment = result.final_segments[0]
    assert segment.start == 0.1
    assert round(segment.end, 2) == 1.08
    assert segment.timing_authority == "draft_fallback"
    assert segment.aligned_token_count == 0


def test_summary_reports_mixed_authority_when_fallbacks_are_used() -> None:
    utterances = [
        DraftUtterance(
            utterance_id=0,
            region_id=0,
            region_start=0.0,
            region_end=1.0,
            local_start=0.0,
            local_end=0.8,
            global_start=0.0,
            global_end=0.8,
            display_text="aligned",
            alignment_text="aligned",
        ),
        DraftUtterance(
            utterance_id=1,
            region_id=1,
            region_start=1.0,
            region_end=2.0,
            local_start=0.0,
            local_end=0.8,
            global_start=1.0,
            global_end=1.8,
            display_text="fallback",
            alignment_text="fallback",
        ),
    ]
    tokens = [AlignedToken(utterance_id=0, region_id=0, text="aligned", global_start=0.1, global_end=0.6)]

    result = build_subtitles(utterances, tokens, SubtitleConfig(hold_ms=0, end_fallback_threshold_ms=2000))

    assert result.summary["timing_authority_summary"] == "mixed"
    assert result.summary["aligned_start_count"] == 1
    assert result.summary["draft_start_fallback_count"] == 1


def test_cleanup_keeps_positive_duration_when_overlap_trims_previous_segment() -> None:
    utterances = [
        DraftUtterance(
            utterance_id=0,
            region_id=0,
            region_start=0.0,
            region_end=1.0,
            local_start=0.0,
            local_end=1.0,
            global_start=0.0,
            global_end=1.0,
            display_text="a",
            alignment_text="a",
        ),
        DraftUtterance(
            utterance_id=1,
            region_id=0,
            region_start=0.0,
            region_end=1.0,
            local_start=0.0,
            local_end=1.0,
            global_start=0.0,
            global_end=1.0,
            display_text="b",
            alignment_text="b",
        ),
    ]
    tokens = [
        AlignedToken(utterance_id=0, region_id=0, text="a", global_start=0.1, global_end=0.5),
        AlignedToken(utterance_id=1, region_id=0, text="b", global_start=0.1, global_end=0.6),
    ]

    result = build_subtitles(utterances, tokens, SubtitleConfig(hold_ms=400, min_duration_ms=220))

    assert all(segment.end > segment.start for segment in result.final_segments)
