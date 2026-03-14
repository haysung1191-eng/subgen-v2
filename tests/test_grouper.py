from subgen.grouper import SubtitleGrouper
from subgen.types import AlignedTranscriptUnit, AlignedWord


def test_grouper_builds_segments_from_aligned_units() -> None:
    unit = AlignedTranscriptUnit(
        unit_id=1,
        source_window_id=0,
        source_region_id=0,
        display_text="안녕하세요",
        alignment_text="안녕하세요",
        rough_start=1.0,
        rough_end=2.0,
        words=[
            AlignedWord("안녕하세요", 1.2, 1.8, 0.9, 1, 0, 0),
        ],
        alignment_applied=True,
    )
    result = SubtitleGrouper().group_units(
        [unit],
        end_tail_padding_ms=90,
        max_end_tail_padding_ms=180,
        enable_acoustic_tail_extension=False,
        acoustic_tail_probe_ms=180,
        max_acoustic_tail_extension_ms=120,
        min_tail_energy_threshold=0.012,
        min_gap_to_next_ms=60,
    )
    assert len(result.segments) == 1
    assert result.segments[0].start == 1.2
    assert round(result.segments[0].end, 2) == 1.89
    assert round(result.avg_end_extension_ms, 1) == 90.0
    assert result.materially_extended_count == 1
    assert result.segments[0].text == "안녕하세요"


def test_grouper_clamps_tail_padding_against_next_start() -> None:
    units = [
        AlignedTranscriptUnit(
            unit_id=1,
            source_window_id=0,
            source_region_id=0,
            display_text="첫번째",
            alignment_text="첫번째",
            rough_start=0.0,
            rough_end=1.0,
            words=[AlignedWord("첫번째", 0.2, 0.8, 0.9, 1, 0, 0)],
            alignment_applied=True,
        ),
        AlignedTranscriptUnit(
            unit_id=2,
            source_window_id=0,
            source_region_id=0,
            display_text="두번째",
            alignment_text="두번째",
            rough_start=0.9,
            rough_end=1.5,
            words=[AlignedWord("두번째", 1.0, 1.3, 0.9, 2, 0, 0)],
            alignment_applied=True,
        ),
    ]
    result = SubtitleGrouper().group_units(
        units,
        end_tail_padding_ms=150,
        max_end_tail_padding_ms=260,
        enable_acoustic_tail_extension=False,
        acoustic_tail_probe_ms=180,
        max_acoustic_tail_extension_ms=120,
        min_tail_energy_threshold=0.012,
        min_gap_to_next_ms=60,
    )
    assert round(result.segments[0].end, 2) == 0.94
    assert round(result.segments[1].end, 2) == 1.45


def test_grouper_extends_tail_from_acoustic_evidence() -> None:
    unit = AlignedTranscriptUnit(
        unit_id=1,
        source_window_id=0,
        source_region_id=0,
        display_text="tail",
        alignment_text="tail",
        rough_start=0.0,
        rough_end=0.5,
        words=[AlignedWord("tail", 0.1, 0.3, 0.9, 1, 0, 0)],
        alignment_applied=True,
    )
    sample_rate = 100
    audio = [0.0] * 60
    for index in range(30, 38):
        audio[index] = 0.05
    result = SubtitleGrouper().group_units(
        [unit],
        end_tail_padding_ms=90,
        max_end_tail_padding_ms=180,
        enable_acoustic_tail_extension=True,
        acoustic_tail_probe_ms=200,
        max_acoustic_tail_extension_ms=120,
        min_tail_energy_threshold=0.01,
        min_gap_to_next_ms=60,
        audio=audio,
        sample_rate=sample_rate,
    )
    assert result.acoustic_tail_extended_count == 1
    assert result.avg_acoustic_tail_extension_ms > 0.0


def test_tail_policy_variants_do_not_change_grouped_start() -> None:
    unit = AlignedTranscriptUnit(
        unit_id=1,
        source_window_id=0,
        source_region_id=0,
        display_text="same start",
        alignment_text="same start",
        rough_start=0.0,
        rough_end=0.5,
        words=[AlignedWord("same", 0.1, 0.3, 0.9, 1, 0, 0)],
        alignment_applied=True,
    )
    a = SubtitleGrouper().group_units(
        [unit],
        end_tail_padding_ms=0,
        max_end_tail_padding_ms=0,
        enable_acoustic_tail_extension=False,
        acoustic_tail_probe_ms=180,
        max_acoustic_tail_extension_ms=120,
        min_tail_energy_threshold=0.012,
        min_gap_to_next_ms=60,
    )
    b = SubtitleGrouper().group_units(
        [unit],
        end_tail_padding_ms=150,
        max_end_tail_padding_ms=260,
        enable_acoustic_tail_extension=True,
        acoustic_tail_probe_ms=220,
        max_acoustic_tail_extension_ms=180,
        min_tail_energy_threshold=0.01,
        min_gap_to_next_ms=45,
        audio=[0.0] * 50,
        sample_rate=100,
    )
    assert a.segments[0].start == b.segments[0].start
