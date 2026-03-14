from subgen.types import DraftTranscriptUnit, DraftWord


def test_draft_transcript_unit_serializes_alignment_and_display_text() -> None:
    unit = DraftTranscriptUnit(
        unit_id=1,
        source_window_id=2,
        source_region_id=3,
        window_local_start=0.1,
        window_local_end=0.9,
        window_global_start=1.1,
        window_global_end=1.9,
        rough_start=0.1,
        rough_end=0.9,
        display_text="오사카 공항!",
        alignment_text="오사카 공항",
        words=[DraftWord(text="오사카", start=0.1, end=0.3, source_window_id=2, source_region_id=3)],
    )
    data = unit.to_dict()
    assert data["window_local_start"] == 0.1
    assert data["window_global_end"] == 1.9
    assert data["display_text"] == "오사카 공항!"
    assert data["alignment_text"] == "오사카 공항"
    assert data["words"][0]["text"] == "오사카"
