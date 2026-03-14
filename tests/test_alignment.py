from subgen.alignment import align_units
from subgen.config import AlignmentConfig
from subgen.types import DraftTranscriptUnit


def test_align_units_returns_fallback_when_disabled() -> None:
    units = [
        DraftTranscriptUnit(
            unit_id=0,
            source_window_id=0,
            source_region_id=0,
            window_local_start=0.0,
            window_local_end=1.0,
            window_global_start=0.0,
            window_global_end=1.0,
            rough_start=0.0,
            rough_end=1.0,
            display_text="안녕",
            alignment_text="안녕",
        )
    ]
    out = align_units(audio=None, sample_rate=16000, units=units, language="ko", config=AlignmentConfig(enabled=False))
    assert len(out.units) == 1
    assert out.applied is False
    assert out.units[0].start == 0.0
    assert out.units[0].end == 1.0
    assert out.skipped_unit_count == 1
