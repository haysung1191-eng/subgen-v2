from subgen.timeline import stabilize_timestamps
from subgen.types import SubtitleSegment


def test_deduplicates_adjacent_overlap_same_text() -> None:
    items = [
        SubtitleSegment(start=1.0, end=2.0, text="hello world"),
        SubtitleSegment(start=1.9, end=2.5, text="hello world"),
    ]
    out = stabilize_timestamps(items)
    assert len(out) == 1
    assert out[0].start == 1.0
    assert out[0].end == 2.5


def test_repair_overlap_keeps_monotonic() -> None:
    items = [
        SubtitleSegment(start=0.0, end=1.2, text="a"),
        SubtitleSegment(start=1.0, end=2.0, text="b"),
    ]
    out = stabilize_timestamps(items)
    assert len(out) == 2
    assert out[0].end <= out[1].start
