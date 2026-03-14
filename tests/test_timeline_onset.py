from subgen.timeline import stabilize_timestamps
from subgen.types import SubtitleSegment


def test_onset_nudge_pulls_segment_forward_without_overlap() -> None:
    items = [
        SubtitleSegment(start=0.5, end=1.0, text="a"),
        SubtitleSegment(start=1.5, end=2.0, text="b"),
    ]
    out = stabilize_timestamps(items, onset_nudge_sec=0.1, hard_gap_sec=0.06)
    assert round(out[0].start, 3) == 0.4
    assert round(out[1].start, 3) == 1.4
    assert out[0].end + 0.06 <= out[1].start
