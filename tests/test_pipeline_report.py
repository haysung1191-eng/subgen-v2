from subgen.pipeline import _compute_onset_error, _count_overlaps
from subgen.types import SubtitleSegment


def test_compute_onset_error() -> None:
    rough = [SubtitleSegment(0.0, 1.0, "a"), SubtitleSegment(2.0, 3.0, "b")]
    aligned = [SubtitleSegment(0.1, 1.0, "a"), SubtitleSegment(2.2, 3.0, "b")]
    avg, n = _compute_onset_error(rough, aligned)
    assert n == 2
    assert round(avg, 1) == 150.0


def test_count_overlaps() -> None:
    segs = [
        SubtitleSegment(0.0, 1.0, "a"),
        SubtitleSegment(0.9, 1.5, "b"),
        SubtitleSegment(1.6, 2.0, "c"),
    ]
    assert _count_overlaps(segs) == 1
