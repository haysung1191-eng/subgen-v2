from subgen.pipeline import _apply_global_shift
from subgen.types import SubtitleSegment


def test_apply_global_shift_positive() -> None:
    items = [SubtitleSegment(start=1.0, end=2.0, text="a")]
    out = _apply_global_shift(items, 250)
    assert out[0].start == 1.25
    assert out[0].end == 2.25


def test_apply_global_shift_negative_clamps_zero() -> None:
    items = [SubtitleSegment(start=0.1, end=0.5, text="a")]
    out = _apply_global_shift(items, -300)
    assert out[0].start == 0.0
    assert out[0].end >= out[0].start
