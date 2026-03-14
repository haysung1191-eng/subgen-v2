from subgen.timeline import stabilize_timestamps
from subgen.types import SubtitleSegment


def test_soft_split_long_segment() -> None:
    text = "첫 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
    items = [SubtitleSegment(start=0.0, end=8.0, text=text)]
    out = stabilize_timestamps(items)
    assert len(out) >= 2
    assert all(seg.end > seg.start for seg in out)
    assert max(seg.end - seg.start for seg in out) <= 4.01
