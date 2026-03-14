from __future__ import annotations

from pathlib import Path

from .types import SubtitleSegment


def write_srt(segments: list[SubtitleSegment], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                str(index),
                f"{_format_time(segment.start)} --> {_format_time(segment.end)}",
                segment.text.strip(),
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _format_time(sec: float) -> str:
    total_ms = max(0, int(round(sec * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
