from __future__ import annotations

from pathlib import Path

from .errors import OutputWriteError
from .types import SubtitleSegment


def _format_ts(total_seconds: float) -> str:
    if total_seconds < 0:
        total_seconds = 0.0
    millis = int(round(total_seconds * 1000))
    hours = millis // 3_600_000
    millis -= hours * 3_600_000
    minutes = millis // 60_000
    millis -= minutes * 60_000
    seconds = millis // 1000
    millis -= seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(segments: list[SubtitleSegment], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_ts(seg.start)} --> {_format_ts(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")

    try:
        output_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        raise OutputWriteError(str(exc)) from exc
