from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TimeSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpeechRegion:
    region_id: int
    raw_start: float
    raw_end: float
    start: float
    end: float
    pre_roll_applied: float
    post_roll_applied: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DraftWord:
    text: str
    start: float
    end: float
    source_window_id: int
    source_region_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DraftTranscriptUnit:
    unit_id: int
    source_window_id: int
    source_region_id: int
    window_local_start: float
    window_local_end: float
    window_global_start: float
    window_global_end: float
    rough_start: float
    rough_end: float
    display_text: str
    alignment_text: str
    words: list[DraftWord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_window_id": self.source_window_id,
            "source_region_id": self.source_region_id,
            "window_local_start": self.window_local_start,
            "window_local_end": self.window_local_end,
            "window_global_start": self.window_global_start,
            "window_global_end": self.window_global_end,
            "rough_start": self.rough_start,
            "rough_end": self.rough_end,
            "display_text": self.display_text,
            "alignment_text": self.alignment_text,
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(slots=True)
class AlignedWord:
    text: str
    start: float
    end: float
    confidence: float | None
    unit_id: int
    source_window_id: int
    source_region_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AlignedTranscriptUnit:
    unit_id: int
    source_window_id: int
    source_region_id: int
    display_text: str
    alignment_text: str
    rough_start: float
    rough_end: float
    words: list[AlignedWord] = field(default_factory=list)
    alignment_applied: bool = False
    fallback_reason: str | None = None

    @property
    def start(self) -> float:
        if self.words:
            return self.words[0].start
        return self.rough_start

    @property
    def end(self) -> float:
        if self.words:
            return self.words[-1].end
        return self.rough_end

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_window_id": self.source_window_id,
            "source_region_id": self.source_region_id,
            "display_text": self.display_text,
            "alignment_text": self.alignment_text,
            "rough_start": self.rough_start,
            "rough_end": self.rough_end,
            "alignment_applied": self.alignment_applied,
            "fallback_reason": self.fallback_reason,
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(slots=True)
class SubtitleSegment:
    start: float
    end: float
    text: str
    source_unit_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
