from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SpeechRegion:
    region_id: int
    raw_start: float
    raw_end: float
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DraftWord:
    text: str
    local_start: float
    local_end: float
    global_start: float
    global_end: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DraftUtterance:
    utterance_id: int
    region_id: int
    region_start: float
    region_end: float
    local_start: float
    local_end: float
    global_start: float
    global_end: float
    display_text: str
    alignment_text: str
    words: list[DraftWord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "region_id": self.region_id,
            "region_start": self.region_start,
            "region_end": self.region_end,
            "local_start": self.local_start,
            "local_end": self.local_end,
            "global_start": self.global_start,
            "global_end": self.global_end,
            "display_text": self.display_text,
            "alignment_text": self.alignment_text,
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(slots=True)
class AlignedToken:
    utterance_id: int
    region_id: int
    text: str
    global_start: float
    global_end: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SubtitleSegment:
    segment_id: int
    utterance_id: int
    region_id: int
    text: str
    start: float
    end: float
    token_start: float
    token_end: float
    draft_start: float
    draft_end: float
    aligned_token_count: int
    start_source: str = "aligned_tokens"
    end_source: str = "aligned_tokens_plus_hold"
    end_fallback_applied: bool = False
    end_gap_ms: float = 0.0
    timing_authority: str = "aligned_tokens"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
