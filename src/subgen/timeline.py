from __future__ import annotations

import re

from .types import SubtitleSegment


_PUNCT = ".?!,;:"


def _norm_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip().lower()
    return re.sub(r"[^\w\s]", "", compact)


def _choose_split_index(text: str) -> int | None:
    if len(text) < 2:
        return None

    low = int(len(text) * 0.3)
    high = int(len(text) * 0.7)
    if low >= high:
        low, high = 1, len(text) - 1

    candidates: list[int] = []
    for i in range(low, high):
        if text[i] in _PUNCT:
            candidates.append(i + 1)

    if not candidates:
        for i in range(low, high):
            if text[i].isspace():
                candidates.append(i)

    if not candidates:
        return None

    target = len(text) // 2
    return min(candidates, key=lambda x: abs(x - target))


def _split_long_segment(seg: SubtitleSegment, max_duration_sec: float, min_duration_sec: float) -> list[SubtitleSegment]:
    queue = [seg]
    out: list[SubtitleSegment] = []

    while queue:
        cur = queue.pop(0)
        if cur.end - cur.start <= max_duration_sec:
            out.append(cur)
            continue

        split_idx = _choose_split_index(cur.text)
        if split_idx is None:
            out.append(cur)
            continue

        left_text = cur.text[:split_idx].strip()
        right_text = cur.text[split_idx:].strip()
        if not left_text or not right_text:
            out.append(cur)
            continue

        ratio = len(left_text) / max(1, len(left_text) + len(right_text))
        mid = cur.start + (cur.end - cur.start) * ratio

        if mid - cur.start < min_duration_sec or cur.end - mid < min_duration_sec:
            out.append(cur)
            continue

        queue.insert(0, SubtitleSegment(start=mid, end=cur.end, text=right_text))
        queue.insert(0, SubtitleSegment(start=cur.start, end=mid, text=left_text))

    return out


def stabilize_timestamps(
    segments: list[SubtitleSegment],
    min_duration_sec: float = 0.20,
    hard_gap_sec: float = 0.06,
    max_duration_sec: float = 4.0,
    onset_nudge_sec: float = 0.0,
) -> list[SubtitleSegment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda x: (x.start, x.end))
    clean: list[SubtitleSegment] = []

    for cur in ordered:
        if not cur.text.strip() or cur.end <= cur.start:
            continue

        if cur.end - cur.start < min_duration_sec:
            cur.end = cur.start + min_duration_sec

        if not clean:
            clean.append(cur)
            continue

        prev = clean[-1]
        same_text = _norm_text(prev.text) == _norm_text(cur.text)
        overlap = max(0.0, min(prev.end, cur.end) - max(prev.start, cur.start))

        # Merge duplicate overlap artifacts only when text matches and time actually overlaps.
        if same_text and overlap > 0.0:
            prev.end = max(prev.end, cur.end)
            continue

        if cur.start < prev.end:
            overlap_span = prev.end - cur.start
            if overlap_span <= 0.25:
                midpoint = (prev.end + cur.start) * 0.5
                prev.end = max(prev.start + min_duration_sec, midpoint)
                cur.start = min(cur.end - min_duration_sec, midpoint)
            else:
                cur.start = prev.end + hard_gap_sec

        if cur.end - cur.start < min_duration_sec:
            cur.end = cur.start + min_duration_sec
        if cur.start >= cur.end:
            continue

        clean.append(cur)

    balanced: list[SubtitleSegment] = []
    for seg in clean:
        balanced.extend(_split_long_segment(seg, max_duration_sec=max_duration_sec, min_duration_sec=min_duration_sec))

    final_segments = sorted(balanced, key=lambda s: (s.start, s.end))
    if onset_nudge_sec <= 0:
        return final_segments

    nudged: list[SubtitleSegment] = []
    for index, seg in enumerate(final_segments):
        earliest = 0.0
        if index > 0:
            earliest = nudged[-1].end + hard_gap_sec
        available = max(0.0, seg.start - earliest)
        shift = min(onset_nudge_sec, available)
        nudged.append(SubtitleSegment(start=seg.start - shift, end=seg.end, text=seg.text))
    return nudged
