from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Protocol

import numpy as np

from .config import AlignmentConfig
from .types import AlignedTranscriptUnit, AlignedWord, DraftTranscriptUnit


@dataclass(slots=True)
class AlignmentResult:
    units: list[AlignedTranscriptUnit]
    applied: bool
    warning: str | None = None
    avg_abs_shift_ms: float = 0.0
    avg_signed_shift_ms: float = 0.0
    median_signed_shift_ms: float = 0.0
    onset_shift_stddev_ms: float = 0.0
    materially_changed_count: int = 0
    skipped_unit_count: int = 0


class TimingAligner(Protocol):
    def align(
        self,
        audio: np.ndarray,
        sample_rate: int,
        units: list[DraftTranscriptUnit],
        language: str,
        config: AlignmentConfig,
    ) -> AlignmentResult:
        ...


class WhisperXTimingAligner:
    def align(
        self,
        audio: np.ndarray,
        sample_rate: int,
        units: list[DraftTranscriptUnit],
        language: str,
        config: AlignmentConfig,
    ) -> AlignmentResult:
        if not config.enabled or not units:
            return _build_alignment_result(units, _fallback_units(units, reason="alignment-disabled"), applied=False)

        try:
            aligned = _align_with_whisperx(audio, sample_rate, units, language, config)
            return _build_alignment_result(units, aligned, applied=True)
        except Exception as exc:
            if config.fallback_on_failure:
                return _build_alignment_result(units, _fallback_units(units, reason=str(exc)), applied=False, warning=f"Alignment skipped: {exc}")
            raise


def align_units(
    audio: np.ndarray,
    sample_rate: int,
    units: list[DraftTranscriptUnit],
    language: str,
    config: AlignmentConfig,
) -> AlignmentResult:
    if config.backend != "whisperx":
        raise RuntimeError(f"Unsupported alignment backend: {config.backend}")
    return WhisperXTimingAligner().align(audio, sample_rate, units, language, config)


def _align_with_whisperx(
    audio: np.ndarray,
    sample_rate: int,
    units: list[DraftTranscriptUnit],
    language: str,
    config: AlignmentConfig,
) -> list[AlignedTranscriptUnit]:
    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Alignment requested but whisperx is not installed. Install with: pip install whisperx") from exc

    lang_code = (language or "ko").lower()
    model_a, metadata = _load_align_model(whisperx, lang_code, config)
    payload = [
        {
            "id": unit.unit_id,
            "start": unit.rough_start,
            "end": unit.rough_end,
            "text": unit.alignment_text or unit.display_text,
        }
        for unit in units
    ]
    aligned = whisperx.align(payload, model_a, metadata, audio, config.device, return_char_alignments=False)
    aligned_list = aligned.get("segments", []) if isinstance(aligned, dict) else []
    by_id: dict[int, dict[str, Any]] = {}
    for i, item in enumerate(aligned_list):
        sid = item.get("id", i)
        if isinstance(sid, int):
            by_id[sid] = item

    out: list[AlignedTranscriptUnit] = []
    for unit in units:
        out.append(_merge_alignment_candidate(unit, by_id.get(unit.unit_id), config))
    return out


def _load_align_model(whisperx: Any, language: str, config: AlignmentConfig) -> tuple[Any, Any]:
    try:
        return whisperx.load_align_model(language_code=language, device=config.device, model_name=config.model_name)
    except Exception:
        if not config.fallback_on_failure or config.device == "cpu":
            raise
        return whisperx.load_align_model(language_code=language, device="cpu", model_name=config.model_name)


def _merge_alignment_candidate(
    unit: DraftTranscriptUnit,
    candidate: dict[str, Any] | None,
    config: AlignmentConfig,
) -> AlignedTranscriptUnit:
    if not candidate:
        return _fallback_unit(unit, reason="missing-alignment")

    words = _extract_aligned_words(candidate, unit, config.min_word_confidence)
    if not words:
        return _fallback_unit(unit, reason="no-confident-words")

    return AlignedTranscriptUnit(
        unit_id=unit.unit_id,
        source_window_id=unit.source_window_id,
        source_region_id=unit.source_region_id,
        display_text=unit.display_text,
        alignment_text=unit.alignment_text,
        rough_start=unit.rough_start,
        rough_end=unit.rough_end,
        words=words,
        alignment_applied=True,
    )


def _extract_aligned_words(candidate: dict[str, Any], unit: DraftTranscriptUnit, min_conf: float) -> list[AlignedWord]:
    items = candidate.get("words")
    if not isinstance(items, list):
        return []

    words: list[AlignedWord] = []
    for item in items:
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        confidence = item.get("score")
        if confidence is not None and float(confidence) < min_conf:
            continue
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            continue
        words.append(
            AlignedWord(
                text=str(item.get("word", "")).strip(),
                start=start_f,
                end=end_f,
                confidence=float(confidence) if confidence is not None else None,
                unit_id=unit.unit_id,
                source_window_id=unit.source_window_id,
                source_region_id=unit.source_region_id,
            )
        )
    return words


def _fallback_unit(unit: DraftTranscriptUnit, reason: str | None = None) -> AlignedTranscriptUnit:
    word = AlignedWord(
        text=unit.alignment_text or unit.display_text,
        start=unit.rough_start,
        end=unit.rough_end,
        confidence=None,
        unit_id=unit.unit_id,
        source_window_id=unit.source_window_id,
        source_region_id=unit.source_region_id,
    )
    return AlignedTranscriptUnit(
        unit_id=unit.unit_id,
        source_window_id=unit.source_window_id,
        source_region_id=unit.source_region_id,
        display_text=unit.display_text,
        alignment_text=unit.alignment_text,
        rough_start=unit.rough_start,
        rough_end=unit.rough_end,
        words=[word],
        alignment_applied=False,
        fallback_reason=reason,
    )


def _fallback_units(units: list[DraftTranscriptUnit], reason: str | None = None) -> list[AlignedTranscriptUnit]:
    return [_fallback_unit(unit, reason) for unit in units]


def _build_alignment_result(
    draft_units: list[DraftTranscriptUnit],
    aligned_units: list[AlignedTranscriptUnit],
    *,
    applied: bool,
    warning: str | None = None,
) -> AlignmentResult:
    shifts_ms: list[float] = []
    signed_shifts_ms: list[float] = []
    materially_changed = 0
    skipped = 0
    for draft, aligned in zip(draft_units, aligned_units):
        signed_shift_ms = (aligned.start - draft.rough_start) * 1000.0
        shift_ms = abs(signed_shift_ms)
        shifts_ms.append(shift_ms)
        signed_shifts_ms.append(signed_shift_ms)
        if shift_ms >= 40.0:
            materially_changed += 1
        if not aligned.alignment_applied:
            skipped += 1

    avg_abs_shift_ms = sum(shifts_ms) / len(shifts_ms) if shifts_ms else 0.0
    avg_signed_shift_ms = sum(signed_shifts_ms) / len(signed_shifts_ms) if signed_shifts_ms else 0.0
    median_signed_shift_ms = median(signed_shifts_ms) if signed_shifts_ms else 0.0
    onset_shift_stddev_ms = _stddev(shifts_ms)
    return AlignmentResult(
        units=aligned_units,
        applied=applied,
        warning=warning,
        avg_abs_shift_ms=avg_abs_shift_ms,
        avg_signed_shift_ms=avg_signed_shift_ms,
        median_signed_shift_ms=median_signed_shift_ms,
        onset_shift_stddev_ms=onset_shift_stddev_ms,
        materially_changed_count=materially_changed,
        skipped_unit_count=skipped,
    )


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)
