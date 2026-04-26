from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .config import AlignmentConfig
from .types import AlignedToken, DraftUtterance


def align_utterances(
    audio: np.ndarray,
    sample_rate: int,
    utterances: list[DraftUtterance],
    config: AlignmentConfig,
    language: str = "ko",
) -> list[AlignedToken]:
    if not utterances:
        return []
    if config.backend != "whisperx":
        raise RuntimeError(f"Unsupported alignment backend: {config.backend}")
    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("whisperx is required for subgen_v2 alignment") from exc

    model_a, metadata = whisperx.load_align_model(
        language_code=language.lower(),
        device=config.device,
        model_name=config.model_name,
    )
    padding_sec = config.utterance_padding_ms / 1000.0
    boundary_epsilon_sec = 0.05
    payload = []
    window_by_utterance: dict[int, tuple[float, float]] = {}
    for utterance in utterances:
        window_start = max(utterance.region_start, utterance.global_start - padding_sec)
        window_end = min(utterance.region_end, utterance.global_end + padding_sec)
        window_by_utterance[utterance.utterance_id] = (window_start, window_end)
        payload.append(
            {
                "id": utterance.utterance_id,
                "start": window_start,
                "end": window_end,
                "text": utterance.alignment_text or utterance.display_text,
            }
        )
    result = whisperx.align(payload, model_a, metadata, audio, config.device, return_char_alignments=False)
    segments = result.get("segments", []) if isinstance(result, dict) else []
    utterance_by_id = {utterance.utterance_id: utterance for utterance in utterances}
    tokens: list[AlignedToken] = []
    for index, segment in enumerate(segments):
        utterance_id = segment.get("id", index)
        utterance = utterance_by_id.get(utterance_id)
        if utterance is None:
            continue
        for item in segment.get("words", []) or []:
            start = item.get("start")
            end = item.get("end")
            if start is None or end is None:
                continue
            confidence = item.get("score")
            start_f = float(start)
            end_f = float(end)
            if end_f <= start_f:
                continue
            window = window_by_utterance.get(utterance.utterance_id)
            if window is not None:
                if end_f < window[0] - boundary_epsilon_sec or start_f > window[1] + boundary_epsilon_sec:
                    continue
                start_f = max(start_f, window[0])
                end_f = min(end_f, window[1])
                if end_f <= start_f:
                    continue
            confidence_f = float(confidence) if confidence is not None else None
            tokens.append(
                AlignedToken(
                    utterance_id=utterance.utterance_id,
                    region_id=utterance.region_id,
                    text=str(item.get("word", "")).strip(),
                    global_start=start_f,
                    global_end=end_f,
                    confidence=confidence_f,
                    low_confidence=confidence_f is not None and confidence_f < config.min_word_confidence,
                    timing_usable=True,
                )
            )
    return _sorted_tokens(tokens)


def tokens_by_utterance(tokens: list[AlignedToken]) -> dict[int, list[AlignedToken]]:
    grouped: dict[int, list[AlignedToken]] = defaultdict(list)
    for token in tokens:
        grouped[token.utterance_id].append(token)
    for utterance_id in grouped:
        grouped[utterance_id].sort(key=lambda item: (item.global_start, item.global_end))
    return dict(grouped)


def _sorted_tokens(tokens: list[AlignedToken]) -> list[AlignedToken]:
    return sorted(tokens, key=lambda item: (item.global_start, item.global_end, item.utterance_id))
