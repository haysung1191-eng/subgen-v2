import sys
from types import SimpleNamespace

import numpy as np

from subgen_v2.align import align_utterances
from subgen_v2.config import AlignmentConfig
from subgen_v2.types import DraftUtterance


def test_alignment_payload_should_be_scoped_to_each_utterance() -> None:
    utterance = DraftUtterance(
        utterance_id=1,
        region_id=5,
        region_start=11.068,
        region_end=13.796,
        local_start=1.18,
        local_end=1.74,
        global_start=12.248,
        global_end=12.808,
        display_text="test",
        alignment_text="test",
    )
    config = AlignmentConfig(utterance_padding_ms=180)
    padding_sec = config.utterance_padding_ms / 1000.0
    start = max(utterance.region_start, utterance.global_start - padding_sec)
    end = min(utterance.region_end, utterance.global_end + padding_sec)
    assert round(start, 3) == 12.068
    assert round(end, 3) == 12.988


def test_alignment_keeps_low_confidence_tokens_for_timing(monkeypatch) -> None:
    utterance = _utterance()
    fake_whisperx = SimpleNamespace(
        load_align_model=lambda **kwargs: ("model", {"meta": True}),
        align=lambda *args, **kwargs: {
            "segments": [
                {
                    "id": 1,
                    "words": [
                        {"word": "low", "start": 0.1, "end": 0.3, "score": 0.1},
                        {"word": "ok", "start": 0.4, "end": 0.6, "score": 0.9},
                    ],
                }
            ]
        },
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    tokens = align_utterances(np.zeros(16000, dtype=np.float32), 16000, [utterance], AlignmentConfig(min_word_confidence=0.35))

    assert len(tokens) == 2
    assert tokens[0].text == "low"
    assert tokens[0].low_confidence is True
    assert tokens[0].timing_usable is True


def test_alignment_drops_tokens_outside_scoped_window(monkeypatch) -> None:
    utterance = _utterance()
    fake_whisperx = SimpleNamespace(
        load_align_model=lambda **kwargs: ("model", {"meta": True}),
        align=lambda *args, **kwargs: {
            "segments": [
                {
                    "id": 1,
                    "words": [
                        {"word": "outside", "start": 2.5, "end": 2.7, "score": 0.9},
                        {"word": "inside", "start": 0.2, "end": 0.4, "score": 0.9},
                    ],
                }
            ]
        },
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    tokens = align_utterances(np.zeros(16000, dtype=np.float32), 16000, [utterance], AlignmentConfig(utterance_padding_ms=100))

    assert [token.text for token in tokens] == ["inside"]


def _utterance() -> DraftUtterance:
    return DraftUtterance(
        utterance_id=1,
        region_id=0,
        region_start=0.0,
        region_end=2.0,
        local_start=0.0,
        local_end=1.0,
        global_start=0.1,
        global_end=1.0,
        display_text="low ok",
        alignment_text="low ok",
    )
