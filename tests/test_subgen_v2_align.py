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
