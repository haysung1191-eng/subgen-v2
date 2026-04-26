import json
from pathlib import Path

from subgen_v2.review import build_review, write_review_outputs


def test_review_flags_draft_fallback_and_writes_outputs(tmp_path: Path) -> None:
    _write_json(tmp_path / "01_regions.json", [{"region_id": 0, "start": 0.0, "end": 2.0}])
    _write_json(
        tmp_path / "02_draft.json",
        [
            {
                "utterance_id": 0,
                "alignment_text": "긴 문장입니다",
                "display_text": "긴 문장입니다",
            }
        ],
    )
    _write_json(
        tmp_path / "03_aligned_tokens.json",
        [
            {
                "utterance_id": 0,
                "text": "긴",
                "global_start": 0.1,
                "global_end": 0.3,
                "low_confidence": True,
            }
        ],
    )
    _write_json(
        tmp_path / "05_subtitles_final.json",
        [
            {
                "segment_id": 0,
                "utterance_id": 0,
                "region_id": 0,
                "text": "긴 문장입니다",
                "start": 0.1,
                "end": 1.0,
                "token_start": 0.1,
                "token_end": 0.3,
                "draft_start": 0.0,
                "draft_end": 1.5,
                "aligned_token_count": 1,
                "start_source": "aligned_tokens",
                "end_source": "draft_end_fallback_plus_hold",
                "end_fallback_applied": True,
                "end_gap_ms": 1200.0,
                "timing_authority": "aligned_start_draft_end_fallback",
                "cleanup_adjusted": True,
                "cleanup_reason": "overlap_trim_previous_end",
                "cleanup_start_delta_ms": 0.0,
                "cleanup_end_delta_ms": -220.0,
            }
        ],
    )

    rows = build_review(tmp_path, top=10)
    paths = write_review_outputs(rows, tmp_path)

    assert rows[0].risk_level == "CRITICAL"
    assert "HIGH_END_GAP" in rows[0].issue_tags
    assert "CLEANUP_TRIMMED_END" in rows[0].issue_tags
    assert paths["md"].exists()
    assert paths["csv"].exists()
    assert paths["json"].exists()
    assert paths["html"].exists()
    assert paths["srt"].exists()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
