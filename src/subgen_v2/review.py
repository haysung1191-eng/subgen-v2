from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .srt import _format_time


@dataclass(slots=True)
class ReviewRow:
    segment_id: int
    utterance_id: int
    region_id: int
    start: float
    end: float
    duration_ms: float
    text_preview: str
    risk_score: int
    risk_level: str
    issue_tags: list[str]
    recommended_action: str
    start_source: str
    end_source: str
    timing_authority: str
    aligned_token_count: int
    end_gap_ms: float
    cleanup_adjusted: bool
    cleanup_reason: str | None
    cleanup_start_delta_ms: float
    cleanup_end_delta_ms: float
    draft_start: float
    draft_end: float
    token_start: float
    token_end: float
    region_start: float | None
    region_end: float | None
    next_start: float | None
    next_gap_ms: float | None
    chars: int
    chars_per_second: float
    alignment_coverage_ratio: float
    first_token_delay_ms: float | None
    tail_gap_ms: float
    final_tail_margin_ms: float
    boundary_start_risk_ms: float | None
    boundary_end_risk_ms: float | None


def build_review(debug_dir: Path, top: int = 40) -> list[ReviewRow]:
    regions = _load_optional(debug_dir / "01_regions.json", [])
    drafts = _load_optional(debug_dir / "02_draft.json", [])
    tokens = _load_optional(debug_dir / "03_aligned_tokens.json", [])
    segments = _load_required(debug_dir / "05_subtitles_final.json")

    region_by_id = {int(item["region_id"]): item for item in regions}
    draft_by_utterance = {int(item["utterance_id"]): item for item in drafts}
    tokens_by_utterance: dict[int, list[dict[str, Any]]] = {}
    for token in tokens:
        tokens_by_utterance.setdefault(int(token["utterance_id"]), []).append(token)

    rows: list[ReviewRow] = []
    ordered = sorted(segments, key=lambda item: (float(item["start"]), float(item["end"])))
    for index, segment in enumerate(ordered):
        next_start = float(ordered[index + 1]["start"]) if index + 1 < len(ordered) else None
        rows.append(_review_segment(segment, next_start, region_by_id, draft_by_utterance, tokens_by_utterance))
    return sorted(rows, key=lambda row: (-row.risk_score, row.start, row.segment_id))[:top]


def write_review_outputs(rows: list[ReviewRow], debug_dir: Path, out_dir: Path | None = None) -> dict[str, Path]:
    target = out_dir or debug_dir
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": target / "timing_review.json",
        "csv": target / "timing_review.csv",
        "md": target / "timing_review.md",
        "html": target / "timing_review.html",
        "srt": target / "timing_review.srt",
    }
    payload = [asdict(row) for row in rows]
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(paths["csv"], rows)
    _write_markdown(paths["md"], rows)
    _write_html(paths["html"], rows)
    _write_review_srt(paths["srt"], rows)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(prog="subgen-v2-review", description="Build a Filmora timing review report from subgen_v2 debug JSON.")
    parser.add_argument("debug_dir", type=Path)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = build_review(args.debug_dir.resolve(), top=args.top)
    paths = write_review_outputs(rows, args.debug_dir.resolve(), args.out_dir.resolve() if args.out_dir else None)
    print(f"review_rows={len(rows)}")
    for name, path in paths.items():
        print(f"{name}={path}")


def _review_segment(
    segment: dict[str, Any],
    next_start: float | None,
    region_by_id: dict[int, dict[str, Any]],
    draft_by_utterance: dict[int, dict[str, Any]],
    tokens_by_utterance: dict[int, list[dict[str, Any]]],
) -> ReviewRow:
    segment_id = int(segment["segment_id"])
    utterance_id = int(segment["utterance_id"])
    region_id = int(segment["region_id"])
    start = float(segment["start"])
    end = float(segment["end"])
    token_start = float(segment["token_start"])
    token_end = float(segment["token_end"])
    draft_start = float(segment["draft_start"])
    draft_end = float(segment["draft_end"])
    text = str(segment.get("text", ""))
    region = region_by_id.get(region_id)
    draft = draft_by_utterance.get(utterance_id)
    utterance_tokens = tokens_by_utterance.get(utterance_id, [])

    chars = len(_visible_text(text))
    duration_ms = max(0.0, (end - start) * 1000.0)
    chars_per_second = chars / max(0.001, end - start)
    next_gap_ms = (next_start - end) * 1000.0 if next_start is not None else None
    region_start = float(region["start"]) if region else None
    region_end = float(region["end"]) if region else None
    first_token_delay_ms = (token_start - region_start) * 1000.0 if region_start is not None else None
    tail_gap_ms = max(0.0, (draft_end - token_end) * 1000.0)
    end_anchor = draft_end if bool(segment.get("end_fallback_applied", False)) else token_end
    final_tail_margin_ms = (end - max(token_end, end_anchor)) * 1000.0
    boundary_start_risk_ms = (start - region_start) * 1000.0 if region_start is not None else None
    boundary_end_risk_ms = (region_end - max(token_end, draft_end)) * 1000.0 if region_end is not None else None
    alignment_coverage_ratio = _coverage_ratio(draft, utterance_tokens)
    dense_handoff = next_gap_ms is not None and next_gap_ms <= 30.0 and end < token_end - 0.05

    score, tags = _score_segment(
        segment=segment,
        duration_ms=duration_ms,
        chars_per_second=chars_per_second,
        alignment_coverage_ratio=alignment_coverage_ratio,
        tail_gap_ms=tail_gap_ms,
        boundary_start_risk_ms=boundary_start_risk_ms,
        boundary_end_risk_ms=boundary_end_risk_ms,
        token_end=token_end,
        end=end,
        dense_handoff=dense_handoff,
        utterance_tokens=utterance_tokens,
    )
    return ReviewRow(
        segment_id=segment_id,
        utterance_id=utterance_id,
        region_id=region_id,
        start=start,
        end=end,
        duration_ms=duration_ms,
        text_preview=text[:80],
        risk_score=score,
        risk_level=_risk_level(score),
        issue_tags=tags,
        recommended_action=_recommend(tags),
        start_source=str(segment.get("start_source", "")),
        end_source=str(segment.get("end_source", "")),
        timing_authority=str(segment.get("timing_authority", "")),
        aligned_token_count=int(segment.get("aligned_token_count", 0)),
        end_gap_ms=float(segment.get("end_gap_ms", 0.0)),
        cleanup_adjusted=bool(segment.get("cleanup_adjusted", False)),
        cleanup_reason=segment.get("cleanup_reason"),
        cleanup_start_delta_ms=float(segment.get("cleanup_start_delta_ms", 0.0)),
        cleanup_end_delta_ms=float(segment.get("cleanup_end_delta_ms", 0.0)),
        draft_start=draft_start,
        draft_end=draft_end,
        token_start=token_start,
        token_end=token_end,
        region_start=region_start,
        region_end=region_end,
        next_start=next_start,
        next_gap_ms=next_gap_ms,
        chars=chars,
        chars_per_second=chars_per_second,
        alignment_coverage_ratio=alignment_coverage_ratio,
        first_token_delay_ms=first_token_delay_ms,
        tail_gap_ms=tail_gap_ms,
        final_tail_margin_ms=final_tail_margin_ms,
        boundary_start_risk_ms=boundary_start_risk_ms,
        boundary_end_risk_ms=boundary_end_risk_ms,
    )


def _score_segment(
    *,
    segment: dict[str, Any],
    duration_ms: float,
    chars_per_second: float,
    alignment_coverage_ratio: float,
    tail_gap_ms: float,
    boundary_start_risk_ms: float | None,
    boundary_end_risk_ms: float | None,
    token_end: float,
    end: float,
    dense_handoff: bool,
    utterance_tokens: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    score = 0
    tags: list[str] = []
    aligned_count = int(segment.get("aligned_token_count", 0))
    if aligned_count == 0:
        score += 90
        tags.append("ZERO_ALIGNED_TOKENS")
    if segment.get("start_source") == "draft_fallback":
        score += 70
        tags.append("DRAFT_START_FALLBACK")
    if bool(segment.get("end_fallback_applied", False)):
        score += 25
        tags.append("DRAFT_END_FALLBACK")
    if tail_gap_ms >= 800:
        score += 45
        tags.append("HIGH_END_GAP")
    elif tail_gap_ms >= 500:
        score += 30
        tags.append("HIGH_END_GAP")
    elif tail_gap_ms >= 250:
        score += 15
        tags.append("MEDIUM_END_GAP")
    cleanup_end_delta = float(segment.get("cleanup_end_delta_ms", 0.0))
    if dense_handoff:
        score += 10
        tags.append("DENSE_SUBTITLE_HANDOFF")
    else:
        if cleanup_end_delta <= -200:
            score += 45
            tags.append("CLEANUP_TRIMMED_END")
        elif cleanup_end_delta <= -100:
            score += 25
            tags.append("CLEANUP_TRIMMED_END")
    if end < token_end - 0.05:
        if dense_handoff:
            score += 10
            tags.append("FINAL_END_BEFORE_TOKEN_END_HANDOFF")
        else:
            score += 80
            tags.append("FINAL_END_BEFORE_TOKEN_END")
    if alignment_coverage_ratio < 0.50:
        score += 70
        tags.append("LOW_ALIGNMENT_COVERAGE")
    elif alignment_coverage_ratio < 0.70:
        score += 40
        tags.append("LOW_ALIGNMENT_COVERAGE")
    elif alignment_coverage_ratio < 0.85:
        score += 20
        tags.append("LOW_ALIGNMENT_COVERAGE")
    if tail_gap_ms > 180 and (_visible_len(str(segment.get("text", ""))) >= 28 or duration_ms >= 4500):
        score += 35
        tags.append("LONG_TAIL_RISK")
    if duration_ms < 500:
        score += 20
        tags.append("VERY_SHORT_SUBTITLE")
    if duration_ms > 8000:
        score += 15
        tags.append("VERY_LONG_SUBTITLE")
    if chars_per_second > 25:
        score += 25
        tags.append("HIGH_READING_SPEED")
    elif chars_per_second > 18:
        score += 10
        tags.append("HIGH_READING_SPEED")
    if boundary_start_risk_ms is not None and boundary_start_risk_ms < 100:
        score += 15
        tags.append("VAD_START_BOUNDARY")
    if boundary_end_risk_ms is not None and boundary_end_risk_ms < 100:
        score += 25
        tags.append("VAD_END_BOUNDARY")
    if any(bool(token.get("low_confidence", False)) for token in utterance_tokens[:1] + utterance_tokens[-1:]):
        score += 10
        tags.append("LOW_CONFIDENCE_EDGE_TOKEN")
    return score, tags or ["PROBABLY_SAFE"]


def _coverage_ratio(draft: dict[str, Any] | None, tokens: list[dict[str, Any]]) -> float:
    if not draft:
        return 1.0 if tokens else 0.0
    draft_chars = _visible_len(str(draft.get("alignment_text") or draft.get("display_text") or ""))
    if draft_chars == 0:
        return 1.0
    token_chars = sum(_visible_len(str(token.get("text", ""))) for token in tokens)
    return min(1.0, token_chars / draft_chars)


def _recommend(tags: list[str]) -> str:
    if "ZERO_ALIGNED_TOKENS" in tags or "DRAFT_START_FALLBACK" in tags:
        return "Check onset; this line used draft start fallback."
    if "FINAL_END_BEFORE_TOKEN_END" in tags or "CLEANUP_TRIMMED_END" in tags:
        return "Check ending; cleanup shortened this line."
    if "DENSE_SUBTITLE_HANDOFF" in tags:
        return "Likely dense handoff; check only if this feels abrupt in Filmora."
    if "LONG_TAIL_RISK" in tags or "HIGH_END_GAP" in tags:
        return "Check ending; likely long Korean tail or weak alignment coverage."
    if "VAD_END_BOUNDARY" in tags:
        return "Check ending; speech may be close to the VAD region boundary."
    if "LOW_ALIGNMENT_COVERAGE" in tags:
        return "Check timing; aligned token coverage is weak."
    if "HIGH_READING_SPEED" in tags:
        return "Check readability; subtitle may be too dense or short."
    return "Probably safe."


def _risk_level(score: int) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


def _write_csv(path: Path, rows: list[ReviewRow]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else ["segment_id"])
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data["issue_tags"] = ";".join(row.issue_tags)
            writer.writerow(data)


def _write_markdown(path: Path, rows: list[ReviewRow]) -> None:
    lines = ["# Timing Review", "", f"Rows: {len(rows)}", ""]
    if rows:
        scores = [row.risk_score for row in rows]
        lines.extend([f"Median risk score: {median(scores):.1f}", f"Max risk score: {max(scores)}", ""])
    lines.append("| # | Time | Risk | Tags | Action | Text |")
    lines.append("| - | - | - | - | - | - |")
    for row in rows:
        time = f"{_format_time(row.start)} - {_format_time(row.end)}"
        tags = ", ".join(row.issue_tags)
        lines.append(f"| {row.segment_id} | {time} | {row.risk_level} {row.risk_score} | {tags} | {row.recommended_action} | {row.text_preview} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(path: Path, rows: list[ReviewRow]) -> None:
    body = [
        "<!doctype html><meta charset='utf-8'><title>Timing Review</title>",
        "<style>body{font-family:Segoe UI,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px}th{background:#f4f4f4}.CRITICAL{background:#ffd7d7}.HIGH{background:#ffe8c2}.MEDIUM{background:#fff7c2}</style>",
        "<h1>Timing Review</h1>",
        f"<p>Rows: {len(rows)}</p>",
        "<table><thead><tr><th>ID</th><th>Time</th><th>Risk</th><th>Tags</th><th>Action</th><th>Text</th></tr></thead><tbody>",
    ]
    for row in rows:
        time = f"{_format_time(row.start)} - {_format_time(row.end)}"
        body.append(
            "<tr class='{level}'><td>{sid}</td><td>{time}</td><td>{level} {score}</td><td>{tags}</td><td>{action}</td><td>{text}</td></tr>".format(
                level=html.escape(row.risk_level),
                sid=row.segment_id,
                time=html.escape(time),
                score=row.risk_score,
                tags=html.escape(", ".join(row.issue_tags)),
                action=html.escape(row.recommended_action),
                text=html.escape(row.text_preview),
            )
        )
    body.append("</tbody></table>")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_review_srt(path: Path, rows: list[ReviewRow]) -> None:
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                str(index),
                f"{_format_time(row.start)} --> {_format_time(row.end)}",
                f"[{row.risk_level} {row.risk_score}] {', '.join(row.issue_tags)}",
                row.text_preview,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_required(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"Required debug file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _visible_text(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def _visible_len(text: str) -> int:
    return len(_visible_text(text))


if __name__ == "__main__":
    main()
