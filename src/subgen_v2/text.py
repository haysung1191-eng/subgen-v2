from __future__ import annotations

import re


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\"'`~!?,.:;()\[\]{}<>]+")


def normalize_alignment_text(text: str) -> str:
    normalized = _SPACE_RE.sub(" ", text.strip())
    normalized = _PUNCT_RE.sub("", normalized)
    return normalized.strip()
