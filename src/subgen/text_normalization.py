from __future__ import annotations

import re
import unicodedata


_PUNCT_RE = re.compile(r"[^\w\s가-힣]")
_MULTISPACE_RE = re.compile(r"\s+")


def normalize_alignment_text(text: str, mode: str = "conservative") -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    if mode == "none":
        return _MULTISPACE_RE.sub(" ", normalized)

    lowered = normalized.lower()
    stripped = _PUNCT_RE.sub(" ", lowered)
    return _MULTISPACE_RE.sub(" ", stripped).strip()
