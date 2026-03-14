from __future__ import annotations

import json
from pathlib import Path


def write_debug_json(output_dir: Path, name: str, payload) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
