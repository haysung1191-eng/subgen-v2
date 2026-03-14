import json
from pathlib import Path

from subgen_v2.debug import write_debug_json


def test_write_debug_json_creates_expected_file(tmp_path: Path) -> None:
    write_debug_json(tmp_path, "01_regions.json", [{"region_id": 0}])
    data = json.loads((tmp_path / "01_regions.json").read_text(encoding="utf-8"))
    assert data[0]["region_id"] == 0
