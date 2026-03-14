from subgen.text_normalization import normalize_alignment_text


def test_conservative_alignment_text_normalization() -> None:
    text = "오사카 공항, Near Gate 12!"
    out = normalize_alignment_text(text, "conservative")
    assert out == "오사카 공항 near gate 12"
