from subgen.srt_writer import _format_ts


def test_format_ts_rounding() -> None:
    assert _format_ts(0.0) == "00:00:00,000"
    assert _format_ts(1.234) == "00:00:01,234"
    assert _format_ts(3661.9) == "01:01:01,900"
