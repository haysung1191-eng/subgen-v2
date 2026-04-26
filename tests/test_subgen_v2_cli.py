from pathlib import Path

from subgen_v2.cli import _unique_temp_wav, build_parser


def test_v2_cli_defaults() -> None:
    args = build_parser().parse_args(["input.mp4"])
    assert args.model == "large-v2"
    assert args.language == "ko"
    assert args.subtitle_hold_ms == 180
    assert args.beam_size == 8
    assert args.end_fallback_threshold_ms == 320


def test_implicit_temp_wav_path_is_unique() -> None:
    first = _unique_temp_wav(Path("sample.mp4"))
    second = _unique_temp_wav(Path("sample.mp4"))
    try:
        assert first != second
        assert first.name == "sample.16k.wav"
        assert second.name == "sample.16k.wav"
    finally:
        first.parent.rmdir()
        second.parent.rmdir()
