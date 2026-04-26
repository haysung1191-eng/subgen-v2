from pathlib import Path

from subgen_v2.cli import _apply_preset, _build_config, _unique_temp_wav, build_parser


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


def test_filmora_preset_applies_defaults() -> None:
    argv = ["input.mp4", "--preset", "filmora-ko"]
    args = build_parser().parse_args(argv)
    _apply_preset(args, argv)
    config = _build_config(args, Path("input.mp4"), Path("out.srt"), Path("temp.wav"), False, None)
    assert config.subtitle.hold_ms == 320
    assert config.subtitle.min_duration_ms == 600
    assert config.vad.post_roll_ms == 250
    assert config.alignment.utterance_padding_ms == 300


def test_preset_does_not_override_explicit_flag() -> None:
    argv = ["input.mp4", "--preset", "filmora-ko", "--subtitle-hold-ms", "180"]
    args = build_parser().parse_args(argv)
    _apply_preset(args, argv)
    config = _build_config(args, Path("input.mp4"), Path("out.srt"), Path("temp.wav"), False, None)
    assert config.subtitle.hold_ms == 180
