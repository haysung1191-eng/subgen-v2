from subgen_v2.cli import build_parser


def test_v2_cli_defaults() -> None:
    args = build_parser().parse_args(["input.mp4"])
    assert args.model == "large-v2"
    assert args.language == "ko"
    assert args.subtitle_hold_ms == 180
    assert args.beam_size == 8
    assert args.end_fallback_threshold_ms == 320
