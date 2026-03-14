import argparse

from subgen.cli import _apply_preset


def test_ko_sync_final_preset_applies_defaults() -> None:
    args = argparse.Namespace(
        preset="ko-sync-final",
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_min_silence_ms=120,
        vad_pad_ms=80,
        vad_merge_gap_ms=140,
        beam_size=7,
        language=None,
        overlap_sec=0.35,
        alignment="off",
        global_shift_ms=0,
        model="large-v3-turbo",
    )
    _apply_preset(args, [])
    assert args.language == "ko"
    assert args.model == "large-v2"
    assert args.global_shift_ms == 0
    assert args.alignment == "on"
    assert args.vad_pre_roll_ms == 95
    assert args.vad_post_roll_ms == 85
    assert args.alignment_normalization_mode == "conservative"


def test_preset_does_not_override_explicit_flags() -> None:
    args = argparse.Namespace(
        preset="ko-sync-final",
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_min_silence_ms=120,
        vad_pad_ms=80,
        vad_merge_gap_ms=140,
        beam_size=7,
        language=None,
        overlap_sec=0.35,
        alignment="off",
        global_shift_ms=0,
        model="large-v3-turbo",
    )
    _apply_preset(args, ["--global-shift-ms", "0", "--model", "large-v3-turbo"])
    assert args.global_shift_ms == 0
    assert args.model == "large-v3-turbo"


def test_tailpad_preset_extends_end_policy_defaults() -> None:
    args = argparse.Namespace(
        preset="ko_sync_final_tailpad",
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_min_silence_ms=120,
        vad_pad_ms=80,
        vad_merge_gap_ms=140,
        beam_size=7,
        language=None,
        overlap_sec=0.35,
        alignment="off",
        global_shift_ms=0,
        model="large-v3-turbo",
    )
    _apply_preset(args, [])
    assert args.end_tail_padding_ms == 150
    assert args.max_end_tail_padding_ms == 260
    assert args.min_gap_to_next_ms == 45


def test_acoustic_tail_preset_enables_dynamic_tail_policy() -> None:
    args = argparse.Namespace(
        preset="ko_sync_final_acoustic_tail",
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_min_silence_ms=120,
        vad_pad_ms=80,
        vad_merge_gap_ms=140,
        beam_size=7,
        language=None,
        overlap_sec=0.35,
        alignment="off",
        global_shift_ms=0,
        model="large-v3-turbo",
    )
    _apply_preset(args, [])
    assert args.enable_acoustic_tail_extension is True
    assert args.acoustic_tail_probe_ms == 220
    assert args.max_acoustic_tail_extension_ms == 180
    assert args.min_tail_energy_threshold == 0.010
