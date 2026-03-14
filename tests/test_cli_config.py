import argparse
import json

from subgen.cli import _apply_config


def test_apply_config_uses_json_defaults(tmp_path) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({"model": "large-v2", "global_shift_ms": 111}), encoding="utf-8")
    args = argparse.Namespace(
        config=config_path,
        preset=None,
        sample_rate=16000,
        global_shift_ms=0,
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_min_silence_ms=120,
        vad_pad_ms=80,
        vad_pre_roll_ms=80,
        vad_post_roll_ms=80,
        vad_merge_gap_ms=140,
        model="large-v3-turbo",
        device="cuda",
        compute_type="float16",
        beam_size=7,
        language=None,
        overlap_sec=0.35,
        alignment="on",
        alignment_backend="whisperx",
        align_model=None,
        align_device="cuda",
        align_min_conf=0.35,
        align_boundary_slack_ms=80,
        alignment_normalization_mode="conservative",
        min_segment_sec=0.2,
        max_segment_sec=4.0,
        hard_gap_ms=60,
        timing_correction_ms=0,
        log_level="INFO",
        debug_export_dir=None,
        batch_concurrency=1,
    )
    _apply_config(args, [])
    assert args.model == "large-v2"
    assert args.global_shift_ms == 111


def test_apply_config_does_not_override_explicit_flags(tmp_path) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({"model": "large-v2"}), encoding="utf-8")
    args = argparse.Namespace(
        config=config_path,
        preset=None,
        sample_rate=16000,
        global_shift_ms=0,
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_min_silence_ms=120,
        vad_pad_ms=80,
        vad_pre_roll_ms=80,
        vad_post_roll_ms=80,
        vad_merge_gap_ms=140,
        model="large-v3-turbo",
        device="cuda",
        compute_type="float16",
        beam_size=7,
        language=None,
        overlap_sec=0.35,
        alignment="on",
        alignment_backend="whisperx",
        align_model=None,
        align_device="cuda",
        align_min_conf=0.35,
        align_boundary_slack_ms=80,
        alignment_normalization_mode="conservative",
        min_segment_sec=0.2,
        max_segment_sec=4.0,
        hard_gap_ms=60,
        timing_correction_ms=0,
        log_level="INFO",
        debug_export_dir=None,
        batch_concurrency=1,
    )
    _apply_config(args, ["--model", "large-v3-turbo"])
    assert args.model == "large-v3-turbo"
