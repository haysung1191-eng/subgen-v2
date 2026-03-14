from __future__ import annotations

from typing import Iterable

import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from .config import VADConfig
from .types import SpeechRegion
from .wav_io import load_mono_wav_float32


def _merge_adjacent_regions(regions: Iterable[SpeechRegion], merge_gap_sec: float) -> list[SpeechRegion]:
    merged: list[SpeechRegion] = []
    for region in regions:
        if not merged:
            merged.append(region)
            continue
        prev = merged[-1]
        if region.start - prev.end <= merge_gap_sec:
            prev.end = max(prev.end, region.end)
            prev.raw_end = max(prev.raw_end, region.raw_end)
            prev.post_roll_applied = max(prev.post_roll_applied, region.post_roll_applied)
        else:
            merged.append(region)
    for index, region in enumerate(merged):
        region.region_id = index
    return merged


def detect_speech_regions(wav_path: str, config: VADConfig, sample_rate: int = 16000) -> list[SpeechRegion]:
    model = load_silero_vad()
    wav_np, wav_sr = load_mono_wav_float32(wav_path)
    if wav_sr != sample_rate:
        raise ValueError(f"Unexpected WAV sample rate: expected {sample_rate}, got {wav_sr}")

    wav = torch.from_numpy(wav_np)
    raw = get_speech_timestamps(
        wav,
        model,
        threshold=config.threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=config.min_speech_ms,
        min_silence_duration_ms=config.min_silence_ms,
        speech_pad_ms=0,
        return_seconds=False,
    )

    total_duration = len(wav_np) / sample_rate
    pre_roll_sec = (config.pre_roll_ms or config.pad_ms) / 1000.0
    post_roll_sec = (config.post_roll_ms or config.pad_ms) / 1000.0

    regions: list[SpeechRegion] = []
    for index, item in enumerate(raw):
        raw_start = item["start"] / sample_rate
        raw_end = item["end"] / sample_rate
        if raw_end <= raw_start:
            continue

        start = max(0.0, raw_start - pre_roll_sec)
        end = min(total_duration, raw_end + post_roll_sec)
        regions.append(
            SpeechRegion(
                region_id=index,
                raw_start=raw_start,
                raw_end=raw_end,
                start=start,
                end=end,
                pre_roll_applied=raw_start - start,
                post_roll_applied=end - raw_end,
            )
        )

    return _merge_adjacent_regions(regions, config.merge_gap_ms / 1000.0)
