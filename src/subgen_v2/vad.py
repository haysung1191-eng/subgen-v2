from __future__ import annotations

from typing import Iterable

import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from .config import VADConfig
from .types import SpeechRegion


def detect_speech_regions(audio, sample_rate: int, config: VADConfig) -> list[SpeechRegion]:
    model = load_silero_vad()
    wav = torch.from_numpy(audio)
    raw_regions = get_speech_timestamps(
        wav,
        model,
        threshold=config.threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=config.min_speech_ms,
        min_silence_duration_ms=config.min_silence_ms,
        speech_pad_ms=0,
        return_seconds=False,
    )
    total_duration = len(audio) / float(sample_rate)
    pre_roll = config.pre_roll_ms / 1000.0
    post_roll = config.post_roll_ms / 1000.0
    regions: list[SpeechRegion] = []
    for index, item in enumerate(raw_regions):
        raw_start = item["start"] / sample_rate
        raw_end = item["end"] / sample_rate
        if raw_end <= raw_start:
            continue
        regions.append(
            SpeechRegion(
                region_id=index,
                raw_start=raw_start,
                raw_end=raw_end,
                start=max(0.0, raw_start - pre_roll),
                end=min(total_duration, raw_end + post_roll),
            )
        )
    return _merge_regions(regions, config.merge_gap_ms / 1000.0)


def _merge_regions(regions: Iterable[SpeechRegion], merge_gap_sec: float) -> list[SpeechRegion]:
    merged: list[SpeechRegion] = []
    for region in regions:
        if not merged:
            merged.append(region)
            continue
        prev = merged[-1]
        if region.start - prev.end <= merge_gap_sec:
            prev.end = max(prev.end, region.end)
            prev.raw_end = max(prev.raw_end, region.raw_end)
        else:
            merged.append(region)
    for index, region in enumerate(merged):
        region.region_id = index
    return merged
