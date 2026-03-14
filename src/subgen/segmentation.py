from __future__ import annotations

from dataclasses import dataclass

from .types import SpeechRegion, TimeSpan


@dataclass(slots=True)
class TranscriptionWindow:
    window_id: int
    region: SpeechRegion
    window: TimeSpan


def build_transcription_windows(
    speech_regions: list[SpeechRegion],
    overlap_sec: float,
    audio_duration_sec: float,
) -> list[TranscriptionWindow]:
    windows: list[TranscriptionWindow] = []
    for index, region in enumerate(speech_regions):
        start = max(0.0, region.start - overlap_sec)
        end = min(audio_duration_sec, region.end + overlap_sec)
        windows.append(TranscriptionWindow(window_id=index, region=region, window=TimeSpan(start, end)))
    return windows
