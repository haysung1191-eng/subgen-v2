from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def load_mono_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    if channels != 1:
        raise ValueError("Expected mono WAV input")
    if sampwidth != 2:
        raise ValueError("Expected 16-bit PCM WAV input")

    pcm = np.frombuffer(raw, dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    return audio, sample_rate


def get_audio_duration_sec(num_samples: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return num_samples / float(sample_rate)
