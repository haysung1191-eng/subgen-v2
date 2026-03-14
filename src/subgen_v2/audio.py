from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class AudioExtractionResult:
    output_path: Path
    command: list[str]
    sample_rate: int
    channels: int
    frame_count: int


def extract_audio(input_path: Path, output_wav: Path, sample_rate: int) -> AudioExtractionResult:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        "-acodec",
        "pcm_s16le",
        str(output_wav),
    ]
    proc = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(_decode_stderr(proc.stderr).strip())
    with wave.open(str(output_wav), "rb") as wf:
        return AudioExtractionResult(
            output_path=output_wav,
            command=command,
            sample_rate=wf.getframerate(),
            channels=wf.getnchannels(),
            frame_count=wf.getnframes(),
        )


def load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("Expected mono WAV")
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV")
        sample_rate = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, sample_rate


def duration_sec(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return len(audio) / float(sample_rate)


def _decode_stderr(raw: bytes) -> str:
    for encoding in ("utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
