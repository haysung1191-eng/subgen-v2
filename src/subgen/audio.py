from __future__ import annotations

from dataclasses import dataclass
import subprocess
import wave
from pathlib import Path

from .errors import FFmpegError


@dataclass(slots=True)
class AudioExtractionResult:
    output_path: Path
    command: list[str]
    sample_rate: int
    channels: int
    sample_width_bytes: int
    frame_count: int


def extract_audio_to_wav(input_path: Path, output_wav: Path, sample_rate: int = 16000) -> AudioExtractionResult:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = _decode_stderr(proc.stderr)
        raise FFmpegError(stderr.strip())

    with wave.open(str(output_wav), "rb") as wav_file:
        return AudioExtractionResult(
            output_path=output_wav,
            command=cmd,
            sample_rate=wav_file.getframerate(),
            channels=wav_file.getnchannels(),
            sample_width_bytes=wav_file.getsampwidth(),
            frame_count=wav_file.getnframes(),
        )


def _decode_stderr(raw: bytes) -> str:
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
