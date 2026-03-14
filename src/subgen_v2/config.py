from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000


@dataclass(slots=True)
class VADConfig:
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 150
    pre_roll_ms: int = 100
    post_roll_ms: int = 100
    merge_gap_ms: int = 150


@dataclass(slots=True)
class ASRConfig:
    model_size: str = "large-v2"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 8
    language: str = "ko"


@dataclass(slots=True)
class AlignmentConfig:
    backend: str = "whisperx"
    device: str = "cuda"
    model_name: str | None = None
    min_word_confidence: float = 0.35
    utterance_padding_ms: int = 180


@dataclass(slots=True)
class SubtitleConfig:
    hold_ms: int = 180
    min_gap_to_next_ms: int = 50
    min_duration_ms: int = 220
    tiny_overlap_fix_ms: int = 20
    end_fallback_threshold_ms: int = 320


@dataclass(slots=True)
class DebugConfig:
    enabled: bool = False
    output_dir: Path | None = None


@dataclass(slots=True)
class PipelineConfig:
    input_path: Path
    output_path: Path
    temp_wav_path: Path
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
