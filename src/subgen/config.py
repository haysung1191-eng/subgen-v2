from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class VADConfig:
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 120
    pad_ms: int = 80
    pre_roll_ms: int = 80
    post_roll_ms: int = 80
    merge_gap_ms: int = 140


@dataclass(slots=True)
class TranscriptionConfig:
    model_size: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 7
    language: str | None = None
    overlap_sec: float = 0.35


@dataclass(slots=True)
class AlignmentConfig:
    enabled: bool = True
    backend: str = "whisperx"
    device: str = "cuda"
    model_name: str | None = None
    min_word_confidence: float = 0.35
    boundary_slack_ms: int = 80
    normalization_mode: str = "conservative"
    fallback_on_failure: bool = True


@dataclass(slots=True)
class TimingConfig:
    min_duration_sec: float = 0.20
    hard_gap_sec: float = 0.06
    max_duration_sec: float = 4.0
    onset_nudge_ms: int = 0
    end_tail_padding_ms: int = 90
    max_end_tail_padding_ms: int = 180
    enable_acoustic_tail_extension: bool = False
    acoustic_tail_probe_ms: int = 180
    max_acoustic_tail_extension_ms: int = 120
    min_tail_energy_threshold: float = 0.012
    min_gap_to_next_ms: int = 60


@dataclass(slots=True)
class RuntimeConfig:
    log_level: str = "INFO"
    log_dir: Path | None = None
    batch_concurrency: int = 1
    debug_export_dir: Path | None = None


@dataclass(slots=True)
class PipelineConfig:
    input_path: Path
    output_path: Path
    temp_wav_path: Path
    sample_rate: int = 16000
    global_shift_ms: int = 0
    vad: VADConfig = field(default_factory=VADConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
