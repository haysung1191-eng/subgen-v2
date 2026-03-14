from pathlib import Path

from subgen.runner import _ensure_unique_output_path, _effective_batch_concurrency
from subgen.config import AlignmentConfig, PipelineConfig, TimingConfig, TranscriptionConfig, VADConfig


def _make_config(path: Path, *, device: str = "cpu", align_device: str = "cpu") -> PipelineConfig:
    return PipelineConfig(
        input_path=path,
        output_path=path.with_suffix(".srt"),
        temp_wav_path=path.with_suffix(".wav"),
        vad=VADConfig(),
        transcription=TranscriptionConfig(device=device),
        alignment=AlignmentConfig(device=align_device),
        timing=TimingConfig(),
    )


def test_effective_batch_concurrency_stays_on_cpu() -> None:
    configs = [_make_config(Path("a.mp4"), device="cpu", align_device="cpu")]
    assert _effective_batch_concurrency(configs, 3) == 3


def test_effective_batch_concurrency_forces_single_on_cuda() -> None:
    configs = [_make_config(Path("a.mp4"), device="cuda", align_device="cuda")]
    assert _effective_batch_concurrency(configs, 3) == 1


def test_unique_output_path_avoids_collision(tmp_path: Path) -> None:
    target = tmp_path / "out.srt"
    target.write_text("x", encoding="utf-8")
    candidate = _ensure_unique_output_path(target)
    assert candidate != target
    assert candidate.name == "out.1.srt"
