from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from faster_whisper import WhisperModel
from tqdm import tqdm

from .config import TranscriptionConfig
from .errors import ModelLoadError, TranscriptionError
from .segmentation import TranscriptionWindow
from .text_normalization import normalize_alignment_text
from .types import DraftTranscriptUnit, DraftWord


@dataclass(slots=True)
class LoadedModel:
    model: WhisperModel
    device: str


@dataclass(slots=True)
class DraftTranscriptionResult:
    units: list[DraftTranscriptUnit]
    device: str
    warning: str | None = None


class DraftTranscriber(Protocol):
    def transcribe_windows(
        self,
        audio: np.ndarray,
        sample_rate: int,
        windows: list[TranscriptionWindow],
        config: TranscriptionConfig,
        normalization_mode: str,
        show_progress: bool = True,
    ) -> DraftTranscriptionResult:
        ...


def _candidate_cuda_bin_dirs() -> list[Path]:
    candidates: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        nvidia_dir = Path(entry) / "nvidia"
        if not nvidia_dir.exists():
            continue
        for sub in ("cudnn", "cublas", "cuda_runtime"):
            bin_dir = nvidia_dir / sub / "bin"
            if bin_dir.exists():
                candidates.append(bin_dir)
    return candidates


def _configure_windows_cuda_dll_path() -> None:
    if os.name != "nt":
        return

    current_path = os.environ.get("PATH", "")
    seen: set[str] = set()
    for bin_dir in _candidate_cuda_bin_dirs():
        resolved = str(bin_dir.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            os.add_dll_directory(resolved)
        except (AttributeError, OSError):
            pass
        if resolved not in current_path:
            current_path = f"{resolved};{current_path}"
    os.environ["PATH"] = current_path


def _cuda_runtime_ready() -> bool:
    _configure_windows_cuda_dll_path()
    try:
        ctypes.WinDLL("cudnn_ops64_9.dll")
        ctypes.WinDLL("cublas64_12.dll")
        return True
    except OSError:
        return False


def load_model(config: TranscriptionConfig) -> LoadedModel:
    if config.device == "cuda" and not _cuda_runtime_ready():
        try:
            fallback = WhisperModel(config.model_size, device="cpu", compute_type="int8")
            return LoadedModel(model=fallback, device="cpu")
        except Exception as exc:
            raise ModelLoadError(f"CUDA runtime unavailable and CPU fallback model load failed: {exc}") from exc

    try:
        model = WhisperModel(config.model_size, device=config.device, compute_type=config.compute_type)
        return LoadedModel(model=model, device=config.device)
    except Exception as exc:
        try:
            fallback = WhisperModel(config.model_size, device="cpu", compute_type="int8")
            return LoadedModel(model=fallback, device="cpu")
        except Exception as fallback_exc:
            raise ModelLoadError(f"Primary model load failed: {exc}; CPU fallback failed: {fallback_exc}") from fallback_exc


def _to_sample_index(sec: float, sample_rate: int, sample_count: int) -> int:
    idx = int(round(sec * sample_rate))
    return max(0, min(sample_count, idx))


class FasterWhisperDraftTranscriber:
    def transcribe_windows(
        self,
        audio: np.ndarray,
        sample_rate: int,
        windows: list[TranscriptionWindow],
        config: TranscriptionConfig,
        normalization_mode: str,
        show_progress: bool = True,
    ) -> DraftTranscriptionResult:
        loaded = load_model(config)
        model = loaded.model
        active_device = loaded.device
        warning: str | None = None
        units: list[DraftTranscriptUnit] = []
        unit_id = 0

        iterator = tqdm(windows, desc=f"Transcribing ({loaded.device})", unit="window") if show_progress else windows
        for item in iterator:
            s_idx = _to_sample_index(item.window.start, sample_rate, len(audio))
            e_idx = _to_sample_index(item.window.end, sample_rate, len(audio))
            if e_idx <= s_idx:
                continue

            chunk = audio[s_idx:e_idx]
            try:
                segments, _ = model.transcribe(
                    chunk,
                    language=config.language,
                    beam_size=config.beam_size,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    word_timestamps=True,
                )
            except Exception as exc:
                if active_device == "cuda":
                    try:
                        model = WhisperModel(config.model_size, device="cpu", compute_type="int8")
                        active_device = "cpu"
                        warning = "CUDA transcription failed and CPU fallback was used."
                        if show_progress and hasattr(iterator, "set_description"):
                            iterator.set_description("Transcribing (cpu-fallback)")
                        segments, _ = model.transcribe(
                            chunk,
                            language=config.language,
                            beam_size=config.beam_size,
                            vad_filter=False,
                            condition_on_previous_text=False,
                            temperature=0.0,
                            word_timestamps=True,
                        )
                    except Exception as fallback_exc:
                        raise TranscriptionError(f"CUDA transcription failed: {exc}; CPU fallback failed: {fallback_exc}") from fallback_exc
                else:
                    raise TranscriptionError(f"CPU transcription failed: {exc}") from exc

            for seg in segments:
                display_text = seg.text.strip()
                if not display_text:
                    continue

                window_local_start = float(seg.start)
                window_local_end = float(seg.end)
                window_global_start = item.window.start + window_local_start
                window_global_end = item.window.start + window_local_end
                rough_start = max(item.window.start, window_global_start)
                rough_end = min(item.window.end, window_global_end)
                if rough_end <= rough_start:
                    continue

                words: list[DraftWord] = []
                for word in getattr(seg, "words", None) or []:
                    if word.start is None or word.end is None:
                        continue
                    start = item.window.start + float(word.start)
                    end = item.window.start + float(word.end)
                    if end <= start:
                        continue
                    words.append(
                        DraftWord(
                            text=str(getattr(word, "word", "")).strip(),
                            start=start,
                            end=end,
                            source_window_id=item.window_id,
                            source_region_id=item.region.region_id,
                        )
                    )

                units.append(
                        DraftTranscriptUnit(
                            unit_id=unit_id,
                            source_window_id=item.window_id,
                            source_region_id=item.region.region_id,
                            window_local_start=window_local_start,
                            window_local_end=window_local_end,
                            window_global_start=window_global_start,
                            window_global_end=window_global_end,
                            rough_start=rough_start,
                            rough_end=rough_end,
                            display_text=display_text,
                        alignment_text=normalize_alignment_text(display_text, normalization_mode),
                        words=words,
                    )
                )
                unit_id += 1

        return DraftTranscriptionResult(units=units, device=active_device, warning=warning)


def transcribe_windows(
    audio: np.ndarray,
    sample_rate: int,
    windows: list[TranscriptionWindow],
    config: TranscriptionConfig,
    normalization_mode: str = "conservative",
    show_progress: bool = True,
) -> DraftTranscriptionResult:
    return FasterWhisperDraftTranscriber().transcribe_windows(
        audio=audio,
        sample_rate=sample_rate,
        windows=windows,
        config=config,
        normalization_mode=normalization_mode,
        show_progress=show_progress,
    )
