from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from .config import ASRConfig
from .text import normalize_alignment_text
from .types import DraftUtterance, DraftWord, SpeechRegion


def transcribe_regions(audio: np.ndarray, sample_rate: int, regions: list[SpeechRegion], config: ASRConfig) -> list[DraftUtterance]:
    model = _load_model(config)
    utterances: list[DraftUtterance] = []
    utterance_id = 0
    for region in regions:
        start_idx = _to_sample_index(region.start, sample_rate, len(audio))
        end_idx = _to_sample_index(region.end, sample_rate, len(audio))
        if end_idx <= start_idx:
            continue
        chunk = audio[start_idx:end_idx]
        segments, _ = model.transcribe(
            chunk,
            language=config.language,
            beam_size=config.beam_size,
            vad_filter=False,
            condition_on_previous_text=True,
            temperature=0.0,
            word_timestamps=True,
        )
        for seg in segments:
            display_text = seg.text.strip()
            if not display_text:
                continue
            local_start = float(seg.start)
            local_end = float(seg.end)
            global_start = region.start + local_start
            global_end = region.start + local_end
            words: list[DraftWord] = []
            for word in getattr(seg, "words", None) or []:
                if word.start is None or word.end is None:
                    continue
                word_local_start = float(word.start)
                word_local_end = float(word.end)
                if word_local_end <= word_local_start:
                    continue
                words.append(
                    DraftWord(
                        text=str(getattr(word, "word", "")).strip(),
                        local_start=word_local_start,
                        local_end=word_local_end,
                        global_start=region.start + word_local_start,
                        global_end=region.start + word_local_end,
                    )
                )
            utterances.append(
                DraftUtterance(
                    utterance_id=utterance_id,
                    region_id=region.region_id,
                    region_start=region.start,
                    region_end=region.end,
                    local_start=local_start,
                    local_end=local_end,
                    global_start=global_start,
                    global_end=global_end,
                    display_text=display_text,
                    alignment_text=normalize_alignment_text(display_text),
                    words=words,
                )
            )
            utterance_id += 1
    return utterances


def _to_sample_index(sec: float, sample_rate: int, sample_count: int) -> int:
    index = int(round(sec * sample_rate))
    return max(0, min(sample_count, index))


def _load_model(config: ASRConfig) -> WhisperModel:
    if config.device == "cuda":
        _configure_windows_cuda_dll_path()
    return WhisperModel(config.model_size, device=config.device, compute_type=config.compute_type)


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
    for bin_dir in _candidate_cuda_bin_dirs():
        resolved = str(bin_dir.resolve())
        try:
            os.add_dll_directory(resolved)
        except (AttributeError, OSError):
            pass
        if resolved not in current_path:
            current_path = f"{resolved};{current_path}"
    os.environ["PATH"] = current_path
    for dll in ("cudnn_ops64_9.dll", "cublas64_12.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            pass
