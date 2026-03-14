from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .align import align_utterances
from .audio import duration_sec, extract_audio, load_mono_wav
from .config import PipelineConfig
from .debug import write_debug_json
from .draft_asr import transcribe_regions
from .srt import write_srt
from .subtitle import build_subtitles
from .vad import detect_speech_regions


@dataclass(slots=True)
class PipelineResult:
    output_path: Path
    debug_dir: Path | None
    timing_authority: str
    region_count: int
    draft_count: int
    aligned_token_count: int
    subtitle_count: int


def run_pipeline(config: PipelineConfig, progress: Callable[[str], None] | None = None) -> PipelineResult:
    _progress(progress, f"extract_audio input={config.input_path}")
    extract_audio(config.input_path, config.temp_wav_path, config.audio.sample_rate)
    _progress(progress, f"extract_audio_done wav={config.temp_wav_path}")

    if config.debug.enabled and config.debug.output_dir is not None:
        config.debug.output_dir.mkdir(parents=True, exist_ok=True)

    _progress(progress, "load_audio")
    audio, sample_rate = load_mono_wav(config.temp_wav_path)
    audio_duration = duration_sec(audio, sample_rate)
    _progress(progress, f"load_audio_done duration={audio_duration:.2f}s")

    _progress(progress, "vad")
    regions = detect_speech_regions(audio, sample_rate, config.vad)
    if config.debug.enabled and config.debug.output_dir is not None:
        write_debug_json(config.debug.output_dir, "01_regions.json", [region.to_dict() for region in regions])
    _progress(progress, f"vad_done regions={len(regions)}")

    _progress(progress, "draft_asr")
    drafts = transcribe_regions(audio, sample_rate, regions, config.asr)
    if config.debug.enabled and config.debug.output_dir is not None:
        write_debug_json(config.debug.output_dir, "02_draft.json", [draft.to_dict() for draft in drafts])
    _progress(progress, f"draft_asr_done drafts={len(drafts)}")

    _progress(progress, "align")
    aligned_tokens = align_utterances(audio, sample_rate, drafts, config.alignment, language=config.asr.language)
    if config.debug.enabled and config.debug.output_dir is not None:
        write_debug_json(config.debug.output_dir, "03_aligned_tokens.json", [token.to_dict() for token in aligned_tokens])
    _progress(progress, f"align_done aligned_tokens={len(aligned_tokens)}")

    _progress(progress, "build_subtitles")
    subtitles = build_subtitles(drafts, aligned_tokens, config.subtitle)
    if config.debug.enabled and config.debug.output_dir is not None:
        write_debug_json(config.debug.output_dir, "04_subtitles_raw.json", [segment.to_dict() for segment in subtitles.raw_segments])
        write_debug_json(config.debug.output_dir, "05_subtitles_final.json", [segment.to_dict() for segment in subtitles.final_segments])
    _progress(progress, f"build_subtitles_done raw={len(subtitles.raw_segments)} final={len(subtitles.final_segments)}")

    _progress(progress, f"write_srt output={config.output_path}")
    write_srt(subtitles.final_segments, config.output_path)
    _progress(progress, "write_srt_done")

    return PipelineResult(
        output_path=config.output_path,
        debug_dir=config.debug.output_dir if config.debug.enabled else None,
        timing_authority="aligned_tokens",
        region_count=len(regions),
        draft_count=len(drafts),
        aligned_token_count=len(aligned_tokens),
        subtitle_count=len(subtitles.final_segments),
    )


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
