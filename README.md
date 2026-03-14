# subgen_v2

Experimental timing-first Korean subtitle drafting pipeline.

This project prioritizes subtitle timing inspectability over polished transcript quality. Whisper is used for draft text, while aligned token timing is treated as the timing authority. The current focus is local/offline experimentation, not production-grade subtitle quality.


Minimal timing-first Korean subtitle drafting pipeline.

This repository is for one practical use case:

- generate Korean subtitle drafts locally
- keep subtitle onset timing inspectable
- separate text generation from timing generation
- use aligned tokens as the only timing authority
- output `.srt` only

The main rule of `v2` is:

> Whisper proposes text, but aligned tokens determine time.

## Status

`subgen_v2` is an experimental but usable CLI pipeline.

It is not a polished end-user product.
It is intended for:

- Korean subtitle drafting
- timing inspection
- debugging alignment behavior
- post-edit workflows where text can be cleaned later by another model or by hand

## What Is Included

The clean `v2` pipeline lives in [src/subgen_v2](/D:/AI/자막생성/src/subgen_v2).

Stages:

1. deterministic `ffmpeg` mono 16k extraction
2. Silero VAD speech regions
3. faster-whisper draft ASR
4. whisperx alignment
5. subtitle assembly from aligned tokens
6. minimal cleanup
7. SRT output

Optional per-stage debug dumps:

- `01_regions.json`
- `02_draft.json`
- `03_aligned_tokens.json`
- `04_subtitles_raw.json`
- `05_subtitles_final.json`

## Timing Authority

`subgen_v2` has exactly one timing authority:

- aligned token timestamps from [align.py](/D:/AI/자막생성/src/subgen_v2/align.py)

Draft ASR timestamps are not the final source of truth.
Post-processing is intentionally minimal and should not invent timing.

## Requirements

- Windows
- Python 3.10+
- NVIDIA GPU recommended
- `ffmpeg` on `PATH`
- PyTorch
- `faster-whisper`
- `silero-vad`
- `whisperx`

Install editable package:

```powershell
cd "D:\AI\자막생성"
pip install -e .
pip install ".[align]"
```

## Main Commands

Run `v2` directly:

```powershell
cd "D:\AI\자막생성"
python -m subgen_v2.cli "D:\path\to\sample.mp4" -o "D:\path\to\sample.v2.srt" --device cuda --align-device cuda --debug-dir "D:\path\to\sample-v2-debug"
```

Or use the file picker helper:

```powershell
cd "D:\AI\자막생성"
.\run_v2.ps1
```

That script:

- opens a file picker
- lets you choose one media file
- writes `filename.v2.srt`
- writes `filename-v2-debug`

## Debugging

If timing looks wrong, inspect these files first:

- `03_aligned_tokens.json`
- `05_subtitles_final.json`

Useful fields in `05_subtitles_final.json`:

- `start`
- `end`
- `token_start`
- `token_end`
- `aligned_token_count`
- `start_source`
- `end_source`
- `end_fallback_applied`
- `end_gap_ms`
- `timing_authority`

These make it easier to see whether a bad subtitle came from:

- poor alignment coverage
- missing aligned tokens
- fallback to draft end
- minimal overlap cleanup

## Current Practical Notes

What `v2` already does well:

- onset timing is easier to reason about than the old pipeline
- timing provenance is easier to inspect
- stage-by-stage JSON makes debugging possible

What is still weak:

- alignment coverage on some long Korean utterances
- draft ASR quality in some clips
- end timing on weak or stretched sentence endings

## Tests

Run focused `v2` tests:

```powershell
cd "D:\AI\자막생성"
python -m pytest -q tests\test_subgen_v2_cli.py tests\test_subgen_v2_debug.py tests\test_subgen_v2_subtitle.py tests\test_subgen_v2_align.py
```

## Project Layout

- [src/subgen_v2](/D:/AI/자막생성/src/subgen_v2): clean timing-first pipeline
- [run_v2.ps1](/D:/AI/자막생성/run_v2.ps1): file picker runner
- [tests/test_subgen_v2_cli.py](/D:/AI/자막생성/tests/test_subgen_v2_cli.py): CLI sanity test
- [tests/test_subgen_v2_debug.py](/D:/AI/자막생성/tests/test_subgen_v2_debug.py): debug file output test
- [tests/test_subgen_v2_subtitle.py](/D:/AI/자막생성/tests/test_subgen_v2_subtitle.py): subtitle assembly tests
- [tests/test_subgen_v2_align.py](/D:/AI/자막생성/tests/test_subgen_v2_align.py): alignment window scoping test

## Before Publishing

See:

- [RELEASE_CHECKLIST.md](/D:/AI/자막생성/RELEASE_CHECKLIST.md)
- [REPO_DESCRIPTION.md](/D:/AI/자막생성/REPO_DESCRIPTION.md)

There is currently no `LICENSE` file in this repository.
Choose a license before making the repository public.
