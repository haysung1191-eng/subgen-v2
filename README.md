# subgen_v2

Experimental timing-first Korean subtitle drafting pipeline.

This project prioritizes subtitle timing inspectability over polished transcript quality. Whisper/faster-whisper proposes draft text. Aligned token timing is the primary timing authority. Draft ASR timing is used only as an explicit fallback and is marked in debug output.

## Status

`subgen_v2` is an experimental but usable CLI pipeline for local Korean subtitle drafting. It is not a polished end-user product.

Primary use case:

- generate Korean subtitle drafts locally
- keep subtitle onset timing inspectable
- separate text generation from timing generation
- output `.srt` only

Out of scope:

- translation
- diarization
- streaming
- GUI polish

## Pipeline

The clean v2 pipeline lives in `src/subgen_v2`.

Stages:

1. deterministic `ffmpeg` mono 16 kHz extraction
2. Silero VAD speech regions
3. faster-whisper draft ASR
4. WhisperX alignment
5. subtitle assembly from aligned tokens, with marked draft fallback when alignment coverage is weak
6. minimal cleanup
7. SRT output

## Timing Policy

The intended policy is:

- `aligned_tokens`: start and end timing came from aligned tokens
- `aligned_start_draft_end_fallback`: start came from aligned tokens, end used draft timing fallback
- `draft_fallback`: no aligned tokens were available for that subtitle, so draft timing was used
- `mixed`: the run used more than one timing source

After each run, the CLI prints fallback counts. When `--debug-dir` is used, `summary.json` and `05_subtitles_final.json` show per-subtitle timing sources.

## Windows Clean Install

Create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Install the package and WhisperX alignment extra:

```powershell
pip install -e ".[align-whisperx]"
```

For CUDA use, install a CUDA-enabled PyTorch build first using the official PyTorch selector, then verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Verify `ffmpeg`:

```powershell
ffmpeg -version
```

Check the local environment:

```powershell
subgen-v2 doctor
```

If `subgen-v2` is not on PATH, use:

```powershell
python -m subgen_v2.cli doctor
```

## Run

Recommended Filmora workflow:

```powershell
python -m subgen_v2.cli "D:\path\to\sample.mp4" -o "D:\path\to\sample.v2.srt" --preset filmora-ko --device cuda --align-device cuda --debug-dir "D:\path\to\sample-v2-debug"
python -m subgen_v2.review "D:\path\to\sample-v2-debug" --top 40
```

Import `sample.v2.srt` into Filmora first, then inspect the highest-risk lines from `timing_review.md` or `timing_review.html`.

Direct CLI:

```powershell
python -m subgen_v2.cli "D:\path\to\sample.mp4" -o "D:\path\to\sample.v2.srt" --device cuda --align-device cuda --debug-dir "D:\path\to\sample-v2-debug"
```

CPU fallback example:

```powershell
python -m subgen_v2.cli "D:\path\to\sample.mp4" -o "D:\path\to\sample.v2.srt" --device cpu --align-device cpu --compute-type int8 --debug-dir "D:\path\to\sample-v2-debug"
```

File picker helper:

```powershell
.\run_v2.ps1
```

The file picker uses `filmora-ko` by default and generates timing review files after the SRT is created.

CPU file picker:

```powershell
.\run_v2.ps1 -Device cpu -AlignDevice cpu -ComputeType int8
```

Tail-safe file picker:

```powershell
.\run_v2.ps1 -Preset filmora-ko-tail-safe
```

## Debug Output

When `--debug-dir` is provided, the pipeline writes:

- `01_regions.json`: padded VAD speech regions
- `02_draft.json`: faster-whisper draft utterances and diagnostic timestamps
- `03_aligned_tokens.json`: aligned tokens from the alignment backend
- `04_subtitles_raw.json`: subtitle segments before final cleanup
- `05_subtitles_final.json`: final SRT segments with timing source fields
- `summary.json`: run-level timing authority and fallback counts

The review command reads those files and writes:

- `timing_review.md`: quick manual review list
- `timing_review.html`: browser-friendly review table
- `timing_review.csv`: spreadsheet-friendly data
- `timing_review.json`: machine-readable rows
- `timing_review.srt`: review-only SRT overlay with risk tags

Important fields in `05_subtitles_final.json`:

- `start`
- `end`
- `token_start`
- `token_end`
- `draft_start`
- `draft_end`
- `aligned_token_count`
- `start_source`
- `end_source`
- `end_fallback_applied`
- `end_gap_ms`
- `timing_authority`
- `cleanup_adjusted`
- `cleanup_reason`
- `cleanup_start_delta_ms`
- `cleanup_end_delta_ms`

Final cleanup enforces positive durations and resolves overlaps deterministically. When no-overlap conflicts with the requested minimum duration, no-overlap wins and the cleanup delta is recorded in debug output.

## Tests

```powershell
python -m pytest -q
```

Focused v2 tests:

```powershell
python -m pytest -q tests\subgen_v2
```

## Current Limitations

- WhisperX is the default alignment backend for v0.1.
- Qwen3-ForcedAligner is a future experimental backend candidate, not the default.
- Some long Korean utterances may still need draft timing fallback when alignment coverage is weak.
- Timing metrics help diagnose problems but do not replace human sync review.

## Filmora Presets

- `current`: original stricter defaults
- `filmora-ko`: recommended editing preset for Korean travel videos
- `filmora-ko-tail-safe`: more generous subtitle endings for long tails
- `filmora-ko-strict`: tighter timing when subtitles feel too sticky

## License

MIT. See `LICENSE`.
