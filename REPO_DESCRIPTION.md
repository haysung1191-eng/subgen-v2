# Repository Description Draft

## GitHub Repository Name Ideas

- `subgen-v2`
- `korean-subtitle-timing`
- `timing-first-subtitles`
- `korean-subtitle-draft-cli`

## Short Description

Minimal timing-first Korean subtitle drafting pipeline with aligned-token timing and per-stage debug JSON.

## Longer Description

`subgen_v2` is an experimental local CLI pipeline for Korean subtitle drafting.
It separates draft text generation from timing generation, uses aligned tokens as the timing authority, and writes per-stage JSON artifacts so subtitle timing can be inspected and debugged directly.

## Suggested GitHub Topics

- `python`
- `subtitles`
- `whisper`
- `whisperx`
- `forced-alignment`
- `korean`
- `speech-to-text`
- `ffmpeg`
- `offline`
- `windows`

## Suggested First Paragraph For README Or Pinned Post

This project is a minimal timing-first Korean subtitle drafting pipeline for local/offline use. It is designed for workflows where subtitle timing needs to be inspectable and stable, while text can be cleaned later. The pipeline keeps timing authority in aligned tokens and dumps stage-by-stage JSON for debugging.
