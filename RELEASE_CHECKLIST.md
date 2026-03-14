# GitHub Release Checklist

Use this before making the repository public.

## Must Do

- Confirm `subgen_v2` is the part you want people to use.
- Keep old `subgen` code only if you want it visible as reference.
- Add a `LICENSE` file.
- Make sure `README.md` is accurate.
- Confirm `ffmpeg` requirement is documented.
- Confirm `whisperx` install step is documented.
- Confirm Windows usage is documented.
- Remove private/local paths from any remaining docs or screenshots.
- Check sample/debug outputs do not contain private file names you do not want to share.

## Recommended

- Add one short demo GIF or screenshot.
- Add one example debug folder tree.
- Add one known-issues section.
- Add one roadmap section.
- Add one small sample clip or a clearly described test clip format.

## Repo Hygiene

- Rename the repository if needed so the purpose is obvious.
- Add a `.gitignore` if missing.
- Exclude generated files:
  - `*-debug/`
  - `*.srt`
  - temp wav files
  - caches
- Check that large local media files are not tracked.

## Suggested Public Positioning

Recommended positioning:

- experimental
- timing-first
- Korean subtitle drafting
- local/offline
- alignment-debuggable

Not recommended positioning:

- production-ready
- high-accuracy general subtitle solution
- polished consumer app

## Quick Validation Before Push

```powershell
cd "D:\AI\자막생성"
python -m pytest -q tests\test_subgen_v2_cli.py tests\test_subgen_v2_debug.py tests\test_subgen_v2_subtitle.py tests\test_subgen_v2_align.py
python -m subgen_v2.cli --help
```
