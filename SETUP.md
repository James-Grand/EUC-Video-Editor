# EUC Studio - setup notes (what's on this PC and how to run it)

Set up 2026-11-13. CLAUDE.md is the spec and style guide; this file is the
practical "how do I run it" sheet.

## What's installed

| Thing | Where | Why |
|---|---|---|
| Python 3.12 | `C:\Users\gleas\AppData\Local\Programs\Python\Python312` | runs the scripts |
| Project venv | `.venv\` (in this folder) | private copy of Python with just this project's packages |
| ffmpeg 8.0.1 (Gyan.dev full build) | installed by winget, on PATH | all video/audio work; has NVENC GPU encoding and the v360 360-reframe filter. **Stay on 8.0.x**: 8.1 and 9.x need NVIDIA driver 610+, this PC has 591 (if the driver gets updated, any version is fine) |
| git 2.56 | `C:\Program Files\Git` | version control for scripts and the style guide (not footage) |
| GitHub CLI 2.104 | `C:\Program Files\GitHub CLI` | signed in as James-Grand; remote is https://github.com/James-Grand/EUC-Video-Editor (public). Push with `git push`. |
| faster-whisper + large-v3 model | venv + `%USERPROFILE%\.cache\huggingface` | speech-to-text on the GPU |
| NVIDIA cuBLAS / cuDNN | inside the venv (pip packages) | lets faster-whisper use the RTX 4050 |
| numpy, scipy, soundfile, Pillow | venv | sync maths, music synthesis, contact sheets |

Hardware: RTX 4050 Laptop (6 GB VRAM), Ryzen 5 7235HS, 16 GB RAM, ~50 GB free on C:.

## Running it

1. Copy footage into `inbox\` (GoPro `.MP4`, Insta360 `.insv`).
2. Export each `.insv` from Insta360 Studio into `stitched\`
   (360 video, equirectangular, highest resolution, keep audio).
3. Double-click `run_ride.bat`, or in a terminal:

```bash
.venv\Scripts\python.exe tools\run_ride.py
```

Useful variants:

```bash
.venv\Scripts\python.exe tools\run_ride.py --no-render
```
(plans the edit but doesn't render; look at `work\<ride>\edit_plan.json` first)

```bash
.venv\Scripts\python.exe tools\run_ride.py 2026-11-05_ride1 --force
```
(one ride, redo transcription and renders)

```bash
.venv\Scripts\python.exe tools\setup_check.py
```
(checks every tool is present and the GPU is visible)

```bash
.venv\Scripts\python.exe tools\selftest.py
```
(builds fake footage in `work\_selftest\` and runs the whole pipeline on it)

## Where things land

- `work\<ride>\` - manifest, audio, transcript, speed.csv, contact sheets, edit_plan.json
- `output\<ride>\` - long.mp4, short_01.mp4..., long.srt, timeline.fcpxml, notes.md, run_log.md
- `music\` - reusable generated beds + catalog.json

## Tuning

Everything adjustable lives in `config.json`: 360 view angles, shot length cap,
silence cut, how many shorts, output resolution (1080p by default; set
`long_width`/`long_height` to 3840/2160 for 4K), audio levels.

## If something breaks

- `output\<ride>\run_log.md` has every step with timings and the full error.
- "ffmpeg not found": open a new terminal (PATH refreshes), or run `setup_check.py`.
- GPU not seen by Whisper: `setup_check.py` shows it; the transcriber falls back
  to CPU automatically (slow but works).
- "NVENC GPU encoder failed ... Driver does not support the required nvenc API
  version": ffmpeg got upgraded past 8.0.x. Either update the NVIDIA driver to
  610+ or reinstall ffmpeg 8.0.1:
  `winget install --id Gyan.FFmpeg -e --version 8.0.1 --source winget`.
  Renders still finish on the CPU encoder in the meantime, just slower.
- Disk full: `work\<ride>\render` and `tmp` are deleted after each render, but
  frames and audio stay. Delete `work\<ride>\` to reclaim space; it's all
  regenerable from inbox + stitched.
