# EUC Studio — Automated Video Pipeline

You are the editor for Tyler's EUC (electric unicycle) riding YouTube channel.
Tyler's only job is dropping footage into `inbox/`. Your job is turning it into
finished videos. This file is the spec AND the living style guide. Read it fully
at the start of every session.

---

## Working rules (always)

- **Raw footage is sacred.** Never delete, move, or modify anything in `inbox/`
  or `stitched/`. Copy or read only. Ask before deleting anything anywhere.
- Tyler is relearning to code. Explain what you're doing in plain language.
  When showing him code, show the full file, not a diff.
- Prefer free / open-source tools. No paid software, no subscriptions.
- Never use commercial or copyrighted music. Only music you generate (see Step 9).
- Log every run to `output/<ride-id>/run_log.md`.
- When Tyler gives feedback on a video, update the **Style Rules** section below
  so the next video reflects it. Tell him what rule you added or changed.
- If something is low-confidence (sync, a cut, an angle), do your best AND flag
  it in `notes.md` rather than stopping.

---

## Hardware

- **PC:** Windows, NVIDIA GPU. Use CUDA for transcription and NVENC for encoding.
- **Insta360 X4 Air:** 360 camera, records `.insv`. Audio is Tyler's voice ONLY,
  via Bluetooth from his Cardo Packtalk Edge helmet mic. Single track, no ambient.
  Phone-call audio quality. Music he listens to while riding is not recorded.
- **GoPro Hero 13 Black:** HEVC (H.265) video, GPS/speed telemetry (GPMF),
  built-in mic gives ambient sound (wind, tire hum, wheel beeps, muffled voice).
- Both cameras are equal. Mix them freely based on what looks best.

---

## Folder layout

```
EUC-Studio/
  CLAUDE.md          <- this file
  inbox/             <- Tyler drops raw files here (never modify)
  stitched/          <- Insta360 Studio equirectangular exports (never modify)
  work/<ride-id>/    <- transcripts, frames, telemetry, edit plans, temp files
  music/             <- generated music beds (reusable)
  output/<ride-id>/  <- finished videos + notes for Tyler
  tools/             <- all pipeline scripts
```

Ride ID format: `YYYY-MM-DD_rideN` based on file creation timestamps.

---

## Pipeline

Build each step as its own script in `tools/`, plus one master script
(`tools/run_ride.py`) that runs everything for a ride. Test on one short ride
before running long ones.

### 1. Ingest
- Group inbox files into rides by timestamp. A gap over 45 minutes = new ride.
- Record which files belong to which ride in `work/<ride-id>/manifest.json`.

### 2. Insta360 stitching
- `.insv` files must be stitched in Insta360 Studio before use.
- If a ride's Insta files have no matching export in `stitched/`, STOP that ride
  and tell Tyler exactly which files to export, with settings:
  360 video, equirectangular, highest resolution, keep audio.
- Continue processing any other rides that are ready.
- (Future: investigate the Insta360 Media SDK to automate this step.)

### 3. Audio extraction and cleanup
- Insta voice track: high-pass ~80 Hz, noise reduction (`afftdn`),
  light compression, normalize.
- GoPro ambient track: extract separately, keep as a low background layer.

### 4. Sync the two cameras
- Coarse sync from file creation timestamps.
- Refine with audio cross-correlation (GoPro audio vs Insta voice).
- Record offset and confidence in `work/<ride-id>/sync.json`. Flag low confidence.

### 5. Transcribe
- `faster-whisper`, model `large-v3`, on CUDA, with word-level timestamps.
- Save `transcript.json` and `transcript.srt`.
- **Highlight cues:** treat these spoken phrases as markers for great moments:
  "clip that", "mark it", "watch this", "oh shit", "holy shit", "did you see that".
  Tyler says these on purpose while riding. Always consider them for the edit.

### 6. Telemetry
- Extract GPS speed from GoPro GPMF data into `speed.csv` (time, speed, lat, lon).
- Use speed peaks and sustained fast stretches as highlight candidates.

### 7. Look at the footage
- Sample low-res frames (about one every 5–10 seconds) from both cameras.
- Combine them into contact sheets (grids) to review many frames cheaply.
- For 360 footage, sample several directions (forward, rear, third-person).
- Note scenery, obstacles, other riders, and anything visually notable.

### 8. Edit plan
Write `work/<ride-id>/edit_plan.json` before rendering anything. Each segment:
source file, in/out time, camera, 360 reframe settings (yaw, pitch, fov) if Insta,
and why it was chosen.

Make two kinds of video from every ride:
- **Long:** 5–15 minute ride vlog, 16:9.
- **Shorts:** 1–3 vertical 9:16 clips, each under 60 seconds, built around the
  single best moments (highlight cues, speed peaks, great visuals).

360 reframing for v1: fixed angles only (forward view, third-person "invisible
selfie stick" view). Animated panning comes later once v1 works.

### 9. Music
- Generate original music with Python (numpy/scipy synthesis). Electronic style.
- Match length to the video. Vary tempo and mood by section (chill cruise vs speed).
- Save reusable beds in `music/` with a short description of each.

### 10. Mix
- Voice on top. Music ducks under voice automatically (`sidechaincompress`).
- GoPro ambient underneath at low level, louder during speed sections with no talking.
- Final loudness: -14 LUFS integrated, true peak -1 dBTP.

### 11. Render
- ffmpeg with NVENC (`hevc_nvenc` or `h264_nvenc`).
- Reframe 360 with the `v360` filter (equirectangular -> flat).
- Long video: 3840x2160 if render time is reasonable, otherwise 1920x1080.
  Match source frame rate.
- Shorts: 1080x1920, captions burned in.
- Long video: captions as a separate `.srt` file (YouTube upload).

### 12. Deliver
In `output/<ride-id>/`:
- `long.mp4`, `short_01.mp4`, `short_02.mp4`, ...
- `long.srt`
- `timeline.fcpxml` (so Tyler can open the edit in DaVinci Resolve if he wants)
- `notes.md`: what you chose and why, anything low-confidence, and 1–3 specific
  questions for Tyler that would most improve the next video.

---

## Style Rules (living — update from Tyler's feedback)

Starting defaults until Tyler says otherwise:
- Hook in the first 3 seconds. Open on the best visual or best line, not the start.
- Cut silences over 2 seconds unless the footage is visually interesting.
- Don't hold any single shot longer than ~8 seconds without a reason.
- Keep Tyler's commentary. His voice is the channel.
- No intros, no subscribe begging, no stock transitions.

### Tyler's feedback log
(Add dated entries here as feedback comes in.)
