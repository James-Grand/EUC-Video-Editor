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
  docs/              <- briefs and reference (music_brief.md = music direction)
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
- Also extract motion from the GoPro gyro / accelerometer / wind meter into
  `motion.csv` (works with GPS off): sharp turns / carving and rough-terrain
  jolts are highlight candidates, stopped stretches are avoided, and filler
  shots go on the busiest riding in each gap.
- Natural exclamations ("whoa", "wow", "let's go", list in `config.json`)
  are soft highlights, weaker than the deliberate cue phrases.

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
**Direction: `docs/music_brief.md`** (written from Tyler's Spotify playlist
"EUC '26"). Read it before changing anything about the music. Summary:
- Generate original music with Python (numpy/scipy synthesis) in
  `tools/make_music.py`. No samples, no AI models, nothing to license.
- Six styles, one per row of the brief's table (BPMs in `config.json`):
  | Style | BPM | Used for |
  |---|---|---|
  | `dnb` drum & bass | 174 | speed peaks, sustained fast stretches |
  | `electro` electro house | 128 | spoken highlight cues ("clip that", "watch this"...) |
  | `funk` funk-hop / electro-swing | 110 | cruising, talking (the bulk of a vlog) |
  | `indie` sunny indie electronic | 118 | scenery fillers |
  | `blues` gritty blues + stomp | 92 | accent only: the long video's cold-open sting |
  | `liquid` liquid drum & bass | 174 | outro (last ~30 s of the long video) |
- Long video follows the playlist arc: blues sting -> funk -> dnb/electro
  where the ride peaks -> indie on scenery -> liquid outro.
- Shorts: one track (dnb, electro or funk), drop lands on the short's best moment.
- Built in 4- and 8-bar phrases so cuts land on downbeats. Music hits within
  the first few seconds. Risers and snare rolls build into each new section.
  Ends clean on a ringing downbeat, not a fade.
- Leave room for voice: 1-4 kHz is dipped (`voice_pocket_db`).
- Sections over 3 min are split into separate "tracks" in new keys.
- Stems (drums, bass, music, vocals) go to `output/<ride-id>/music_stems/`.
- Run `python tools/make_music.py --library` to rebuild the demo of every style
  in `music/` (with `music/catalog.json` describing each).

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
- `music_stems/` (long video's music as drums / bass / music / vocals, for Resolve)
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

**Music** (from `docs/music_brief.md`):
- High energy and forward momentum. Fun and a bit cocky, not epic-cinematic or emotional.
- Short and punchy. Get to the hook fast; no ambient intros longer than ~8 bars.
- Put a drop or clean downbeat where the ride peaks.
- Vocals only as short chops or generic hype shouts ("hey!"). Never imitate a real artist.
- Blues is an accent (intro, tension, breakdown), never a whole track.
- Never use: trap, lo-fi chill, orchestral "epic trailer", sad piano, country,
  or anything slow or moody for long stretches.
- Reference artists in the brief are for understanding the vibe only. Never
  copy their melodies, hooks, vocal lines, signature sounds or lyrics.

### Tyler's feedback log
(Add dated entries here as feedback comes in.)
- **2026-11-13** — Music brief added (`docs/music_brief.md`, from the "EUC '26"
  playlist). Replaced the old chill/speed music with six brief-based styles,
  the blues -> funk -> dnb -> indie -> liquid arc, drops on highlight moments,
  a voice pocket, and stems. Added the Music rules above.
