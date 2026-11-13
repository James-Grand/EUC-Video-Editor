"""
extract_audio.py - Step 3. Pull the audio out of both cameras and clean it up.

Plain-language summary:
  * A ride can be several files per camera. We lay every file's audio onto one
    shared "ride timeline" that starts at the earliest file's start time, so
    one long WAV per camera covers the whole ride (silence fills the gaps).
    The coarse positions come from file timestamps; sync_cameras.py refines
    the GoPro's position later.
  * Insta360 voice track -> work/<ride>/audio/voice_clean.wav (48 kHz) and
    voice_16k.wav (16 kHz, what Whisper wants). Cleanup chain:
      high-pass 80 Hz (cut rumble) -> afftdn noise reduction ->
      gentle compression -> loudness normalise.
  * GoPro ambient track -> ambient.wav (48 kHz) and ambient_16k.wav, untouched
    except for a high-pass to tame wind rumble.
  * If a ride has no Insta360 footage, the GoPro track doubles as the voice
    track (muffled, but it's what we have).
  * Writes audio.json describing where every file sits on the ride timeline.

Run:  python tools/extract_audio.py <ride-id>
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from common import (CONFIG, FFMPEG, ffmpeg, log, read_json, run, run_log,
                    run_log_header, work_dir, write_json)


def _iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def ride_layout(ride_id: str) -> dict:
    """Work out, for every source file, where it starts on the ride timeline."""
    manifest = read_json(work_dir(ride_id) / "manifest.json")
    stitched = read_json(work_dir(ride_id) / "stitched.json", {"matches": []})
    t0 = _iso(manifest["ride_start"])

    sources = []
    # Insta360: use the stitched export (it has the voice audio and the video we edit).
    for m in stitched.get("matches", []):
        start = _iso(m["raw_start"])
        sources.append({
            "camera": "insta",
            "path": m["stitched_path"],
            "name": Path(m["stitched_path"]).name,
            "duration": m["duration"],
            "fps": m["fps"],
            "width": m["width"],
            "height": m["height"],
            "has_audio": m["has_audio"],
            "offset_in_ride": (start - t0).total_seconds(),
        })
    for f in manifest["files"]:
        if f["camera"] == "gopro":
            sources.append({
                "camera": "gopro",
                "path": f["path"],
                "name": f["name"],
                "duration": f["duration"],
                "fps": f["fps"],
                "width": f["width"],
                "height": f["height"],
                "has_audio": f["has_audio"],
                "offset_in_ride": (_iso(f["start"]) - t0).total_seconds(),
            })
    ride_end = max(s["offset_in_ride"] + s["duration"] for s in sources) if sources else 0
    return {"ride_id": ride_id, "ride_t0": manifest["ride_start"], "ride_duration": ride_end,
            "sources": sources}


def assemble_track(sources: list[dict], out_path: Path, sample_rate: int, total_s: float,
                   filters: str = "") -> None:
    """
    Build one mono WAV covering the whole ride from several files.
    Each input is delayed to its position (adelay), all are summed (amix),
    then padded/trimmed to the ride length.
    """
    if not sources:
        return
    args = []
    for s in sources:
        args += ["-i", s["path"]]
    chains = []
    for i, s in enumerate(sources):
        delay_ms = max(0, int(round(s["offset_in_ride"] * 1000)))
        chains.append(f"[{i}:a]aformat=channel_layouts=mono,aresample={sample_rate},"
                      f"adelay={delay_ms}|{delay_ms}[a{i}]")
    inputs = "".join(f"[a{i}]" for i in range(len(sources)))
    mix = f"{inputs}amix=inputs={len(sources)}:normalize=0:dropout_transition=0[mix]" if len(sources) > 1 \
        else f"[a0]acopy[mix]"
    tail = f"[mix]apad=whole_dur={total_s:.3f},atrim=0:{total_s:.3f}"
    if filters:
        tail += "," + filters
    tail += "[out]"
    graph = ";".join(chains + [mix, tail])
    ffmpeg([*args, "-filter_complex", graph, "-map", "[out]", "-ac", "1",
            "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out_path)])


def main(ride_id: str) -> dict:
    cfg = CONFIG["audio"]
    sr = cfg["sample_rate"]
    layout = ride_layout(ride_id)
    adir = work_dir(ride_id) / "audio"
    adir.mkdir(exist_ok=True)
    total = layout["ride_duration"]

    insta = [s for s in layout["sources"] if s["camera"] == "insta" and s["has_audio"]]
    gopro = [s for s in layout["sources"] if s["camera"] == "gopro" and s["has_audio"]]

    run_log_header(ride_id, "Audio extraction")

    voice_filters = (
        f"highpass=f={cfg['voice_highpass_hz']},"
        f"afftdn=nf={cfg['voice_noise_floor_db']}:nr=12:nt=w,"
        "acompressor=threshold=-20dB:ratio=3:attack=10:release=200:makeup=2,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    ambient_filters = "highpass=f=60"

    voice_source = "insta" if insta else ("gopro" if gopro else None)
    layout["voice_source"] = voice_source

    if insta:
        assemble_track(insta, adir / "voice_clean.wav", sr, total, voice_filters)
        run_log(ride_id, f"voice_clean.wav from {len(insta)} Insta360 file(s), {total:.0f}s")
    if gopro:
        assemble_track(gopro, adir / "ambient.wav", sr, total, ambient_filters)
        run_log(ride_id, f"ambient.wav from {len(gopro)} GoPro file(s), {total:.0f}s")
        if not insta:
            # GoPro-only ride: clean the GoPro track as the voice track too.
            assemble_track(gopro, adir / "voice_clean.wav", sr, total, voice_filters)
            run_log(ride_id, "no Insta360 audio - GoPro track used as voice track (expect muffled voice)")

    # 16 kHz mono copies for Whisper and for the sync step.
    for name in ("voice_clean", "ambient"):
        src = adir / f"{name}.wav"
        if src.exists():
            ffmpeg(["-i", str(src), "-ar", "16000", "-ac", "1", str(adir / f"{name.replace('_clean', '')}_16k.wav")])

    write_json(work_dir(ride_id) / "audio.json", layout)
    if voice_source is None:
        run_log(ride_id, "WARNING: no audio found on any camera")
    return layout


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: extract_audio.py <ride-id>")
    main(sys.argv[1])
