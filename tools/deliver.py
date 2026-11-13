"""
deliver.py - Step 12. Write output/<ride-id>/notes.md for Tyler.

Plain-language summary:
  * Summarises the ride (length, cameras, how much talking, top speed).
  * Explains the hook and each short, and why.
  * Lists everything low-confidence (sync, fillers, missing data).
  * Asks 1-3 specific questions whose answers would most improve the next cut.
  * Lists the delivered files.

Run:  python tools/deliver.py <ride-id>
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from common import CONFIG, fmt_time, human_size, output_dir, read_json, run_log, run_log_header, work_dir


def main(ride_id: str) -> Path:
    wd, od = work_dir(ride_id), output_dir(ride_id)
    manifest = read_json(wd / "manifest.json", {})
    sync = read_json(wd / "sync.json", {})
    plan = read_json(wd / "edit_plan.json", {})
    transcript = read_json(wd / "transcript.json", {})
    cues = read_json(wd / "cues.json", [])
    speed = read_json(wd / "speed_highlights.json", {})
    frames = read_json(wd / "frames.json", {})

    spoken = sum(i["end"] - i["start"] for i in transcript.get("speech_islands", []))
    ride_len = plan.get("ride_duration", 0)
    long = plan.get("long", {})
    shorts = plan.get("shorts", [])
    low = list(plan.get("low_confidence", []))
    if sync.get("low_confidence"):
        low.insert(0, f"Camera sync confidence {sync.get('confidence')} (threshold {CONFIG['sync']['min_confidence']}).")

    L = []
    L.append(f"# {ride_id} - editor's notes\n")
    L.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by the EUC Studio pipeline (v1)._\n")
    L.append("## The ride\n")
    L.append(f"- Footage: {fmt_time(ride_len)} across {len(manifest.get('files', []))} file(s), cameras: "
             f"{', '.join(manifest.get('cameras', []))}")
    L.append(f"- Talking: {fmt_time(spoken)} ({(spoken / ride_len * 100) if ride_len else 0:.0f}% of the ride)")
    if speed.get("max_speed_mps"):
        L.append(f"- Top speed: {speed['max_speed_mps']*2.237:.1f} mph, average while moving "
                 f"{speed.get('avg_speed_mps', 0)*2.237:.1f} mph")
    else:
        L.append("- Speed: no GPS data available")
    if sync.get("method") and sync["method"] != "single-camera":
        L.append(f"- Sync: GoPro offset {sync['ambient_lag_s']:+.2f}s, confidence {sync['confidence']}")
    L.append(f"- Highlight cues heard: {len(cues)}" +
             (" - " + ", ".join(f"'{c['cue']}' at {fmt_time(c['time'])}" for c in cues) if cues else ""))
    L.append("")

    L.append("## What I made\n")
    L.append(f"**long.mp4** - {fmt_time(long.get('duration', 0))}, {len(long.get('segments', []))} shots.")
    if long.get("hook"):
        L.append(f"Opens on: {'; '.join(long['hook']['reasons'])}.")
    cams = {}
    for s in long.get("segments", []):
        key = s["camera"] + (f"/{s['view']}" if s.get("view") else "")
        cams[key] = cams.get(key, 0) + s["duration"]
    if cams:
        L.append("Camera time: " + ", ".join(f"{k} {fmt_time(v)}" for k, v in sorted(cams.items(), key=lambda x: -x[1])))
    L.append("")
    for sh in shorts:
        L.append(f"**{sh['id']}.mp4** - {fmt_time(sh['duration'])}, built around {fmt_time(sh['anchor'])}: "
                 f"{'; '.join(sh['reasons'])}.")
    if not shorts:
        L.append("No shorts this time: nothing scored high enough. Say a cue ('clip that') and I'll build one around it.")
    L.append("")

    # Music: what each video got (written by render.py, styles from docs/music_brief.md).
    music_lines = []
    for which in ["long"] + [s["id"] for s in shorts]:
        mp = read_json(wd / f"music_plan_{which}.json", {})
        if mp.get("sections"):
            parts, t = [], 0.0
            for s in mp["sections"]:
                drop = f", drop at {fmt_time(t + s['drop_at'])}" if "drop_at" in s else ""
                parts.append(f"{fmt_time(t)} {s['style']} ({s['bpm']} BPM{drop})")
                t += s["duration"]
            music_lines.append(f"- **{which}**: " + " -> ".join(parts))
    if music_lines:
        L.append("## Music\n")
        L.append("All original, generated in `tools/make_music.py` from the EUC '26 brief (`docs/music_brief.md`).")
        L += music_lines
        stems = od / "music_stems"
        if stems.exists():
            L.append(f"- Stems for Resolve (drums / bass / music / vocals, full level, before ducking): `music_stems/`")
        L.append("")

    L.append("## Low confidence / please check\n")
    L += [f"- {n}" for n in low] or ["- Nothing flagged."]
    L.append("")

    # Questions: pick the most useful ones for this ride.
    qs = []
    if any(s.get("camera") == "insta" for s in plan.get("sources", [])):
        qs.append("360 angle: I used yaw 0 as 'forward' and yaw 180 / pitch -25 as the third-person view. "
                  "Do they actually point forward and back at you? If not, tell me the camera's mount position "
                  "(e.g. 'stick pointing up behind my head') and I'll fix the angles in config.json.")
    if cues:
        qs.append("When you say 'clip that', how far back should the clip reach? I'm using 12 seconds before you say it.")
    else:
        qs.append("No cue words were heard this ride. Were you saying them, or should the edit rely more on speed and scenery?")
    if any("filler" in n for n in low):
        qs.append("Fillers: for quiet stretches, do you prefer scenery shots, speed shots, or just cutting them out?")
    if sync.get("low_confidence"):
        qs.append("Do the GoPro shots look in sync with your voice? If not, note a timestamp where it's off.")
    qs.append("Pace: shots are capped at 8 s and silences over 2 s are cut. Too fast, too slow, or about right?")
    L.append("## Questions for Tyler\n")
    L += [f"{i}. {q}" for i, q in enumerate(qs[:3], 1)]
    L.append("")

    L.append("## Files\n")
    for p in sorted(od.iterdir()):
        if p.is_file() and p.name != "notes.md":
            L.append(f"- `{p.name}` ({human_size(p.stat().st_size)})")
    if frames.get("sheets"):
        L.append(f"- Contact sheets for review: `work/{ride_id}/frames/sheets/` ({len(frames['sheets'])} images)")
    L.append(f"- Edit plan (every cut and why): `work/{ride_id}/edit_plan.json`")
    L.append("")

    out = od / "notes.md"
    out.write_text("\n".join(L), encoding="utf-8")
    run_log_header(ride_id, "Deliver")
    run_log(ride_id, "notes.md written")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: deliver.py <ride-id>")
    main(sys.argv[1])
