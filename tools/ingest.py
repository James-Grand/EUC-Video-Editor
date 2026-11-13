"""
ingest.py - Step 1. Look at everything in inbox/ and group it into rides.

Plain-language summary:
  * Every video file in inbox/ gets a start time. We try, in order:
      1. the creation_time ffprobe reads from inside the file (most reliable),
      2. the date and time in the filename (Insta360 names look like
         VID_20261105_143022_00_001.insv),
      3. the file's modified time on disk (least reliable; copying changes it).
  * Files are sorted by start time. A gap bigger than 45 minutes between the
    end of one file and the start of the next means a new ride.
  * Each ride gets an ID like 2026-11-05_ride1 and a manifest.json in
    work/<ride-id>/ listing its files, camera, duration and start time.

inbox/ is read-only here. Nothing is moved, renamed or deleted.

Run:  python tools/ingest.py
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common import (CONFIG, log, media_info, path_for, read_json, run_log,
                    run_log_header, work_dir, write_json)

INSTA_NAME = re.compile(r"(?:VID|LRV|PRO_VID)_(\d{8})_(\d{6})_(\d{2})_(\d{3})", re.IGNORECASE)


def camera_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in CONFIG["ingest"]["insta_extensions"]:
        return "insta"
    if ext in CONFIG["ingest"]["gopro_extensions"]:
        return "gopro"
    return None


def parse_iso(ts: str) -> datetime | None:
    """ffprobe gives e.g. '2026-11-05T14:30:22.000000Z'. Return naive local-ish time."""
    try:
        ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is not None:
            # GoPro writes UTC. Convert to this PC's local time so both cameras
            # line up roughly (Insta360 filenames are camera-local time).
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def start_time_for(path: Path, info: dict) -> tuple[datetime, str]:
    """Return (start datetime, which method produced it)."""
    m = INSTA_NAME.search(path.name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"), "filename"
        except ValueError:
            pass
    if info.get("creation_time"):
        dt = parse_iso(info["creation_time"])
        if dt and dt.year > 2010:
            return dt, "metadata"
    return datetime.fromtimestamp(path.stat().st_mtime) - timedelta(seconds=info.get("duration", 0)), "mtime"


def insta_recording_key(path: Path) -> str | None:
    """VID_20261105_143022_00_001.insv -> 'VID_20261105_143022'. Chunks of one
    recording (_001, _002 ...) share a key, so we can match them to one export."""
    m = INSTA_NAME.search(path.name)
    if not m:
        return None
    return f"VID_{m.group(1)}_{m.group(2)}"


def scan_inbox() -> list[dict]:
    inbox = path_for("inbox")
    entries = []
    for p in sorted(inbox.rglob("*")):
        if not p.is_file():
            continue
        cam = camera_for(p)
        if cam is None:
            continue
        # Skip Insta360 low-res proxies; they're not footage.
        if p.name.upper().startswith("LRV_") or p.suffix.lower() == ".lrv":
            continue
        try:
            info = media_info(p)
        except Exception as e:  # unreadable file: record it, don't crash
            log(f"WARNING: ffprobe could not read {p.name}: {e}")
            continue
        start, method = start_time_for(p, info)
        entries.append({
            "path": str(p),
            "name": p.name,
            "camera": cam,
            "start": start,
            "end": start + timedelta(seconds=info["duration"]),
            "duration": info["duration"],
            "fps": info["fps"],
            "width": info["width"],
            "height": info["height"],
            "video_codec": info["video_codec"],
            "has_audio": info["has_audio"],
            "size_bytes": info["size_bytes"],
            "time_source": method,
            "recording_key": insta_recording_key(p) if cam == "insta" else p.stem,
        })
    entries.sort(key=lambda e: e["start"])
    return entries


def group_rides(entries: list[dict]) -> list[list[dict]]:
    gap = timedelta(minutes=CONFIG["ingest"]["ride_gap_minutes"])
    rides: list[list[dict]] = []
    for e in entries:
        if rides and e["start"] - max(x["end"] for x in rides[-1]) <= gap:
            rides[-1].append(e)
        else:
            rides.append([e])
    return rides


def assign_ride_ids(rides: list[list[dict]]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    per_day: dict[str, int] = {}
    for files in rides:
        day = files[0]["start"].strftime("%Y-%m-%d")
        per_day[day] = per_day.get(day, 0) + 1
        out[f"{day}_ride{per_day[day]}"] = files
    return out


def write_manifest(ride_id: str, files: list[dict]) -> Path:
    wd = work_dir(ride_id)
    manifest_path = wd / "manifest.json"
    existing = read_json(manifest_path)
    manifest = {
        "ride_id": ride_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "ride_start": min(f["start"] for f in files).isoformat(timespec="seconds"),
        "ride_end": max(f["end"] for f in files).isoformat(timespec="seconds"),
        "cameras": sorted({f["camera"] for f in files}),
        "files": [
            {**f, "start": f["start"].isoformat(timespec="seconds"),
             "end": f["end"].isoformat(timespec="seconds")}
            for f in files
        ],
    }
    if existing and existing.get("files") == manifest["files"]:
        return manifest_path  # unchanged, don't churn the log
    write_json(manifest_path, manifest)
    run_log_header(ride_id, "Ingest")
    run_log(ride_id, f"manifest written: {len(files)} file(s), cameras={manifest['cameras']}, "
                     f"{manifest['ride_start']} -> {manifest['ride_end']}")
    for f in files:
        run_log(ride_id, f"  {f['camera']:5s} {f['name']} ({f['duration']:.0f}s, time from {f['time_source']})")
    return manifest_path


def main() -> list[str]:
    entries = scan_inbox()
    if not entries:
        log("inbox/ is empty (or has no .insv / .mp4 files). Nothing to do.")
        return []
    rides = assign_ride_ids(group_rides(entries))
    for ride_id, files in rides.items():
        write_manifest(ride_id, files)
    log(f"Found {len(entries)} file(s) in {len(rides)} ride(s): {', '.join(rides)}")
    return list(rides)


if __name__ == "__main__":
    main()
