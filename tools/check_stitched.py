"""
check_stitched.py - Step 2. Make sure every Insta360 .insv file has a
stitched (equirectangular) export in stitched/.

Plain-language summary:
  * Raw .insv files are two fisheye circles; we can't edit them directly.
    Tyler exports them from Insta360 Studio into stitched/.
  * For every .insv in a ride's manifest we look for an export whose filename
    contains the same VID_date_time key, in stitched/ or stitched/<ride-id>/.
  * If any are missing, the ride is marked BLOCKED and we print exact
    instructions for which files to export and with what settings.
  * The result is saved to work/<ride-id>/stitched.json so later steps know
    which stitched file goes with which raw file.

Run:  python tools/check_stitched.py <ride-id>
"""

from __future__ import annotations

import sys
from pathlib import Path

from common import CONFIG, log, media_info, path_for, read_json, run_log, run_log_header, work_dir, write_json

EXPORT_SETTINGS = (
    "Insta360 Studio export settings:\n"
    "    - Type: 360 video (NOT reframed / flat)\n"
    "    - Projection: equirectangular\n"
    "    - Resolution: highest available\n"
    "    - Keep audio: ON\n"
    "    - Save into the stitched/ folder (filename can keep the default name)"
)


def find_export(key: str, ride_id: str) -> Path | None:
    stitched = path_for("stitched")
    exts = tuple(CONFIG["ingest"]["stitched_extensions"])
    candidates = []
    for folder in (stitched / ride_id, stitched):
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in exts and key.lower() in p.name.lower():
                candidates.append(p)
    if not candidates:
        return None
    # Prefer the biggest file (highest resolution export) if there are several.
    return max(candidates, key=lambda p: p.stat().st_size)


def main(ride_id: str) -> dict:
    manifest = read_json(work_dir(ride_id) / "manifest.json")
    if not manifest:
        raise SystemExit(f"No manifest for {ride_id}. Run ingest.py first.")

    insta_files = [f for f in manifest["files"] if f["camera"] == "insta"]
    result = {"ride_id": ride_id, "blocked": False, "missing": [], "matches": []}

    # Group raw chunks by recording key so one export can cover several chunks.
    by_key: dict[str, list[dict]] = {}
    for f in insta_files:
        by_key.setdefault(f["recording_key"], []).append(f)

    for key, chunks in by_key.items():
        export = find_export(key, ride_id)
        raw_total = sum(c["duration"] for c in chunks)
        if export is None:
            result["missing"].append({"recording_key": key, "raw_files": [c["name"] for c in chunks],
                                      "raw_duration": raw_total})
            continue
        info = media_info(export)
        note = ""
        if abs(info["duration"] - raw_total) > 5:
            note = (f"duration mismatch: raw {raw_total:.0f}s vs export {info['duration']:.0f}s "
                    f"(export may be trimmed or partial)")
        result["matches"].append({
            "recording_key": key,
            "raw_files": [c["name"] for c in chunks],
            "raw_start": chunks[0]["start"],
            "stitched_path": str(export),
            "duration": info["duration"],
            "fps": info["fps"],
            "width": info["width"],
            "height": info["height"],
            "has_audio": info["has_audio"],
            "note": note,
        })

    result["blocked"] = bool(result["missing"])
    write_json(work_dir(ride_id) / "stitched.json", result)

    run_log_header(ride_id, "Stitched check")
    if result["blocked"]:
        names = ", ".join(", ".join(m["raw_files"]) for m in result["missing"])
        run_log(ride_id, f"BLOCKED - missing stitched export for: {names}")
        print()
        print(f"=== {ride_id} is waiting on Insta360 Studio exports ===")
        print("Please export these files from Insta360 Studio into stitched/:")
        for m in result["missing"]:
            print(f"  - {', '.join(m['raw_files'])}  ({m['raw_duration']/60:.1f} min)")
        print(EXPORT_SETTINGS)
        print()
    else:
        for m in result["matches"]:
            run_log(ride_id, f"matched {m['recording_key']} -> {Path(m['stitched_path']).name} "
                             f"{m['width']}x{m['height']} @ {m['fps']:.2f}fps {m['note']}".rstrip())
        if not insta_files:
            run_log(ride_id, "no Insta360 files in this ride (GoPro only)")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: check_stitched.py <ride-id>")
    main(sys.argv[1])
