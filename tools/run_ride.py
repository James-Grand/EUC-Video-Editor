"""
run_ride.py - The master script. Runs the whole pipeline for every ride in
inbox/ (or just the rides you name).

Plain-language summary:
  1. ingest        group inbox files into rides
  For each ride:
  2. check_stitched  stop this ride if Insta360 exports are missing (others continue)
  3. extract_audio   voice + ambient tracks on one ride timeline
  4. sync_cameras    line up GoPro with Insta360 by sound
  5. transcribe      Whisper on the GPU, find highlight cues
  6. telemetry       GPS speed from GoPro
  7. sample_frames   contact sheets for review
  8. edit_plan       decide the cut
  9-11. render       music, mix, encode long + shorts
  12. export_fcpxml + deliver  timeline for Resolve, notes.md

  Everything is logged to output/<ride-id>/run_log.md. A failure in one step
  is recorded and the next ride continues.

Usage:
  python tools/run_ride.py                 all rides
  python tools/run_ride.py 2026-11-05_ride1
  python tools/run_ride.py --no-render     plan only (fast; check edit_plan.json first)
  python tools/run_ride.py --force         redo transcription and renders even if they exist
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime

from common import (Timer, free_space_gb, log, output_dir, read_json, run_log, run_log_header,
                    work_dir)
import ingest
import check_stitched
import extract_audio
import sync_cameras
import transcribe
import telemetry
import sample_frames
import edit_plan
import render
import export_fcpxml
import deliver


def step(ride_id: str, name: str, fn, *args):
    log(f"--- {ride_id}: {name}")
    with Timer() as t:
        result = fn(*args)
    run_log(ride_id, f"{name} done in {t.elapsed:.0f}s")
    return result


def process_ride(ride_id: str, do_render: bool, force: bool) -> str:
    wd = work_dir(ride_id)
    run_log_header(ride_id, f"Run started {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    try:
        st = step(ride_id, "check stitched", check_stitched.main, ride_id)
        if st["blocked"]:
            return "blocked (needs Insta360 Studio exports)"
        step(ride_id, "extract audio", extract_audio.main, ride_id)
        step(ride_id, "sync cameras", sync_cameras.main, ride_id)
        if force or not (wd / "transcript.json").exists():
            step(ride_id, "transcribe", transcribe.main, ride_id)
        else:
            run_log(ride_id, "transcribe skipped (transcript.json exists; use --force to redo)")
        step(ride_id, "telemetry", telemetry.main, ride_id)
        step(ride_id, "sample frames", sample_frames.main, ride_id)
        step(ride_id, "edit plan", edit_plan.main, ride_id)
        if do_render:
            if force or not (output_dir(ride_id) / "long.mp4").exists():
                step(ride_id, "render", render.main, ride_id, "all")
            else:
                run_log(ride_id, "render skipped (long.mp4 exists; use --force to redo)")
            step(ride_id, "export fcpxml", export_fcpxml.main, ride_id)
        step(ride_id, "deliver", deliver.main, ride_id)
        return "done" if do_render else "planned (no render)"
    except Exception as e:
        run_log(ride_id, f"ERROR: {e}")
        with open(output_dir(ride_id) / "run_log.md", "a", encoding="utf-8") as f:
            f.write("\n```\n" + traceback.format_exc() + "```\n")
        return f"FAILED: {e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the EUC Studio pipeline.")
    ap.add_argument("rides", nargs="*", help="ride IDs to process (default: all found in inbox)")
    ap.add_argument("--no-render", action="store_true", help="stop after the edit plan")
    ap.add_argument("--force", action="store_true", help="redo transcription and renders")
    args = ap.parse_args(argv)

    log(f"free disk: {free_space_gb():.1f} GB")
    found = ingest.main()
    rides = args.rides or found
    if not rides:
        log("No rides to process.")
        return 0
    results = {}
    for ride_id in rides:
        if not (work_dir(ride_id) / "manifest.json").exists():
            results[ride_id] = "unknown ride id (no manifest)"
            continue
        with Timer() as t:
            results[ride_id] = process_ride(ride_id, not args.no_render, args.force)
        results[ride_id] += f"  [{t.elapsed/60:.1f} min]"

    print("\n=== Summary ===")
    for r, s in results.items():
        print(f"  {r}: {s}")
    return 0 if all("FAILED" not in s for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
