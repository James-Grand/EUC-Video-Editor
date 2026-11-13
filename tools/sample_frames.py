"""
sample_frames.py - Step 7. Grab small frames from every source and tile them
into contact sheets so the footage can be reviewed cheaply.

Plain-language summary:
  * Every N seconds (config frames.interval_s) we save a small JPEG from each
    source file.
  * Insta360 (360 video) is decoded once to small equirectangular frames,
    then re-projected into several flat views (forward, rear, third-person,
    left, right) using ffmpeg's v360 filter on the image sequence. One decode
    pass, several cheap re-projections.
  * Frames are tiled into grids (contact sheets) with the ride-timeline time
    stamped on each thumbnail. Those PNGs live in work/<ride>/frames/sheets/.
  * Writes frames.json: which sheet shows which times, so notes can refer to
    "sheet 3, 14:20".

Run:  python tools/sample_frames.py <ride-id>
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import (CONFIG, cuda_decode_args, ffmpeg, fmt_time, log, read_json, run_log,
                    run_log_header, work_dir, write_json)


def font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def extract_frames(src: str, out_dir: Path, interval: float, width: int, extra_vf: str = "",
                   gpu: bool = True) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps=1/{interval},scale={width}:-2"
    if extra_vf:
        vf = extra_vf + "," + vf
    args = ["-i", src, "-vf", vf, "-q:v", "4", str(out_dir / "f_%05d.jpg")]
    try:
        ffmpeg([*(cuda_decode_args() if gpu else []), *args])
    except RuntimeError:
        # Some clips (or odd resolutions) refuse GPU decode; CPU decode always works.
        ffmpeg(args)
    return len(list(out_dir.glob("f_*.jpg")))


def reproject(eq_dir: Path, out_dir: Path, view: dict, width: int) -> None:
    """Turn a folder of equirectangular frames into flat frames for one view."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = (f"v360=input=e:output=flat:yaw={view['yaw']}:pitch={view['pitch']}:"
          f"h_fov={view['fov']}:v_fov={view['fov']*9/16:.1f}:w={width}:h={int(width*9/16)}")
    ffmpeg(["-framerate", "1", "-i", str(eq_dir / "f_%05d.jpg"), "-vf", vf, "-q:v", "4",
            str(out_dir / "f_%05d.jpg")])


def contact_sheets(frame_dir: Path, sheet_dir: Path, label: str, interval: float,
                   offset: float, cols: int, rows: int) -> list[dict]:
    frames = sorted(frame_dir.glob("f_*.jpg"))
    if not frames:
        return []
    sheet_dir.mkdir(parents=True, exist_ok=True)
    per = cols * rows
    sheets = []
    fnt = font(14)
    for n in range(math.ceil(len(frames) / per)):
        batch = frames[n * per:(n + 1) * per]
        w, h = Image.open(batch[0]).size
        sheet = Image.new("RGB", (cols * w, rows * h + 24), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        t_first = offset + (n * per) * interval
        t_last = offset + (n * per + len(batch) - 1) * interval
        draw.text((6, 4), f"{label}   {fmt_time(t_first)} - {fmt_time(t_last)}", fill=(230, 230, 230), font=fnt)
        for i, fp in enumerate(batch):
            img = Image.open(fp)
            x, y = (i % cols) * w, 24 + (i // cols) * h
            sheet.paste(img, (x, y))
            t = offset + (n * per + i) * interval
            draw.rectangle([x, y, x + 58, y + 18], fill=(0, 0, 0))
            draw.text((x + 3, y + 2), fmt_time(t), fill=(255, 255, 0), font=fnt)
        out = sheet_dir / f"{label}_sheet_{n+1:02d}.png"
        sheet.save(out, optimize=True)
        sheets.append({"path": str(out), "label": label, "from_s": round(t_first, 1), "to_s": round(t_last, 1),
                       "frames": len(batch)})
    return sheets


def main(ride_id: str) -> dict:
    cfg = CONFIG["frames"]
    wd = work_dir(ride_id)
    sync = read_json(wd / "sync.json")
    fdir = wd / "frames"
    sheet_dir = fdir / "sheets"
    if sheet_dir.exists():
        shutil.rmtree(sheet_dir)
    run_log_header(ride_id, "Frame sampling")

    interval, width = cfg["interval_s"], cfg["thumb_width"]
    cols, rows = cfg["sheet_columns"], cfg["sheet_rows"]
    result = {"interval_s": interval, "sheets": []}

    for i, s in enumerate(sync["sources"]):
        offset = s["offset_in_ride_corrected"]
        tag = f"{s['camera']}{i+1}"
        if s["camera"] == "gopro":
            d = fdir / tag
            n = extract_frames(s["path"], d, interval, width)
            result["sheets"] += contact_sheets(d, sheet_dir, tag, interval, offset, cols, rows)
            run_log(ride_id, f"{s['name']}: {n} frames")
        else:
            # Decode the 360 once at modest equirectangular size, then reproject.
            eq = fdir / f"{tag}_equirect"
            n = extract_frames(s["path"], eq, interval, 1280)
            for view_name, view in cfg["insta_views"].items():
                d = fdir / f"{tag}_{view_name}"
                reproject(eq, d, view, width)
                result["sheets"] += contact_sheets(d, sheet_dir, f"{tag}_{view_name}", interval, offset, cols, rows)
            run_log(ride_id, f"{s['name']}: {n} frames x {len(cfg['insta_views'])} views")

    write_json(wd / "frames.json", result)
    run_log(ride_id, f"{len(result['sheets'])} contact sheet(s) in {sheet_dir}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: sample_frames.py <ride-id>")
    main(sys.argv[1])
