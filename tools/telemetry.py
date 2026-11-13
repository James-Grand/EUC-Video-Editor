"""
telemetry.py - Step 6. Read GPS speed out of the GoPro files.

Plain-language summary:
  * GoPro hides a data track inside each MP4 called GPMF ("GoPro Metadata
    Format"). It holds GPS, accelerometer, gyro and more.
  * We pull that track out with ffmpeg, then decode it ourselves. GPMF is a
    simple nested format: 4-letter key, type, size, count, then the data.
    Older cameras store GPS as GPS5 (lat, lon, alt, 2D speed, 3D speed); the
    Hero 11 and newer use GPS9 (adds date/time and fix quality). We read both.
  * Speeds are placed on the ride timeline using each GoPro file's corrected
    position from sync.json.
  * Output:
      speed.csv            - time_s, speed_mps, speed_mph, lat, lon, alt_m, fix
      speed_highlights.json - speed peaks and sustained fast stretches
  * Motion (works even with GPS switched off): the same data track has the
    gyro, accelerometer, gravity direction and GoPro's wind meter. Once per
    second we measure:
      turn  - how fast the camera is turning left/right (carving, sharp turns)
      bump  - how much the ride is shaking (trail, gravel, curbs, jumps)
      wind  - GoPro's wind meter 0-100 (moving vs standing still)
    Output:
      motion.csv            - time_s, wind, turn_rad_s, bump
      motion_highlights.json - sharp-turn and jolt peaks, rough stretches,
                               stopped stretches, and a 0-1 "interest" score
                               per second (used to place filler shots)
  * No GPS lock (indoors, first minute after power-on) -> those samples are
    skipped. No GPMF track at all -> step is skipped with a note.

Run:  python tools/telemetry.py <ride-id>
"""

from __future__ import annotations

import csv
import struct
import sys
from pathlib import Path

import numpy as np

from common import (CONFIG, FFMPEG, FFPROBE, ffprobe_json, log, read_json, run, run_log,
                    run_log_header, work_dir, write_json)

# GPMF type letter -> struct format code (big-endian)
TYPE_FMT = {
    "b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I",
    "f": "f", "d": "d", "j": "q", "J": "Q", "q": "i", "Q": "q",
}


def parse_klv(data: bytes, type_def: str | None = None) -> list[tuple[str, object]]:
    """Decode one level of GPMF. Returns a list of (key, value) pairs.
    Nested blocks (type 0) are returned as nested lists."""
    out = []
    pos = 0
    n = len(data)
    while pos + 8 <= n:
        key = data[pos:pos + 4].decode("latin-1")
        t = data[pos + 4]
        size = data[pos + 5]
        repeat = struct.unpack(">H", data[pos + 6:pos + 8])[0]
        length = size * repeat
        payload = data[pos + 8:pos + 8 + length]
        pos += 8 + ((length + 3) // 4) * 4
        if key == "\x00\x00\x00\x00":
            break
        if t == 0:
            out.append((key, parse_klv(payload)))
            continue
        typ = chr(t)
        if typ == "c":
            out.append((key, payload.decode("latin-1").rstrip("\x00")))
        elif typ == "U":
            out.append((key, payload.decode("latin-1")))
        elif typ == "F":
            out.append((key, [payload[i:i + 4].decode("latin-1") for i in range(0, length, 4)]))
        elif typ == "?" and type_def:
            fmt = ">" + "".join(TYPE_FMT.get(ch, "x") for ch in type_def)
            fsize = struct.calcsize(fmt)
            rows = [list(struct.unpack(fmt, payload[i:i + fsize])) for i in range(0, length, size) if i + fsize <= length]
            out.append((key, rows))
        elif typ in TYPE_FMT:
            code = TYPE_FMT[typ]
            per = struct.calcsize(code)
            cols = max(1, size // per)
            fmt = ">" + code * cols
            rows = [list(struct.unpack(fmt, payload[i:i + size])) for i in range(0, length, size)]
            out.append((key, rows if cols > 1 else [r[0] for r in rows]))
        else:
            out.append((key, payload))
        # A TYPE entry describes the layout of the following '?' entries in this stream.
        if key == "TYPE":
            type_def = out[-1][1]
    return out


def find_gpmf_stream(path: str) -> int | None:
    info = ffprobe_json(path)
    for s in info.get("streams", []):
        if s.get("codec_tag_string") == "gpmd" or "GoPro MET" in str(s.get("tags", {}).get("handler_name", "")):
            return s["index"]
    return None


def extract_packets(path: str, stream_index: int, tmp: Path) -> list[tuple[float, bytes]]:
    """Return [(pts_seconds, payload_bytes)] for every GPMF packet (one per ~second)."""
    raw = tmp / "gpmf.bin"
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", path,
         "-map", f"0:{stream_index}", "-c", "copy", "-f", "rawvideo", str(raw)])
    proc = run([FFPROBE, "-v", "error", "-select_streams", str(stream_index), "-show_packets",
                "-show_entries", "packet=pts_time,size", "-of", "csv=p=0", path])
    data = raw.read_bytes()
    packets, pos = [], 0
    for line in proc.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            pts, size = float(parts[0]), int(parts[1])
        except ValueError:
            continue
        packets.append((pts, data[pos:pos + size]))
        pos += size
    return packets


def gps_from_packet(payload: bytes) -> list[dict]:
    """Pull GPS samples (lat, lon, alt, speed2d, speed3d, fix) out of one packet."""
    samples = []
    for key, devc in parse_klv(payload):
        if key != "DEVC" or not isinstance(devc, list):
            continue
        for k2, strm in devc:
            if k2 != "STRM" or not isinstance(strm, list):
                continue
            d = {}
            for k3, v in strm:
                d.setdefault(k3, v)
            scal = d.get("SCAL")
            if "GPS9" in d:
                rows = d["GPS9"]
                sc = list(scal) if isinstance(scal, list) else [scal]
                sc = [s if isinstance(s, (int, float)) else s[0] for s in sc]
                for r in rows:
                    vals = [x / s for x, s in zip(r, sc)] if len(sc) >= 5 else r
                    fix = r[8] if len(r) > 8 else 3
                    if fix == 0:
                        continue
                    samples.append({"lat": vals[0], "lon": vals[1], "alt": vals[2],
                                    "speed2d": vals[3], "speed3d": vals[4], "fix": fix})
            elif "GPS5" in d:
                fix = d.get("GPSF", [3])
                fix = fix[0] if isinstance(fix, list) else fix
                if fix == 0:
                    continue
                rows = d["GPS5"]
                sc = scal if isinstance(scal, list) else [scal] * 5
                sc = [s if isinstance(s, (int, float)) else s[0] for s in sc]
                for r in rows:
                    vals = [x / s for x, s in zip(r, sc)]
                    samples.append({"lat": vals[0], "lon": vals[1], "alt": vals[2],
                                    "speed2d": vals[3], "speed3d": vals[4], "fix": fix})
    return samples


def read_gopro_speed(path: str, tmp: Path, packets: list | None = None) -> list[dict]:
    """All GPS samples from one file with time in seconds from the file's start."""
    if packets is None:
        idx = find_gpmf_stream(path)
        if idx is None:
            return []
        packets = extract_packets(path, idx, tmp)
    out = []
    for i, (pts, payload) in enumerate(packets):
        samples = gps_from_packet(payload)
        if not samples:
            continue
        # Packets are ~1 s apart; spread this packet's samples evenly across it.
        next_pts = packets[i + 1][0] if i + 1 < len(packets) else pts + 1.0
        step = (next_pts - pts) / len(samples)
        for j, s in enumerate(samples):
            s["t"] = pts + j * step
            out.append(s)
    return out


def motion_from_packet(payload: bytes) -> dict | None:
    """One packet (~1 s) -> {wind, turn, bump}. None if there's no gyro/accel."""
    found = {}
    for key, devc in parse_klv(payload):
        if key != "DEVC" or not isinstance(devc, list):
            continue
        for k2, strm in devc:
            if k2 != "STRM" or not isinstance(strm, list):
                continue
            scal = None
            for k3, v in strm:
                if k3 == "SCAL":
                    scal = v
                elif k3 in ("GYRO", "ACCL", "GRAV", "WNDM") and k3 not in found:
                    found[k3] = (v, scal)
    if "GYRO" not in found or "ACCL" not in found:
        return None

    def scaled(k):
        v, sc = found[k]
        arr = np.array(v, float)
        if arr.ndim == 1:
            arr = arr[None, :]
        sc = np.array(sc if sc is not None else 1, float).ravel()
        return arr / (sc[0] if sc.size == 1 else sc[: arr.shape[1]])

    gyro, accl = scaled("GYRO"), scaled("ACCL")
    # "Turning" = rotation around the up/down axis, whichever way the camera is mounted.
    if "GRAV" in found:
        g = np.array(found["GRAV"][0], float).reshape(-1, 3).mean(0)
        up = g / (np.linalg.norm(g) or 1)
        turn = float(np.mean(np.abs(gyro @ up)))
    else:
        turn = float(np.mean(np.linalg.norm(gyro, axis=1)))
    bump = float(np.std(np.linalg.norm(accl, axis=1)))
    wind = float("nan")
    if "WNDM" in found:
        wind = float(np.mean(np.array(found["WNDM"][0], float).reshape(-1, 2)[:, 1]))
    return {"wind": wind, "turn": turn, "bump": bump}


def find_motion_highlights(times: np.ndarray, wind: np.ndarray, turn: np.ndarray, bump: np.ndarray) -> dict:
    """Turn per-second sensor readings into moments worth showing."""
    cfg = CONFIG["telemetry"]
    if len(times) < 20:
        return {"peaks": [], "stretches": [], "stopped": [], "interest": []}

    def sm(x):
        return np.convolve(np.nan_to_num(x), np.ones(3) / 3, mode="same")

    wind_s, turn_s, bump_s = sm(wind), sm(turn), sm(bump)
    moving = bump_s > cfg["moving_bump"]
    if not np.all(np.isnan(wind)):
        moving |= wind_s > cfg["moving_wind"]

    def pct_rank(x):
        if not moving.any():
            return np.zeros_like(x)
        ref = np.sort(x[moving])
        return np.where(moving, np.searchsorted(ref, x) / len(ref), 0.0)

    # 0-1 score: how much turning + shaking compared with the rest of this ride.
    interest = 0.5 * pct_rank(turn_s) + 0.5 * pct_rank(bump_s)

    win = cfg["peak_window_s"]
    peaks = []
    for kind, series, label in (("turn", turn_s, "sharp turn / carving"), ("jolt", bump_s, "rough terrain jolt")):
        if not moving.any():
            break
        thresh = float(np.percentile(series[moving], cfg["motion_peak_percentile"]))
        top = float(series[moving].max()) or 1.0
        cands = sorted((i for i in range(len(series)) if moving[i] and series[i] >= thresh), key=lambda i: -series[i])
        kept = []
        for i in cands:
            if all(abs(times[i] - times[j]) >= win for j in kept):
                kept.append(i)
            if len(kept) >= cfg["motion_max_peaks"]:
                break
        for i in kept:
            peaks.append({"time": round(float(times[i]), 1), "kind": kind, "label": label,
                          "value": round(float(series[i]), 2), "strength": round(float(series[i] / top), 2)})
    peaks.sort(key=lambda p: p["time"])

    def runs(mask, min_s):
        out, start = [], None
        for i in range(len(mask) + 1):
            on = i < len(mask) and mask[i]
            if on and start is None:
                start = i
            elif not on and start is not None:
                if times[i - 1] - times[start] >= min_s:
                    out.append((start, i))
                start = None
        return out

    # Rough stretches: judged on a slower (9 s) average so one smooth second doesn't break a trail section.
    bump_slow = np.convolve(np.nan_to_num(bump), np.ones(9) / 9, mode="same")
    rough_t = float(np.percentile(bump_slow[moving], cfg["sustained_percentile"])) if moving.any() else np.inf
    stretches = [{"start": round(float(times[a]), 1), "end": round(float(times[b - 1]), 1), "kind": "rough",
                  "avg_bump": round(float(bump_slow[a:b].mean()), 2)}
                 for a, b in runs(moving & (bump_slow >= rough_t), cfg["sustained_min_s"])]
    stopped = [[round(float(times[a]), 1), round(float(times[b - 1]), 1)] for a, b in runs(~moving, 5)]
    return {"peaks": peaks, "stretches": stretches, "stopped": stopped,
            "interest": [[round(float(t), 1), round(float(v), 3)] for t, v in zip(times, interest)],
            "moving_s": int(moving.sum())}


def find_highlights(times: np.ndarray, speed: np.ndarray) -> dict:
    cfg = CONFIG["telemetry"]
    if len(times) < 10:
        return {"peaks": [], "sustained": [], "max_speed_mps": 0, "avg_speed_mps": 0}
    # Smooth over ~3 s so GPS jitter doesn't create fake peaks.
    k = max(1, int(3 / max(np.median(np.diff(times)), 1e-3)))
    kernel = np.ones(k) / k
    smooth = np.convolve(speed, kernel, mode="same")
    moving = smooth > 1.0  # ignore standing still

    # Peaks: highest point in each window, if clearly above the ride average.
    win = cfg["peak_window_s"]
    avg = float(smooth[moving].mean()) if moving.any() else 0.0
    peaks = []
    i = 0
    while i < len(times):
        j = np.searchsorted(times, times[i] + win)
        seg = smooth[i:j]
        if len(seg):
            m = int(np.argmax(seg))
            if seg[m] > avg * 1.15:
                peaks.append({"time": round(float(times[i + m]), 1), "speed_mps": round(float(seg[m]), 2)})
        i = j if j > i else i + 1
    # Keep peaks at least `win` apart and take the strongest ones.
    peaks.sort(key=lambda p: -p["speed_mps"])
    kept = []
    for p in peaks:
        if all(abs(p["time"] - q["time"]) >= win for q in kept):
            kept.append(p)
    kept = sorted(kept[:12], key=lambda p: p["time"])

    # Sustained: above the Nth percentile for at least N seconds.
    thresh = float(np.percentile(smooth[moving], cfg["sustained_percentile"])) if moving.any() else 0
    fast = smooth >= thresh
    sustained, start = [], None
    for idx in range(len(fast) + 1):
        on = idx < len(fast) and fast[idx]
        if on and start is None:
            start = idx
        elif not on and start is not None:
            if times[idx - 1] - times[start] >= cfg["sustained_min_s"]:
                sustained.append({"start": round(float(times[start]), 1), "end": round(float(times[idx - 1]), 1),
                                  "avg_speed_mps": round(float(smooth[start:idx].mean()), 2)})
            start = None
    return {"peaks": kept, "sustained": sustained, "max_speed_mps": round(float(smooth.max()), 2),
            "avg_speed_mps": round(avg, 2), "fast_threshold_mps": round(thresh, 2)}


def main(ride_id: str) -> dict:
    wd = work_dir(ride_id)
    sync = read_json(wd / "sync.json")
    tmp = wd / "tmp"
    tmp.mkdir(exist_ok=True)
    run_log_header(ride_id, "Telemetry")

    rows, motion = [], []
    for s in sync["sources"]:
        if s["camera"] != "gopro":
            continue
        idx = find_gpmf_stream(s["path"])
        packets = extract_packets(s["path"], idx, tmp) if idx is not None else []
        samples = read_gopro_speed(s["path"], tmp, packets)
        off = s["offset_in_ride_corrected"]
        for smp in samples:
            rows.append((smp["t"] + off, smp["speed2d"], smp["lat"], smp["lon"], smp["alt"], smp["fix"]))
        n_motion = 0
        for pts, payload in packets:
            m = motion_from_packet(payload)
            if m:
                motion.append((pts + off, m["wind"], m["turn"], m["bump"]))
                n_motion += 1
        gps_note = "" if samples else " (no GPS data - is GPS switched off on the camera?)"
        run_log(ride_id, f"{s['name']}: {len(samples)} GPS samples{gps_note}, {n_motion} s of motion data")

    motion.sort()
    if motion:
        with open(wd / "motion.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "wind", "turn_rad_s", "bump"])
            for t, wnd, trn, bmp in motion:
                w.writerow([f"{t:.2f}", f"{wnd:.0f}", f"{trn:.3f}", f"{bmp:.2f}"])
        arr = np.array(motion)
        mh = find_motion_highlights(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
    else:
        mh = {"peaks": [], "stretches": [], "stopped": [], "interest": []}
    write_json(wd / "motion_highlights.json", mh)
    run_log(ride_id, f"motion: {len(mh['peaks'])} peak(s) (turns/jolts), {len(mh['stretches'])} rough stretch(es), "
                     f"{len(mh['stopped'])} stopped stretch(es)")

    rows.sort()
    if not rows:
        run_log(ride_id, "no telemetry - speed highlights skipped")
        write_json(wd / "speed_highlights.json", {"peaks": [], "sustained": [], "max_speed_mps": 0})
        return {}

    to_mph = 2.23694
    with open(wd / "speed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "speed_mps", "speed_mph", "lat", "lon", "alt_m", "fix"])
        for t, v, lat, lon, alt, fix in rows:
            w.writerow([f"{t:.2f}", f"{v:.2f}", f"{v*to_mph:.1f}", f"{lat:.6f}", f"{lon:.6f}", f"{alt:.1f}", fix])

    times = np.array([r[0] for r in rows])
    speed = np.array([r[1] for r in rows])
    hl = find_highlights(times, speed)
    hl["units_note"] = "speeds in metres/second; multiply by 2.237 for mph"
    write_json(wd / "speed_highlights.json", hl)
    run_log(ride_id, f"max {hl['max_speed_mps']*to_mph:.1f} mph, avg moving {hl['avg_speed_mps']*to_mph:.1f} mph, "
                     f"{len(hl['peaks'])} peak(s), {len(hl['sustained'])} sustained stretch(es)")
    return hl


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: telemetry.py <ride-id>")
    main(sys.argv[1])
