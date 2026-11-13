"""
edit_plan.py - Step 8. Decide what goes in the video, and from which camera.
Nothing is rendered here; this writes work/<ride>/edit_plan.json.

Plain-language summary (v1 rules, all tunable in config.json "edit"):
  1. Collect "moments" worth showing:
       - highlight cues Tyler spoke ("clip that" etc.)  -> highest score
       - GPS speed peaks and sustained fast stretches    -> medium score
       - natural exclamations ("whoa", "wow", "let's go") -> medium score
       - GoPro motion sensors: sharp turns / carving and
         rough-terrain jolts (gyro + accelerometer)       -> medium score
       - every stretch where he's talking                -> keep (his voice is the channel)
     Filler shots (when the ride is short on all of the above) go on the
     busiest riding in each gap, never on a stretch where he's stopped.
  2. Turn each moment into a time window on the ride timeline. A cue like
     "clip that" refers to what just happened, so its window reaches back
     ~12 s; "watch this" points forward, so its window reaches ahead ~15 s.
  3. Merge overlapping windows. Cut silences over 2 s (they're simply not in
     any window). If the total is under 5 min, add scenery fillers; if over
     15 min, drop the weakest windows (never a cue).
  4. Hook: the single best moment goes FIRST (cold open), then everything else
     in ride order.
  5. Every window is chopped into shots of at most ~8 s, alternating between
     cameras/views that cover that time: Insta forward -> GoPro -> Insta
     third-person -> GoPro ... so no shot drags.
  6. Shorts: the top 1-3 moments, each a window of under 60 s around the
     moment, snapped outward to sentence boundaries, framed vertical.

Every segment records WHY it was chosen, and anything uncertain is listed in
"low_confidence" so notes.md can flag it for Tyler.

Run:  python tools/edit_plan.py <ride-id>
"""

from __future__ import annotations

import sys
from pathlib import Path

from common import CONFIG, fmt_time, log, read_json, run_log, run_log_header, work_dir, write_json

CUE_SCORES = {"clip that": 10, "mark it": 10, "watch this": 9, "holy shit": 8, "oh shit": 8, "did you see that": 8}
FORWARD_CUES = {"watch this"}


# ----------------------------------------------------------------------------
# Helpers for the ride timeline
# ----------------------------------------------------------------------------

def covering_sources(sources: list[dict], t0: float, t1: float) -> list[dict]:
    """Sources whose footage fully covers ride time [t0, t1]."""
    out = []
    for s in sources:
        a = s["offset_in_ride_corrected"]
        b = a + s["duration"]
        if a <= t0 and b >= t1:
            out.append(s)
    return out


def merge_windows(wins: list[dict]) -> list[dict]:
    wins = sorted(wins, key=lambda w: w["start"])
    merged: list[dict] = []
    for w in wins:
        if merged and w["start"] <= merged[-1]["end"]:
            m = merged[-1]
            m["end"] = max(m["end"], w["end"])
            # The merged window's anchor (the moment a short is built around)
            # follows whichever piece scored higher. Compare BEFORE updating score.
            if w["score"] > m["score"]:
                m["anchor"] = w["anchor"]
            m["score"] = max(m["score"], w["score"])
            m["reasons"] = sorted(set(m["reasons"] + w["reasons"]))
            m["is_cue"] = m["is_cue"] or w["is_cue"]
        else:
            merged.append(dict(w))
    return merged


def total(wins: list[dict]) -> float:
    return sum(w["end"] - w["start"] for w in wins)


# ----------------------------------------------------------------------------
# Build candidate windows
# ----------------------------------------------------------------------------

def find_exclamations(words: list[dict]) -> list[dict]:
    """Natural excitement ("whoa", "wow", "let's go") - softer than the
    deliberate cue phrases, but still a sign something happened."""
    import re
    norm = [re.sub(r"[^a-z0-9' ]+", " ", w["word"].lower()).strip() for w in words]
    hits = []
    for phrase in CONFIG["transcribe"].get("excitement_words", []):
        target = phrase.split()
        k = len(target)
        for i in range(len(norm) - k + 1):
            if norm[i:i + k] == target:
                hits.append({"word": phrase, "time": words[i]["start"], "end": words[i + k - 1]["end"]})
    hits.sort(key=lambda h: h["time"])
    out = []
    for h in hits:  # one per 5 s is plenty
        if not out or h["time"] - out[-1]["time"] >= 5:
            out.append(h)
    return out


def interest_at(motion: dict, t: float) -> float:
    pts = motion.get("interest", [])
    if not pts:
        return 0.0
    i = min(range(len(pts)), key=lambda k: abs(pts[k][0] - t))
    return pts[i][1]


def build_windows(ride_len: float, islands: list[dict], cues: list[dict], speed: dict,
                  motion: dict | None = None, exclaims: list[dict] | None = None) -> list[dict]:
    cfg = CONFIG["edit"]
    pad = cfg["speech_pad_s"]
    motion = motion or {}
    wins = []

    # Talk blocks: speech islands merged across pauses shorter than silence_cut_s.
    blocks: list[dict] = []
    for isl in islands:
        if blocks and isl["start"] - blocks[-1]["end"] <= cfg["silence_cut_s"]:
            blocks[-1]["end"] = isl["end"]
        else:
            blocks.append({"start": isl["start"], "end": isl["end"]})
    for b in blocks:
        wins.append({"start": max(0, b["start"] - pad), "end": min(ride_len, b["end"] + pad),
                     "score": 3, "reasons": ["commentary"], "is_cue": False, "anchor": b["start"]})

    for c in cues:
        if c["cue"] in FORWARD_CUES:
            s, e = c["time"] - 2, c["end"] + 15
        else:
            s, e = c["time"] - 12, c["end"] + 4
        wins.append({"start": max(0, s), "end": min(ride_len, e), "score": CUE_SCORES.get(c["cue"], 8),
                     "reasons": [f"cue '{c['cue']}' at {fmt_time(c['time'])}"], "is_cue": True,
                     "anchor": c["time"]})

    vmax = max(speed.get("max_speed_mps", 0), 0.1)
    for p in speed.get("peaks", []):
        wins.append({"start": max(0, p["time"] - 6), "end": min(ride_len, p["time"] + 6),
                     "score": 5 + 3 * p["speed_mps"] / vmax,
                     "reasons": [f"speed peak {p['speed_mps']*2.237:.0f} mph at {fmt_time(p['time'])}"],
                     "is_cue": False, "anchor": p["time"]})
    for st in speed.get("sustained", []):
        mid = (st["start"] + st["end"]) / 2
        wins.append({"start": max(0, mid - 8), "end": min(ride_len, mid + 8), "score": 4,
                     "reasons": [f"sustained speed {st['avg_speed_mps']*2.237:.0f} mph {fmt_time(st['start'])}-{fmt_time(st['end'])}"],
                     "is_cue": False, "anchor": mid})

    # Exclamations react to what just happened, so reach back like "clip that" (but shorter).
    for x in exclaims or []:
        wins.append({"start": max(0, x["time"] - 8), "end": min(ride_len, x["end"] + 4), "score": 6,
                     "reasons": [f"exclamation '{x['word']}' at {fmt_time(x['time'])}"], "is_cue": False,
                     "anchor": x["time"]})

    # GoPro motion sensors: sharp turns / carving and rough-terrain jolts (score 4-6,
    # so the strongest can become shorts), plus sustained rough stretches.
    for p in motion.get("peaks", []):
        wins.append({"start": max(0, p["time"] - 6), "end": min(ride_len, p["time"] + 6),
                     "score": 4 + 2 * p["strength"],
                     "reasons": [f"motion: {p['label']} at {fmt_time(p['time'])}"],
                     "is_cue": False, "anchor": p["time"]})
    for st in motion.get("stretches", []):
        mid = (st["start"] + st["end"]) / 2
        wins.append({"start": max(0, mid - 8), "end": min(ride_len, mid + 8), "score": 3.5,
                     "reasons": [f"motion: rough stretch {fmt_time(st['start'])}-{fmt_time(st['end'])}"],
                     "is_cue": False, "anchor": mid})
    return [w for w in wins if w["end"] - w["start"] >= 1.0]


def fit_to_length(wins: list[dict], ride_len: float, notes: list[str], motion: dict | None = None) -> list[dict]:
    cfg = CONFIG["edit"]
    motion = motion or {}
    stopped = motion.get("stopped", [])
    wins = merge_windows(wins)

    def filler_spot(a: float, b: float) -> float:
        """Most interesting moment (turning + shaking) in the gap, never while
        stopped. Falls back to the middle when there's no sensor data."""
        if not motion.get("interest"):
            return (a + b) / 2
        best, best_v = (a + b) / 2, -1.0
        for t, v in motion["interest"]:
            if a + 3 <= t <= b - 3 and not any(s0 <= t <= s1 for s0, s1 in stopped) and v > best_v:
                best, best_v = t, v
        return best

    if total(wins) < cfg["long_min_s"] and ride_len > 0:
        # Fillers: 6-second scenery chunks spread through uncovered time.
        gaps, cursor = [], 0.0
        for w in wins:
            if w["start"] - cursor >= 10:
                gaps.append((cursor, w["start"]))
            cursor = w["end"]
        if ride_len - cursor >= 10:
            gaps.append((cursor, ride_len))
        need = cfg["long_min_s"] - total(wins)
        fillers = []
        while need > 0 and gaps:
            new_gaps = []
            for a, b in gaps:
                if need <= 0:
                    new_gaps.append((a, b))
                    continue
                mid = filler_spot(a, b)
                reason = "scenery filler" + (f" (busiest riding in gap, interest {interest_at(motion, mid):.2f})"
                                             if motion.get("interest") else "")
                fillers.append({"start": mid - 3, "end": mid + 3, "score": 1, "reasons": [reason],
                                "is_cue": False, "anchor": mid})
                need -= 6
                if mid - 3 - a >= 10:
                    new_gaps.append((a, mid - 3))
                if b - (mid + 3) >= 10:
                    new_gaps.append((mid + 3, b))
            gaps = new_gaps
        if fillers:
            notes.append(f"Ride was short on talk/highlights; added {len(fillers)} scenery fillers to reach "
                         f"{cfg['long_min_s']//60} min. Pick better filler spots once we know what Tyler likes.")
        wins = merge_windows(wins + fillers)

    while total(wins) > cfg["long_max_s"]:
        droppable = [w for w in wins if not w["is_cue"]]
        if not droppable:
            break
        victim = min(droppable, key=lambda w: w["score"])
        wins.remove(victim)
    return wins


# ----------------------------------------------------------------------------
# Shots
# ----------------------------------------------------------------------------

def make_shots(win: dict, sources: list[dict], vertical: bool, notes: list[str],
               rot: dict | None = None) -> list[dict]:
    """Chop a window into <= max_shot_s shots, rotating cameras/views.
    Pass the same rot dict for every window of a video so the rotation carries
    on from one moment to the next instead of restarting on Insta forward."""
    cfg = CONFIG["edit"]
    views = CONFIG["frames"]["insta_views"]
    max_shot = cfg["max_shot_s"]
    shots = []
    t = win["start"]
    rotation = [("insta", cfg["insta_default_view"]), ("gopro", None),
                ("insta", cfg["insta_alt_view"]), ("gopro", None)]
    k = rot["k"] if rot else 0
    while t < win["end"] - 0.25:
        end = min(win["end"], t + max_shot)
        # If what remains after this shot is a stub under 2 s, absorb it.
        if win["end"] - end < 2:
            end = win["end"]
        cover = covering_sources(sources, t, end)
        choice = None
        for _ in range(len(rotation)):
            cam, view = rotation[k % len(rotation)]
            k += 1
            src = next((s for s in cover if s["camera"] == cam), None)
            if src:
                choice = (src, view)
                break
        if choice is None:
            # Nothing covers this exact span (e.g. one camera stopped early).
            cover_any = [s for s in sources if s["offset_in_ride_corrected"] <= t < s["offset_in_ride_corrected"] + s["duration"]]
            if not cover_any:
                notes.append(f"No footage covers {fmt_time(t)}-{fmt_time(end)}; skipped.")
                t = end
                continue
            src = cover_any[0]
            end = min(end, src["offset_in_ride_corrected"] + src["duration"])
            choice = (src, cfg["insta_default_view"] if src["camera"] == "insta" else None)
        src, view = choice
        off = src["offset_in_ride_corrected"]
        shot = {
            "timeline_in": round(t, 3), "timeline_out": round(end, 3), "duration": round(end - t, 3),
            "camera": src["camera"], "source": src["path"], "source_name": src["name"],
            "source_in": round(t - off, 3), "source_out": round(end - off, 3),
            "view": view, "v360": None, "reasons": win["reasons"], "score": round(win["score"], 2),
        }
        if src["camera"] == "insta":
            v = dict(views[view])
            if vertical:
                v["fov"] = max(60, v["fov"] - 30)  # tighter for 9:16
            shot["v360"] = v
        shots.append(shot)
        t = end
    if rot is not None:
        rot["k"] = k
    return shots


def plan_long(wins: list[dict], sources: list[dict], notes: list[str]) -> dict:
    if not wins:
        return {"segments": [], "duration": 0, "hook": None}
    hook_src = max(wins, key=lambda w: w["score"])
    rest = [w for w in wins if w is not hook_src]
    # Hook: up to 10 s around the moment's anchor, then the ride in order.
    a = hook_src["anchor"]
    h_start = max(hook_src["start"], a - 6)
    h_end = min(hook_src["end"], h_start + 10)
    hook = {**hook_src, "start": h_start, "end": h_end, "reasons": hook_src["reasons"] + ["HOOK (cold open)"]}
    ordered = [hook]
    # Put back whatever of the hook window wasn't used, in order.
    leftovers = []
    if h_start - hook_src["start"] >= 2:
        leftovers.append({**hook_src, "end": h_start})
    if hook_src["end"] - h_end >= 2:
        leftovers.append({**hook_src, "start": h_end})
    ordered += sorted(rest + leftovers, key=lambda w: w["start"])

    segments, cursor = [], 0.0
    rot = {"k": 0}
    for w in ordered:
        for sh in make_shots(w, sources, vertical=False, notes=notes, rot=rot):
            sh["out_in"] = round(cursor, 3)
            sh["out_out"] = round(cursor + sh["duration"], 3)
            cursor += sh["duration"]
            segments.append(sh)
    return {"segments": segments, "duration": round(cursor, 3),
            "hook": {"time": a, "reasons": hook_src["reasons"]}}


def plan_shorts(wins: list[dict], islands: list[dict], sources: list[dict], ride_len: float, notes: list[str]) -> list[dict]:
    cfg = CONFIG["edit"]
    cands = sorted([w for w in wins if w["score"] >= 4], key=lambda w: -w["score"])
    chosen = []
    for w in cands:
        if all(abs(w["anchor"] - c["anchor"]) >= 60 for c in chosen):
            chosen.append(w)
        if len(chosen) >= cfg["short_count"]:
            break
    shorts = []
    for i, w in enumerate(chosen, 1):
        a = w["anchor"]
        s, e = max(0, a - cfg["short_pre_s"]), min(ride_len, a + cfg["short_post_s"])
        # Snap outward to whole sentences if that keeps us under the limit.
        for isl in islands:
            if isl["start"] < s < isl["end"] and e - isl["start"] <= cfg["short_max_s"]:
                s = max(0, isl["start"] - 0.3)
            if isl["start"] < e < isl["end"] and isl["end"] - s <= cfg["short_max_s"]:
                e = min(ride_len, isl["end"] + 0.3)
        e = min(e, s + cfg["short_max_s"])
        win = {**w, "start": s, "end": e}
        segments, cursor = [], 0.0
        for sh in make_shots(win, sources, vertical=True, notes=notes):
            sh["out_in"] = round(cursor, 3)
            sh["out_out"] = round(cursor + sh["duration"], 3)
            cursor += sh["duration"]
            segments.append(sh)
        if not segments:
            continue
        shorts.append({"id": f"short_{i:02d}", "anchor": a, "start": s, "end": e, "duration": round(cursor, 3),
                       "reasons": w["reasons"], "segments": segments})
    return shorts


def pick_fps(sources: list[dict]) -> float:
    insta = [s for s in sources if s["camera"] == "insta" and s.get("fps")]
    pool = insta or [s for s in sources if s.get("fps")]
    fps = pool[0]["fps"] if pool else 30.0
    # Snap to the nearest standard rate (30 vs 29.97 matters for clean cuts).
    nearest = min((23.976, 24, 25, 29.97, 30, 50, 59.94, 60), key=lambda s: abs(fps - s))
    return nearest if abs(fps - nearest) < 0.05 else round(fps, 3)


def main(ride_id: str) -> dict:
    wd = work_dir(ride_id)
    sync = read_json(wd / "sync.json")
    audio = read_json(wd / "audio.json")
    transcript = read_json(wd / "transcript.json", {"speech_islands": []})
    cues = read_json(wd / "cues.json", [])
    speed = read_json(wd / "speed_highlights.json", {})
    sources = sync["sources"]
    ride_len = audio["ride_duration"]
    notes: list[str] = []
    run_log_header(ride_id, "Edit plan")

    if sync.get("low_confidence"):
        notes.append("Camera sync confidence was low; GoPro shots may be slightly out of step with the voice.")
    motion = read_json(wd / "motion_highlights.json", {})
    exclaims = find_exclamations(transcript.get("words", []))
    if not cues:
        notes.append("No spoken highlight cues found. Edit leans on exclamations, motion sensors, speed data "
                     "and commentary instead.")
    if not speed.get("peaks"):
        notes.append("No GPS speed data; speed highlights unavailable for this ride.")
    if motion.get("peaks"):
        notes.append(f"Used {len(motion['peaks'])} GoPro motion moment(s) (sharp turns / rough-terrain jolts). "
                     "These are guesses from the gyro/accelerometer: check they're actually interesting.")
    if exclaims:
        notes.append("Exclamations used as soft highlights: " +
                     ", ".join(f"'{x['word']}' at {fmt_time(x['time'])}" for x in exclaims))

    wins = build_windows(ride_len, transcript.get("speech_islands", []), cues, speed, motion, exclaims)
    wins = fit_to_length(wins, ride_len, notes, motion)
    long = plan_long(wins, sources, notes)
    shorts = plan_shorts(wins, transcript.get("speech_islands", []), sources, ride_len, notes)
    if not shorts:
        notes.append("No moment scored high enough for a short; none planned.")

    plan = {
        "ride_id": ride_id, "fps": pick_fps(sources), "ride_duration": ride_len,
        "windows": wins, "long": long, "shorts": shorts, "low_confidence": notes,
        "sources": sources,
    }
    write_json(wd / "edit_plan.json", plan)
    run_log(ride_id, f"long: {len(long['segments'])} shots, {fmt_time(long['duration'])}; "
                     f"hook = {long['hook']['reasons'] if long['hook'] else 'none'}")
    for sh in shorts:
        run_log(ride_id, f"{sh['id']}: {fmt_time(sh['duration'])} around {fmt_time(sh['anchor'])} ({'; '.join(sh['reasons'])})")
    for n in notes:
        run_log(ride_id, f"note: {n}")
    return plan


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: edit_plan.py <ride-id>")
    main(sys.argv[1])
