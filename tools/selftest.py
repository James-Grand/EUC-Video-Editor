"""
selftest.py - Build fake footage and run the whole pipeline on it.
Nothing in the real inbox/ or stitched/ is touched: everything happens inside
work/_selftest/.

Plain-language summary:
  * Makes a 60 s "Insta360" clip (test pattern + a synthetic voice saying
    highlight cues like "clip that") and a 55 s "GoPro" clip (test pattern +
    wind-like noise + the same voice, muffled, offset by a known 2.0 s).
  * The GoPro's timestamp is deliberately set 1 s wrong, so the sync step has
    something real to correct.
  * Runs run_ride.py against that fake inbox, then checks: the ride was found,
    sync landed near 2.0 s, cues were heard, and long.mp4 / shorts exist.
  * There's no GPS data in fake files, so the telemetry step should report
    "no telemetry" and carry on.

Run:  python tools/selftest.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402  (uses the real project root for ffmpeg + config)

ROOT = common.PROJECT_ROOT / "work" / "_selftest"
TRUE_OFFSET = 2.0        # GoPro really started 2.0 s after the Insta360
CLOCK_ERROR = 1.0        # ...but its clock says 3.0 s
SR = 48000

SPEECH = (
    "Alright, we're heading out for a ride on the wheel this morning. The weather is perfect. "
    "<break time='3s'/> Okay, watch this. <break time='2500ms'/> Oh, that was smooth. Clip that. "
    "<break time='4s'/> Coming up on the trail now. Lots of gravel here, so I'm going to take it easy. "
    "<break time='3s'/> Holy shit, did you see that squirrel? <break time='2s'/> Okay, almost home. Good ride."
)


def tts(out_wav: Path) -> None:
    """Windows' built-in text-to-speech, via PowerShell. 16 kHz mono WAV."""
    ssml = f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>{SPEECH}</speak>"
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, 'Sixteen', 'Mono'); "
        f"$s.SetOutputToWaveFile('{out_wav}', $fmt); "
        f"$s.SpeakSsml(@'\n{ssml}\n'@); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=True,
                   capture_output=True, text=True)


def build_fixtures() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    for d in ("inbox", "stitched", "work", "output", "music", "fixtures"):
        (ROOT / d).mkdir(parents=True)
    fx = ROOT / "fixtures"

    tts(fx / "speech16k.wav")
    speech, sr = sf.read(str(fx / "speech16k.wav"), dtype="float32")
    speech = signal.resample_poly(speech, SR // 1000, sr // 1000).astype(np.float32)
    speech /= max(1e-6, np.abs(speech).max())

    insta_len, gopro_len, speech_at = 60.0, 55.0, 4.0
    voice = np.zeros(int(insta_len * SR), np.float32)
    s0 = int(speech_at * SR)
    voice[s0:s0 + len(speech)] += speech[: len(voice) - s0] * 0.6
    voice += np.random.randn(len(voice)).astype(np.float32) * 0.003  # faint hiss, like a phone call
    sf.write(str(fx / "voice.wav"), voice, SR)

    # GoPro: wind-ish noise + muffled voice, starting TRUE_OFFSET later.
    g = np.random.randn(int(gopro_len * SR)).astype(np.float32)
    g = signal.sosfilt(signal.butter(2, 400, "low", fs=SR, output="sos"), g).astype(np.float32) * 0.08
    muffled = signal.sosfilt(signal.butter(4, 1800, "low", fs=SR, output="sos"), speech).astype(np.float32)
    g0 = int((speech_at - TRUE_OFFSET) * SR)
    g[g0:g0 + len(muffled)] += muffled[: len(g) - g0] * 0.25
    sf.write(str(fx / "gopro_audio.wav"), g, SR)

    ff = [common.FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    # "Stitched" 360 export: 2:1 equirectangular-shaped test pattern.
    subprocess.run([*ff, "-f", "lavfi", "-i", f"testsrc2=size=1920x960:rate=30:duration={insta_len}",
                    "-i", str(fx / "voice.wav"), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(ROOT / "stitched" / "VID_20260601_100000_00_001.mp4")], check=True)
    # Raw .insv stand-in: a tiny mp4 with the right name (ffprobe reads it for duration).
    subprocess.run([*ff, "-f", "lavfi", "-i", f"testsrc2=size=320x160:rate=30:duration={insta_len}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-f", "mp4",
                    str(ROOT / "inbox" / "VID_20260601_100000_00_001.insv")], check=True)
    # GoPro file with a creation_time that's CLOCK_ERROR seconds off.
    local_start = datetime(2026, 6, 1, 10, 0, 0) .astimezone()  # Insta filename time, in this PC's zone
    gopro_utc = (local_start.timestamp() + TRUE_OFFSET + CLOCK_ERROR)
    ct = datetime.fromtimestamp(gopro_utc, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    subprocess.run([*ff, "-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=30:duration={gopro_len}",
                    "-i", str(fx / "gopro_audio.wav"), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", "-metadata", f"creation_time={ct}",
                    str(ROOT / "inbox" / "GX010001.MP4")], check=True)
    print(f"fixtures built in {ROOT}")


def run_pipeline() -> int:
    env = dict(os.environ, EUC_ROOT=str(ROOT))
    proc = subprocess.run([sys.executable, str(common.TOOLS_DIR / "run_ride.py"), "--force"], env=env)
    return proc.returncode


def verify() -> bool:
    ok = True

    def expect(cond: bool, msg: str):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        ok = ok and cond

    rides = [d for d in (ROOT / "work").iterdir() if (d / "manifest.json").exists()]
    expect(len(rides) == 1, f"one ride found ({[r.name for r in rides]})")
    if not rides:
        return False
    wd = rides[0]

    def load(name, default):
        p = wd / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    sync = load("sync.json", {})
    lag = sync.get("ambient_lag_s", 0)
    expect(abs(lag - CLOCK_ERROR) < 0.15, f"sync found clock error {lag:+.2f}s (expected {CLOCK_ERROR:+.2f}s), "
                                          f"confidence {sync.get('confidence')}")
    cues = load("cues.json", [])
    heard = {c["cue"] for c in cues}
    expect({"clip that", "watch this"} <= heard, f"cues heard: {sorted(heard)}")
    plan = load("edit_plan.json", {"long": {"segments": []}, "shorts": []})
    expect(len(plan["long"]["segments"]) > 0, f"long plan has {len(plan['long']['segments'])} shots")
    expect(len(plan["shorts"]) >= 1, f"{len(plan['shorts'])} short(s) planned")
    od = ROOT / "output" / wd.name
    for name in ["long.mp4", "long.srt", "timeline.fcpxml", "notes.md", "run_log.md"] + \
                [f"{s['id']}.mp4" for s in plan["shorts"]]:
        p = od / name
        expect(p.exists() and p.stat().st_size > 0, f"output {name}")
    if (od / "long.mp4").exists():
        info = common.media_info(od / "long.mp4")
        expect(info["has_audio"] and info["duration"] > 5, f"long.mp4 {info['width']}x{info['height']} "
                                                            f"{info['duration']:.1f}s audio={info['has_audio']}")
    sheets = list((wd / "frames" / "sheets").glob("*.png"))
    expect(len(sheets) >= 2, f"{len(sheets)} contact sheets")
    return ok


if __name__ == "__main__":
    print("== building fixtures")
    build_fixtures()
    print("== running pipeline")
    rc = run_pipeline()
    print(f"== pipeline exit code {rc}")
    print("== verifying")
    good = verify() and rc == 0
    print("\nSELFTEST", "PASSED" if good else "FAILED")
    sys.exit(0 if good else 1)
