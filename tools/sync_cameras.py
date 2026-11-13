"""
sync_cameras.py - Step 4. Line up the GoPro with the Insta360 using sound.

Plain-language summary:
  * File timestamps get us close (the "coarse" sync in audio.json), but camera
    clocks drift, so we refine by listening.
  * Tyler's voice is on both tracks: clear on the Insta360 (helmet mic), muffled
    on the GoPro. We turn each track into a "loudness envelope" (100 values per
    second) and slide one against the other to find the shift where they match
    best. That's cross-correlation.
  * Result: ambient_lag_s. Positive means the GoPro track is running LATE
    compared to the voice track by that many seconds, so GoPro files get moved
    earlier on the ride timeline by that amount.
  * Confidence = how much the best match stands out from the noise. Under the
    threshold in config.json it's flagged low-confidence (we still use it, but
    notes.md will tell Tyler to double-check).
  * Writes work/<ride>/sync.json with the lag, confidence, and each GoPro
    file's corrected position on the ride timeline.

Run:  python tools/sync_cameras.py <ride-id>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from common import CONFIG, log, read_json, run_log, run_log_header, work_dir, write_json


def envelope(path: Path, env_hz: int) -> np.ndarray:
    """Load a 16 kHz mono WAV and return its loudness envelope at env_hz samples/sec."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Keep the voice band (300-3000 Hz) so wind rumble and tire hum don't dominate.
    sos = signal.butter(4, [300, 3000], btype="bandpass", fs=sr, output="sos")
    audio = signal.sosfilt(sos, audio)
    block = sr // env_hz
    n = len(audio) // block
    env = np.abs(audio[: n * block]).reshape(n, block).mean(axis=1)
    # Compress dynamic range and remove slow trends so correlation keys on
    # onsets (syllables) rather than overall level.
    env = np.log1p(env * 50)
    env -= signal.medfilt(env, kernel_size=env_hz * 2 + 1) if len(env) > env_hz * 2 + 1 else env.mean()
    std = env.std()
    return (env - env.mean()) / std if std > 1e-8 else env


def best_lag(ref: np.ndarray, other: np.ndarray, env_hz: int, window_s: float | None) -> tuple[float, float, float]:
    """
    Cross-correlate. Returns (lag_seconds, confidence, peak_value).
    lag > 0 means `other` is delayed relative to `ref`.
    """
    corr = signal.correlate(other, ref, mode="full", method="fft")
    lags = signal.correlation_lags(len(other), len(ref), mode="full")
    if window_s is not None:
        w = int(window_s * env_hz)
        keep = np.abs(lags) <= w
        corr, lags = corr[keep], lags[keep]
    i = int(np.argmax(corr))
    peak = float(corr[i])
    # Robust noise estimate: median absolute deviation of the correlation,
    # ignoring +-1 s around the peak.
    mask = np.abs(lags - lags[i]) > env_hz
    rest = corr[mask] if mask.any() else corr
    mad = np.median(np.abs(rest - np.median(rest))) * 1.4826 + 1e-9
    confidence = (peak - np.median(rest)) / mad
    return lags[i] / env_hz, float(confidence), peak


def main(ride_id: str) -> dict:
    cfg = CONFIG["sync"]
    wd = work_dir(ride_id)
    layout = read_json(wd / "audio.json")
    adir = wd / "audio"
    voice_p, amb_p = adir / "voice_16k.wav", adir / "ambient_16k.wav"

    result = {"ride_id": ride_id, "method": None, "ambient_lag_s": 0.0, "confidence": None,
              "low_confidence": False, "notes": [], "sources": layout["sources"]}
    run_log_header(ride_id, "Camera sync")

    have_both = voice_p.exists() and amb_p.exists() and layout.get("voice_source") == "insta"
    if not have_both:
        result["method"] = "single-camera"
        result["notes"].append("Only one camera has audio; no sync needed.")
        run_log(ride_id, "single camera - sync skipped")
        for s in result["sources"]:
            s["offset_in_ride_corrected"] = s["offset_in_ride"]
        write_json(wd / "sync.json", result)
        return result

    env_hz = cfg["envelope_hz"]
    ref = envelope(voice_p, env_hz)
    other = envelope(amb_p, env_hz)
    n = min(len(ref), len(other))
    ref, other = ref[:n], other[:n]

    lag, conf, _ = best_lag(ref, other, env_hz, cfg["search_window_s"])
    result["method"] = f"cross-correlation (+-{cfg['search_window_s']}s window)"
    if conf < cfg["min_confidence"]:
        # Try the whole range in case the clocks are off by more than the window.
        lag2, conf2, _ = best_lag(ref, other, env_hz, None)
        result["notes"].append(f"windowed search weak (conf {conf:.1f}); full search gave lag {lag2:.2f}s conf {conf2:.1f}")
        if conf2 > conf:
            lag, conf = lag2, conf2
            result["method"] = "cross-correlation (full range)"

    result["ambient_lag_s"] = round(float(lag), 3)
    result["confidence"] = round(conf, 2)
    result["low_confidence"] = conf < cfg["min_confidence"]
    for s in result["sources"]:
        if s["camera"] == "gopro":
            s["offset_in_ride_corrected"] = s["offset_in_ride"] - lag
        else:
            s["offset_in_ride_corrected"] = s["offset_in_ride"]

    msg = f"GoPro lag {lag:+.2f}s vs voice track, confidence {conf:.1f} ({result['method']})"
    if result["low_confidence"]:
        msg += "  <-- LOW CONFIDENCE, flagged for notes.md"
        result["notes"].append("Sync confidence is low. If GoPro shots look out of step with the voice, "
                               "the offset in sync.json may be wrong.")
    run_log(ride_id, msg)
    write_json(wd / "sync.json", result)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: sync_cameras.py <ride-id>")
    main(sys.argv[1])
