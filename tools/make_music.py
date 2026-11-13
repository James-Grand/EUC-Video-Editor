"""
make_music.py - Step 9. Generate original electronic music with numpy.
No samples, no copyrighted material: every sound is synthesised from math.

Plain-language summary:
  * Two moods:
      chill  (96 BPM)  - soft kick, warm pad, slow bass. For cruising/talking.
      speed  (128 BPM) - four-on-the-floor kick, hats, pumping bass, arpeggio.
  * A "bed" is a list of sections, e.g. [chill 90 s, speed 40 s, chill 60 s].
    Each section is rendered at its own tempo and the sections are
    crossfaded, so the music follows the video's energy.
  * Key: A minor, chords Am - F - C - G (one chord per bar, 4-bar loop).
  * Output: 48 kHz stereo WAV, peak-normalised to -3 dBFS (the mixer will set
    the final level and duck it under Tyler's voice).
  * Reusable 60-second beds of each mood are saved to music/ with a line in
    music/catalog.json describing them.

Run standalone:  python tools/make_music.py out.wav chill:30 speed:20 chill:10
Used by render.py for each video's exact section list.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from common import CONFIG, PROJECT_ROOT, log, path_for, read_json, write_json

SR = CONFIG["music"]["sample_rate"]

# A minor: Am, F, C, G  (MIDI note numbers; 57 = A3)
PROGRESSION = [
    [45, 57, 60, 64],   # Am  : root A2, chord A3 C4 E4
    [41, 53, 57, 60],   # F   : root F2, chord F3 A3 C4
    [48, 60, 64, 67],   # C   : root C3, chord C4 E4 G4
    [43, 55, 59, 62],   # G   : root G2, chord G3 B3 D4
]


def midi_hz(n: float) -> float:
    return 440.0 * 2 ** ((n - 69) / 12)


def env_exp(n: int, decay_s: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (decay_s * SR))


def env_adsr(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    s_n = max(0, n - a_n - d_n - r_n)
    e = np.concatenate([
        np.linspace(0, 1, a_n, endpoint=False),
        np.linspace(1, s, d_n, endpoint=False),
        np.full(s_n, s),
        np.linspace(s, 0, r_n),
    ])
    return e[:n] if len(e) >= n else np.pad(e, (0, n - len(e)))


def lowpass(x: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    cutoff = min(cutoff, SR / 2 - 100)
    sos = signal.butter(order, cutoff, btype="low", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def highpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    sos = signal.butter(2, cutoff, btype="high", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def saw(freq: float, n: int, detune: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    f = freq * (1 + detune)
    return 2 * ((t * f) % 1.0) - 1


def place(buf: np.ndarray, start: int, sound: np.ndarray) -> None:
    end = min(len(buf), start + len(sound))
    if end > start:
        buf[start:end] += sound[: end - start]


# ----------------------------------------------------------------------------
# Instruments
# ----------------------------------------------------------------------------

def kick(hard: bool) -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 50 + (170 if hard else 110) * np.exp(-t * 40)
    phase = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(phase) * env_exp(n, 0.18 if hard else 0.25)
    click = np.random.randn(n) * env_exp(n, 0.005) * (0.4 if hard else 0.15)
    return np.tanh((body + click) * (1.8 if hard else 1.3))


def clap() -> np.ndarray:
    n = int(0.25 * SR)
    noise = np.random.randn(n)
    sos = signal.butter(2, [900, 4500], btype="band", fs=SR, output="sos")
    noise = signal.sosfilt(sos, noise)
    e = env_exp(n, 0.08)
    for k in (0.012, 0.024):  # flam for a clap feel
        e[int(k * SR):] += env_exp(n - int(k * SR), 0.06) * 0.7
    return noise * e * 0.5


def hat(open_: bool = False) -> np.ndarray:
    n = int((0.18 if open_ else 0.06) * SR)
    noise = highpass(np.random.randn(n), 6000)
    return noise * env_exp(n, 0.07 if open_ else 0.02) * 0.35


def bass_note(freq: float, dur: float, hard: bool) -> np.ndarray:
    n = int(dur * SR)
    x = saw(freq, n) + 0.5 * saw(freq / 2, n)
    cutoff = 90 + (900 if hard else 350) * env_exp(n, 0.25 if hard else 0.6)
    # Simple time-varying lowpass: filter in short blocks.
    out = np.zeros(n)
    blk = 512
    for i in range(0, n, blk):
        c = float(cutoff[min(i, n - 1)])
        out[i:i + blk] = lowpass(x[i:i + blk], c)
    return np.tanh(out * 1.5) * env_adsr(n, 0.005, 0.1, 0.8, 0.05)


def pad_chord(notes: list[int], dur: float, bright: bool) -> np.ndarray:
    n = int(dur * SR)
    x = np.zeros(n)
    for m in notes:
        f = midi_hz(m)
        for det in (-0.004, 0.0, 0.004):
            x += saw(f, n, det)
    x = lowpass(x / (3 * len(notes)), 1800 if bright else 1000)
    return x * env_adsr(n, 0.4, 0.5, 0.7, 0.6)


def arp_note(freq: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sign(np.sin(2 * np.pi * freq * t)) * 0.5 + saw(freq, n) * 0.5
    x = lowpass(x, 2500)
    return x * env_adsr(n, 0.003, 0.08, 0.3, 0.05) * 0.5


def echo(x: np.ndarray, delay_s: float, fb: float = 0.35, mix: float = 0.3) -> np.ndarray:
    d = int(delay_s * SR)
    out = x.copy()
    buf = x.copy()
    for _ in range(4):
        shifted = np.zeros_like(buf)
        shifted[d:] = buf[:-d] * fb
        out += shifted * mix
        buf = shifted
    return out


# ----------------------------------------------------------------------------
# Section renderer
# ----------------------------------------------------------------------------

def render_section(mood: str, duration: float, seed: int = 0) -> np.ndarray:
    """Return stereo float32 array (n, 2) for one section."""
    np.random.seed(seed)
    hard = mood == "speed"
    bpm = CONFIG["music"]["speed_bpm"] if hard else CONFIG["music"]["chill_bpm"]
    beat = 60.0 / bpm
    bar = beat * 4
    n_bars = int(np.ceil(duration / bar)) + 1
    n = int(n_bars * bar * SR)
    drums, bass, pad, lead = (np.zeros(n) for _ in range(4))

    k_hard, k_soft, cl, h_c, h_o = kick(True), kick(False), clap(), hat(False), hat(True)
    for b in range(n_bars):
        chord = PROGRESSION[b % 4]
        root_hz = midi_hz(chord[0])
        bar_start = int(b * bar * SR)
        # --- drums
        for beat_i in range(4):
            bs = bar_start + int(beat_i * beat * SR)
            if hard:
                place(drums, bs, k_hard)
                if beat_i in (1, 3):
                    place(drums, bs, cl)
                for sub in range(4):
                    g = 1.0 if sub % 2 == 0 else 0.6
                    place(drums, bs + int(sub * beat / 4 * SR), h_c * g)
                place(drums, bs + int(beat / 2 * SR), h_o * 0.8)
            else:
                if beat_i in (0, 2) or (beat_i == 3 and b % 2 == 1):
                    place(drums, bs, k_soft)
                if beat_i in (1, 3):
                    place(drums, bs, cl * 0.4)
                place(drums, bs + int(beat / 2 * SR), h_c * 0.6)
        # --- bass
        if hard:
            for e in range(8):
                f = root_hz * (2 if e % 2 else 1)
                place(bass, bar_start + int(e * beat / 2 * SR), bass_note(f, beat / 2 * 0.9, True))
        else:
            place(bass, bar_start, bass_note(root_hz, bar * 0.95, False))
        # --- pad
        place(pad, bar_start, pad_chord(chord[1:], bar * 1.05, bright=hard))
        # --- arpeggio (speed only)
        if hard:
            tones = chord[1:] + [chord[1] + 12]
            for s in range(16):
                f = midi_hz(tones[s % len(tones)] + 12)
                place(lead, bar_start + int(s * beat / 4 * SR), arp_note(f, beat / 4 * 0.9))

    if hard:
        lead = echo(lead, beat * 0.75)
        # Sidechain-style pump: duck pad/bass on every kick.
        pump = np.ones(n)
        for b in range(n_bars * 4):
            s = int(b * beat * SR)
            m = min(n, s + int(0.25 * SR))
            pump[s:m] *= 0.35 + 0.65 * (1 - env_exp(m - s, 0.12))
        pad *= pump
        bass *= 0.7 + 0.3 * pump

    mono_mix = {
        "drums": drums * (1.0 if hard else 0.7),
        "bass": bass * (0.8 if hard else 0.7),
        "pad": pad * (0.45 if hard else 0.75),
        "lead": lead * 0.35,
    }
    left = mono_mix["drums"] + mono_mix["bass"] + mono_mix["pad"] + mono_mix["lead"]
    right = left.copy()
    # Width: slight Haas delay on pad and lead for the right channel.
    d = int(0.012 * SR)
    right[d:] += (mono_mix["pad"][:-d] + mono_mix["lead"][:-d]) * 0.4
    left += (mono_mix["pad"] + mono_mix["lead"]) * 0.4 * 0.3
    stereo = np.stack([left, right], axis=1)
    stereo = np.tanh(stereo * 0.8)
    n_out = int(duration * SR)
    return stereo[:n_out].astype(np.float32)


def render_bed(sections: list[dict], out_path: Path, fade_out: float = 2.0) -> Path:
    """sections = [{"mood": "chill"|"speed", "duration": seconds}, ...]"""
    xfade = int(1.0 * SR)
    pieces = []
    for i, sec in enumerate(sections):
        pieces.append(render_section(sec["mood"], sec["duration"] + 1.0, seed=i))
    if not pieces:
        raise ValueError("no sections")
    out = pieces[0]
    for p in pieces[1:]:
        fade = np.linspace(0, 1, xfade)[:, None]
        head = out[-xfade:] * (1 - fade) + p[:xfade] * fade
        out = np.concatenate([out[:-xfade], head, p[xfade:]])
    total_n = int(sum(s["duration"] for s in sections) * SR)
    out = out[:total_n]
    fo = min(len(out), int(fade_out * SR))
    if fo:
        out[-fo:] *= np.linspace(1, 0, fo)[:, None]
    peak = np.abs(out).max() or 1.0
    out = out / peak * 10 ** (-3 / 20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, SR, subtype="PCM_16")
    return out_path


def ensure_library() -> None:
    """Save a 60 s bed of each mood into music/ once, with a description."""
    music = path_for("music")
    catalog_path = music / "catalog.json"
    catalog = read_json(catalog_path, [])
    have = {c["file"] for c in catalog}
    for mood, desc in (("chill", "96 BPM, A minor, soft kick, warm pad, slow bass. Cruising / talking."),
                       ("speed", "128 BPM, A minor, four-on-the-floor, hats, pumping bass, arpeggio. Fast sections.")):
        name = f"bed_{mood}_60s.wav"
        if name not in have:
            render_bed([{"mood": mood, "duration": 60}], music / name)
            catalog.append({"file": name, "mood": mood, "duration_s": 60, "description": desc,
                            "generator": "tools/make_music.py (numpy synthesis, original)"})
            log(f"saved reusable bed music/{name}")
    write_json(catalog_path, catalog)


def parse_sections(args: list[str]) -> list[dict]:
    secs = []
    for a in args:
        mood, _, dur = a.partition(":")
        secs.append({"mood": mood, "duration": float(dur)})
    return secs


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: make_music.py out.wav chill:30 speed:20 ...")
    render_bed(parse_sections(sys.argv[2:]), Path(sys.argv[1]))
    print("wrote", sys.argv[1])
