"""
make_music.py - Step 9. Generate original music with numpy, following
docs/music_brief.md (the brief written from Tyler's "EUC '26" playlist).
No samples, no AI model, no copyrighted material: every sound is built from
maths in this file, so there is nothing to license and it is safe for
monetised YouTube.

Plain-language summary:
  * Six styles, straight from the brief's table (BPMs live in config.json):
      dnb      174  drum & bass: rolling breaks, reese bass, rave stabs.   Speed runs.
      electro  128  electro house: four-on-the-floor, wobble bass, saw stabs.  Big moments.
      funk     110  funk-hop / electro-swing: swung breakbeat, slap bass,
                    brass stabs, "hey!" shouts, record scratches.          Cruising.
      indie    118  sunny indie electronic: soft four-on-the-floor, warm pads, arps.  Scenery.
      blues    92   gritty blues riff over a foot-stomp beat.  Accent only (intro / tension).
      liquid   174  liquid drum & bass: rolling but uplifting, pads, wordless vocals.  Outro.
  * A "bed" is a list of sections, e.g. [blues 8 s, funk 90 s, dnb 40 s, liquid 30 s].
    Each section is built bar by bar in 4- and 8-bar phrases so video cuts
    can land on downbeats.
  * Hook fast: music hits within the first ~4 seconds, no long ambient intros.
  * Drops: a section can carry "drop_at" (seconds). The bars are shifted so a
    downbeat + crash + full groove land exactly there, with a riser and
    snare roll building into it. Used to put the drop on a short's best moment.
  * Between sections the outgoing one builds (riser + snare roll) and the new
    one starts on a downbeat with a crash, like cutting between tracks.
  * Sections longer than max_track_s are split into "tracks" in different keys
    so a 10-minute vlog doesn't loop one groove forever.
  * The bed ends clean on a downbeat hit that rings out, not a fade.
  * The 1-4 kHz range is dipped a few dB ("voice pocket") so Tyler's voice and
    the GoPro audio have room.
  * Four stems can be written (drums, bass, music, vocals) for DaVinci Resolve.
    The stems add up exactly to the full mix.
  * Output: 48 kHz stereo WAV, peak-normalised to -3 dBFS. The mixer sets the
    final level, ducks it under Tyler's voice, and masters to -14 LUFS.

Run standalone:
  python tools/make_music.py out.wav funk:30 dnb:20@8 liquid:20
      (style:seconds, optional @seconds = where the drop lands in that section)
  python tools/make_music.py --library
      (writes a 45-second demo of every style into music/ for listening)
Used by render.py for each video's exact section list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from common import CONFIG, log, path_for, read_json, write_json

MCFG = CONFIG["music"]
SR = MCFG["sample_rate"]
STEMS = ("drums", "bass", "music", "vocals")
STYLES = tuple(MCFG["styles"].keys())
LEGACY = {"chill": "funk", "speed": "dnb"}       # old section names still work
HIGH_ENERGY = {"dnb", "electro"}

# Chords per style. Each entry is [bass root, chord tones...] as MIDI note
# numbers (60 = middle C). One chord per bar, 4-bar loop.
PROGRESSIONS = {
    "dnb":     [[41, 53, 56, 60], [37, 53, 56, 61], [44, 51, 56, 60], [39, 51, 55, 58]],  # Fm Db Ab Eb
    "electro": [[45, 57, 60, 64], [41, 53, 57, 60], [48, 55, 60, 64], [43, 55, 59, 62]],  # Am F C G
    "funk":    [[40, 52, 55, 59, 62], [45, 55, 57, 61, 64], [40, 52, 55, 59, 62], [47, 54, 57, 59, 62]],  # Em7 A7 Em7 Bm7
    "indie":   [[38, 62, 66, 69], [45, 61, 64, 69], [47, 62, 66, 71], [43, 62, 67, 71]],  # D A Bm G
    "blues":   [[40, 52, 56, 59, 62], [40, 52, 56, 59, 62], [45, 55, 57, 61, 64], [40, 52, 56, 59, 62]],  # E7 E7 A7 E7
    "liquid":  [[37, 56, 60, 61, 65], [36, 55, 58, 60, 63], [41, 56, 60, 63, 67], [34, 56, 58, 61, 65]],  # Dbmaj7 Cm7 Fm9 Bbm7
}
SWING = {"funk": 0.22, "blues": 0.33}            # fraction of a 16th note the off-16ths are pushed late
INTRO_BARS = {"dnb": 2, "liquid": 2, "electro": 2, "funk": 1, "indie": 1, "blues": 0}
TRANSPOSE_BY_TRACK = [0, 2, -2, 3]               # key change for each extra "track" in a long section


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

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


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = signal.butter(2, [lo, min(hi, SR / 2 - 100)], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def sweep_lowpass(x: np.ndarray, cutoffs: np.ndarray, block: int = 256) -> np.ndarray:
    """Lowpass whose cutoff moves over time (filter sweeps, wobbles, plucks).
    The filter memory is carried from block to block so it doesn't click."""
    out = np.empty_like(x)
    zi = np.zeros((1, 2))
    for i in range(0, len(x), block):
        c = float(np.clip(cutoffs[min(i, len(cutoffs) - 1)], 30, SR / 2 - 200))
        sos = signal.butter(2, c, btype="low", fs=SR, output="sos")
        out[i:i + block], zi = signal.sosfilt(sos, x[i:i + block], zi=zi)
    return out


def peaking(x: np.ndarray, f0: float, gain_db: float, q: float) -> np.ndarray:
    """Bell EQ (boost or cut around f0). Used for the voice pocket."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return signal.lfilter(b, den, x)


def saw(freq: float, n: int, detune: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    return 2 * ((t * freq * (1 + detune)) % 1.0) - 1


def osc_phase(freqs: np.ndarray) -> np.ndarray:
    """Running phase (in cycles) for a frequency that changes over time."""
    return np.cumsum(freqs) / SR


def place(buf: np.ndarray, start: int, sound: np.ndarray) -> None:
    if start < 0:
        sound, start = sound[-start:], 0
    end = min(len(buf), start + len(sound))
    if end > start:
        buf[start:end] += sound[: end - start]


def echo(x: np.ndarray, delay_s: float, fb: float = 0.35, mix: float = 0.35) -> np.ndarray:
    d = int(delay_s * SR)
    out = x.copy()
    buf = x
    for _ in range(4):
        shifted = np.zeros_like(buf)
        shifted[d:] = buf[:-d] * fb
        out += shifted * mix
        buf = shifted
    return out


# ----------------------------------------------------------------------------
# Drums
# ----------------------------------------------------------------------------

def make_kit(rng: np.random.Generator) -> dict:
    def noise(sec):
        return rng.standard_normal(int(sec * SR))

    def kick(hard):
        n = int(0.35 * SR)
        t = np.arange(n) / SR
        f = 50 + (170 if hard else 110) * np.exp(-t * 40)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 0.18 if hard else 0.25)
        click = noise(0.35) * env_exp(n, 0.005) * (0.4 if hard else 0.15)
        return np.tanh((body + click) * (1.8 if hard else 1.3))

    n = int(0.22 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 185 * t) * env_exp(n, 0.06) + 0.5 * np.sin(2 * np.pi * 330 * t) * env_exp(n, 0.04)
    snare = np.tanh(1.6 * (0.6 * tone + bandpass(noise(0.22), 1500, 9000) * env_exp(n, 0.1) * 0.8)) * 0.7

    n = int(0.25 * SR)
    e = env_exp(n, 0.08)
    for k in (0.012, 0.024):  # three quick hits = a clap
        e[int(k * SR):] += env_exp(n - int(k * SR), 0.06) * 0.7
    clap = bandpass(noise(0.25), 900, 4500) * e * 0.5

    hat = highpass(noise(0.06), 6000) * env_exp(int(0.06 * SR), 0.02) * 0.35
    ohat = highpass(noise(0.18), 6000) * env_exp(int(0.18 * SR), 0.07) * 0.35
    n = int(0.08 * SR)
    shaker = bandpass(noise(0.08), 4000, 10000) * env_adsr(n, 0.01, 0.03, 0.2, 0.03) * 0.3
    n = int(1.6 * SR)
    crash = (highpass(noise(1.6), 4000) + 0.5 * bandpass(noise(1.6), 6000, 8000)) * env_exp(n, 0.45) * 0.3

    n = int(0.4 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * np.cumsum(45 + 70 * np.exp(-t * 30)) / SR) * env_exp(n, 0.2)
    thud = lowpass(noise(0.4), 400) * env_exp(n, 0.05)
    stomp = np.tanh(1.5 * (body + 0.6 * thud))

    return {"kick": kick(True), "kick_soft": kick(False), "snare": snare, "clap": clap, "hat": hat,
            "ohat": ohat, "shaker": shaker, "crash": crash, "stomp": stomp}


def riser(dur: float, rng: np.random.Generator) -> np.ndarray:
    """White-noise sweep plus a rising tone. Builds tension into a drop."""
    n = max(1, int(dur * SR))
    t = np.arange(n) / SR
    y = highpass(sweep_lowpass(rng.standard_normal(n), 300 * 40 ** (t / dur), 512), 200)
    tone = np.sin(2 * np.pi * osc_phase(200 * 6 ** (t / dur))) * 0.3
    return (y * 0.5 + tone) * (t / dur) ** 2 * 0.6


# ----------------------------------------------------------------------------
# Bass
# ----------------------------------------------------------------------------

def reese(m: int, dur: float) -> np.ndarray:
    """Drum & bass "reese": detuned saws that phase against each other, plus a sub."""
    n = int(dur * SR)
    f = midi_hz(m)
    t = np.arange(n) / SR
    x = (saw(f, n, -0.007) + saw(f, n, 0.007) + saw(f, n)) / 3
    x = sweep_lowpass(x, 350 + 500 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)))
    sub = np.sin(2 * np.pi * f / 2 * t) * 0.8
    return np.tanh(2.2 * x + sub) * env_adsr(n, 0.005, 0.05, 0.9, 0.03) * 0.8


def wobble(m: int, dur: float, rate_hz: float) -> np.ndarray:
    """Electro / dubstep wobble: a filter opening and closing in time with the beat."""
    n = int(dur * SR)
    f = midi_hz(m)
    t = np.arange(n) / SR
    x = saw(f, n) * 0.7 + np.sign(np.sin(2 * np.pi * f / 2 * t)) * 0.3
    y = sweep_lowpass(x, 120 + 2200 * (0.5 - 0.5 * np.cos(2 * np.pi * rate_hz * t)))
    y = np.tanh(3 * y) * 0.7 + np.sin(2 * np.pi * f / 2 * t) * 0.4
    return y * env_adsr(n, 0.005, 0.05, 0.9, 0.03)


def slap(m: int, dur: float, pop: bool) -> np.ndarray:
    """Funky slap/pop bass: bright, fast-decaying pluck."""
    n = int(dur * SR)
    f = midi_hz(m)
    t = np.arange(n) / SR
    x = saw(f, n) * 0.6 + np.sign(np.sin(2 * np.pi * f * t)) * 0.4
    y = sweep_lowpass(x, 250 + (5000 if pop else 2500) * env_exp(n, 0.04 if pop else 0.06), 128)
    amp = env_adsr(n, 0.002, 0.15, 0.5, 0.03)
    out = np.tanh(2 * y * amp) * 0.8
    if not pop:
        out += np.sin(2 * np.pi * f * t) * amp * 0.3
    return out


def sub(m: int, dur: float, ring: bool = False) -> np.ndarray:
    """Smooth sub bass (liquid, blues, and the final ring-out)."""
    n = int(dur * SR)
    f = midi_hz(m)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t) + 0.25 * lowpass(saw(f, n), 250)
    amp = env_exp(n, 0.6) if ring else env_adsr(n, 0.01, 0.1, 0.85, 0.08)
    return np.tanh(1.3 * x) * amp * 0.8


def soft_bass(m: int, dur: float) -> np.ndarray:
    """Round, gentle bass for the indie style."""
    n = int(dur * SR)
    f = midi_hz(m)
    x = saw(f, n) + 0.5 * saw(f / 2, n)
    x = sweep_lowpass(x, 90 + 350 * env_exp(n, 0.3))
    return np.tanh(x * 1.5) * env_adsr(n, 0.005, 0.1, 0.7, 0.05)


# ----------------------------------------------------------------------------
# Chords, leads, guitar
# ----------------------------------------------------------------------------

def pad_chord(notes: tuple, dur: float, bright: bool, ring: bool = False) -> np.ndarray:
    n = int(dur * SR)
    x = np.zeros(n)
    for m in notes:
        for det in (-0.004, 0.0, 0.004):
            x += saw(midi_hz(m), n, det)
    x = lowpass(x / (3 * len(notes)), 1800 if bright else 1000)
    return x * (env_exp(n, 0.9) * env_adsr(n, 0.01, 0, 1, 0.2) if ring else env_adsr(n, 0.4, 0.5, 0.7, 0.6))


def arp_note(m: int, dur: float) -> np.ndarray:
    n = int(dur * SR)
    f = midi_hz(m)
    t = np.arange(n) / SR
    x = np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + saw(f, n) * 0.5
    return lowpass(x, 2500) * env_adsr(n, 0.003, 0.08, 0.3, 0.05) * 0.5


def supersaw(notes: tuple, dur: float) -> np.ndarray:
    """Big distorted saw chord stab (electro house)."""
    n = int(dur * SR)
    x = np.zeros(n)
    for m in notes:
        for det in (-0.012, -0.006, 0.0, 0.006, 0.012):
            x += saw(midi_hz(m), n, det)
    x = sweep_lowpass(x / (5 * len(notes)), 800 + 5000 * env_exp(n, 0.3))
    return np.tanh(2 * x * env_adsr(n, 0.003, 0.1, 0.5, 0.05)) * 0.6


def rave_stab(notes: tuple, dur: float) -> np.ndarray:
    """Short, punchy rave chord with a tiny pitch drop (drum & bass)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in notes:
        ph = osc_phase(midi_hz(m) * (1 + 0.03 * np.exp(-t / 0.02)))
        x += np.sign(np.sin(2 * np.pi * ph)) * 0.5 + (2 * (ph % 1) - 1) * 0.5
    x = sweep_lowpass(x / len(notes), 600 + 7000 * np.exp(-t / 0.12))
    return np.tanh(1.8 * x * env_adsr(n, 0.002, 0.08, 0.4, 0.04)) * 0.6


def brass(notes: tuple, dur: float) -> np.ndarray:
    """Synth brass stab: scoops up into pitch, filter swells open. Funk / electro-swing."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in notes:
        f = midi_hz(m) * 2 ** (-0.3 * np.exp(-t / 0.03) / 12)
        for det in (-0.003, 0.003):
            x += 2 * (osc_phase(f * (1 + det)) % 1) - 1
    cut = 800 + 3000 * (1 - np.exp(-t / 0.03)) * np.exp(-t / 0.25)
    x = sweep_lowpass(x / (2 * len(notes)), cut)
    return np.tanh(1.5 * x * env_adsr(n, 0.008, 0.12, 0.6, 0.06)) * 0.6


def siren(dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * osc_phase(900 + 300 * np.sin(2 * np.pi * 2 * t)))
    return lowpass(np.tanh(2 * x), 3000) * env_adsr(n, 0.05, 0.1, 0.8, 0.3) * 0.3


def guitar(m: int, dur: float, slide: float = 0) -> np.ndarray:
    """Dirty electric blues note. slide = semitones it slides up from."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = midi_hz(m) * 2 ** (-slide * np.exp(-t / 0.06) / 12)
    ph = 2 * np.pi * osc_phase(f)
    x = sum(np.sin(k * ph) / k for k in range(1, 13))
    x = x * env_exp(n, 0.7) * env_adsr(n, 0.002, 0, 1, 0.04)
    return lowpass(np.tanh(4 * x) * 0.5, 3200)


def scratch(rng: np.random.Generator) -> np.ndarray:
    """Record scratch: a brass hit played back and forth like a DJ on vinyl."""
    src = brass((52, 55, 59), 0.5)
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    pos = (0.12 + 0.1 * np.sin(2 * np.pi * 5 * t)) * SR
    x = np.interp(pos, np.arange(len(src)), src)
    grit = bandpass(rng.standard_normal(n), 1000, 5000) * 0.08
    return (x * 1.5 + grit) * env_adsr(n, 0.005, 0.05, 0.9, 0.05)


# ----------------------------------------------------------------------------
# Vocals (synthesised: a buzzy source pushed through vowel-shaped filters)
# ----------------------------------------------------------------------------

VOWELS = {  # (formant Hz, gain, bandwidth Hz)
    "a": [(730, 1.0, 80), (1090, 0.5, 90), (2440, 0.25, 120)],
    "e": [(530, 1.0, 60), (1840, 0.6, 100), (2480, 0.3, 120)],
    "i": [(300, 1.0, 50), (2300, 0.5, 100), (3000, 0.3, 120)],
    "o": [(570, 1.0, 60), (840, 0.45, 80), (2410, 0.15, 120)],
}


def formant(src: np.ndarray, vowel: str) -> np.ndarray:
    out = np.zeros_like(src)
    for f, g, bw in VOWELS[vowel]:
        b, a = signal.iirpeak(f, f / bw, fs=SR)
        out += signal.lfilter(b, a, src) * g
    return out


def vocal_hey(f0: float = 210) -> np.ndarray:
    """Generic hype shout, "hey!". Not anyone's voice; it's a filtered buzz."""
    n = int(0.32 * SR)
    t = np.arange(n) / SR
    f = np.linspace(f0 * 1.15, f0 * 0.9, n)
    src = lowpass(2 * (osc_phase(f) % 1) - 1, 5000)
    xf = np.clip((t - 0.08) / 0.15, 0, 1)
    voiced = formant(src, "e") * (1 - xf) + formant(src, "i") * xf
    voiced *= env_adsr(n, 0.04, 0.1, 0.7, 0.12)
    h = highpass(np.random.default_rng(7).standard_normal(n), 1500) * env_exp(n, 0.03) * 0.15
    x = voiced + h
    return np.tanh(1.5 * x / (np.abs(x).max() or 1)) * 0.6


def vocal_pad(notes: tuple, dur: float, vowel: str = "a", attack: float = 0.25) -> np.ndarray:
    """Wordless "aah"/"ooh" vocal on chord tones (liquid DnB)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    src = np.zeros(n)
    for m in notes:
        f = midi_hz(m) * (1 + 0.004 * np.sin(2 * np.pi * 5 * t))
        src += 2 * (osc_phase(f) % 1) - 1
    x = formant(lowpass(src / len(notes), 4000), vowel)
    x = x / (np.abs(x).max() or 1)
    return x * env_adsr(n, attack, 0.1, 0.8, min(0.3, dur / 3)) * 0.5


# Sounds are rebuilt only when their settings change, so a 4-minute section
# reuses the same few bass notes and chords instead of re-synthesising them.
_CACHE: dict = {}


def snd(kind: str, *args) -> np.ndarray:
    key = (kind, *args)
    if key not in _CACHE:
        fn = {"reese": reese, "wobble": wobble, "slap": slap, "sub": sub, "soft_bass": soft_bass,
              "pad": pad_chord, "arp": arp_note, "supersaw": supersaw, "rave": rave_stab, "brass": brass,
              "guitar": guitar, "hey": vocal_hey, "vpad": vocal_pad, "siren": siren}[kind]
        _CACHE[key] = fn(*args).astype(np.float32)
    return _CACHE[key]


# ----------------------------------------------------------------------------
# The bar-by-bar arranger
# ----------------------------------------------------------------------------

class Ctx:
    """Holds the buses for one section and turns (bar, 16th-step) into samples."""

    def __init__(self, style: str, n: int, rng: np.random.Generator):
        self.style = style
        self.beat = 60.0 / MCFG["styles"][style]["bpm"]
        self.bar = self.beat * 4
        self.step = self.beat / 4
        self.swing = SWING.get(style, 0.0)
        self.bus = {k: np.zeros(n, np.float32) for k in (*STEMS, "echo")}
        self.kicks: list[float] = []
        self.rng = rng

    def t(self, b: int, s: float) -> float:
        late = self.swing * self.step if float(s).is_integer() and int(s) % 2 == 1 else 0.0
        return b * self.bar + s * self.step + late

    def put(self, bus: str, b: int, s: float, sound: np.ndarray, gain: float = 1.0) -> None:
        place(self.bus[bus], int(self.t(b, s) * SR), sound * gain)

    def kick(self, b: int, s: float, sound: np.ndarray, gain: float = 1.0) -> None:
        self.put("drums", b, s, sound, gain)
        self.kicks.append(self.t(b, s))


def build_drums(c: Ctx, b: int, j: int, m: int, kit: dict, soft: bool) -> None:
    """Snare roll that speeds up into the drop: quarters, then 8ths, then 16ths."""
    g = 0.6 if soft else 1.0
    kick = kit["kick_soft"] if soft else kit["kick"]
    for s in ((0,) if j == m - 1 else (0, 4, 8, 12)):     # four-on-the-floor pulse, then just the downbeat
        c.kick(b, s, kick, 0.7 * g)
    if j == m - 1:
        steps = range(16)
    elif j >= m - 2:
        steps = range(0, 16, 2)
    else:
        steps = range(0, 16, 4)
    for s in steps:
        c.put("drums", b, s, kit["snare"], g * (0.35 + 0.6 * (j + s / 16) / m))
    for s in range(0, 16, 2):
        c.put("drums", b, s, kit["hat"], 0.3 * g)


def sty_dnb(c, b, role, ch, fill, kit, build):
    root, tones = ch[0], tuple(ch[1:])
    if role == "full":
        c.kick(b, 0, kit["kick"])
        c.kick(b, 10, kit["kick"], 0.9)
        for s, g in ((4, 1), (12, 1), (7, 0.18), (15, 0.14)):
            c.put("drums", b, s, kit["snare"], g)
        for s in range(0, 16, 2):
            c.put("drums", b, s, kit["hat"], 0.35 if s % 4 == 0 else 0.55)
        for s in range(1, 16, 2):
            c.put("drums", b, s, kit["shaker"], 0.25)
        if fill:
            for s in (13, 14, 15):
                c.put("drums", b, s, kit["snare"], 0.5)
        c.put("bass", b, 0, snd("reese", root, round(10 * c.step, 4)))
        c.put("bass", b, 10, snd("reese", root, round(6 * c.step, 4)))
        stab = snd("rave", tones, round(1.5 * c.step, 4))
        for s in ((0, 6) if b % 2 == 0 else (3, 10)):
            c.put("music", b, s, stab, 0.55)
        c.put("music", b, 0, snd("pad", tones, round(c.bar * 1.05, 4), False), 0.3)
    elif role in ("intro", "breakdown"):
        c.put("music", b, 0, snd("pad", tones, round(c.bar * 1.05, 4), False), 0.6)
        c.put("music", b, 0, snd("rave", tones, round(1.5 * c.step, 4)), 0.5)
        c.put("echo", b, 6, snd("rave", tones, round(1.5 * c.step, 4)), 0.4)
        c.kick(b, 0, kit["kick"], 0.8)
        c.put("drums", b, 12, kit["snare"], 0.6)
        for s in range(0, 16, 2):
            c.put("drums", b, s, kit["hat"], 0.45)
    elif role == "build":
        build_drums(c, b, *build, kit, soft=False)
        c.put("music", b, 0, snd("pad", tones, round(c.bar * 1.05, 4), False), 0.4)


def sty_electro(c, b, role, ch, fill, kit, build):
    root, tones = ch[0], tuple(ch[1:])
    if role == "full":
        for s in (0, 4, 8, 12):
            c.kick(b, s, kit["kick"])
        for s in (4, 12):
            c.put("drums", b, s, kit["clap"])
        for s in range(16):
            c.put("drums", b, s, kit["ohat"] if s % 4 == 2 else kit["hat"], 0.5 if s % 4 == 2 else 0.25)
        if fill:
            for s in (13, 14, 15):
                c.put("drums", b, s, kit["clap"], 0.6)
        rate = (4 if b % 4 == 3 else 2) / c.beat      # 8th-note wobble, 16ths on every 4th bar
        c.put("bass", b, 0, snd("wobble", root - 12, round(c.bar, 4), round(rate, 4)))
        stab = snd("supersaw", tones, round(1.5 * c.step, 4))
        for s in (0, 3, 6, 10, 12):
            c.put("music", b, s, stab, 0.45)
    elif role in ("intro", "breakdown"):
        c.put("music", b, 0, snd("pad", tones, round(c.bar * 1.05, 4), True), 0.6)
        c.kick(b, 0, kit["kick"], 0.8)
        c.kick(b, 8, kit["kick"], 0.6)
        for s in (4, 12):
            c.put("drums", b, s, kit["clap"], 0.6)
        for s in range(2, 16, 4):
            c.put("drums", b, s, kit["ohat"], 0.3)
    elif role == "build":
        build_drums(c, b, *build, kit, soft=False)
        c.put("music", b, 0, snd("pad", tones, round(c.bar * 1.05, 4), True), 0.4)


SLAP_PATTERN = [  # (16th step, semitones above root, length in 16ths, pop?)
    (0, 0, 2, False), (3, 12, 1, True), (4, 0, 1, False), (6, 7, 1, False),
    (8, 0, 2, False), (10, 12, 1, True), (11, 10, 1, False), (14, 7, 2, False),
]


def sty_funk(c, b, role, ch, fill, kit, build):
    root, tones = ch[0], tuple(ch[1:])
    drums = role in ("full", "intro", "breakdown")
    if drums:
        for s in (0, 7, 10):
            c.kick(b, s, kit["kick"], 0.8)
        for s in (4, 12):
            c.put("drums", b, s, kit["snare"], 0.8)
            c.put("drums", b, s, kit["clap"], 0.6)
        for s, g in ((2, 0.15), (9, 0.2), (14, 0.2)):
            c.put("drums", b, s, kit["snare"], g)
        for s in range(16):
            if s == 14:
                c.put("drums", b, s, kit["ohat"], 0.3)
            else:
                c.put("drums", b, s, kit["hat"], 0.35 if s % 2 == 0 else 0.18)
        if fill:
            for s in (13, 14, 15):
                c.put("drums", b, s, kit["snare"], 0.4)
            if b % 16 == 15:
                c.put("music", b, 8, scratch(c.rng), 0.5)
    if role in ("full", "breakdown"):
        for s, iv, ln, pop in SLAP_PATTERN:
            c.put("bass", b, s, snd("slap", root + iv, round(ln * c.step * 0.9, 4), pop), 0.8 if pop else 1.0)
    if role in ("full", "intro"):
        hit = snd("brass", tones, round(1.5 * c.step, 4))
        if b % 2 == 0:
            for s in (3, 6):
                c.put("music", b, s, hit, 0.6)
        elif b % 4 == 1:
            c.put("music", b, 14, snd("brass", tones, round(2 * c.step, 4)), 0.6)
        if role == "intro":
            c.put("music", b, 0, hit, 0.7)
        c.put("music", b, 0, snd("pad", tones, round(c.bar * 1.05, 4), False), 0.2)
        if b % 4 == 3 and not fill:
            c.put("vocals", b, 12, snd("hey", 210.0), 0.6)
    if role == "build":
        build_drums(c, b, *build, kit, soft=False)
        c.put("music", b, 0, snd("pad", tones, round(c.bar * 1.05, 4), False), 0.4)


def sty_indie(c, b, role, ch, fill, kit, build):
    root, tones = ch[0], tuple(ch[1:])
    if role == "full":
        for s in (0, 4, 8, 12):
            c.kick(b, s, kit["kick_soft"], 0.8)
        for s in (4, 12):
            c.put("drums", b, s, kit["clap"], 0.45)
        for s in range(2, 16, 4):
            c.put("drums", b, s, kit["ohat"], 0.2)
        for e in range(8):
            c.put("bass", b, e * 2, snd("soft_bass", root, round(2 * c.step * 0.9, 4)), 1.0 if e % 2 == 0 else 0.7)
    if role in ("intro", "breakdown"):
        for s in (0, 8):
            c.kick(b, s, kit["kick_soft"], 0.6)
    if role in ("full", "intro", "breakdown", "build"):
        for s in range(16):
            c.put("drums", b, s, kit["shaker"], 0.2)
        c.put("music", b, 0, snd("pad", tones, round(c.bar * 1.05, 4), True), 0.5)
        arp = list(tones) + [tones[0] + 12]
        for s in range(16):
            c.put("echo", b, s, snd("arp", arp[s % len(arp)] + 12, round(c.step * 0.9, 4)), 0.4)
    if role == "build":
        build_drums(c, b, *build, kit, soft=True)


BLUES_RIFF = [  # (16th step, semitones above root, length in 16ths, slide-up semitones)
    (0, 0, 3, 2), (3, 3, 1, 0), (4, 5, 2, 1), (6, 6, 1, 0),
    (7, 7, 1, 0), (8, 0, 3, 0), (11, 10, 1, 0), (12, 7, 2, 1), (14, 5, 2, 0),
]


def sty_blues(c, b, role, ch, fill, kit, build):
    root = ch[0]
    if role in ("full", "intro", "breakdown"):
        for s in (0, 8, 10):
            c.kick(b, s, kit["stomp"], 1.0 if s != 10 else 0.6)
        for s in (4, 12):
            c.put("drums", b, s, kit["clap"], 0.6)
        for s in range(0, 16, 2):
            c.put("drums", b, s, kit["shaker"], 0.25)
        for s, iv, ln, sl in BLUES_RIFF:
            c.put("music", b, s, snd("guitar", root + 12 + iv, round(ln * c.step * 1.1, 4), sl), 0.5)
    if role == "full":
        for s in (0, 8):
            c.put("bass", b, s, snd("sub", root, round(8 * c.step * 0.95, 4)))
    if role == "build":
        build_drums(c, b, *build, kit, soft=False)
        for s in range(0, 16, 2):
            c.put("music", b, s, snd("guitar", root + 12, round(2 * c.step, 4), 0), 0.4)


def sty_liquid(c, b, role, ch, fill, kit, build):
    root, tones = ch[0], tuple(ch[1:])
    if role == "full":
        c.kick(b, 0, kit["kick"], 0.8)
        c.kick(b, 10, kit["kick"], 0.7)
        for s, g in ((4, 0.7), (12, 0.7), (2, 0.12), (7, 0.15), (9, 0.12), (15, 0.15)):
            c.put("drums", b, s, kit["snare"], g)
        for s in range(2, 16, 4):
            c.put("drums", b, s, kit["ohat"], 0.15)
        c.put("bass", b, 0, snd("sub", root, round(10 * c.step, 4)))
        c.put("bass", b, 10, snd("sub", root, round(6 * c.step, 4)))
        for s in (6, 14):
            c.put("vocals", b, s, snd("vpad", (tones[-1] + 12,), round(1.5 * c.step, 4), "o", 0.01), 0.35)
    elif role in ("intro", "breakdown"):
        c.kick(b, 0, kit["kick"], 0.6)
        c.put("drums", b, 12, kit["snare"], 0.5)
        c.put("bass", b, 0, snd("sub", root, round(10 * c.step, 4)), 0.6)
    if role in ("full", "intro", "breakdown", "build"):
        for s in range(16):
            c.put("drums", b, s, kit["shaker"], 0.2)
        c.put("music", b, 0, snd("pad", tones, round(c.bar * 1.05, 4), True), 0.45)
        c.put("vocals", b, 0, snd("vpad", tones, round(c.bar * 1.05, 4), "a", 0.25), 0.3)
        for s in range(0, 16, 2):
            c.put("echo", b, s, snd("arp", tones[(s // 2) % len(tones)] + 12, round(c.step * 1.5, 4)), 0.25)
    if role == "build":
        build_drums(c, b, *build, kit, soft=True)


ARRANGERS = {"dnb": sty_dnb, "electro": sty_electro, "funk": sty_funk,
             "indie": sty_indie, "blues": sty_blues, "liquid": sty_liquid}


def plan_bars(n_bars: int, bar: float, span: float, style: str, drop_bar: int | None,
              first: bool, build_out: bool, end_clean: bool) -> tuple[list[str], set, set, int | None]:
    """Decide what each bar does: intro / build / full / breakdown / end.
    Returns (roles, crash bars, fill bars, final-hit bar)."""
    roles = ["full"] * n_bars
    crash: set[int] = set()
    if drop_bar is not None:
        for b in range(drop_bar):
            roles[b] = "build" if b >= drop_bar - 4 else "intro"
        f0 = drop_bar
    elif first:
        f0 = INTRO_BARS[style]
        for b in range(f0):
            roles[b] = "build" if (f0 >= 2 and b == f0 - 1) else "intro"
    else:
        f0 = 0
    crash.add(f0)
    # Long sections: every 32 bars, 4 bars of breakdown then 4 of build, then crash back in.
    if n_bars - f0 >= 40:
        for b in range(f0, n_bars):
            k = (b - f0) % 32
            if 24 <= k < 28:
                roles[b] = "breakdown"
            elif k >= 28:
                roles[b] = "build"
            elif k == 0:
                crash.add(b)
    fills = {b for b in range(f0, n_bars) if roles[b] == "full" and (b - f0) % 8 == 7}
    # Build into whatever comes next.
    if build_out:
        nb = 2 if span >= 8 * bar else 1 if span >= 3 * bar else 0
        for b in range(n_bars):
            if span - nb * bar - 1e-6 <= b * bar < span and nb:
                roles[b] = "build"
    final = None
    if end_clean and span > 2 * bar:
        final = int(np.floor((span - 1.0) / bar))
        for b in range(final, n_bars):
            roles[b] = "end"
        crash.discard(final)
    return roles, crash, fills - {b for b in range(n_bars) if roles[b] != "full"}, final


def render_section(style: str, duration: float, seed: int = 0, drop_at: float | None = None,
                   first: bool = False, build_out: bool = False, end_clean: bool = False,
                   transpose: int = 0) -> dict:
    """Return {stem: stereo float32 array (n, 2)} for one section."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / MCFG["styles"][style]["bpm"]
    bar = beat * 4
    drop_bar, offset = None, 0.0
    if drop_at is not None and drop_at >= bar:
        drop_bar = int(np.ceil(drop_at / bar - 1e-6))
        offset = drop_bar * bar - drop_at   # shift the grid so that bar lands exactly on drop_at
    span = duration + offset
    n_bars = int(np.ceil(span / bar)) + 1
    n = int((n_bars * bar + 3) * SR)
    c = Ctx(style, n, rng)
    kit = make_kit(rng)
    prog = [[m + transpose for m in ch] for ch in PROGRESSIONS[style]]
    roles, crash, fills, final = plan_bars(n_bars, bar, span, style, drop_bar, first, build_out, end_clean)

    # Number the bars inside each build so the snare roll can accelerate.
    build_pos: dict[int, tuple[int, int]] = {}
    runs, cur = [], []
    for b, r in enumerate(roles):
        if r == "build":
            cur.append(b)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    for run in runs:
        for j, b in enumerate(run):
            build_pos[b] = (j, len(run))
        t0, t1 = run[0] * bar, min((run[-1] + 1) * bar, span)
        if t1 - t0 > 0.2:
            g = 0.5 if style in HIGH_ENERGY or style == "funk" else 0.3
            place(c.bus["music"], int(t0 * SR), (riser(t1 - t0, rng) * g).astype(np.float32))

    arranger = ARRANGERS[style]
    for b in range(n_bars):
        if roles[b] != "end":
            arranger(c, b, roles[b], prog[b % len(prog)], b in fills, kit, build_pos.get(b, (0, 1)))
    for b in crash:
        if b < n_bars and roles[b] != "end":
            c.put("drums", b, 0, kit["crash"], 0.5)
            if style in HIGH_ENERGY:
                c.put("vocals", b, 0, snd("hey", 200.0), 0.5)
            if style == "electro":
                c.put("music", b, 0, snd("siren", round(bar, 4)), 0.6)

    if final is not None:
        # Clean ending: everything stops on the downbeat and one big chord rings out.
        s0 = int(final * bar * SR)
        fade = int(0.015 * SR)
        for k, x in c.bus.items():
            x[s0:s0 + fade] *= np.linspace(1, 0, len(x[s0:s0 + fade]))
            x[s0 + fade:] = 0
        ch = prog[final % len(prog)]
        place(c.bus["drums"], s0, kit["kick"] + 0)
        place(c.bus["drums"], s0, kit["crash"] * 0.6)
        place(c.bus["bass"], s0, sub(ch[0], 2.5, ring=True))
        ring = guitar(ch[0] + 12, 2.5, 2) if style == "blues" else pad_chord(tuple(ch[1:]), 2.5, True, ring=True)
        place(c.bus["music"], s0, ring * 0.7)
        if style == "funk":
            place(c.bus["music"], s0, brass(tuple(ch[1:]), 0.6) * 0.6)

    if style == "electro":
        # Sidechain "pump": pads and bass duck on every kick (classic house feel).
        pump = np.ones(n, np.float32)
        for kt in c.kicks:
            s = int(kt * SR)
            m = min(n, s + int(0.25 * SR))
            if m > s:
                pump[s:m] *= 0.35 + 0.65 * (1 - env_exp(m - s, 0.12))
        c.bus["music"] *= pump
        c.bus["bass"] *= 0.7 + 0.3 * pump
    c.bus["music"] += echo(c.bus.pop("echo"), beat * 0.75).astype(np.float32)

    pocket = MCFG.get("voice_pocket_db", 0)
    if pocket:
        for k in ("music", "vocals"):
            c.bus[k] = peaking(c.bus[k], 2500, pocket, 0.5).astype(np.float32)

    gains = {"drums": 1.0, "bass": 0.8, "music": 0.6, "vocals": 0.7}
    s0, s1 = int(offset * SR), int(offset * SR) + int(duration * SR)
    out = {}
    for k in STEMS:
        x = c.bus[k][s0:s1] * gains[k]
        out[k] = widen(x, k)
    return out


def widen(x: np.ndarray, stem: str) -> np.ndarray:
    """Mono -> stereo. Drums and bass stay centred; music and vocals get a
    little width from two short, different delays (still sounds fine in mono)."""
    if stem in ("drums", "bass"):
        return np.stack([x, x], axis=1).astype(np.float32)
    def delayed(ms):
        d = int(ms * SR / 1000)
        y = np.zeros_like(x)
        y[d:] = x[:-d]
        return y
    return np.stack([0.7 * x + 0.3 * delayed(7), 0.7 * x + 0.3 * delayed(13)], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------
# Whole beds
# ----------------------------------------------------------------------------

def normalise_sections(sections: list[dict]) -> list[dict]:
    """Accept old 'mood' names, and split anything longer than max_track_s
    into separate "tracks" in different keys."""
    max_s = MCFG.get("max_track_s", 180)
    out = []
    for sec in sections:
        style = sec.get("style") or LEGACY.get(sec.get("mood", ""), "funk")
        if style not in STYLES:
            raise ValueError(f"unknown music style '{style}' (choose from {', '.join(STYLES)})")
        d = float(sec["duration"])
        k = max(1, int(np.ceil(d / max_s)))
        for i in range(k):
            part = {**sec, "style": style, "duration": d / k, "track": i}
            if i > 0:
                part.pop("drop_at", None)
            out.append(part)
    return out


def render_bed(sections: list[dict], out_path: Path, stems_prefix: Path | None = None,
               end_clean: bool = True) -> list[dict]:
    """sections = [{"style": "funk", "duration": 30, "drop_at": 12 (optional)}, ...]
    Writes the mix to out_path (and stems if stems_prefix is given).
    Returns the section list actually rendered (after splitting long ones)."""
    secs = normalise_sections(sections)
    if not secs:
        raise ValueError("no sections")
    parts = {k: [] for k in STEMS}
    fade = int(0.02 * SR)
    for i, sec in enumerate(secs):
        st = render_section(sec["style"], sec["duration"], seed=i, drop_at=sec.get("drop_at"),
                            first=(i == 0), build_out=(i < len(secs) - 1),
                            end_clean=end_clean and i == len(secs) - 1,
                            transpose=TRANSPOSE_BY_TRACK[sec["track"] % len(TRANSPOSE_BY_TRACK)])
        for k in STEMS:
            x = st[k]
            if i < len(secs) - 1 and len(x) > fade:   # tiny fade so the cut between sections doesn't click
                x[-fade:] *= np.linspace(1, 0, fade)[:, None]
            parts[k].append(x)
    stems = {k: np.concatenate(v) for k, v in parts.items()}
    mix = sum(stems.values())
    if not end_clean:
        fo = min(len(mix), int(2.0 * SR))
        ramp = np.linspace(1, 0, fo)[:, None]
        for x in (mix, *stems.values()):
            x[-fo:] *= ramp
    gain = 10 ** (-3 / 20) / (np.abs(mix).max() or 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mix * gain, SR, subtype="PCM_16")
    if stems_prefix is not None:
        stems_prefix.parent.mkdir(parents=True, exist_ok=True)
        for k, x in stems.items():
            sf.write(str(stems_prefix.parent / f"{stems_prefix.name}_{k}.wav"), x * gain, SR, subtype="PCM_16")
    return [{"style": s["style"], "duration": round(s["duration"], 2), "bpm": MCFG["styles"][s["style"]]["bpm"],
             **({"drop_at": round(s["drop_at"], 2)} if s.get("drop_at") is not None else {})} for s in secs]


def build_library(seconds: float = 45) -> None:
    """A demo bed of every style in music/, plus a full-arc demo, with a catalog."""
    music = path_for("music")
    catalog = []
    for style, info in MCFG["styles"].items():
        name = f"demo_{style}_{int(seconds)}s.wav"
        render_bed([{"style": style, "duration": seconds, "drop_at": 12}], music / name)
        catalog.append({"file": name, "style": style, "bpm": info["bpm"], "duration_s": seconds,
                        "description": info["use"], "generator": "tools/make_music.py (numpy synthesis, original)"})
        log(f"saved music/{name}")
    arc = [{"style": "blues", "duration": 10}, {"style": "funk", "duration": 40},
           {"style": "dnb", "duration": 30}, {"style": "indie", "duration": 30}, {"style": "liquid", "duration": 30}]
    render_bed(arc, music / "demo_arc_140s.wav", stems_prefix=music / "demo_arc_140s")
    catalog.append({"file": "demo_arc_140s.wav", "style": "arc", "duration_s": 140,
                    "description": "The brief's long-video arc: blues sting -> funk -> dnb -> indie -> liquid outro. "
                                   "Stems alongside (demo_arc_140s_drums/bass/music/vocals.wav).",
                    "generator": "tools/make_music.py (numpy synthesis, original)"})
    log("saved music/demo_arc_140s.wav (+ stems)")
    write_json(music / "catalog.json", catalog)


def parse_sections(args: list[str]) -> list[dict]:
    secs = []
    for a in args:
        style, _, rest = a.partition(":")
        dur, _, drop = rest.partition("@")
        sec = {"style": style, "duration": float(dur)}
        if drop:
            sec["drop_at"] = float(drop)
        secs.append(sec)
    return secs


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--library":
        build_library()
    elif len(sys.argv) >= 3:
        done = render_bed(parse_sections(sys.argv[2:]), Path(sys.argv[1]))
        print("wrote", sys.argv[1], done)
    else:
        raise SystemExit("usage: make_music.py out.wav funk:30 dnb:20@8 ...   or   make_music.py --library")
