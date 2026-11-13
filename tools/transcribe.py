"""
transcribe.py - Step 5. Turn Tyler's voice track into text with timestamps,
and find the spoken highlight cues.

Plain-language summary:
  * Uses faster-whisper (OpenAI's Whisper model, sped-up) with the large-v3
    model on the NVIDIA GPU. First run downloads the model (~3 GB) once.
  * Input is work/<ride>/audio/voice_16k.wav, so all times are on the ride
    timeline already.
  * Output:
      transcript.json  - segments and every word with start/end times
      transcript.srt   - same text as subtitles
      cues.json        - every time Tyler says a highlight phrase
                         ("clip that", "mark it", "watch this", ...)
  * Also builds a list of "speech islands" (stretches where he's talking,
    merged across pauses under 1 s). The edit planner uses these to keep the
    commentary and cut long silences.
  * If the GPU fails for any reason it falls back to CPU (much slower but works).

Run:  python tools/transcribe.py <ride-id>
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from common import (CONFIG, enable_cuda_libs, log, read_json, run_log, run_log_header,
                    srt_time, work_dir, write_json)


def load_model():
    cfg = CONFIG["transcribe"]
    enable_cuda_libs()
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel(cfg["model"], device=cfg["device"], compute_type=cfg["compute_type"])
        return model, f"{cfg['model']} on {cfg['device']} ({cfg['compute_type']})"
    except Exception as e:  # CUDA missing or out of memory -> CPU fallback
        log(f"WARNING: GPU load failed ({e}); falling back to CPU int8. This will be slow.")
        model = WhisperModel(cfg["model"], device="cpu", compute_type="int8")
        return model, f"{cfg['model']} on cpu (int8) - GPU fallback"


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()


def find_cues(words: list[dict], phrases: list[str]) -> list[dict]:
    """Slide each cue phrase over the word list and record where it appears."""
    norm_words = [normalise(w["word"]) for w in words]
    cues = []
    for phrase in phrases:
        target = normalise(phrase).split()
        k = len(target)
        for i in range(len(norm_words) - k + 1):
            if norm_words[i:i + k] == target:
                start = words[i]["start"]
                ctx_lo, ctx_hi = max(0, i - 8), min(len(words), i + k + 8)
                cues.append({
                    "cue": phrase,
                    "time": round(start, 2),
                    "end": round(words[i + k - 1]["end"], 2),
                    "context": " ".join(w["word"].strip() for w in words[ctx_lo:ctx_hi]),
                })
    cues.sort(key=lambda c: c["time"])
    # Drop duplicates within 2 seconds (e.g. "oh shit" said twice).
    deduped = []
    for c in cues:
        if deduped and c["time"] - deduped[-1]["time"] < 2 and c["cue"] == deduped[-1]["cue"]:
            continue
        deduped.append(c)
    return deduped


def speech_islands(segments: list[dict], max_gap: float = 1.0) -> list[dict]:
    islands = []
    for s in segments:
        if islands and s["start"] - islands[-1]["end"] <= max_gap:
            islands[-1]["end"] = s["end"]
            islands[-1]["text"] += " " + s["text"].strip()
        else:
            islands.append({"start": s["start"], "end": s["end"], "text": s["text"].strip()})
    return islands


def write_srt(segments: list[dict], path: Path, offset: float = 0.0) -> None:
    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{srt_time(s['start'] + offset)} --> {srt_time(s['end'] + offset)}",
                  s["text"].strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(ride_id: str) -> dict:
    cfg = CONFIG["transcribe"]
    wd = work_dir(ride_id)
    audio = wd / "audio" / "voice_16k.wav"
    run_log_header(ride_id, "Transcription")
    if not audio.exists():
        run_log(ride_id, "no voice track - transcription skipped")
        empty = {"segments": [], "words": [], "speech_islands": [], "model": None}
        write_json(wd / "transcript.json", empty)
        write_json(wd / "cues.json", [])
        return empty

    model, desc = load_model()
    run_log(ride_id, f"model loaded: {desc}")
    t0 = time.time()
    # Read the WAV ourselves (soundfile) and pass samples in. faster-whisper's
    # built-in reader depends on PyAV, whose API changes between versions.
    import soundfile as sf
    samples, sr = sf.read(str(audio), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f"{audio.name} is {sr} Hz; expected 16000")
    seg_iter, info = model.transcribe(
        samples, language=cfg.get("language"), beam_size=5, word_timestamps=True,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    segments, words = [], []
    next_report = 60
    for s in seg_iter:
        seg_words = [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3),
                      "prob": round(w.probability, 3)} for w in (s.words or [])]
        segments.append({"start": round(s.start, 3), "end": round(s.end, 3),
                         "text": s.text.strip(), "words": seg_words})
        words.extend(seg_words)
        if s.end >= next_report:
            log(f"  ...transcribed {s.end/60:.1f} min of {info.duration/60:.1f}")
            next_report += 120

    cues = find_cues(words, cfg["highlight_cues"])
    islands = speech_islands(segments)
    result = {"model": desc, "language": info.language, "audio_duration": info.duration,
              "segments": segments, "words": words, "speech_islands": islands}
    write_json(wd / "transcript.json", result)
    write_json(wd / "cues.json", cues)
    write_srt(segments, wd / "transcript.srt")

    spoken = sum(i["end"] - i["start"] for i in islands)
    run_log(ride_id, f"{len(segments)} segments, {len(words)} words, {spoken/60:.1f} min of speech "
                     f"in {info.duration/60:.1f} min, took {time.time()-t0:.0f}s")
    run_log(ride_id, f"{len(cues)} highlight cue(s): " +
            (", ".join(f"'{c['cue']}' @ {c['time']:.0f}s" for c in cues) or "none"))
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: transcribe.py <ride-id>")
    main(sys.argv[1])
