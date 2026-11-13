"""
common.py - shared helpers used by every pipeline script.

Plain-language summary:
  * Finds the project folders (inbox, work, output...) from config.json.
  * Finds ffmpeg / ffprobe even if the shell PATH hasn't been refreshed.
  * Makes the NVIDIA CUDA libraries visible to faster-whisper.
  * Runs ffmpeg commands and raises a clear error if they fail.
  * Reads and writes JSON files.
  * Appends lines to output/<ride-id>/run_log.md so every run is recorded.

Every other script does `from common import *` style imports of what it needs.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Project root and config
# --------------------------------------------------------------------------

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent

# You can point the whole pipeline at a different folder tree with the
# EUC_ROOT environment variable. selftest.py uses this so it never touches
# the real inbox.
ROOT = Path(os.environ.get("EUC_ROOT", PROJECT_ROOT)).resolve()


def load_config() -> dict:
    """Read config.json from the project root (always the real one, not EUC_ROOT)."""
    cfg_path = PROJECT_ROOT / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()


def path_for(key: str) -> Path:
    """Return the absolute folder for 'inbox', 'work', etc., creating it if needed."""
    p = Path(CONFIG["paths"][key])
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def work_dir(ride_id: str) -> Path:
    d = path_for("work") / ride_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(ride_id: str) -> Path:
    d = path_for("output") / ride_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------

def read_json(path: Path | str, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    """Print a timestamped line to the console."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def run_log(ride_id: str, msg: str) -> None:
    """Print AND append to output/<ride-id>/run_log.md."""
    log(f"{ride_id}: {msg}")
    path = output_dir(ride_id) / "run_log.md"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- `{stamp}` {msg}\n")


def run_log_header(ride_id: str, title: str) -> None:
    path = output_dir(ride_id) / "run_log.md"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n")


# --------------------------------------------------------------------------
# ffmpeg / ffprobe discovery
# --------------------------------------------------------------------------

def _find_exe(name: str) -> str:
    """Find ffmpeg or ffprobe. Checks PATH, then the winget install folder."""
    found = shutil.which(name)
    if found:
        return found
    # winget puts Gyan.FFmpeg here and adds it to PATH, but a shell opened
    # before the install won't see it yet.
    pattern = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "ffmpeg-*", "bin", name + ".exe",
    )
    hits = sorted(glob.glob(pattern))
    if hits:
        return hits[-1]
    raise FileNotFoundError(
        f"{name} not found. Install with: winget install --id Gyan.FFmpeg -e --source winget"
    )


FFMPEG = _find_exe("ffmpeg")
FFPROBE = _find_exe("ffprobe")


def run(cmd: list[str], quiet: bool = True, check: bool = True, cwd=None) -> subprocess.CompletedProcess:
    """Run a command. On failure, show the last part of its output so the error is readable."""
    cmd = [str(c) for c in cmd]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=cwd)
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-3000:]
        raise RuntimeError(f"Command failed ({proc.returncode}):\n  {' '.join(cmd)}\n{tail}")
    if not quiet and proc.stdout:
        print(proc.stdout)
    return proc


def ffmpeg(args: list[str], cwd=None) -> None:
    """Run ffmpeg with sensible defaults (overwrite output, hide banner, errors only)."""
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], cwd=cwd)


def ffprobe_json(path: Path | str) -> dict:
    """Return ffprobe's full description of a media file as a dictionary."""
    proc = run([FFPROBE, "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path)])
    return json.loads(proc.stdout)


def media_info(path: Path | str) -> dict:
    """Simplified info: duration, fps, width, height, creation time, audio present, codec."""
    info = ffprobe_json(path)
    fmt = info.get("format", {})
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = None
    if video and video.get("r_frame_rate"):
        num, _, den = video["r_frame_rate"].partition("/")
        try:
            fps = float(num) / float(den or 1)
        except ZeroDivisionError:
            fps = None
    creation = fmt.get("tags", {}).get("creation_time") or (video or {}).get("tags", {}).get("creation_time")
    return {
        "duration": float(fmt.get("duration", 0) or 0),
        "fps": fps,
        "width": int(video["width"]) if video else None,
        "height": int(video["height"]) if video else None,
        "video_codec": video.get("codec_name") if video else None,
        "has_audio": audio is not None,
        "audio_channels": int(audio["channels"]) if audio else 0,
        "creation_time": creation,
        "size_bytes": int(fmt.get("size", 0) or 0),
        "streams": info.get("streams", []),
    }


def cuda_decode_args() -> list[str]:
    """ffmpeg input flags for GPU decoding, if enabled in config."""
    if CONFIG["render"].get("use_cuda_decode", True):
        return ["-hwaccel", "cuda"]
    return []


# --------------------------------------------------------------------------
# CUDA libraries for faster-whisper (ctranslate2)
# --------------------------------------------------------------------------

def enable_cuda_libs() -> None:
    """
    faster-whisper needs NVIDIA's cuBLAS and cuDNN DLLs. We installed them as
    pip packages, so they live inside the venv. Windows won't find them unless
    we add their folders to the DLL search path first.
    """
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nv = Path(sp) / "nvidia"
        if not nv.is_dir():
            continue
        for sub in nv.iterdir():
            b = sub / "bin"
            if b.is_dir():
                try:
                    os.add_dll_directory(str(b))
                except (AttributeError, OSError):
                    pass
                os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    """Seconds -> 'MM:SS' or 'H:MM:SS'."""
    seconds = max(0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


def srt_time(seconds: float) -> str:
    """Seconds -> SRT timestamp 'HH:MM:SS,mmm'."""
    seconds = max(0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d},{ms:03d}"


def free_space_gb(path: Path | str = ROOT) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


class Timer:
    """with Timer() as t: ...; t.elapsed -> seconds."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.start


def list_rides() -> list[str]:
    """Ride IDs that have a manifest in work/."""
    rides = []
    for d in sorted(path_for("work").iterdir()):
        if d.is_dir() and (d / "manifest.json").exists() and not d.name.startswith("_"):
            rides.append(d.name)
    return rides
