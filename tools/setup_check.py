"""
setup_check.py - Verify this PC has everything the pipeline needs.
Prints a PASS/FAIL line per item. Safe to run any time.

Run:  python tools/setup_check.py
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  - {detail}" if detail else ""))


def main() -> int:
    print("EUC Studio setup check\n")
    v = sys.version_info
    check("Python 3.10+", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro} at {sys.executable}")
    check("Running inside .venv", ".venv" in sys.executable, "" if ".venv" in sys.executable else
          "use .venv\\Scripts\\python.exe (or run_ride.bat)")

    for mod in ("numpy", "scipy", "soundfile", "PIL", "faster_whisper", "ctranslate2"):
        try:
            m = importlib.import_module(mod)
            check(f"package {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:
            check(f"package {mod}", False, str(e))

    try:
        import common
        check("ffmpeg found", True, common.FFMPEG)
        enc = subprocess.run([common.FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        check("NVENC encoders", "hevc_nvenc" in enc and "h264_nvenc" in enc)
        flt = subprocess.run([common.FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        for f in ("v360", "afftdn", "sidechaincompress", "loudnorm", "subtitles"):
            check(f"ffmpeg filter {f}", f" {f} " in flt)
    except Exception as e:
        check("ffmpeg found", False, str(e))
        common = None

    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        check("NVIDIA GPU", bool(out), out)
    else:
        check("NVIDIA GPU", False, "nvidia-smi not found")

    try:
        if common:
            common.enable_cuda_libs()
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        check("faster-whisper sees CUDA", n > 0, f"{n} device(s); types {sorted(ctranslate2.get_supported_compute_types('cuda')) if n else ''}")
    except Exception as e:
        check("faster-whisper sees CUDA", False, str(e))

    try:
        from faster_whisper import download_model
        p = download_model("large-v3", local_files_only=True)
        check("Whisper large-v3 downloaded", True, p)
    except Exception:
        check("Whisper large-v3 downloaded", False, "will download (~3 GB) on first transcription")

    if common:
        for key in ("inbox", "stitched", "work", "music", "output"):
            p = common.path_for(key)
            check(f"folder {key}/", p.is_dir(), str(p))
        free = common.free_space_gb()
        check("free disk >= 30 GB", free >= 30, f"{free:.1f} GB free")

    studio = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Insta360 Studio"
    check("Insta360 Studio installed", studio.is_dir(), str(studio))
    resolve = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Blackmagic Design" / "DaVinci Resolve"
    check("DaVinci Resolve installed (optional)", resolve.is_dir())

    fails = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(fails)} passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
