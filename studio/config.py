"""Paths and settings. Everything can be overridden with environment variables or .env."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# Model weights live on their own drive; code and engines live next to the repo.
MODELS_DIR = Path(os.getenv("STUDIO_MODELS_DIR", r"N:\Models"))
ENGINES_DIR = Path(os.getenv("STUDIO_ENGINES_DIR", str(ROOT / "runtime" / "engines")))
DATA_DIR = Path(os.getenv("STUDIO_DATA_DIR", str(ROOT / "data")))

HOST = os.getenv("STUDIO_HOST", "127.0.0.1")
PORT = int(os.getenv("STUDIO_PORT", "8000"))
# Optional. When set, /api and /mcp require "Authorization: Bearer <key>" or "X-API-Key: <key>".
API_KEY = os.getenv("STUDIO_API_KEY", "")

# Threads for CPU-bound engines (whisper.cpp, text encoders kept on the CPU).
CPU_THREADS = int(os.getenv("STUDIO_CPU_THREADS", str(os.cpu_count() or 4)))

# Where sd.cpp decodes images and video frames: "gpu" (default; tiled and throttled like sampling)
# or "cpu" (no GPU load at all, but a 544 px decode takes over a minute).
VAE_DEVICE = os.getenv("STUDIO_VAE_DEVICE", "gpu").lower()

# Conservative sd.cpp settings for the RX 580 (flash attention, direct conv and prefetch off).
SD_SAFE = os.getenv("STUDIO_SD_SAFE", "1") != "0"

# GPU rest between sd.cpp sampling steps, as a multiple of how long the step took: 1.0 rests as
# long as it worked (about 50% duty); 0 turns it off. The RX 580 froze the PC after 30-50 s of
# continuous full load, never in short bursts.
GPU_REST = float(os.getenv("STUDIO_GPU_REST", "1.0"))
# Seconds of GPU work to build up before a rest. Sampling steps are longer than this, so they
# rest after every step; short VAE tiles are grouped.
GPU_MIN_WORK = float(os.getenv("STUDIO_GPU_MIN_WORK", "4"))
# Vulkan device index for sd.cpp / llama.cpp. 0 is normally the RX 580 (the Intel iGPU has no Vulkan driver).
VULKAN_DEVICE = os.getenv("STUDIO_VULKAN_DEVICE", "")

# Auto-detect answers "hi" for most Urdu speech, which comes back in Devanagari.
# When on, an auto-detected "hi" is re-run as "ur" (a full second pass with whisper.cpp).
WHISPER_HINDI_AS_URDU = os.getenv("WHISPER_HINDI_AS_URDU", "1") != "0"

MAX_UPLOAD_MB = int(os.getenv("STUDIO_MAX_UPLOAD_MB", "500"))
