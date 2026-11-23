"""Music: ACE-Step 1.5 via acestep.cpp's two CLI stages (ace-lm, then ace-synth).

acestep.cpp has no official release; put a Vulkan build of it in runtime/engines/acestep
(see README). Until then the music tool reports itself as unavailable.
"""

import json

from .. import models
from .common import Context, exe, run


def available() -> bool:
    try:
        exe("acestep", "ace-lm.exe"), exe("acestep", "ace-synth.exe")
        return True
    except Exception:
        return False


def generate(ctx: Context, caption: str, lyrics: str = "", duration: float = 30, seed: int = -1, instrumental: bool = False) -> dict:
    model_dir = models.path("ace-lm").parent
    req = {
        "caption": caption,
        # "[Instrumental]" is how ACE-Step is told "no vocals"; empty lets the LM write lyrics.
        "lyrics": "[Instrumental]" if instrumental else lyrics,
        "duration": float(duration),
        "seed": int(seed),
        "lm_model": models.path("ace-lm").stem,
        "synth_model": models.path("ace-dit").stem,
        "output_format": "mp3",
    }
    req_path = ctx.workdir / "request.json"
    req_path.write_text(json.dumps(req, ensure_ascii=False), encoding="utf-8")

    # Stage 1: the LM plans the song (metadata, lyrics, audio codes) -> request0.json
    run([exe("acestep", "ace-lm.exe"), "--models", model_dir, "--request", req_path], ctx, label="composing", span=(0.0, 0.4))
    planned = ctx.workdir / "request0.json"
    # Stage 2: DiT + VAE render it -> request00.mp3
    run([exe("acestep", "ace-synth.exe"), "--models", model_dir, "--request", planned], ctx, label="rendering", span=(0.4, 1.0))

    mp3 = ctx.workdir / "request00.mp3"
    final = ctx.workdir / "music.mp3"
    mp3.replace(final)
    plan = json.loads(planned.read_text(encoding="utf-8"))
    return {
        "files": [final.name, planned.name],
        "lyrics": plan.get("lyrics", ""),
        "bpm": plan.get("bpm"),
        "keyscale": plan.get("keyscale"),
        "duration": plan.get("duration"),
    }
