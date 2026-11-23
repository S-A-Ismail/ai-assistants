"""Background removal with BiRefNet (the model behind not-lain/background-removal), ONNX + DirectML."""

from pathlib import Path

import numpy as np
from PIL import Image

from .. import models
from .common import Context
from .voice import _lock, _session, _sessions

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 1024


def remove_background(ctx: Context, image: Path) -> dict:
    src = Image.open(image).convert("RGB")
    x = np.asarray(src.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]

    with _lock:
        ctx.progress(0.1, "loading BiRefNet")
        if "birefnet" not in _sessions:
            _sessions["birefnet"] = _session(models.path("birefnet"))
        sess = _sessions["birefnet"]
        ctx.progress(0.3, "segmenting")
        # The last output is the full-resolution prediction (logits).
        pred = sess.run(None, {sess.get_inputs()[0].name: x})[-1]

    m = 1 / (1 + np.exp(-pred[0, 0]))
    mask = Image.fromarray((m * 255).clip(0, 255).astype(np.uint8)).resize(src.size, Image.BILINEAR)
    cutout = src.convert("RGBA")
    cutout.putalpha(mask)

    out, mask_out = ctx.workdir / "cutout.png", ctx.workdir / "mask.png"
    cutout.save(out)
    mask.save(mask_out)
    return {"files": [out.name, mask_out.name], "width": src.width, "height": src.height}
