"""Image and video generation through stable-diffusion.cpp (sd-cli.exe, Vulkan build).

Each function builds one sd-cli command line. Weights are loaded per job and freed when the
process exits, so only one model ever occupies the 8 GB card; sd.cpp's --auto-fit (on by
default) spills what doesn't fit into system RAM.
"""

import random
from pathlib import Path

from PIL import Image, ImageFilter

from .. import config, models
from .common import Context, exe, run, webm_to_mp4

# Wan's recommended negative prompt (it was trained with Chinese negatives).
WAN_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

ASPECTS = {"1:1": (1, 1), "16:9": (16, 9), "9:16": (9, 16), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3)}


def seed_or_random(seed: int | None) -> int:
    return random.randint(0, 2**31 - 1) if seed is None or seed < 0 else int(seed)


def size_for(aspect: str, megapixels: float, multiple: int = 32) -> tuple[int, int]:
    """Width/height for an aspect ratio at roughly the given pixel count, snapped to `multiple`."""
    aw, ah = ASPECTS[aspect]
    h = (megapixels * 1e6 * ah / aw) ** 0.5
    w = h * aw / ah
    return snap(w, multiple), snap(h, multiple)


def size_like(image: Path, megapixels: float, multiple: int = 32) -> tuple[int, int]:
    """Keep an input image's aspect ratio, scaled to about `megapixels`."""
    with Image.open(image) as im:
        w, h = im.size
    scale = (megapixels * 1e6 / (w * h)) ** 0.5
    return snap(w * scale, multiple), snap(h * scale, multiple)


def snap(v: float, multiple: int) -> int:
    return max(multiple, int(round(v / multiple)) * multiple)


def _sd(ctx: Context, args: list, out: Path, label: str, span=(0.0, 1.0)) -> Path:
    cmd = [exe("sdcpp", "sd-cli.exe"), *args, "-t", config.CPU_THREADS, "-o", out]
    run(cmd, ctx, label=label, span=span)
    if not out.exists():
        raise RuntimeError(f"sd-cli finished but wrote no {out.name}")
    return out


def _common(prompt: str, negative: str, w: int, h: int, steps: int, cfg: float, seed: int) -> list:
    args = ["-p", prompt, "-W", w, "-H", h, "--steps", steps, "--cfg-scale", cfg, "-s", seed, "--sampling-method", "euler"]
    if negative:
        args += ["-n", negative]
    return args


# ---------------------------------------------------------------- text to image


def qwen_image(ctx, prompt, negative="", aspect="16:9", steps=20, cfg=4.0, seed=None, turbo=False) -> dict:
    seed = seed_or_random(seed)
    w, h = size_for(aspect, 1.0)
    args = [
        "--diffusion-model", models.path("qwen-image-turbo" if turbo else "qwen-image"),
        "--vae", models.path("qwen-image-vae"),
        "--llm", models.path("qwen3-vl-8b"),
        "--offload-to-cpu",
    ]
    if turbo:
        # Turbo uses a fixed 8-step schedule and no CFG.
        args += _common(prompt, "", w, h, 8, 1.0, seed) + ["--sigmas", "qwen-image-2.1-turbo"]
    else:
        args += _common(prompt, negative, w, h, steps, cfg, seed)
    out = _sd(ctx, args, ctx.workdir / "image.png", "generating")
    return {"files": [out.name], "seed": seed, "width": w, "height": h}


def flux_krea(ctx, prompt, aspect="1:1", megapixels=0.6, steps=24, guidance=4.5, seed=None) -> dict:
    seed = seed_or_random(seed)
    w, h = size_for(aspect, megapixels, 16)
    args = _flux_parts("flux-krea") + _common(prompt, "", w, h, steps, 1.0, seed) + ["--guidance", guidance]
    out = _sd(ctx, args, ctx.workdir / "image.png", "generating")
    return {"files": [out.name], "seed": seed, "width": w, "height": h}


def z_image(ctx, prompt, aspect="1:1", megapixels=1.0, steps=8, seed=None) -> dict:
    seed = seed_or_random(seed)
    w, h = size_for(aspect, megapixels, 16)
    args = [
        "--diffusion-model", models.path("z-image-turbo"),
        "--vae", models.path("flux-vae"),
        "--llm", models.path("qwen3-4b"),
        *_common(prompt, "", w, h, steps, 1.0, seed),
    ]
    out = _sd(ctx, args, ctx.workdir / "image.png", "generating")
    return {"files": [out.name], "seed": seed, "width": w, "height": h}


def sdxl(ctx, prompt, negative="", aspect="1:1", steps=30, cfg=7.0, seed=None) -> dict:
    seed = seed_or_random(seed)
    w, h = size_for(aspect, 1.0, 64)
    args = ["-m", models.path("sdxl"), "--vae", models.path("sdxl-vae"), *_common(prompt, negative, w, h, steps, cfg, seed)]
    out = _sd(ctx, args, ctx.workdir / "image.png", "generating")
    return {"files": [out.name], "seed": seed, "width": w, "height": h}


# ---------------------------------------------------------------- editing


def _flux_parts(model_key: str) -> list:
    return [
        "--diffusion-model", models.path(model_key),
        "--vae", models.path("flux-vae"),
        "--clip_l", models.path("clip-l"),
        "--t5xxl", models.path("t5xxl"),
        "--offload-to-cpu",
    ]


def qwen_edit(ctx, image: Path, prompt, steps=20, cfg=4.0, seed=None, turbo=False, megapixels=1.0) -> dict:
    seed = seed_or_random(seed)
    w, h = size_like(image, megapixels, 32)
    args = [
        "--diffusion-model", models.path("qwen-image-turbo" if turbo else "qwen-image"),
        "--vae", models.path("qwen-image-vae"),
        "--llm", models.path("qwen3-vl-8b"),
        "--llm_vision", models.path("qwen3-vl-8b-vision"),
        "--offload-to-cpu",
        "-r", image,
    ]
    if turbo:
        args += _common(prompt, "", w, h, 8, 1.0, seed) + ["--sigmas", "qwen-image-2.1-turbo"]
    else:
        args += _common(prompt, "", w, h, steps, cfg, seed)
    out = _sd(ctx, args, ctx.workdir / "edited.png", "editing")
    return {"files": [out.name], "seed": seed, "width": w, "height": h}


# Photo-Mate adapters (same LoRAs the Photo-Mate-i2i Space loads), each with the trigger
# prompt from its model card. "{what}" is filled from the user's prompt where it makes sense.
PHOTO_MATE = {
    "restore": (
        "lora-photo-restore",
        "[photo content], restore and enhance the image by repairing any damage, scratches, or fading. "
        "Colorize the photo naturally while preserving authentic textures and details, maintaining a "
        "realistic and historically accurate look.",
    ),
    "unblur-upscale": (
        "lora-unblur-upscale",
        "[photo content], upscale the low-quality image to 4K resolution, enhancing sharpness, clarity, and "
        "fine details while preserving the original texture, colors, lighting, and natural appearance. Remove "
        "noise, blur, and compression artifacts without over-smoothing or distorting facial or object features.",
    ),
    "remove-watermark": (
        "lora-watermark-remover",
        "[photo content], remove any watermark text or logos from the image while preserving the background, "
        "texture, lighting, and overall realism. Ensure the edited areas blend seamlessly with surrounding "
        "details, leaving no visible traces of watermark removal.",
    ),
    "cleanse": (
        "lora-photo-cleanser",
        "[photo content], remove {what} from the image while preserving the background and remaining "
        "elements, maintaining realism and original details.",
    ),
}


def kontext(ctx, image: Path, prompt, steps=20, guidance=2.5, seed=None, adapter: str = "", megapixels=0.7) -> dict:
    seed = seed_or_random(seed)
    w, h = size_like(image, megapixels, 16)
    args = _flux_parts("flux-kontext") + ["-r", image]
    if adapter:
        key, template = PHOTO_MATE[adapter]
        lora = models.path(key)
        if "{what}" in template:
            # For object removal the user names the object: "the floor drain", "all people".
            prompt = template.format(what=prompt or "the specified object(s)")
        elif not prompt:
            prompt = template
        prompt += f"<lora:{lora.stem}:1>"
        args += ["--lora-model-dir", lora.parent]
    args += _common(prompt, "", w, h, steps, 1.0, seed) + ["--guidance", guidance]
    out = _sd(ctx, args, ctx.workdir / "edited.png", "editing")
    return {"files": [out.name], "seed": seed, "width": w, "height": h}


def outpaint(ctx, image: Path, aspect="9:16", align="middle", prompt="", steps=30, cfg=6.0, seed=None, megapixels=0.9) -> dict:
    """Extend an image to a new aspect ratio without touching the original pixels.

    The source is placed on a larger canvas, only the new border is generated (SDXL
    inpainting, quantized), and the original is pasted back on top so the product in the
    photo is never redrawn. A short feathered overlap blends the seam.
    """
    seed = seed_or_random(seed)
    cw, ch = size_for(aspect, megapixels, 64)
    src = Image.open(image).convert("RGB")
    aw, ah = ASPECTS[aspect]
    if abs((src.width / src.height) / (aw / ah) - 1) < 0.04:
        raise ValueError("The image already has this aspect ratio; nothing to extend.")
    scale = min(cw / src.width, ch / src.height)
    sw, sh = max(1, int(src.width * scale)), max(1, int(src.height * scale))
    placed = src.resize((sw, sh), Image.LANCZOS)

    x = {"left": 0, "right": cw - sw}.get(align, (cw - sw) // 2)
    y = {"top": 0, "bottom": ch - sh}.get(align, (ch - sh) // 2)

    canvas = Image.new("RGB", (cw, ch), (127, 127, 127))
    # Seed the empty area with stretched edge colours so the model starts from something plausible.
    canvas.paste(src.resize((cw, ch), Image.BILINEAR).filter(ImageFilter.GaussianBlur(40)))
    canvas.paste(placed, (x, y))

    # White = generate. The model also repaints a thin strip inside the original's edge
    # so the seam blends; everything inside that strip is restored afterwards.
    overlap = max(8, int(min(sw, sh) * 0.04))
    mask = Image.new("L", (cw, ch), 255)
    mask.paste(Image.new("L", (max(1, sw - 2 * overlap), max(1, sh - 2 * overlap)), 0), (x + overlap, y + overlap))

    canvas_path, mask_path = ctx.workdir / "canvas.png", ctx.workdir / "mask.png"
    canvas.save(canvas_path)
    mask.save(mask_path)

    text = prompt or "seamless continuation of the scene, same lighting and style, photorealistic"
    args = [
        "-m", models.path("sdxl-inpaint"), "--vae", models.path("sdxl-vae"),
        "-i", canvas_path, "--mask", mask_path, "--strength", 1.0,
        *_common(text, "text, watermark, frame, border, blurry, distorted", cw, ch, steps, cfg, seed),
    ]
    raw = _sd(ctx, args, ctx.workdir / "outpaint_raw.png", "extending", span=(0.0, 0.97))

    # Paste the untouched original back, feathered over the overlap strip.
    result = Image.open(raw).convert("RGB")
    alpha = Image.new("L", (sw, sh), 0)
    alpha.paste(255, (overlap, overlap, sw - overlap, sh - overlap))
    alpha = alpha.filter(ImageFilter.GaussianBlur(overlap / 2))
    result.paste(placed, (x, y), alpha)
    out = ctx.workdir / "outpainted.png"
    result.save(out)
    return {"files": [out.name, mask_path.name], "seed": seed, "width": cw, "height": ch}


def upscale(ctx, image: Path, repeats: int = 1) -> dict:
    args = ["-M", "upscale", "--upscale-model", models.path("realesrgan-x4"), "-i", image, "--upscale-repeats", repeats]
    out = _sd(ctx, args, ctx.workdir / "upscaled.png", "upscaling")
    with Image.open(out) as im:
        w, h = im.size
    return {"files": [out.name], "width": w, "height": h}


# ---------------------------------------------------------------- video


def _video(ctx, args: list, fps: int) -> dict:
    webm = ctx.workdir / "video.webm"
    _sd(ctx, ["-M", "vid_gen", *args, "--fps", fps], webm, "animating", span=(0.0, 0.97))
    mp4 = webm_to_mp4(webm, ctx.workdir / "video.mp4", ctx)
    return {"files": [mp4.name, webm.name]}


def _video_size(image: Path | None, long_side: int) -> tuple[int, int]:
    if image is None:
        return long_side, snap(long_side * 9 / 16, 16)
    with Image.open(image) as im:
        w, h = im.size
    s = long_side / max(w, h)
    return snap(w * s, 16), snap(h * s, 16)


def wan_i2v(ctx, image: Path, prompt, end_image: Path | None = None, frames=33, long_side=832, steps=4, seed=None) -> dict:
    """Wan2.2 I2V A14B, 4-step distilled. With end_image it morphs first frame -> last frame."""
    seed = seed_or_random(seed)
    w, h = _video_size(image, long_side)
    half = max(1, steps // 2)
    args = [
        "--diffusion-model", models.path("wan22-i2v-low"),
        "--high-noise-diffusion-model", models.path("wan22-i2v-high"),
        "--vae", models.path("wan21-vae"),
        "--t5xxl", models.path("umt5-xxl"),
        "--offload-to-cpu", "--vae-tiling",
        "-p", prompt, "-n", WAN_NEGATIVE,
        "--init-img", image,
        "--cfg-scale", 1.0, "--sampling-method", "euler",
        "--steps", half, "--high-noise-steps", steps - half,
        "--flow-shift", 5.0, "-W", w, "-H", h, "--video-frames", frames, "-s", seed,
    ]
    if end_image is not None:
        args += ["--end-img", end_image]
    return {**_video(ctx, args, 16), "seed": seed, "width": w, "height": h, "frames": frames}


def wan_5b_i2v(ctx, image: Path, prompt, frames=49, long_side=832, steps=20, cfg=5.0, seed=None) -> dict:
    """Wan2.2 TI2V 5B: the quicker image-to-video option."""
    seed = seed_or_random(seed)
    w, h = _video_size(image, long_side)
    args = [
        "--diffusion-model", models.path("wan22-ti2v-5b"),
        "--vae", models.path("wan22-vae"),
        "--t5xxl", models.path("umt5-xxl"),
        "--offload-to-cpu", "--vae-tiling",
        "-p", prompt, "-n", WAN_NEGATIVE, "-i", image,
        "--cfg-scale", cfg, "--sampling-method", "euler", "--steps", steps,
        "--flow-shift", 3.0, "-W", w, "-H", h, "--video-frames", frames, "-s", seed,
    ]
    return {**_video(ctx, args, 24), "seed": seed, "width": w, "height": h, "frames": frames}


def wan_t2v(ctx, prompt, frames=33, aspect="16:9", steps=20, cfg=6.0, seed=None) -> dict:
    """Wan2.1 T2V 1.3B: text to video, small enough to sit entirely in VRAM."""
    seed = seed_or_random(seed)
    w, h = (832, 480) if aspect == "16:9" else (480, 832)
    args = [
        "--diffusion-model", models.path("wan21-t2v-1.3b"),
        "--vae", models.path("wan21-vae"),
        "--t5xxl", models.path("umt5-xxl"),
        "-p", prompt, "-n", WAN_NEGATIVE,
        "--cfg-scale", cfg, "--sampling-method", "euler", "--steps", steps,
        "--flow-shift", 3.0, "-W", w, "-H", h, "--video-frames", frames, "-s", seed,
    ]
    return {**_video(ctx, args, 16), "seed": seed, "width": w, "height": h, "frames": frames}
