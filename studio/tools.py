"""The tool catalogue. The web page, the REST API and the MCP server are all built from this list.

A tool is one model doing one job. Its `fields` describe its inputs (the page renders a form
from them; the API validates against them) and `run` does the work inside a job.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import models
from .engines import birefnet, music, sdcpp, translate, voice, whisper
from .engines.common import Context

ASPECTS = list(sdcpp.ASPECTS)


@dataclass
class Field:
    name: str
    type: str  # text | textarea | image | audio | select | number | bool
    label: str
    default: Any = None
    required: bool = False
    options: list | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    help: str = ""


@dataclass
class Tool:
    id: str
    category: str  # image | edit | animate | voice | music | translate | transcribe
    title: str
    model: str
    summary: str
    license: str
    commercial: bool
    speed: str  # rough, on an RX 580; shown so nobody is surprised
    requires: list[str]
    fields: list[Field]
    run: Callable[[Context, dict], dict]
    engine: str = "sdcpp"
    extra_check: Callable[[], str | None] = field(default=lambda: None)

    def missing(self) -> list[str]:
        return models.missing(*self.requires)

    def status(self) -> str | None:
        """None when ready, otherwise a reason it can't run."""
        if m := self.missing():
            return f"waiting for model files: {', '.join(m)}"
        return self.extra_check()

    def describe(self) -> dict:
        return {
            "id": self.id, "category": self.category, "title": self.title, "model": self.model,
            "summary": self.summary, "license": self.license, "commercial": self.commercial,
            "speed": self.speed, "ready": self.status() is None, "status": self.status(),
            "fields": [f.__dict__ for f in self.fields],
        }


def _seed() -> Field:
    return Field("seed", "number", "Seed", -1, help="-1 picks a random seed; reuse a seed to reproduce a result")


def _prompt(label="Prompt", required=True, default="") -> Field:
    return Field("prompt", "textarea", label, default, required=required)


def _aspect(default="1:1") -> Field:
    return Field("aspect", "select", "Aspect ratio", default, options=ASPECTS)


# Language choices shown in forms (codes TranslateGemma accepts; any of its 500+ codes work via the API).
COMMON_LANGS = ["en", "ur", "hi", "pa", "ar", "fa", "ps", "bn", "zh", "es", "fr", "de", "it", "pt", "ru", "tr", "ja", "ko", "id", "ms"]

TOOLS: list[Tool] = [
    # ------------------------------------------------------------------ image
    Tool(
        "qwen-image", "image", "Qwen-Image 2.1", "Qwen-Image-2.1 · Q4_K GGUF",
        "Main image generator. Best at putting readable text (signs, labels, posters) into images.",
        "Apache-2.0", True, "~20-40 min",
        ["qwen-image", "qwen-image-vae", "qwen3-vl-8b"],
        [_prompt(), Field("negative", "text", "Negative prompt", "blurry, low quality, watermark, distorted text"),
         _aspect("16:9"), Field("steps", "number", "Steps", 20, min=8, max=50), Field("cfg", "number", "CFG", 4.0, min=1, max=10, step=0.5), _seed()],
        lambda c, a: sdcpp.qwen_image(c, a["prompt"], a.get("negative", ""), a["aspect"], int(a["steps"]), float(a["cfg"]), int(a["seed"])),
    ),
    Tool(
        "qwen-image-fast", "image", "Qwen-Image 2.1 Turbo", "Qwen-Image-2.1-Turbo · Q4_K_M GGUF",
        "Quicker drafts with the same text skill: a fixed 8-step schedule, no CFG.",
        "Apache-2.0", True, "~10-15 min",
        ["qwen-image-turbo", "qwen-image-vae", "qwen3-vl-8b"],
        [_prompt(), _aspect("16:9"), _seed()],
        lambda c, a: sdcpp.qwen_image(c, a["prompt"], aspect=a["aspect"], seed=int(a["seed"]), turbo=True),
    ),
    Tool(
        "flux-krea", "image", "FLUX.1 Krea", "FLUX.1-Krea-dev · Q4_K_S GGUF",
        "The most photo-realistic of the set: real-looking homes, people and interiors.",
        "FLUX.1-dev non-commercial", False, "~10-20 min",
        ["flux-krea", "flux-vae", "clip-l", "t5xxl"],
        [_prompt(), _aspect("1:1"), Field("megapixels", "number", "Megapixels", 0.6, min=0.3, max=1.0, step=0.1),
         Field("steps", "number", "Steps", 24, min=8, max=40), Field("guidance", "number", "Guidance", 4.5, min=1, max=8, step=0.5), _seed()],
        lambda c, a: sdcpp.flux_krea(c, a["prompt"], a["aspect"], float(a["megapixels"]), int(a["steps"]), float(a["guidance"]), int(a["seed"])),
    ),
    Tool(
        "z-image", "image", "Z-Image Turbo", "Z-Image-Turbo · Q4_K GGUF",
        "Fast, good-looking general images in 8 steps. Good for quick ideas.",
        "Apache-2.0", True, "~3-6 min",
        ["z-image-turbo", "flux-vae", "qwen3-4b"],
        [_prompt(), _aspect("1:1"), Field("megapixels", "number", "Megapixels", 1.0, min=0.3, max=1.5, step=0.1), _seed()],
        lambda c, a: sdcpp.z_image(c, a["prompt"], a["aspect"], float(a["megapixels"]), seed=int(a["seed"])),
    ),
    Tool(
        "sdxl", "image", "Stable Diffusion XL", "SDXL 1.0 base · Q8_0 GGUF",
        "Classic Stable Diffusion XL, quantized to 8-bit. Fastest of the full-size models.",
        "CreativeML Open RAIL++-M", True, "~2-5 min",
        ["sdxl", "sdxl-vae"],
        [_prompt(), Field("negative", "text", "Negative prompt", "blurry, low quality, watermark"),
         _aspect("1:1"), Field("steps", "number", "Steps", 30, min=10, max=60), Field("cfg", "number", "CFG", 7.0, min=1, max=12, step=0.5), _seed()],
        lambda c, a: sdcpp.sdxl(c, a["prompt"], a.get("negative", ""), a["aspect"], int(a["steps"]), float(a["cfg"]), int(a["seed"])),
    ),
    # ------------------------------------------------------------------ edit
    Tool(
        "qwen-edit", "edit", "Qwen-Image Edit", "Qwen-Image-2.1 + Qwen3-VL vision · Q4 GGUF",
        "Edit by instruction, e.g. \"remove the floor drain\" or \"change the label to 'Scala2'\". Commercial-safe alternative to Kontext.",
        "Apache-2.0", True, "~20-40 min (turbo ~10-15)",
        ["qwen-image", "qwen-image-turbo", "qwen-image-vae", "qwen3-vl-8b", "qwen3-vl-8b-vision"],
        [Field("image", "image", "Image", required=True), _prompt("Instruction"),
         Field("turbo", "bool", "Turbo (8 steps)", True), Field("steps", "number", "Steps (non-turbo)", 20, min=8, max=50), _seed()],
        lambda c, a: sdcpp.qwen_edit(c, a["image"], a["prompt"], int(a["steps"]), seed=int(a["seed"]), turbo=bool(a["turbo"])),
    ),
    Tool(
        "flux-kontext", "edit", "FLUX.1 Kontext", "FLUX.1-Kontext-dev · Q4_K_S GGUF",
        "Edits images by instruction, e.g. \"remove the logo\".",
        "FLUX.1-dev non-commercial", False, "~15-25 min",
        ["flux-kontext", "flux-vae", "clip-l", "t5xxl"],
        [Field("image", "image", "Image", required=True), _prompt("Instruction"),
         Field("steps", "number", "Steps", 20, min=8, max=40), Field("guidance", "number", "Guidance", 2.5, min=1, max=6, step=0.5), _seed()],
        lambda c, a: sdcpp.kontext(c, a["image"], a["prompt"], int(a["steps"]), float(a["guidance"]), int(a["seed"])),
    ),
    Tool(
        "photo-mate", "edit", "Photo-Mate", "FLUX.1-Kontext-dev + Photo-Mate LoRAs",
        "Restore, unblur/upscale, remove watermarks, or remove named objects, using the same adapters as the Photo-Mate-i2i Space.",
        "FLUX.1-dev non-commercial", False, "~20-30 min",
        ["flux-kontext", "flux-vae", "clip-l", "t5xxl", "lora-photo-restore", "lora-unblur-upscale", "lora-watermark-remover", "lora-photo-cleanser"],
        [Field("image", "image", "Image", required=True),
         Field("adapter", "select", "Adapter", "unblur-upscale", options=list(sdcpp.PHOTO_MATE)),
         _prompt("Prompt (optional; for 'cleanse' name what to remove)", required=False),
         Field("steps", "number", "Steps", 28, min=10, max=40), _seed()],
        lambda c, a: sdcpp.kontext(c, a["image"], a.get("prompt", ""), int(a["steps"]), 2.5, int(a["seed"]), adapter=a["adapter"]),
    ),
    Tool(
        "outpaint", "edit", "Outpaint", "SDXL 1.0 inpainting · Q8_0 GGUF",
        "Extends a photo to a new aspect ratio (e.g. landscape to 9:16) without cropping. The original pixels are pasted back untouched, so the product is never redrawn.",
        "CreativeML Open RAIL++-M", True, "~2-5 min",
        ["sdxl-inpaint", "sdxl-vae"],
        [Field("image", "image", "Image", required=True), _aspect("9:16"),
         Field("align", "select", "Place original", "middle", options=["middle", "top", "bottom", "left", "right"]),
         _prompt("Describe the surroundings (optional)", required=False),
         Field("steps", "number", "Steps", 30, min=10, max=60), _seed()],
        lambda c, a: sdcpp.outpaint(c, a["image"], a["aspect"], a["align"], a.get("prompt", ""), int(a["steps"]), seed=int(a["seed"])),
    ),
    Tool(
        "upscale", "edit", "Upscale 4x", "Real-ESRGAN x4plus",
        "Plain 4x upscale. Fast and faithful; use Photo-Mate 'unblur-upscale' when the source is blurry.",
        "BSD-3-Clause", True, "~1 min",
        ["realesrgan-x4"],
        [Field("image", "image", "Image", required=True), Field("repeats", "number", "Passes (each is 4x)", 1, min=1, max=2)],
        lambda c, a: sdcpp.upscale(c, a["image"], int(a["repeats"])),
    ),
    Tool(
        "remove-background", "edit", "Remove background", "BiRefNet · ONNX",
        "Clean product cutouts: transparent PNG plus the mask.",
        "MIT", True, "seconds",
        ["birefnet"],
        [Field("image", "image", "Image", required=True)],
        lambda c, a: birefnet.remove_background(c, a["image"]),
        engine="onnx",
    ),
    # ------------------------------------------------------------------ animate
    Tool(
        "animate", "animate", "Image to video (Wan2.2 14B)", "Wan2.2-I2V-A14B, 4-step distilled · Q4_K_S GGUF",
        "Turns a still image into a short clip. Highest quality; slow on this card.",
        "Apache-2.0", True, "~1 hour+",
        ["wan22-i2v-high", "wan22-i2v-low", "wan21-vae", "umt5-xxl"],
        [Field("image", "image", "Image", required=True), _prompt(default="make this image come alive, cinematic motion"),
         Field("frames", "number", "Frames (16 fps)", 33, min=17, max=81, step=16), Field("long_side", "number", "Long side (px)", 832, min=480, max=1280, step=16), _seed()],
        lambda c, a: sdcpp.wan_i2v(c, a["image"], a["prompt"], None, int(a["frames"]), int(a["long_side"]), seed=int(a["seed"])),
    ),
    Tool(
        "first-last-frame", "animate", "First/last frame video", "Wan2.2-I2V-A14B, 4-step distilled · Q4_K_S GGUF",
        "Video that moves from one image to another. Good for before/after shots.",
        "Apache-2.0", True, "~1 hour+",
        ["wan22-i2v-high", "wan22-i2v-low", "wan21-vae", "umt5-xxl"],
        [Field("image", "image", "First frame", required=True), Field("end_image", "image", "Last frame", required=True), _prompt(),
         Field("frames", "number", "Frames (16 fps)", 33, min=17, max=81, step=16), Field("long_side", "number", "Long side (px)", 832, min=480, max=1280, step=16), _seed()],
        lambda c, a: sdcpp.wan_i2v(c, a["image"], a["prompt"], a["end_image"], int(a["frames"]), int(a["long_side"]), seed=int(a["seed"])),
    ),
    Tool(
        "animate-fast", "animate", "Image to video (Wan2.2 5B)", "Wan2.2-TI2V-5B · Q4_K_M GGUF",
        "Quicker image-to-video. Less detailed than the 14B model.",
        "Apache-2.0", True, "~30-60 min",
        ["wan22-ti2v-5b", "wan22-vae", "umt5-xxl"],
        [Field("image", "image", "Image", required=True), _prompt(default="make this image come alive, cinematic motion"),
         Field("frames", "number", "Frames (24 fps)", 49, min=25, max=121, step=24), Field("steps", "number", "Steps", 20, min=8, max=40), _seed()],
        lambda c, a: sdcpp.wan_5b_i2v(c, a["image"], a["prompt"], int(a["frames"]), steps=int(a["steps"]), seed=int(a["seed"])),
    ),
    Tool(
        "text-to-video", "animate", "Text to video", "Wan2.1-T2V-1.3B · Q8_0 GGUF",
        "Short clip from a text prompt. Small model; fits entirely on the GPU.",
        "Apache-2.0", True, "~15-30 min",
        ["wan21-t2v-1.3b", "wan21-vae", "umt5-xxl"],
        [_prompt(), Field("aspect", "select", "Aspect ratio", "16:9", options=["16:9", "9:16"]),
         Field("frames", "number", "Frames (16 fps)", 33, min=17, max=81, step=16), Field("steps", "number", "Steps", 20, min=8, max=40), _seed()],
        lambda c, a: sdcpp.wan_t2v(c, a["prompt"], int(a["frames"]), a["aspect"], int(a["steps"]), seed=int(a["seed"])),
    ),
    # ------------------------------------------------------------------ voice
    Tool(
        "voice-chatterbox", "voice", "Voice-over (Chatterbox)", "Chatterbox Multilingual · ONNX",
        "Expressive voice-over in 23 languages. Upload 5-20 s of someone speaking to clone their voice. No Urdu; Hindi is the closest.",
        "MIT", True, "~1-3 min per 10 s of speech",
        ["chatterbox-lm", "chatterbox-lm-data", "chatterbox-decoder", "chatterbox-decoder-data", "chatterbox-embed",
         "chatterbox-embed-data", "chatterbox-speech-encoder", "chatterbox-speech-encoder-data", "chatterbox-tokenizer", "chatterbox-default-voice"],
        [Field("text", "textarea", "Script", required=True), Field("language", "select", "Language", "en", options=list(voice.CHATTERBOX_LANGS)),
         Field("reference", "audio", "Voice to clone (optional)"), Field("exaggeration", "number", "Expressiveness", 0.5, min=0.25, max=1.5, step=0.05)],
        lambda c, a: voice.chatterbox(c, a["text"], a["language"], a.get("reference"), float(a["exaggeration"])),
        engine="onnx",
    ),
    Tool(
        "voice-kokoro", "voice", "Voice-over (Kokoro)", "Kokoro-82M · ONNX",
        "Fast, clean narration with 54 built-in voices (English, Hindi, Spanish, French, Italian, Japanese, Portuguese, Chinese).",
        "Apache-2.0", True, "faster than real time",
        ["kokoro"],
        [Field("text", "textarea", "Script", required=True), Field("voice", "select", "Voice", "af_heart", options=[]),
         Field("speed", "number", "Speed", 1.0, min=0.5, max=2.0, step=0.05)],
        lambda c, a: voice.kokoro(c, a["text"], a["voice"], float(a["speed"])),
        engine="onnx",
    ),
    # ------------------------------------------------------------------ music
    Tool(
        "music", "music", "Music", "ACE-Step 1.5 turbo · GGUF",
        "Songs or instrumentals from a description, with optional lyrics.",
        "Apache-2.0", True, "~5-10 min per 30 s",
        ["ace-text-encoder", "ace-lm", "ace-dit", "ace-vae"],
        [Field("caption", "textarea", "Describe the music", required=True), Field("lyrics", "textarea", "Lyrics (optional; blank = written for you)"),
         Field("instrumental", "bool", "Instrumental", False), Field("duration", "number", "Seconds", 30, min=10, max=240), _seed()],
        lambda c, a: music.generate(c, a["caption"], a.get("lyrics", ""), float(a["duration"]), int(a["seed"]), bool(a["instrumental"])),
        engine="acestep",
        extra_check=lambda: None if music.available() else "acestep.cpp engine not installed (see README)",
    ),
    # ------------------------------------------------------------------ translate / transcribe
    Tool(
        "translate", "translate", "Translate", "TranslateGemma 4B · Q4_K_M GGUF",
        "Text translation across 55 languages, including Urdu.",
        "Gemma terms", True, "seconds per paragraph",
        ["translategemma-4b"],
        [Field("text", "textarea", "Text", required=True), Field("source", "select", "From", "ur", options=COMMON_LANGS),
         Field("target", "select", "To", "en", options=COMMON_LANGS)],
        lambda c, a: translate.translate(c, a["text"], a["source"], a["target"]),
        engine="llamacpp",
    ),
    Tool(
        "transcribe", "transcribe", "Transcribe", "Whisper large-v3 / large-v3-turbo · q5_0 GGML (whisper.cpp)",
        "Speech to text with SRT subtitles. Handles mixed Urdu/English; Urdu auto-detected as Hindi is decoded as Urdu.",
        "MIT", True, "CPU: roughly real time to 2x",
        ["whisper-large-v3-q5", "whisper-large-v3-turbo-q5"],
        [Field("audio", "audio", "Audio or video file", required=True),
         Field("model", "select", "Model", "large-v3", options=list(whisper.MODELS)),
         Field("language", "select", "Language", "auto", options=["auto", "ur", "en", "hi", "pa", "ar", "fa"]),
         Field("prompt", "text", "Initial prompt (optional)", "", help="e.g. 'Meeting mein Urdu aur English dono hain.'"),
         Field("translate_to", "select", "Also translate to", "", options=["", *COMMON_LANGS])],
        lambda c, a: _transcribe(c, a),
        engine="whispercpp",
    ),
]

BY_ID = {t.id: t for t in TOOLS}


def _transcribe(ctx: Context, a: dict) -> dict:
    result = whisper.transcribe(ctx, a["audio"], a["model"], a["language"], a.get("prompt", ""))
    target = a.get("translate_to") or ""
    if target and target != result["language"] and result["text"]:
        src = result["language"] if result["language"] in translate.LANGUAGES else "ur"
        sub = Context(ctx.workdir, lambda p, m: ctx.progress(0.9 + 0.1 * p, m), ctx.cancel)
        tr = translate.translate(sub, result["text"], src, target)
        result["translation"] = tr["text"]
        result["files"].append("translation.txt")
    return result


def coerce(tool: Tool, raw: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and type-convert form/API values. File fields arrive as Paths already."""
    out: dict[str, Any] = {}
    for f in tool.fields:
        v = raw.get(f.name)
        if v in (None, ""):
            if f.required:
                raise ValueError(f"'{f.name}' is required")
            out[f.name] = f.default
            continue
        if f.type == "number":
            v = float(v)
            if f.min is not None and v < f.min or f.max is not None and v > f.max:
                raise ValueError(f"'{f.name}' must be between {f.min} and {f.max}")
        elif f.type == "bool":
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
        elif f.type == "select" and f.options and v not in f.options and f.name not in ("source", "target", "translate_to", "voice"):
            raise ValueError(f"'{f.name}' must be one of {f.options}")
        elif f.type in ("image", "audio"):
            v = Path(v)
        out[f.name] = v
    return out


def kokoro_voice_options() -> None:
    BY_ID["voice-kokoro"].fields[1].options = voice.kokoro_voices() or ["af_heart"]
