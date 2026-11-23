"""MCP server: image generation, animation, voice-over, translation and transcription as tools.

Served over streamable HTTP at /mcp by the same process as the web app, so MCP jobs share the
one GPU queue with jobs from the page. Generation is slow on this machine, so every tool
returns after at most `wait_seconds`; if the job isn't finished by then, the reply carries
its job_id and the client calls `get_job` later.
"""

import base64
import binascii
import re
import urllib.request
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from . import config
from .jobs import JobManager
from .tools import BY_ID, TOOLS

INSTRUCTIONS = """Local AI studio running open models on the user's own GPU (Radeon RX 580, 8 GB).
Everything is slow by cloud standards: images take minutes, videos can take an hour.
Each tool queues a job and waits up to `wait_seconds`; if it isn't done, call get_job(job_id)
later instead of resubmitting. Jobs run one at a time.
Image/audio inputs accept a local file path, an http(s) URL, a data: URI, or "job:<job_id>/<filename>"
to reuse an earlier result (e.g. generate an image, then animate it).
FLUX tools (flux-krea, flux-kontext, photo-mate) carry a non-commercial licence; prefer the
Qwen/SDXL tools for commercial work."""


def build(manager: JobManager) -> MCPServer:
    mcp = MCPServer("ai-studio", instructions=INSTRUCTIONS)

    def media(value: str | None, name: str) -> tuple[dict, dict]:
        """Turn a path / URL / data URI / job ref into (raw params, uploaded files)."""
        if not value:
            return {}, {}
        if value.startswith("job:"):
            return {name: value}, {}
        if value.startswith("data:"):
            m = re.match(r"data:[^;,]*(?:;base64)?,(.*)", value, re.S)
            try:
                data = base64.b64decode(m.group(1)) if m else b""
            except binascii.Error as e:
                raise ValueError(f"{name}: bad base64 data URI") from e
            ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "audio/wav": ".wav", "audio/mpeg": ".mp3"}
            mime = value[5:value.find(";")] if ";" in value else ""
            return {}, {name: (f"{name}{ext.get(mime, '.bin')}", data)}
        if value.startswith(("http://", "https://")):
            req = urllib.request.Request(value, headers={"User-Agent": "ai-studio"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
            return {}, {name: (Path(value.split("?")[0]).name or name, data)}
        p = Path(value).expanduser()
        if not p.is_file():
            raise ValueError(f"{name}: no such file {value}")
        return {}, {name: (p.name, p.read_bytes())}

    def submit(tool_id: str, params: dict[str, Any], files_in: dict[str, str | None], wait_seconds: float) -> dict:
        raw = {k: v for k, v in params.items() if v is not None}
        files: dict = {}
        for name, value in files_in.items():
            r, f = media(value, name)
            raw.update(r)
            files.update(f)
        job = manager.submit(tool_id, raw, files)
        return report(manager.wait(job.id, max(0.0, min(wait_seconds, 3000))) or job)

    def report(job) -> dict:
        tool = BY_ID[job.tool]
        out = {
            "job_id": job.id,
            "tool": job.tool,
            "model": tool.model,
            "status": job.status,
            "progress": round(job.progress * 100),
            "message": job.message,
        }
        if job.status == "queued":
            out["jobs_ahead"] = manager.position(job.id)
        if job.status in ("queued", "running"):
            out["typical_time"] = tool.speed
            out["next"] = f"call get_job('{job.id}') later"
        if job.status == "error":
            out["error"] = job.error
        if job.status == "done":
            base = f"http://{config.HOST}:{config.PORT}"
            out["files"] = [
                {"name": n, "path": str(job.dir / n), "url": f"{base}/api/jobs/{job.id}/files/{n}", "ref": f"job:{job.id}/{n}"}
                for n in job.result.get("files", [])
            ]
            out.update({k: v for k, v in job.result.items() if k != "files"})
        return out

    # ------------------------------------------------------------- discovery and jobs

    @mcp.tool()
    def list_models(category: str = "") -> list[dict]:
        """List the available models (tools), with readiness, licence and typical run time.
        category: image | edit | animate | voice | music | translate | transcribe (blank = all)."""
        return [
            {k: d[k] for k in ("id", "category", "title", "model", "summary", "license", "commercial", "speed", "ready", "status")}
            for d in (t.describe() for t in TOOLS)
            if not category or d["category"] == category
        ]

    @mcp.tool()
    def get_job(job_id: str, wait_seconds: float = 30) -> dict:
        """Status of a job; waits up to wait_seconds for it to finish. Finished jobs list their output files."""
        job = manager.wait(job_id, max(0.0, min(wait_seconds, 3000)))
        if job is None:
            raise ValueError(f"No job {job_id}")
        return report(job)

    @mcp.tool()
    def list_jobs(limit: int = 20) -> list[dict]:
        """Recent jobs, newest first."""
        return [report(j) for j in manager.list(limit)]

    @mcp.tool()
    def cancel_job(job_id: str) -> dict:
        """Cancel a queued or running job."""
        return {"job_id": job_id, "cancelled": manager.cancel(job_id)}

    # ------------------------------------------------------------- image generation

    @mcp.tool()
    def generate_image(
        prompt: str,
        model: Literal["qwen-image-fast", "qwen-image", "flux-krea", "z-image", "sdxl"] = "qwen-image-fast",
        aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"] = "1:1",
        negative_prompt: str | None = None,
        seed: int = -1,
        wait_seconds: float = 50,
    ) -> dict:
        """Generate an image from text.
        qwen-image / qwen-image-fast: best for readable text in images. flux-krea: most photo-realistic
        (non-commercial licence). z-image: fast general images. sdxl: classic Stable Diffusion XL, quantized."""
        params: dict[str, Any] = {"prompt": prompt, "aspect": aspect_ratio, "seed": seed}
        if negative_prompt is not None and model in ("qwen-image", "sdxl"):
            params["negative"] = negative_prompt
        return submit(model, params, {}, wait_seconds)

    @mcp.tool()
    def edit_image(
        image: str,
        instruction: str = "",
        model: Literal["qwen-edit", "flux-kontext", "photo-mate"] = "qwen-edit",
        photo_mate_adapter: Literal["restore", "unblur-upscale", "remove-watermark", "cleanse"] = "unblur-upscale",
        seed: int = -1,
        wait_seconds: float = 50,
    ) -> dict:
        """Edit an image by instruction, e.g. "remove the floor drain" or "remove the logo".
        qwen-edit: commercial-safe. flux-kontext: alternative editor. photo-mate: Kontext with a
        restoration adapter (restore, unblur-upscale, remove-watermark, or cleanse = remove the
        object named in `instruction`)."""
        params: dict[str, Any] = {"prompt": instruction, "seed": seed}
        if model == "photo-mate":
            params["adapter"] = photo_mate_adapter
        elif not instruction:
            raise ValueError("instruction is required for this model")
        return submit(model, params, {"image": image}, wait_seconds)

    @mcp.tool()
    def extend_image(
        image: str,
        aspect_ratio: Literal["9:16", "16:9", "1:1", "4:3", "3:4", "3:2", "2:3"] = "9:16",
        align: Literal["middle", "top", "bottom", "left", "right"] = "middle",
        prompt: str = "",
        seed: int = -1,
        wait_seconds: float = 50,
    ) -> dict:
        """Outpaint: extend a photo to a new aspect ratio (e.g. landscape to vertical 9:16) without
        cropping. The original pixels are kept exactly; only the new area is generated."""
        return submit("outpaint", {"aspect": aspect_ratio, "align": align, "prompt": prompt, "seed": seed}, {"image": image}, wait_seconds)

    @mcp.tool()
    def upscale_image(image: str, passes: int = 1, wait_seconds: float = 120) -> dict:
        """Upscale 4x per pass with Real-ESRGAN."""
        return submit("upscale", {"repeats": passes}, {"image": image}, wait_seconds)

    @mcp.tool()
    def remove_background(image: str, wait_seconds: float = 120) -> dict:
        """Cut out the subject: transparent PNG plus its mask (BiRefNet)."""
        return submit("remove-background", {}, {"image": image}, wait_seconds)

    # ------------------------------------------------------------- animation

    @mcp.tool()
    def animate_image(
        image: str,
        prompt: str = "make this image come alive, cinematic motion",
        end_image: str | None = None,
        model: Literal["animate", "animate-fast"] = "animate",
        frames: int | None = None,
        seed: int = -1,
        wait_seconds: float = 50,
    ) -> dict:
        """Turn a still image into a short video (MP4 + WebM).
        animate: Wan2.2 14B, best quality (~1 hour). animate-fast: Wan2.2 5B (~30-60 min).
        Give end_image to morph from the first image to the second (14B only; good for before/after)."""
        if end_image:
            tool_id, files = "first-last-frame", {"image": image, "end_image": end_image}
        else:
            tool_id, files = model, {"image": image}
        params: dict[str, Any] = {"prompt": prompt, "seed": seed}
        if frames:
            params["frames"] = frames
        return submit(tool_id, params, files, wait_seconds)

    @mcp.tool()
    def text_to_video(
        prompt: str,
        aspect_ratio: Literal["16:9", "9:16"] = "16:9",
        frames: int = 33,
        seed: int = -1,
        wait_seconds: float = 50,
    ) -> dict:
        """Short video clip from a text prompt (Wan2.1 1.3B, 16 fps)."""
        return submit("text-to-video", {"prompt": prompt, "aspect": aspect_ratio, "frames": frames, "seed": seed}, {}, wait_seconds)

    # ------------------------------------------------------------- voice-over

    @mcp.tool()
    def voice_over(
        text: str,
        language: str = "en",
        engine: Literal["chatterbox", "kokoro"] = "chatterbox",
        voice: str = "af_heart",
        reference_audio: str | None = None,
        expressiveness: float = 0.5,
        speed: float = 1.0,
        wait_seconds: float = 300,
    ) -> dict:
        """Speak a script (MP3 + WAV).
        chatterbox: 23 languages (no Urdu; Hindi is closest), expressive, and clones the voice in
        reference_audio (5-20 s of clean speech). kokoro: fast narration with built-in voices
        (`voice`, e.g. af_heart, am_michael, bf_emma, hf_alpha for Hindi)."""
        if engine == "kokoro":
            return submit("voice-kokoro", {"text": text, "voice": voice, "speed": speed}, {}, wait_seconds)
        return submit("voice-chatterbox", {"text": text, "language": language, "exaggeration": expressiveness}, {"reference": reference_audio}, wait_seconds)

    # ------------------------------------------------------------- translation / transcription

    @mcp.tool()
    def translate_text(text: str, source_language: str, target_language: str, wait_seconds: float = 600) -> dict:
        """Translate text with TranslateGemma. Languages are ISO codes: en, ur, hi, ar, fr, zh, ...
        (regional forms like en-GB or pt-BR also work)."""
        return submit("translate", {"text": text, "source": source_language, "target": target_language}, {}, wait_seconds)

    @mcp.tool()
    def transcribe_audio(
        audio: str,
        language: str = "auto",
        model: Literal["large-v3", "large-v3-turbo"] = "large-v3",
        initial_prompt: str = "",
        translate_to: str = "",
        wait_seconds: float = 600,
    ) -> dict:
        """Transcribe speech (any audio/video file) to text plus SRT subtitles, with quantized Whisper.
        language: auto, ur, en, hi, ... Urdu auto-detected as Hindi is decoded as Urdu.
        translate_to: optionally translate the transcript (e.g. "en")."""
        params = {"language": language, "model": model, "prompt": initial_prompt, "translate_to": translate_to}
        return submit("transcribe", params, {"audio": audio}, wait_seconds)

    return mcp
