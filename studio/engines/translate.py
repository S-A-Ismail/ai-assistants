"""Translation: TranslateGemma 4B (Q4_K_M GGUF) on llama.cpp's Vulkan build.

TranslateGemma's chat template only accepts a structured user turn (source/target language
codes plus text), which an OpenAI-style chat request can't express. So this builds the exact
prompt the template would produce and sends it to llama-server's raw /completion endpoint.

llama-server stays up between consecutive translations (loading takes a while) and is shut
down before any other GPU job runs; see `stop()`.
"""

import json
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

from .. import models
from .common import Context, EngineError, exe

LANGUAGES: dict[str, str] = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "translategemma_languages.json").read_text(encoding="utf-8")
)

CHUNK_CHARS = 1200  # the model's input context is 2K tokens; Urdu/Hindi tokenise densely

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_port: int | None = None
_last_used = 0.0
IDLE_SECONDS = 300


def language_name(code: str) -> str:
    code = code.replace("_", "-")
    if code not in LANGUAGES:
        raise ValueError(f"Unsupported language code {code!r}")
    return LANGUAGES[code]


def build_prompt(text: str, source: str, target: str) -> str:
    """Byte-for-byte what TranslateGemma's chat template renders for a text turn (minus BOS,
    which llama-server adds)."""
    src, tgt = source.replace("_", "-"), target.replace("_", "-")
    s, t = language_name(src), language_name(tgt)
    return (
        "<start_of_turn>user\n"
        f"You are a professional {s} ({src}) to {t} ({tgt}) translator. Your goal is to accurately convey "
        f"the meaning and nuances of the original {s} text while adhering to {t} grammar, vocabulary, and "
        "cultural sensitivities.\n"
        f"Produce only the {t} translation, without any additional explanations or commentary. Please "
        f"translate the following {s} text into {t}:\n\n\n"
        f"{text.strip()}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def chunks(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split on paragraph, then sentence boundaries, keeping each piece under `limit` chars."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        if len(para) <= limit:
            out.append(para)
            continue
        cur = ""
        for sent in re.split(r"(?<=[.!?۔؟।])\s+", para):
            if cur and len(cur) + len(sent) + 1 > limit:
                out.append(cur)
                cur = ""
            while len(sent) > limit:  # one enormous sentence: hard split
                out.append(sent[:limit])
                sent = sent[limit:]
            cur = f"{cur} {sent}".strip()
        if cur:
            out.append(cur)
    return [c for c in out if c.strip()]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_server(ctx: Context | None) -> int:
    global _proc, _port
    if _proc is not None and _proc.poll() is None:
        return _port  # type: ignore[return-value]
    _port = _free_port()
    _proc = subprocess.Popen(
        [
            str(exe("llamacpp", "llama-server.exe")),
            "-m", str(models.path("translategemma-4b")),
            "--host", "127.0.0.1", "--port", str(_port),
            "-c", "4096", "-ngl", "99", "--no-webui",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.time() + 300
    while time.time() < deadline:
        if ctx:
            ctx.check()
        if _proc.poll() is not None:
            raise EngineError("llama-server exited while loading TranslateGemma")
        try:
            if httpx.get(f"http://127.0.0.1:{_port}/health", timeout=2).status_code == 200:
                return _port
        except httpx.HTTPError:
            pass
        time.sleep(1)
    stop()
    raise EngineError("TranslateGemma did not finish loading within 5 minutes")


def stop() -> None:
    """Free the GPU. Called before any other engine runs, and after IDLE_SECONDS idle."""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(10)
            except subprocess.TimeoutExpired:
                _proc.kill()
        _proc = None


def stop_if_idle() -> None:
    if _proc is not None and time.time() - _last_used > IDLE_SECONDS:
        stop()


def translate(ctx: Context, text: str, source: str, target: str) -> dict:
    global _last_used
    language_name(source), language_name(target)  # validate before loading anything
    pieces = chunks(text)
    if not pieces:
        raise ValueError("Nothing to translate.")
    with _lock:
        ctx.progress(0.02, "loading TranslateGemma")
        port = _ensure_server(ctx)
        out = []
        for i, piece in enumerate(pieces):
            ctx.check()
            ctx.progress(0.1 + 0.9 * i / len(pieces), f"translating part {i + 1} of {len(pieces)}")
            r = httpx.post(
                f"http://127.0.0.1:{port}/completion",
                json={
                    "prompt": build_prompt(piece, source, target),
                    "n_predict": 2048,
                    "temperature": 0,
                    "stop": ["<end_of_turn>"],
                    "cache_prompt": False,
                },
                timeout=900,
            )
            r.raise_for_status()
            out.append(r.json()["content"].strip())
        _last_used = time.time()
    result = "\n\n".join(out)
    (ctx.workdir / "translation.txt").write_text(result, encoding="utf-8")
    return {"text": result, "source": source, "target": target, "chunks": len(pieces), "files": ["translation.txt"]}
