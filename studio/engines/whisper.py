"""Transcription: whisper.cpp with quantized (q5_0) Whisper weights, on the CPU.

whisper.cpp publishes no Windows Vulkan build, so this is the official OpenBLAS CPU build.
Swapping in a self-built Vulkan whisper-cli.exe later needs no code change.
"""

import re
from pathlib import Path

from .. import config, models
from .common import Context, exe, run, to_wav16k

MODELS = {
    "large-v3-turbo": "whisper-large-v3-turbo-q5",  # 4 decoder layers: much faster, slightly weaker
    "large-v3": "whisper-large-v3-q5",  # best for Urdu and code-switched speech
}

_DETECTED = re.compile(r"auto-detected language:\s*([a-z]{2,3})\s*\(p\s*=\s*([0-9.]+)\)")


def detect_language(wav: Path, model: Path, ctx: Context) -> tuple[str, float]:
    out = run(
        [exe("whispercpp", "whisper-cli.exe"), "-m", model, "-f", wav, "-l", "auto", "-dl", "-t", config.CPU_THREADS],
        ctx,
        label="detecting language",
        span=(0.05, 0.1),
        parse=lambda _: None,
    )
    m = _DETECTED.search(out)
    if not m:
        return "auto", 0.0
    return m.group(1), float(m.group(2))


def transcribe(
    ctx: Context,
    audio: Path,
    model: str = "large-v3",
    language: str = "auto",
    prompt: str = "",
) -> dict:
    weights = models.path(MODELS[model])
    wav = to_wav16k(audio, ctx.workdir / "audio16k.wav", ctx)
    ctx.progress(0.05, "decoded audio")

    detected, prob, remapped = language, None, False
    if language == "auto":
        # Detection only runs the encoder on the first 30 s, so it's cheap; it lets the
        # Hindi->Urdu fix happen up front instead of as a second full transcription.
        detected, prob = detect_language(wav, weights, ctx)
        if detected == "hi" and config.WHISPER_HINDI_AS_URDU:
            detected, remapped = "ur", True

    base = ctx.workdir / "transcript"
    cmd = [
        exe("whispercpp", "whisper-cli.exe"),
        "-m", weights, "-f", wav, "-l", detected,
        "-t", config.CPU_THREADS, "-bs", 5, "-pp",
        "-otxt", "-osrt", "-oj", "-of", base,
    ]
    if models.present("silero-vad"):
        # Same job as faster-whisper's vad_filter: skip silence, which is where Whisper hallucinates.
        cmd += ["--vad", "-vm", models.path("silero-vad"), "-vsd", 500]
    if prompt:
        cmd += ["--prompt", prompt]
    run(cmd, ctx, label="transcribing", span=(0.1, 1.0))

    text = base.with_suffix(".txt").read_text(encoding="utf-8").strip()
    return {
        "text": text,
        "language": detected,
        "language_probability": prob,
        "hindi_remapped": remapped,
        "files": [base.with_suffix(".txt").name, base.with_suffix(".srt").name, base.with_suffix(".json").name],
    }
