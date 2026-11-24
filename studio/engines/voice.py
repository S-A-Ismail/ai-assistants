"""Voice-over: Kokoro-82M (ONNX Runtime + DirectML) and Chatterbox Multilingual (ONNX, CPU).

DirectML is how ONNX Runtime reaches the RX 580 on Windows. Kokoro runs there. Chatterbox
runs on the CPU instead: DirectML can't execute its language model's GroupQueryAttention
nodes, and its decoder takes ~4-5 minutes just to build a session on either device. So the
Chatterbox sessions are built once, kept for the life of the server (they use no VRAM), and
warmed up in the background at startup.
"""

import re
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from .. import config, models
from .common import Context, to_wav, wav_to_mp3

_sessions: dict[str, object] = {}  # GPU sessions: dropped before other engines run
_lock = threading.Lock()
_cpu_sessions: dict[str, object] = {}  # CPU sessions: kept for the life of the process
_cpu_lock = threading.Lock()


def _session(path: Path, gpu: bool = True) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    # DirectML requires these two settings.
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = config.CPU_THREADS
    wanted = ("DmlExecutionProvider", "CPUExecutionProvider") if gpu else ("CPUExecutionProvider",)
    providers = [p for p in wanted if p in ort.get_available_providers()]
    try:
        return ort.InferenceSession(str(path), opts, providers=providers)
    except Exception:
        return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def release() -> None:
    """Drop loaded sessions so their GPU memory is free for the next engine."""
    with _lock:
        _sessions.clear()


def _sentences(text: str, limit: int) -> list[str]:
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?۔؟।])\s+|\n+", text.strip()):
        s = s.strip()
        if not s:
            continue
        if cur and len(cur) + len(s) + 1 > limit:
            parts.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def _finish(ctx: Context, audio: np.ndarray, rate: int) -> dict:
    wav = ctx.workdir / "voice.wav"
    sf.write(wav, audio, rate)
    mp3 = wav_to_mp3(wav, ctx.workdir / "voice.mp3", ctx)
    return {"files": [mp3.name, wav.name], "seconds": round(len(audio) / rate, 2), "sample_rate": rate}


# ---------------------------------------------------------------- Kokoro

# Voice-name prefix -> espeak language. a/b = US/UK English, h = Hindi, etc.
KOKORO_LANGS = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "j": "ja", "p": "pt-br", "z": "cmn"}


def kokoro_voices() -> list[str]:
    d = models.path("kokoro").parent / "voices"
    return sorted(p.stem for p in d.glob("*.bin")) if d.exists() else []


def _kokoro():
    from kokoro_onnx import Kokoro

    if "kokoro" not in _sessions:
        voices_dir = models.path("kokoro").parent / "voices"
        npz = voices_dir.parent / "voices.npz"
        if not npz.exists():
            # The Hugging Face export ships one raw float32 file per voice, shape (510, 1, 256);
            # kokoro-onnx wants them in a single .npz.
            arrays = {p.stem: np.fromfile(p, dtype=np.float32).reshape(-1, 1, 256) for p in voices_dir.glob("*.bin")}
            np.savez(npz, **arrays)
        _sessions["kokoro"] = Kokoro.from_session(_session(models.path("kokoro")), str(npz))
    return _sessions["kokoro"]


def kokoro(ctx: Context, text: str, voice: str = "af_heart", speed: float = 1.0) -> dict:
    lang = KOKORO_LANGS.get(voice[:1], "en-us")
    with _lock:
        ctx.progress(0.05, "loading Kokoro")
        tts = _kokoro()
        pieces = _sentences(text, 400)
        out = []
        for i, piece in enumerate(pieces):
            ctx.check()
            ctx.progress(0.1 + 0.85 * i / len(pieces), f"speaking part {i + 1} of {len(pieces)}")
            audio, rate = tts.create(piece, voice=voice, speed=speed, lang=lang)
            out += [audio, np.zeros(int(rate * 0.15), dtype=np.float32)]
    return {**_finish(ctx, np.concatenate(out), rate), "voice": voice}


# ---------------------------------------------------------------- Chatterbox

CHATTERBOX_LANGS = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}
SR = 24000
START, STOP = 6561, 6562
LAYERS, KV_HEADS, HEAD_DIM = 30, 16, 64


CHATTERBOX_FILES = ("chatterbox-speech-encoder", "chatterbox-embed", "chatterbox-lm", "chatterbox-decoder", "chatterbox-tokenizer")


def _chatterbox():
    from tokenizers import Tokenizer

    with _cpu_lock:  # a warm-up may already be building it; wait for that instead of building twice
        if "chatterbox" not in _cpu_sessions:
            _cpu_sessions["chatterbox"] = {
                "speech_encoder": _session(models.path("chatterbox-speech-encoder"), gpu=False),
                "embed": _session(models.path("chatterbox-embed"), gpu=False),
                "lm": _session(models.path("chatterbox-lm"), gpu=False),
                "decoder": _session(models.path("chatterbox-decoder"), gpu=False),  # the ~4-5 min one
                "tokenizer": Tokenizer.from_file(str(models.path("chatterbox-tokenizer"))),
            }
        return _cpu_sessions["chatterbox"]


def chatterbox_loaded() -> bool:
    return "chatterbox" in _cpu_sessions


def warm_up_chatterbox() -> None:
    """Build the Chatterbox sessions in the background so the first voice-over doesn't wait."""
    if models.present(*CHATTERBOX_FILES, "chatterbox-lm-data", "chatterbox-decoder-data") and not chatterbox_loaded():
        threading.Thread(target=_chatterbox, daemon=True, name="chatterbox-warmup").start()


def _dtype(session: ort.InferenceSession, name: str):
    t = next(i.type for i in session.get_inputs() if i.name == name)
    return np.float16 if "float16" in t else np.float32


def _hangul_to_jamo(text: str) -> str:
    def split(c: str) -> str:
        if not ("가" <= c <= "힣"):
            return c
        b = ord(c) - 0xAC00
        return chr(0x1100 + b // 588) + chr(0x1161 + (b % 588) // 28) + (chr(0x11A7 + b % 28) if b % 28 else "")

    return "".join(split(c) for c in text)


def _speak_chunk(m: dict, text: str, lang: str, ref: np.ndarray, exaggeration: float, max_tokens: int, ctx: Context):
    if lang == "ko":
        text = _hangul_to_jamo(text)
    ids = np.array([m["tokenizer"].encode(f"[{lang}]{text}").ids], dtype=np.int64)
    pos = np.where(ids >= START, 0, np.arange(ids.shape[1])[None, :] - 1).astype(np.int64)
    embed_in = {"input_ids": ids, "position_ids": pos, "exaggeration": np.array([exaggeration], dtype=np.float32)}

    lm = m["lm"]
    emb_t = _dtype(lm, "inputs_embeds")
    kv_t = _dtype(lm, "past_key_values.0.key")
    tokens = np.array([[START]], dtype=np.int64)
    past: dict = {}
    mask = None
    prompt_token = speaker = feats = None

    for i in range(max_tokens):
        ctx.check()
        emb = m["embed"].run(None, embed_in)[0]
        if i == 0:
            cond, prompt_token, speaker, feats = m["speech_encoder"].run(None, {"audio_values": ref})
            emb = np.concatenate((cond, emb), axis=1)
            past = {
                f"past_key_values.{layer}.{kv}": np.zeros([1, KV_HEADS, 0, HEAD_DIM], dtype=kv_t)
                for layer in range(LAYERS)
                for kv in ("key", "value")
            }
            mask = np.ones((1, emb.shape[1]), dtype=np.int64)
        logits, *present = lm.run(None, {"inputs_embeds": emb.astype(emb_t), "attention_mask": mask, **past})
        logits = logits[:, -1, :].astype(np.float32)
        # Repetition penalty 1.2, as in the reference implementation.
        seen = np.take_along_axis(logits, tokens, axis=1)
        np.put_along_axis(logits, tokens, np.where(seen < 0, seen * 1.2, seen / 1.2), axis=1)
        nxt = np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)
        tokens = np.concatenate((tokens, nxt), axis=-1)
        if int(nxt[0, 0]) == STOP:
            break
        embed_in["input_ids"] = nxt
        embed_in["position_ids"] = np.full((1, 1), i + 1, dtype=np.int64)
        mask = np.concatenate([mask, np.ones((1, 1), dtype=np.int64)], axis=1)
        for j, k in enumerate(past):
            past[k] = present[j]

    speech = np.concatenate([prompt_token, tokens[:, 1:-1] if int(tokens[0, -1]) == STOP else tokens[:, 1:]], axis=1)
    wav = m["decoder"].run(None, {"speech_tokens": speech, "speaker_embeddings": speaker, "speaker_features": feats})[0]
    return np.squeeze(wav, axis=0).astype(np.float32)


def chatterbox(
    ctx: Context,
    text: str,
    language: str = "en",
    reference: Path | None = None,
    exaggeration: float = 0.5,
) -> dict:
    """Voice-over in 23 languages. Give `reference` (5-20 s of clean speech) to clone that voice."""
    if language not in CHATTERBOX_LANGS:
        raise ValueError(f"Chatterbox doesn't support {language!r}. Supported: {', '.join(CHATTERBOX_LANGS)}")
    ref_path = reference or models.path("chatterbox-default-voice")
    ref_wav = to_wav(ref_path, ctx.workdir / "reference24k.wav", ctx, SR)
    ref, _ = sf.read(ref_wav, dtype="float32")
    ref = ref[None, : SR * 20]  # the encoder only needs a short sample

    # CPU-only and kept loaded, so no GPU lock is needed; jobs are serialised by the queue anyway.
    ctx.progress(0.05, "speaking" if chatterbox_loaded() else "loading Chatterbox (one-time, ~5 min after a restart)")
    m = _chatterbox()
    pieces = _sentences(text, 250)
    out = []
    for i, piece in enumerate(pieces):
        ctx.progress(0.1 + 0.85 * i / len(pieces), f"speaking part {i + 1} of {len(pieces)}")
        out += [_speak_chunk(m, piece, language, ref, exaggeration, 1000, ctx), np.zeros(int(SR * 0.2), dtype=np.float32)]
    return {**_finish(ctx, np.concatenate(out), SR), "language": language, "cloned": reference is not None}
