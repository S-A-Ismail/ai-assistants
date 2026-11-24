"""Shared plumbing for engines: running native binaries with progress and cancellation."""

import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import imageio_ffmpeg

from ..config import ENGINES_DIR

ProgressFn = Callable[[float, str], None]


class Cancelled(Exception):
    pass


class EngineError(RuntimeError):
    pass


class Context:
    """What an engine gets from the job runner: where to write, how to report, how to stop."""

    def __init__(self, workdir: Path, progress: ProgressFn, cancel: threading.Event):
        self.workdir = workdir
        self.progress = progress
        self.cancel = cancel

    def check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()


def exe(engine: str, name: str) -> Path:
    p = ENGINES_DIR / engine / name
    if not p.exists():
        raise EngineError(f"{p} is missing. Run: python scripts/fetch_engines.py {engine}")
    return p


# "12/20", as printed by sd.cpp, llama.cpp and whisper.cpp progress bars.
_STEP = re.compile(r"(\d+)\s*/\s*(\d+)")
_PCT = re.compile(r"progress\s*=\s*(\d+)%")


def run(
    cmd: list[str],
    ctx: Context,
    *,
    label: str = "",
    span: tuple[float, float] = (0.0, 1.0),
    parse: Callable[[str], float | None] | None = None,
    cwd: Path | None = None,
    env: dict | None = None,
    duty: tuple[float, float] | None = None,
) -> str:
    """Run a native engine, mapping its own progress output onto [span[0], span[1]].

    `duty` = (ratio, min_work): whenever a progress step finishes (a sampling step, a VAE tile)
    and at least `min_work` seconds of work have built up since the last rest, the process is
    suspended for `ratio` times that work, so the GPU gets regular breaks (see _Pause).

    Returns the combined output. Raises Cancelled if the job is cancelled (the process
    is killed), or EngineError with the tail of the output if it exits non-zero.
    """
    lo, hi = span
    tail: deque[str] = deque(maxlen=40)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env={**os.environ, **(env or {})},
        creationflags=flags,
    )

    def watch_cancel() -> None:
        ctx.cancel.wait()
        if proc.poll() is None:
            proc.kill()

    threading.Thread(target=watch_cancel, daemon=True).start()

    pause = _Pause(proc) if duty and os.name == "nt" else None
    bar_total, last_step, work, mark = 0, 0, 0.0, time.monotonic()

    # Everything the engine prints goes to engine.log in the job folder, forced to disk line by
    # line: when a GPU hang takes the whole PC down, this is the only record of where it stopped.
    log = open(ctx.workdir / "engine.log", "a", encoding="utf-8")
    log.write(f"$ {' '.join(str(c) for c in cmd)}\n")

    def record(line: str) -> None:
        log.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        log.flush()
        os.fsync(log.fileno())

    # Progress bars redraw with \r, so split on both \r and \n.
    buf = b""
    assert proc.stdout is not None
    try:
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                tail.append(line)
                record(line)
                if pause:
                    if "completed" in line and _step_done(line) is None:
                        # e.g. "loading tensors completed": start timing work from here, not
                        # from before a load or a CPU stage.
                        mark = time.monotonic()
                    elif step := _step_done(line):
                        n, total = step
                        if total != bar_total or n < last_step:
                            # A new bar (sampling -> VAE tiles). The GPU sat idle in between
                            # (loading the next model), so count work afresh.
                            bar_total, last_step, work = total, 0, 0.0
                        if n > last_step:
                            # A step or tile just finished, so the GPU has handed its result back
                            # and is idle: rest it here. (Mid-step it can't: a whole step is queued.)
                            work += time.monotonic() - mark
                            last_step = n
                            if n < total and work >= duty[1]:
                                rest = min(90.0, duty[0] * work)
                                record(f"[studio] GPU rest {rest:.1f}s after {n}/{total} ({work:.1f}s of work)")
                                pause.rest(rest, ctx.cancel)
                                work = 0.0
                            mark = time.monotonic()
                frac = parse(line) if parse else default_parse(line)
                if frac is not None:
                    ctx.progress(lo + (hi - lo) * max(0.0, min(1.0, frac)), label)
        if buf:
            tail.append(buf.decode("utf-8", "replace").strip())
            record(tail[-1])
        code = proc.wait()
        record(f"exit code {code}")
    finally:
        if pause:
            pause.close()
        log.close()
    if ctx.cancel.is_set():
        raise Cancelled()
    out = "\n".join(tail)
    if code != 0:
        raise EngineError(f"{Path(cmd[0]).name} exited with {code}:\n{out}")
    return out


_STEP_LINE = re.compile(r"\|\s*(\d+)\s*/\s*(\d+)\s*-\s*[\d.]+\s*(?:s/it|it/s)")


def _step_done(line: str) -> tuple[int, int] | None:
    """(step, total) from a sampling progress line like '| 3/8 - 7.57s/it', else None."""
    m = _STEP_LINE.search(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


class _Pause:
    """Suspend and resume a process (Windows), to give the GPU breaks between sampling steps.

    The RX 580 froze the PC every time it ran flat out for 30-50 s, never in short bursts. A
    suspended engine submits no new GPU work. It is always resumed, and can still be killed while
    suspended (which is how cancelling works).
    """

    def __init__(self, proc: subprocess.Popen):
        import ctypes

        self._k, self._nt = ctypes.windll.kernel32, ctypes.windll.ntdll
        self._h = self._k.OpenProcess(0x0800 | 0x1000, False, proc.pid)  # SUSPEND_RESUME | QUERY_LIMITED

    def rest(self, seconds: float, cancel: threading.Event) -> None:
        if not self._h:
            return
        self._nt.NtSuspendProcess(self._h)
        try:
            cancel.wait(seconds)
        finally:
            self._nt.NtResumeProcess(self._h)

    def close(self) -> None:
        if self._h:
            self._k.CloseHandle(self._h)
            self._h = None


def default_parse(line: str) -> float | None:
    m = _PCT.search(line)
    if m:
        return int(m.group(1)) / 100
    # sd.cpp's tensor-loading bar ("123/453 - 180MB/s") isn't job progress: counting it made a job
    # read 99.9% before sampling had even started.
    if "MB/s" in line or "GB/s" in line:
        return None
    # Only trust "a/b" on lines that look like a progress bar, not on arbitrary log text.
    if "it]" in line or "s/it" in line or "it/s" in line or "|" in line:
        m = _STEP.search(line)
        if m and int(m.group(2)) > 0:
            return int(m.group(1)) / int(m.group(2))
    return None


def ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def to_wav(src: Path, dst: Path, ctx: Context, rate: int = 16000) -> Path:
    """Decode any audio/video ffmpeg understands (incl. WhatsApp .ogg/opus) to mono PCM WAV."""
    run([ffmpeg(), "-y", "-i", src, "-ac", "1", "-ar", rate, "-c:a", "pcm_s16le", dst], ctx, parse=lambda _: None)
    return dst


def to_wav16k(src: Path, dst: Path, ctx: Context) -> Path:
    return to_wav(src, dst, ctx, 16000)


def webm_to_mp4(src: Path, dst: Path, ctx: Context) -> Path:
    run(
        [ffmpeg(), "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", dst],
        ctx,
        parse=lambda _: None,
    )
    return dst


def wav_to_mp3(src: Path, dst: Path, ctx: Context) -> Path:
    run([ffmpeg(), "-y", "-i", src, "-c:a", "libmp3lame", "-q:a", "2", dst], ctx, parse=lambda _: None)
    return dst
