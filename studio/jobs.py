"""Job queue. One worker, one job at a time: every model here wants the whole 8 GB card.

Each job lives in DATA_DIR/jobs/<id>/ with its inputs, outputs and a job.json, so the history
(and the gallery built from it) survives restarts. Jobs that were queued or running when the
server stopped come back as failed, with a note saying why.
"""

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .engines import translate, voice
from .engines.common import Cancelled, Context
from .tools import BY_ID, coerce

JOBS_DIR = config.DATA_DIR / "jobs"


@dataclass
class Job:
    id: str
    tool: str
    params: dict[str, Any]
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def public(self) -> dict:
        d = asdict(self)
        d["files"] = [
            {"name": n, "url": f"/api/jobs/{self.id}/files/{n}"} for n in self.result.get("files", []) if (self.dir / n).exists()
        ]
        d["elapsed"] = round((self.finished or time.time()) - self.started, 1) if self.started else None
        return d

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.dir / "job.json")


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.cancels: dict[str, threading.Event] = {}
        self.lock = threading.Lock()
        self.last_engine: str | None = None
        self._load()
        threading.Thread(target=self._worker, daemon=True, name="job-worker").start()

    def _load(self) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        for p in JOBS_DIR.glob("*/job.json"):
            try:
                job = Job(**json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
            if job.status in ("queued", "running"):
                job.status, job.error = "error", "The server stopped before this job finished. Submit it again."
                job.save()
            self.jobs[job.id] = job

    # ---- public API

    def submit(self, tool_id: str, raw: dict[str, Any], files: dict[str, tuple[str, bytes]] | None = None) -> Job:
        tool = BY_ID.get(tool_id)
        if tool is None:
            raise KeyError(f"Unknown tool {tool_id!r}")
        if reason := tool.status():
            raise RuntimeError(f"{tool.title} isn't ready: {reason}")
        job = Job(id=uuid.uuid4().hex[:12], tool=tool_id, params={})
        job.dir.mkdir(parents=True, exist_ok=True)
        raw = dict(raw)
        for name, (filename, data) in (files or {}).items():
            safe = Path(filename or name).name or name
            dest = job.dir / f"input_{name}{Path(safe).suffix.lower() or '.bin'}"
            dest.write_bytes(data)
            raw[name] = dest
        for f in tool.fields:
            # A file field can also point at an earlier result: "job:<id>/<filename>".
            v = raw.get(f.name)
            if f.type in ("image", "audio") and isinstance(v, str) and v.startswith("job:"):
                raw[f.name] = self.resolve_ref(v)
        try:
            params = coerce(tool, raw)
        except Exception:
            shutil.rmtree(job.dir, ignore_errors=True)
            raise
        job.params = {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()}
        with self.lock:
            self.jobs[job.id] = job
            self.cancels[job.id] = threading.Event()
        job.save()
        self.queue.put(job.id)
        return job

    def resolve_ref(self, ref: str) -> Path:
        job_id, _, name = ref[4:].partition("/")
        job = self.jobs.get(job_id)
        if not job or not name or name not in job.result.get("files", []):
            raise ValueError(f"No such result: {ref}")
        return job.dir / name

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self, limit: int = 100, tool: str | None = None) -> list[Job]:
        items = [j for j in self.jobs.values() if tool is None or j.tool == tool]
        return sorted(items, key=lambda j: j.created, reverse=True)[:limit]

    def position(self, job_id: str) -> int:
        """How many jobs are ahead of this one (0 = running or next)."""
        queued = sorted((j for j in self.jobs.values() if j.status == "queued"), key=lambda j: j.created)
        ids = [j.id for j in queued]
        return ids.index(job_id) if job_id in ids else 0

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status not in ("queued", "running"):
            return False
        self.cancels.get(job_id, threading.Event()).set()
        if job.status == "queued":
            job.status, job.finished, job.message = "cancelled", time.time(), "cancelled"
            job.save()
        return True

    def delete(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status in ("queued", "running"):
            self.cancel(job_id)
        with self.lock:
            self.jobs.pop(job_id, None)
        shutil.rmtree(job.dir, ignore_errors=True)
        return True

    def wait(self, job_id: str, timeout: float) -> Job | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.jobs.get(job_id)
            if job is None or job.status in ("done", "error", "cancelled"):
                return job
            time.sleep(1)
        return self.jobs.get(job_id)

    # ---- worker

    def _free_gpu_for(self, engine: str) -> None:
        # Persistent engines hold VRAM between jobs; drop them before a different engine runs.
        if engine != "llamacpp":
            translate.stop()
        if engine != "onnx":
            voice.release()

    def _worker(self) -> None:
        while True:
            try:
                job_id = self.queue.get(timeout=30)
            except queue.Empty:
                translate.stop_if_idle()
                continue
            job = self.jobs.get(job_id)
            cancel = self.cancels.get(job_id)
            if job is None or cancel is None or job.status != "queued":
                continue
            tool = BY_ID[job.tool]
            job.status, job.started, job.message = "running", time.time(), "starting"
            job.save()

            last_save = [0.0]

            def progress(frac: float, msg: str) -> None:
                job.progress = round(max(job.progress, min(frac, 0.999)), 4)
                if msg:
                    job.message = msg
                if time.time() - last_save[0] > 5:
                    last_save[0] = time.time()
                    job.save()

            try:
                self._free_gpu_for(tool.engine)
                params = {}
                for f in tool.fields:
                    v = job.params.get(f.name)
                    params[f.name] = Path(v) if f.type in ("image", "audio") and v else v
                job.result = tool.run(Context(job.dir, progress, cancel), params) or {}
                job.status, job.progress, job.message = "done", 1.0, "done"
            except Cancelled:
                job.status, job.message = "cancelled", "cancelled"
            except Exception as e:
                job.status, job.error = "error", str(e) or e.__class__.__name__
                (job.dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                job.finished = time.time()
                job.save()
                self.cancels.pop(job_id, None)
