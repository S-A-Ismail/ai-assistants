"""Web app + REST API + MCP endpoint, all in one process sharing one GPU job queue.

    GET    /                         the page
    GET    /api/health               engines, models on disk, queue
    GET    /api/tools                every tool with its form fields and readiness
    POST   /api/jobs                 multipart: tool=<id>, fields..., files...  (?wait=<s> to block)
    GET    /api/jobs                 history, newest first
    GET    /api/jobs/{id}            status; ?wait=<s> blocks until done or timeout
    GET    /api/jobs/{id}/files/{n}  a result file
    POST   /api/jobs/{id}/cancel
    DELETE /api/jobs/{id}
    *      /mcp                      MCP (streamable HTTP)
"""

import asyncio
import contextlib
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

from . import config, mcp_server, models
from .engines import translate
from .jobs import JobManager
from .tools import TOOLS, kokoro_voice_options

STATIC = config.ROOT / "static"

manager = JobManager()
mcp = mcp_server.build(manager)
mcp_app = mcp.streamable_http_app(streamable_http_path="/mcp", host=config.HOST)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    async with mcp.session_manager.run():
        yield
    translate.stop()


app = FastAPI(title="AI Studio", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.middleware("http")
async def api_key(request: Request, call_next):
    """With STUDIO_API_KEY set, /api and /mcp need it. The page itself stays open and asks for it."""
    path = request.url.path
    if config.API_KEY and (path.startswith("/api/") or path.startswith("/mcp")) and path not in ("/api/docs", "/api/openapi.json"):
        given = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        given = given or request.query_params.get("key", "")
        if given != config.API_KEY:
            return JSONResponse({"detail": "Missing or wrong API key"}, status_code=401)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/api/health")
def health():
    engines = {}
    for name, exe in [("sdcpp", "sd-cli.exe"), ("llamacpp", "llama-server.exe"), ("whispercpp", "whisper-cli.exe"), ("acestep", "ace-lm.exe")]:
        version = config.ENGINES_DIR / name / "VERSION"
        engines[name] = {
            "installed": (config.ENGINES_DIR / name / exe).exists(),
            "version": version.read_text().strip() if version.exists() else None,
        }
    present = [f for f in models.FILES if f.local.exists()]
    usage = shutil.disk_usage(config.MODELS_DIR) if config.MODELS_DIR.exists() else None
    return {
        "models_dir": str(config.MODELS_DIR),
        "models_present": len(present),
        "models_total": len(models.FILES),
        "models_missing_gb": round(sum(f.size for f in models.FILES if not f.local.exists()) / 1e9, 1),
        "models_free_gb": round(usage.free / 1e9, 1) if usage else None,
        "engines": engines,
        "queue": {
            "running": [j.id for j in manager.list(1000) if j.status == "running"],
            "queued": sum(1 for j in manager.list(1000) if j.status == "queued"),
        },
        "tools_ready": sum(1 for t in TOOLS if t.status() is None),
        "tools_total": len(TOOLS),
    }


@app.get("/api/tools")
def tools():
    kokoro_voice_options()
    return [t.describe() for t in TOOLS]


@app.post("/api/jobs")
async def create_job(request: Request, wait: float = 0):
    form = await request.form()
    tool_id = str(form.get("tool") or "")
    raw: dict = {}
    files: dict = {}
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    for key, value in form.multi_items():
        if key == "tool":
            continue
        if isinstance(value, UploadFile):
            if not value.filename:
                continue
            data = await value.read()
            if len(data) > limit:
                raise HTTPException(413, f"{value.filename} is over {config.MAX_UPLOAD_MB} MB")
            if data:
                files[key] = (value.filename, data)
        else:
            raw[key] = value
    try:
        job = manager.submit(tool_id, raw, files)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e
    if wait > 0:
        job = await asyncio.to_thread(manager.wait, job.id, min(wait, 3600)) or job
    return job.public()


@app.get("/api/jobs")
def list_jobs(limit: int = 100, tool: str | None = None):
    return [j.public() for j in manager.list(limit, tool)]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, wait: float = 0):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    if wait > 0:
        job = await asyncio.to_thread(manager.wait, job_id, min(wait, 3600)) or job
    d = job.public()
    if job.status == "queued":
        d["jobs_ahead"] = manager.position(job_id)
    return d


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str, download: bool = False):
    job = manager.get(job_id)
    if job is None or Path(name).name != name:
        raise HTTPException(404, "No such file")
    allowed = set(job.result.get("files", [])) | {p.name for p in job.dir.glob("input_*")}
    path = job.dir / name
    if name not in allowed or not path.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(path, filename=name if download else None)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not manager.cancel(job_id):
        raise HTTPException(409, "Job isn't queued or running")
    return {"cancelled": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    if not manager.delete(job_id):
        raise HTTPException(404, "No such job")
    return {"deleted": True}


# Last: everything unmatched above goes to the MCP app, which serves /mcp.
app.mount("/", mcp_app)
