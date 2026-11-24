"""Tests that need no model weights or GPU: engines are stubbed where they'd run binaries.

    .venv\\Scripts\\python.exe -m pytest -q
"""

import io
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

# Isolate job data before the studio modules read their config.
os.environ["STUDIO_DATA_DIR"] = tempfile.mkdtemp(prefix="studio-test-")
os.environ["STUDIO_WARMUP"] = "0"  # don't spend minutes loading Chatterbox during tests

from jinja2 import Environment  # noqa: E402
from PIL import Image  # noqa: E402

from studio import tools  # noqa: E402
from studio.engines import sdcpp, translate  # noqa: E402
from studio.engines.common import Context, default_parse  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def ctx(tmp: Path) -> Context:
    return Context(tmp, lambda p, m: None, threading.Event())


# ---------------------------------------------------------------- translation prompt


@pytest.mark.parametrize("src,tgt", [("ur", "en"), ("en", "ur"), ("hi", "en-GB"), ("pt_BR", "fr")])
def test_prompt_matches_official_template(src, tgt):
    """build_prompt must equal what TranslateGemma's own chat template renders."""
    tpl = Environment().from_string((FIXTURES / "translategemma_chat_template.jinja").read_text(encoding="utf-8"))
    text = "یہ ایک ٹیسٹ ہے۔ Pump on hai."
    rendered = tpl.render(
        bos_token="<bos>",
        add_generation_prompt=True,
        messages=[{"role": "user", "content": [{"type": "text", "source_lang_code": src, "target_lang_code": tgt, "text": text}]}],
    )
    assert rendered == "<bos>" + translate.build_prompt(text, src, tgt)


def test_unknown_language_rejected():
    with pytest.raises(ValueError):
        translate.build_prompt("hi", "xx", "en")


def test_chunks_respect_limit_and_keep_text():
    para = "۔ ".join(["یہ ایک لمبا جملہ ہے"] * 200)
    text = "Short intro.\n\n" + para
    pieces = translate.chunks(text, 300)
    assert pieces[0] == "Short intro."
    assert all(len(p) <= 300 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace("\n\n", "").replace(" ", "")


# ---------------------------------------------------------------- progress parsing


def test_progress_parse():
    assert default_parse("  |=====>     | 3/8 - 41.20s/it") == pytest.approx(3 / 8)
    assert default_parse("whisper_print_progress_callback: progress =  45%") == pytest.approx(0.45)
    assert default_parse("loaded 3/8 tensors") is None  # not a progress bar
    assert default_parse("|############    | 123/453 - 181.58MB/s") is None  # weight loading, not progress


@pytest.mark.skipif(os.name != "nt", reason="suspend/resume is Windows-only")
def test_throttle_pauses_engine(tmp_path):
    """The GPU rest must suspend the engine between steps and tiles, grouping short ones."""
    import sys

    from studio.engines.common import run

    # Mimics sd.cpp: a loading bar, 3 sampling steps of 0.5 s, then 6 VAE tiles of 0.2 s. The work
    # burns CPU time, which (unlike a sleep) stops while the process is suspended.
    child = (
        "import time\n"
        "def work(s):\n"
        "    t = time.process_time()\n"
        "    while time.process_time() - t < s: pass\n"
        "for i in range(5): print(f'|##   | {i}/5 - 100.0MB/s', flush=True); time.sleep(0.2)\n"
        "print('loading tensors completed', flush=True)\n"
        "for i in range(1, 4): work(0.5); print(f'|===>  | {i}/3 - 0.50s/it', flush=True)\n"
        "print('sampling completed', flush=True)\n"
        "for i in range(1, 7): work(0.2); print(f'|=>    | {i}/6 - 0.20s/it', flush=True)\n"
        "print('computing vae decode graph completed', flush=True)\n"
    )
    script = tmp_path / "child.py"
    script.write_text(child)
    # The venv's python.exe is a launcher that starts the real interpreter as a child; suspend that.
    python = getattr(sys, "_base_executable", sys.executable)
    t0 = time.monotonic()
    run([python, "-u", str(script)], ctx(tmp_path), duty=(1.0, 0.35))
    took = time.monotonic() - t0
    log = (tmp_path / "engine.log").read_text()
    rests = [line for line in log.splitlines() if "GPU rest" in line]
    # After every sampling step but the last; tiles grouped in pairs (0.4 s of work >= 0.35);
    # nothing for the loading bar or after the final step or tile.
    assert [r.split("after ")[1].split(" ")[0] for r in rests] == ["1/3", "2/3", "2/6", "4/6"], rests
    assert took > 1 + 2.7 + 1.8  # loading + work + rests
    assert "exit code 0" in log


# ---------------------------------------------------------------- input validation


def test_coerce_defaults_and_types():
    t = tools.BY_ID["sdxl"]
    out = tools.coerce(t, {"prompt": "a pump", "steps": "12", "seed": "-1"})
    assert out["steps"] == 12.0 and out["aspect"] == "1:1" and out["cfg"] == 7.0


def test_coerce_rejects_bad_values():
    t = tools.BY_ID["sdxl"]
    with pytest.raises(ValueError):
        tools.coerce(t, {})  # prompt missing
    with pytest.raises(ValueError):
        tools.coerce(t, {"prompt": "x", "steps": "999"})
    with pytest.raises(ValueError):
        tools.coerce(t, {"prompt": "x", "aspect": "7:5"})


def test_every_tool_requires_known_files():
    from studio import models

    for t in tools.TOOLS:
        for key in t.requires:
            assert key in models.BY_KEY, (t.id, key)


# ---------------------------------------------------------------- sizes and outpaint geometry


def test_sizes_snap_and_keep_aspect():
    w, h = sdcpp.size_for("9:16", 0.9, 64)
    assert w % 64 == 0 and h % 64 == 0 and h > w
    assert abs(w / h - 9 / 16) < 0.06


def test_outpaint_keeps_original_pixels(tmp_path, monkeypatch):
    src = Image.new("RGB", (640, 360), (200, 30, 30))
    for x in range(0, 640, 40):  # some structure so a shift would show
        src.paste((10, 10, 200), (x, 0, x + 10, 360))
    src_path = tmp_path / "src.png"
    src.save(src_path)

    def fake_sd(c, args, out, label, span=(0, 1)):
        # Pretend the model painted the whole canvas green.
        canvas = Image.open(args[args.index("-i") + 1])
        Image.new("RGB", canvas.size, (0, 255, 0)).save(out)
        mask = Image.open(args[args.index("--mask") + 1])
        assert mask.size == canvas.size
        return out

    monkeypatch.setattr(sdcpp, "_sd", fake_sd)
    res = sdcpp.outpaint(ctx(tmp_path), src_path, "9:16", "middle", seed=1)
    out = Image.open(tmp_path / res["files"][0]).convert("RGB")
    w, h = out.size
    assert h > w
    # Centre of the image: original pixels, not the "generated" green.
    assert out.getpixel((w // 2, h // 2))[1] < 100
    # Far top and bottom: generated.
    assert out.getpixel((w // 2, 2)) == (0, 255, 0)
    assert out.getpixel((w // 2, h - 3)) == (0, 255, 0)


def test_outpaint_refuses_same_aspect(tmp_path):
    p = tmp_path / "s.png"
    Image.new("RGB", (576, 1024)).save(p)
    with pytest.raises(ValueError):
        sdcpp.outpaint(ctx(tmp_path), p, "9:16")


# ---------------------------------------------------------------- jobs + API (fake tool)


@pytest.fixture(scope="module")
def app_client():
    # One app lifespan per process: the MCP session manager can only be started once.
    from fastapi.testclient import TestClient

    from studio import app as appmod

    # The MCP transport rejects unknown Host headers (DNS-rebinding protection), so use the real one.
    with TestClient(appmod.app, base_url="http://127.0.0.1:8000") as c:
        yield c


@pytest.fixture()
def client(app_client, monkeypatch):
    def fake_run(c, a):
        c.progress(0.5, "halfway")
        if a["prompt"] == "boom":
            raise RuntimeError("engine exploded")
        if a["prompt"] == "slow":
            for _ in range(100):
                c.check()
                time.sleep(0.05)
        (c.workdir / "image.png").write_bytes(b"png")
        return {"files": ["image.png"], "seed": 7}

    t = tools.BY_ID["sdxl"]
    monkeypatch.setattr(t, "run", fake_run)
    monkeypatch.setattr(t, "requires", [])
    return app_client


def test_job_roundtrip(client):
    j = client.post("/api/jobs?wait=10", data={"tool": "sdxl", "prompt": "a pump"}).json()
    assert j["status"] == "done", j
    assert j["result"]["seed"] == 7
    f = client.get(j["files"][0]["url"])
    assert f.status_code == 200 and f.content == b"png"
    assert any(x["id"] == j["id"] for x in client.get("/api/jobs").json())
    assert client.delete(f"/api/jobs/{j['id']}").status_code == 200
    assert client.get(f"/api/jobs/{j['id']}").status_code == 404


def test_job_error_is_reported(client):
    j = client.post("/api/jobs?wait=10", data={"tool": "sdxl", "prompt": "boom"}).json()
    assert j["status"] == "error" and "exploded" in j["error"]


def test_cancel_running_job(client):
    j = client.post("/api/jobs", data={"tool": "sdxl", "prompt": "slow"}).json()
    time.sleep(0.5)
    assert client.post(f"/api/jobs/{j['id']}/cancel").status_code == 200
    j = client.get(f"/api/jobs/{j['id']}?wait=10").json()
    assert j["status"] == "cancelled"


def test_validation_errors_are_400(client):
    assert client.post("/api/jobs", data={"tool": "sdxl"}).status_code == 400
    assert client.post("/api/jobs", data={"tool": "nope", "prompt": "x"}).status_code == 404


def test_tool_not_ready_is_400(client, monkeypatch):
    # Force "not ready" whatever is on disk: a real tool here would start a real GPU job.
    t = tools.BY_ID["flux-krea"]
    monkeypatch.setattr(t, "status", lambda: "model files missing")
    monkeypatch.setattr(t, "run", lambda c, a: pytest.fail("a real engine was started"))
    r = client.post("/api/jobs", data={"tool": "flux-krea", "prompt": "x"})
    assert r.status_code == 400 and "isn't ready" in r.json()["detail"]


def test_upload_and_job_reference(client, monkeypatch):
    seen = {}

    def fake_upscale(c, a):
        seen["image"] = Path(a["image"]).read_bytes()
        (c.workdir / "upscaled.png").write_bytes(b"big")
        return {"files": ["upscaled.png"]}

    up = tools.BY_ID["upscale"]
    monkeypatch.setattr(up, "run", fake_upscale)
    monkeypatch.setattr(up, "requires", [])
    j = client.post("/api/jobs?wait=10", files={"image": ("x.png", io.BytesIO(b"orig"), "image/png")}, data={"tool": "upscale"}).json()
    assert j["status"] == "done" and seen["image"] == b"orig"
    # Feed that result into another job by reference.
    ref = f"job:{j['id']}/upscaled.png"
    j2 = client.post("/api/jobs?wait=10", data={"tool": "upscale", "image": ref}).json()
    assert j2["status"] == "done" and seen["image"] == b"big"


def test_api_key(client, monkeypatch):
    from studio import config

    monkeypatch.setattr(config, "API_KEY", "s3cret")
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/api/health", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/").status_code == 200  # the page itself stays reachable


def test_mcp_lists_tools(client):
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    r = client.post("/mcp", headers=h, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}})
    assert r.status_code == 200
    sid = r.headers["mcp-session-id"]
    client.post("/mcp", headers={**h, "Mcp-Session-Id": sid}, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    r = client.post("/mcp", headers={**h, "Mcp-Session-Id": sid}, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = r.text
    for name in ("generate_image", "animate_image", "voice_over", "translate_text", "transcribe_audio"):
        assert name in names
