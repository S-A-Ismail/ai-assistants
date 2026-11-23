# AI Studio

Open models from Hugging Face, running on my own GPU (Radeon RX 580, 8 GB). One browser page, one REST
API and one MCP server for image generation and editing, animation, voice-over, music, translation and
transcription. Nothing goes to the cloud.

The card is old and slow, so everything runs as a background job: queue it, close the tab, come back.
Images take minutes. Videos can take an hour.

## What's in it

| Tab | Tool | Model (quantized) | Engine | Licence | Typical time |
| --- | --- | --- | --- | --- | --- |
| Image | Qwen-Image 2.1 | Q4_K GGUF | stable-diffusion.cpp, Vulkan | Apache-2.0 | 20-40 min |
| Image | Qwen-Image 2.1 Turbo | Q4_K_M GGUF | sd.cpp | Apache-2.0 | 10-15 min |
| Image | FLUX.1 Krea | Q4_K_S GGUF | sd.cpp | **non-commercial** | 10-20 min |
| Image | Z-Image Turbo | Q4_K GGUF | sd.cpp | Apache-2.0 | 3-6 min |
| Image | Stable Diffusion XL | Q8_0 GGUF | sd.cpp | OpenRAIL++ | 2-5 min |
| Edit | Qwen-Image Edit | Q4 GGUF + Qwen3-VL vision | sd.cpp | Apache-2.0 | 10-40 min |
| Edit | FLUX.1 Kontext | Q4_K_S GGUF | sd.cpp | **non-commercial** | 15-25 min |
| Edit | Photo-Mate (restore, unblur-upscale, watermark, object removal) | Kontext + the Space's LoRAs | sd.cpp | **non-commercial** | 20-30 min |
| Edit | Outpaint (e.g. landscape to 9:16) | SDXL inpainting Q8_0 | sd.cpp | OpenRAIL++ | 2-5 min |
| Edit | Upscale 4x | Real-ESRGAN x4plus | sd.cpp | BSD-3 | ~1 min |
| Edit | Remove background | BiRefNet ONNX | ONNX Runtime, DirectML | MIT | seconds |
| Animate | Image to video | Wan2.2 I2V A14B 4-step, Q4_K_S | sd.cpp | Apache-2.0 | 1 h+ |
| Animate | First/last frame video | same | sd.cpp | Apache-2.0 | 1 h+ |
| Animate | Image to video (fast) | Wan2.2 TI2V 5B, Q4_K_M | sd.cpp | Apache-2.0 | 30-60 min |
| Animate | Text to video | Wan2.1 T2V 1.3B, Q8_0 | sd.cpp | Apache-2.0 | 15-30 min |
| Voice-over | Chatterbox (23 languages, voice cloning) | ONNX | ONNX Runtime, DirectML | MIT | 1-3 min per 10 s |
| Voice-over | Kokoro (54 voices) | ONNX | ONNX Runtime, DirectML | Apache-2.0 | faster than real time |
| Music | ACE-Step 1.5 turbo | GGUF | acestep.cpp, Vulkan | Apache-2.0 | 5-10 min per 30 s |
| Translate | TranslateGemma 4B | Q4_K_M GGUF | llama.cpp, Vulkan | Gemma terms | seconds |
| Transcribe | Whisper large-v3 / large-v3-turbo | q5_0 GGML | whisper.cpp, CPU | MIT | ~real time |

Times are rough estimates for the RX 580 until I've benchmarked each one.

**Licences matter for client work.** FLUX.1 Krea, Kontext and Photo-Mate (built on Kontext) are under Black
Forest Labs' non-commercial licence. For anything commercial, use Qwen-Image, Qwen-Image Edit, SDXL or
Z-Image, or buy BFL's commercial licence. The page marks the non-commercial tools.

**Why these engines.** The RX 580 has no CUDA and ROCm dropped it years ago. On Windows the GPU paths that
work are Vulkan (stable-diffusion.cpp, llama.cpp, acestep.cpp) and DirectML (ONNX Runtime). whisper.cpp
publishes no Windows Vulkan build, so transcription uses its CPU build for now; a self-built Vulkan
`whisper-cli.exe` dropped into `runtime/engines/whispercpp` would be picked up with no code change.

## Layout

```
studio/
  app.py          FastAPI: page, REST API, mounts MCP at /mcp
  mcp_server.py   MCP tools
  tools.py        the tool catalogue (forms, API and MCP are all built from it)
  jobs.py         one-at-a-time GPU job queue, saved to data/jobs/<id>/
  models.py       every weight file: Hugging Face repo, file, size, licence
  engines/        sdcpp, whisper, translate, voice, birefnet, music
static/           the page (plain HTML/JS, no build step)
scripts/          download_models.py, fetch_engines.py
tests/            no GPU or weights needed
```

Model weights live in `N:\Models` (set `STUDIO_MODELS_DIR` to move them). Engines go in
`runtime/engines`, job outputs in `data/`. Neither is committed.

## Setup

Everything runs on **Windows**, not WSL: the engines are Windows programs and WSL can't reach the AMD
card. Git can stay in WSL.

```powershell
# 1. Python 3.12 venv (uv puts its Python and cache under %LOCALAPPDATA%)
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe --no-deps kokoro-onnx

# 2. Engines: official Windows builds from GitHub releases
.venv\Scripts\python.exe scripts\fetch_engines.py

# 3. Models: about 92 GB into N:\Models. Resumable; re-run if it stops.
.venv\Scripts\python.exe scripts\download_models.py
.venv\Scripts\python.exe scripts\download_models.py --list     # what's there and what isn't
```

`kokoro-onnx` goes in with `--no-deps` because it depends on the CPU-only `onnxruntime`, which would
replace `onnxruntime-directml`.

A Hugging Face token in `.env` (`HF_TOKEN=...`) speeds downloads up. Without one they ran at about 5 MB/s.

### Music engine

acestep.cpp has no official release. Either build it with its `buildvulkan.cmd` (needs the Vulkan SDK
and a compiler), or use the Windows build its author links from the README (hosted on their own server).
Put `ace-lm.exe`, `ace-synth.exe` and their DLLs in `runtime/engines/acestep/`. The Music tab says "not
ready" until then.

## Run

```powershell
.\run.ps1
```

Open <http://127.0.0.1:8000>. Tools whose models haven't finished downloading show "not ready" and switch
on by themselves as files arrive.

Results stay in `data/jobs/` across restarts. Any image result has a **Use in…** menu that sends it to
Edit or Animate, so you can generate, outpaint to 9:16, then animate.

## MCP

The MCP server is part of the same process, at `http://127.0.0.1:8000/mcp` (streamable HTTP). This repo's
`.mcp.json` points Claude Code at it. For another client, add the same URL as an HTTP MCP server.

| MCP tool | What it does |
| --- | --- |
| `generate_image` | Qwen-Image (fast or full), FLUX Krea, Z-Image, SDXL |
| `edit_image` | Qwen-Image Edit, Kontext, or Photo-Mate adapters |
| `extend_image` | outpaint to another aspect ratio, original pixels kept |
| `upscale_image`, `remove_background` | |
| `animate_image` | Wan2.2 14B or 5B; give `end_image` for a first/last-frame morph |
| `text_to_video` | Wan2.1 1.3B |
| `voice_over` | Chatterbox (with voice cloning) or Kokoro |
| `translate_text` | TranslateGemma |
| `transcribe_audio` | Whisper, with optional translation of the transcript |
| `list_models`, `get_job`, `list_jobs`, `cancel_job` | |

Generation is slow, so each tool waits at most `wait_seconds` and then returns a `job_id`; call `get_job`
later rather than resubmitting. Image and audio inputs take a local path, a URL, a `data:` URI, or
`job:<id>/<file>` to reuse an earlier result.

## REST API

Interactive docs at <http://127.0.0.1:8000/api/docs>.

```bash
# queue a job (form fields match the tool's fields in GET /api/tools)
curl -F tool=qwen-image-fast -F prompt="a blue water pump on a white table" http://127.0.0.1:8000/api/jobs

# wait up to 10 minutes for it
curl "http://127.0.0.1:8000/api/jobs/<id>?wait=600"
```

| Method | Path | |
| --- | --- | --- |
| GET | `/api/health` | engines, models on disk, queue |
| GET | `/api/tools` | every tool, its fields, whether it's ready |
| POST | `/api/jobs` | multipart: `tool`, fields, files. `?wait=<s>` blocks |
| GET | `/api/jobs` | history |
| GET | `/api/jobs/{id}` | status. `?wait=<s>` blocks until done |
| GET | `/api/jobs/{id}/files/{name}` | a result (`?download=1` to save) |
| POST | `/api/jobs/{id}/cancel` | |
| DELETE | `/api/jobs/{id}` | removes the job and its files |

With `STUDIO_API_KEY` set in `.env`, `/api` and `/mcp` need `X-API-Key: <key>` or
`Authorization: Bearer <key>`. Leave the server on 127.0.0.1 unless a key is set; there's no other auth.

## Notes

- **One job at a time.** Every model wants the whole card. llama-server (translation) stays loaded between
  back-to-back translations and is shut down before any other job, or after 5 idle minutes.
- **Big models spill into RAM.** sd.cpp places what fits on the GPU and keeps the rest in the 32 GB of
  system RAM. The 14B Wan model needs about 20 GB of RAM free.
- **Urdu.** Whisper's language detection calls most Urdu "Hindi" and writes Devanagari. With auto-detect
  the studio detects first (cheap) and, if it says Hindi, transcribes as Urdu. Chatterbox has no Urdu voice;
  Hindi is the closest, written in Devanagari.
- **Outpainting** never redraws the original: only the new border is generated and the source pixels are
  pasted back over the result, feathered across a thin seam.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

They stub the engines, so they need no weights or GPU. Among other things they check that the
TranslateGemma prompt is byte-identical to what Google's chat template renders, that outpainting keeps the
original pixels, the job lifecycle (done, error, cancel), file uploads and `job:` references, the API key,
and that the MCP endpoint lists its tools.
