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
| Image | Z-Image Turbo | Q4_K GGUF | sd.cpp | Apache-2.0 | ~2 min at 0.3 MP (measured) |
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
| Voice-over | Chatterbox (23 languages, voice cloning) | ONNX, 4-bit language model | ONNX Runtime, CPU | MIT | ~1 min per 7 s |
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

## Step-by-step guides

### 1. Set up a fresh machine

1. Install [uv](https://docs.astral.sh/uv/) and clone this repo (the code lives in `M:\Code Base\ai-assistants`).
2. Run the three commands under [Setup](#setup): venv, engines, models. The model download is about
   92 GB; leave it running and re-run it if it stops. `--list` shows what's still missing.
3. Raise the GPU timeout so long GPU jobs can't freeze the display (see
   [Image generation froze the display](#image-generation-froze-the-display)), then restart Windows.
4. Optional: put `HF_TOKEN=...` in `.env` for faster downloads, and `STUDIO_API_KEY=...` if anything
   other than this PC will call the API.

### 2. Start and stop the studio

1. Open PowerShell in the repo folder and run `.\run.ps1`. Leave the window open.
2. Wait about 5 minutes before the first voice-over: Chatterbox loads in the background after a start.
   Everything else is usable straight away.
3. Open <http://127.0.0.1:8000>. The top-right corner shows how many tools are ready.
4. To stop, press `Ctrl+C` in that window. A job that was running is marked as failed on the next start;
   queue it again.

### 3. Make your first image

Start small so a problem shows up quickly instead of after 40 minutes.

1. **Image** tab, pick **Z-Image Turbo**.
2. Write a prompt, e.g. `a blue water pump on a white table, studio lighting, product photo`.
3. Set **Megapixels** to `0.5` for the first try. Leave **Seed** at `-1`.
4. Click **Queue job**. The job appears on the right with a progress bar; you can close the tab.
5. When it's done, click the image to open it full size, or download it.
6. Liked it? Note its seed (shown on the job) and re-run with the same seed and a bigger size.

For commercial work use Qwen-Image, Z-Image or SDXL. FLUX Krea is non-commercial.

### 4. Product photo to a 9:16 social video with voice-over

1. **Edit** tab, **Remove background** (optional): upload the product photo, queue it. You get
   `cutout.png` (transparent) and `mask.png`.
2. **Edit** tab, **Outpaint**: upload the landscape photo (or pick a result with **Use in… → Outpaint**).
   Aspect ratio `9:16`, **Place original** `middle`, and describe the surroundings in the prompt if the
   default isn't right. The product's pixels are kept exactly; only the new top and bottom are generated.
3. On the outpainted result, choose **Use in… → Image to video (Wan2.2 5B)**. Prompt the motion, e.g.
   `slow camera push-in, soft light moving across the pump`. Expect 30-60 minutes. The 14B tool looks
   better and takes over an hour.
4. **Voice-over** tab, **Voice-over (Chatterbox)**: paste the script, pick the language, queue it. Spell
   brand names phonetically ("Grund-foss").
5. The studio doesn't merge video and audio yet. Put `video.mp4` and `voice.mp3` together in any video
   editor, or with the ffmpeg that came with the venv (run from the folder holding both files):
   ```powershell
   $ff = & "M:\Code Base\ai-assistants\.venv\Scripts\python.exe" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"; & $ff -i video.mp4 -i voice.mp3 -c:v copy -c:a aac -shortest final.mp4
   ```
   `-shortest` cuts at whichever is shorter; a Wan clip is only 2-5 seconds, so keep the script short or
   make several clips.

### 5. Restore or clean up a photo

1. **Edit** tab, **Photo-Mate** (non-commercial). Pick an **Adapter**: `restore` for old or damaged
   photos, `unblur-upscale` for soft ones, `remove-watermark` to remove watermarks, `cleanse` to remove an
   object (name the object in the prompt).
2. For commercial use instead: **Qwen-Image Edit** with a plain instruction such as `remove the
   watermark` or `make the background pure white`.
3. Finish with **Upscale 4x** if you need a bigger file. It takes about a minute.

### 6. Voice-over in a cloned voice

1. Record 5-20 seconds of clean speech: one speaker, no music, little room echo. Any audio or video
   format works.
2. **Voice-over** tab, **Voice-over (Chatterbox)**: paste the script, choose the language, and upload the
   recording under **Voice to clone**.
3. **Expressiveness** around `0.5` sounds natural; higher is more dramatic.
4. Queue it. You get `voice.mp3` and `voice.wav`. Long scripts are split into sentences and joined.

Urdu isn't supported; use Hindi (written in Devanagari) for the closest result. For English with a
choice of 54 stock voices and faster output, use **Voice-over (Kokoro)**.

### 7. Transcribe a meeting and translate it

1. **Transcribe** tab: upload the recording (audio or video).
2. **Model**: `large-v3` is more accurate, `large-v3-turbo` is faster.
3. **Language**: leave `auto`, or pick it if you know it. Auto-detected "Hindi" is transcribed as Urdu.
4. Optional: an **Initial prompt** with names and terms that come up ("Grundfos, Scala2, booster pump")
   helps spelling.
5. Optional: **Also translate to** `en` to get an English translation alongside.
6. You get `transcript.txt`, `transcript.srt` (subtitles) and `transcript.json`.

### 8. Translate text

1. **Translate** tab: paste the text, set **From** and **To**, queue it. Seconds once the model is loaded.
2. Check Urdu output before using it (see [Translation quality](#translation-quality)).

### 9. Use it from Claude Code (MCP)

1. Start the studio (guide 2).
2. Open Claude Code in this repo folder: `.mcp.json` already points at `http://127.0.0.1:8000/mcp`.
   Elsewhere, add that URL as an HTTP MCP server.
3. Ask in plain words: "generate a 9:16 image of a pump in a kitchen with Qwen-Image Turbo", "transcribe
   this file", "voice-over this script in Hindi".
4. Long jobs come back with a job id. Ask Claude to check the job later instead of asking again, which
   would queue a duplicate.

### 10. Call it from a script (REST API)

1. `GET /api/tools` lists each tool's id and form fields.
2. Queue a job with a multipart POST to `/api/jobs` (`tool=<id>` plus the fields; files as uploads).
3. Poll `GET /api/jobs/<id>?wait=600` until `status` is `done`, then fetch the files from the URLs in
   `files`. See [REST API](#rest-api) for curl examples.
4. To chain jobs, pass `job:<id>/<file>` in place of an upload.

### 11. If the screen freezes or a job gets stuck

1. Wait up to a minute: with the raised GPU timeout, Windows recovers the display by itself.
2. If it doesn't come back, hold the power button. On the next start, run
   `Get-WinEvent -FilterHashtable @{LogName='System'; Id=4101} -MaxEvents 5` in PowerShell; event 4101
   means the display driver timed out.
3. Start the studio again. The job that was running is marked failed; delete it.
4. Queue it again smaller (lower **Megapixels** or **Long side**, fewer frames) and note what crashed.
5. A job that seems stuck but the screen is fine: check Task Manager for `sd-cli.exe` using the GPU. If
   nothing is running, **Cancel** the job and queue it again.

### 12. Update engines or re-check models

1. `.venv\Scripts\python.exe scripts\fetch_engines.py` pulls the newest Windows builds of sd.cpp,
   llama.cpp and whisper.cpp. Stop the studio first. A new sd.cpp build can change flags, so run one small
   image afterwards.
2. `.venv\Scripts\python.exe scripts\download_models.py --list` shows any missing or partial model files;
   running it without `--list` fetches them.

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

## Running on the RX 580: findings

What testing on this machine turned up, and what was changed because of it.

### Image generation froze the display

**Short version.** The RX 580 freezes the whole PC after roughly 30-50 seconds of continuous full GPU
load. Short bursts are fine. The studio now rests the GPU between sampling steps and decode tiles, and
images generate without crashing. A 544 px Z-Image takes about 2 minutes.

**The crashes, in order** (all hard freezes that needed the power button; Windows logged event 41 with
BugcheckCode 0, so no bluescreen and no crash dump):

| # | Job | Settings | What happened |
| --- | --- | --- | --- |
| 1 | Qwen-Image-Fast, 16:9 | defaults | froze; event 4101 "display driver stopped responding and recovered", then a hang |
| 2 | Z-Image, 0.3 MP, 8 steps | + VAE tiling, 60 s GPU timeout | froze, no 4101 |
| 3 | same | + VAE on CPU | "device lost on Vulkan0" ~27 s into step 1, driver reset (4101), PC hung 2 min later |
| 4 | Z-Image, 256 px, 2 steps | + flash attention, direct conv, prefetch off; text encoder on CPU | **worked** (~4 s of GPU compute) |
| 5 | Z-Image, 0.3 MP, 8 steps | same | steps 1-4 fine at 7.5 s each, froze during step 5 (~46 s of load) |
| 6 | same | + pausing the process every 3 s | no effect (steps still 7.5 s apart); stopped by hand before step 5 |
| 7 | same | + rest between steps, as long as the step took | **worked**: all 8 steps, no driver reset (and again on a second run) |
| 8 | same | + VAE back on the GPU, tiled, rests between tiles | **worked**: decode 17 s instead of 49-88 s on the CPU |

**What it means.** Identical steps ran fine four times and the fifth froze, so it isn't a particular
piece of GPU code failing. Failing only after sustained load points to the card itself: most likely
heat, or clocks that become unstable as it warms up. The display driver usually can't recover from that
and the whole PC locks up. Smaller models wouldn't help: the weights are already 4-bit, and a smaller
model only shortens each step.

**A wrong turn worth knowing about.** Crashes 1 and 2 seemed to happen "at 99.9%, after sampling", which
pointed at the final VAE decode. That number was really sd.cpp's *weight-loading* bar, which the progress
parser counted as job progress. The freezes happened during sampling. The parser now ignores loading
bars ("MB/s" lines).

**What the studio does now** (`studio/engines/sdcpp.py`, `studio/engines/common.py`, `studio/config.py`):

- **Rests the GPU between steps and tiles.** Whenever a sampling step or a VAE tile finishes and at
  least 4 s of GPU work have built up since the last rest, the engine is suspended for as long as that
  work took: about 50% duty. Sampling steps (7.5 s for Z-Image at 544 px) rest after every step; short
  decode tiles (~0.5 s) are grouped. It has to happen at those boundaries: sd.cpp queues a whole step's
  GPU work at once, so pausing mid-step does nothing (crash 6). Settings: `STUDIO_GPU_REST` (ratio,
  default `1.0`, `0` = off) and `STUDIO_GPU_MIN_WORK` (seconds, default `4`). It roughly doubles
  sampling time.
- **Turns off sd.cpp's automatic fast paths**: `--flash-attn off --conv-direct off --disable-prefetch`
  (`STUDIO_SD_SAFE=0` turns them back on). They weren't proven to be the cause, but the run that first
  worked had them off, and the speed was the same afterwards.
- **Text encoder on the CPU** (`--backend te=cpu`): it runs once per job, and the diffusion model gets
  the VRAM to itself.
- **VAE on the GPU, always tiled**, so each tile is a short GPU job the throttle can rest between (an
  untiled decode would be one long one). `STUDIO_VAE_DEVICE=cpu` moves it to the CPU (no GPU load,
  untiled there, but ~50 s instead of ~15 s at 544 px).
- **Writes each engine's output to `engine.log`** in the job folder, forced to disk line by line, so a
  crash leaves a record of where it stopped. Rests show up there as `[studio] GPU rest ...` lines.

**The limit that's left.** A single sampling step can't be split. Z-Image at 0.3 MP is 7.5 s a step. If a
bigger model or image makes one step run longer than about 30 s, it may still freeze. Qwen-Image at
1 MP and the Wan 14B video model are the ones at risk. Test those small first.

**GPU timeout (still set, as a safety net).** Windows' default is 2 seconds. It's raised to 60, in an
administrator PowerShell, then a restart:

```powershell
reg add "HKLM\System\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f; reg add "HKLM\System\CurrentControlSet\Control\GraphicsDrivers" /v TdrDdiDelay /t REG_DWORD /d 60 /f
```

Delete the two values to go back to the default.

**Options not taken (for now).** Plugging the monitor into the motherboard's video port (Intel HD P4600)
would stop the RX 580 driving the display, and a GPU fault might then not take the desktop with it. That
needs a BIOS setting. Lowering the card's power limit or clocks in AMD Software (Performance → Tuning)
would attack the heat directly, without the BIOS, and would let the rests be shorter.

### Chatterbox runs on the CPU

- DirectML can't run the `GroupQueryAttention` nodes in Chatterbox's language model ("The parameter is
  incorrect"), so the GPU path doesn't work at all.
- The fp16 language model is replaced by the 4-bit one (`language_model_q4.onnx`, ~350 MB), which runs
  fine on the CPU.
- Building the decoder's ONNX session takes 4-5 minutes on either device, even with graph optimisation
  off. So all Chatterbox sessions are CPU-only, built once in a background thread when the server starts,
  and kept until it stops. They use RAM, not VRAM. Set `STUDIO_WARMUP=0` to skip the warm-up (the tests
  do).
- Tested: English and Hindi voice-overs read back correctly through Whisper. A 7 s Hindi line took 62 s
  once warm. Brand names get mangled ("Grundfos" came out as "Grenfaux"/"Grunfo"), so spell them
  phonetically in the script.

### Translation quality

- Urdu to English: accurate. 15 s the first time (includes loading the model onto the GPU), ~3 s after.
- English to Urdu: usable but flawed. Scripts leak in ("Grundfos" partly in Telugu, a Devanagari letter in
  "دباو"), and some words are transliterated or wrong ("tap" became "ٹپ"). This is the small 4-bit 4B
  model; TranslateGemma 4B at Q8_0 or the 12B model should do better. Check Urdu output before using it.

### Transcription

An 8 s English clip came back word for word in 33 s with large-v3-turbo on the CPU.

### Downloads

- Hugging Face's Xet transfer failed repeatedly on this connection ("CAS Client Error") and stalled.
  `download_models.py` retries, and after two Xet failures switches to plain HTTP (`HF_HUB_DISABLE_XET=1`).
  Plain HTTP ran at ~7 MB/s versus ~0.5 MB/s through Xet.
- All 49 files (plus 55 Kokoro voices) are downloaded.

### Tests must never start an engine

Once every model was on disk, `test_tool_not_ready_is_400` stopped being skipped in practice: FLUX Krea
was "ready", so the test queued a real job and started `sd-cli.exe` on the GPU. Three were left running
before it was caught. The test now forces the tool to "not ready" and fails if an engine is called. Any
new test that submits a job must stub the tool's `run`.

### Where things stand

| Verified on this machine | Not yet run on the GPU |
| --- | --- |
| Transcribe, Translate, Chatterbox, Z-Image (544 px, ~2 min) | the other image tools, every edit and video tool, Kokoro |

Music needs the acestep.cpp engine installed first (see Setup).

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

They stub the engines, so they need no weights or GPU. Among other things they check that the
TranslateGemma prompt is byte-identical to what Google's chat template renders, that outpainting keeps the
original pixels, the job lifecycle (done, error, cancel), file uploads and `job:` references, the API key,
and that the MCP endpoint lists its tools.
