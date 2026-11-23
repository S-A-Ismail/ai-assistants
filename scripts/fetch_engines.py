"""Download the official Windows builds of the inference engines into runtime/engines.

    python scripts/fetch_engines.py          # all
    python scripts/fetch_engines.py sdcpp    # one

Each engine goes in its own folder with a VERSION file naming the release it came from.
"""

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.config import ENGINES_DIR  # noqa: E402

# name -> (GitHub repo, predicate picking the asset)
ENGINES = {
    # Image + video. Vulkan build: the only GPU path an RX 580 has on Windows.
    "sdcpp": ("leejet/stable-diffusion.cpp", lambda n: n.endswith("-bin-win-vulkan-x64.zip")),
    # Translation (TranslateGemma). Vulkan build.
    "llamacpp": ("ggml-org/llama.cpp", lambda n: n.endswith("-bin-win-vulkan-x64.zip")),
    # Transcription. No official Windows Vulkan build exists, so this is the CPU build with OpenBLAS.
    "whispercpp": ("ggml-org/whisper.cpp", lambda n: n == "whisper-blas-bin-x64.zip"),
}


def latest_asset(repo: str, pick) -> tuple[str, str, str]:
    # Walk recent releases: some repos (whisper.cpp) tag a release without assets.
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases?per_page=10",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "ai-studio"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        releases = json.load(r)
    for rel in releases:
        for a in rel.get("assets", []):
            if pick(a["name"]):
                return rel["tag_name"], a["name"], a["browser_download_url"]
    raise RuntimeError(f"no matching asset in the last 10 releases of {repo}")


def fetch(name: str) -> None:
    repo, pick = ENGINES[name]
    tag, asset, url = latest_asset(repo, pick)
    dest = ENGINES_DIR / name
    version = dest / "VERSION"
    if version.exists() and version.read_text().split()[0] == tag:
        print(f"{name}: {tag} already installed")
        return
    print(f"{name}: downloading {asset} ({tag})", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r:
        data = r.read()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            # Flatten: some zips nest everything under Release/ or build/bin/.
            target = dest / Path(member.filename).name
            target.write_bytes(z.read(member))
    version.write_text(f"{tag} {asset}\n")
    print(f"{name}: installed to {dest}")


def main() -> int:
    names = sys.argv[1:] or list(ENGINES)
    for n in names:
        fetch(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
