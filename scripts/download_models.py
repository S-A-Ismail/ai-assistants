"""Download every model in studio/models.py into STUDIO_MODELS_DIR (default N:\\Models).

Safe to re-run: finished files are skipped and interrupted ones resume.

    python scripts/download_models.py            # everything
    python scripts/download_models.py --list     # what's present / missing
    python scripts/download_models.py --only whisper translate   # groups or keys
"""

import argparse
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

from studio.config import MODELS_DIR  # noqa: E402
from studio.models import FILES, VOICE_DIRS, File  # noqa: E402

STAGING = MODELS_DIR / ".staging"


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def selected(only: list[str]) -> list[File]:
    if not only:
        return FILES
    return [f for f in FILES if f.key in only or any(f.group == o or f.group.startswith(o + "/") for o in only)]


def fetch(f: File, attempts: int = 5) -> None:
    """Download with retries. Xet transfers have failed mid-file here ("error decoding response
    body"); after two such failures, fall back to plain HTTP for the rest of the run."""
    for n in range(1, attempts + 1):
        try:
            return _fetch(f)
        except Exception as e:
            if n == attempts:
                raise
            if "CAS" in str(e) or "Xet" in str(e) or "xet" in str(e):
                if n >= 2 and os.environ.get("HF_HUB_DISABLE_XET") != "1":
                    print("   switching to plain HTTP downloads (HF_HUB_DISABLE_XET=1)", flush=True)
                    os.environ["HF_HUB_DISABLE_XET"] = "1"
                    import huggingface_hub.constants as c

                    c.HF_HUB_DISABLE_XET = True
            wait = 15 * n
            print(f"   attempt {n} failed ({str(e)[:120]}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def _fetch(f: File) -> None:
    f.local.parent.mkdir(parents=True, exist_ok=True)
    if f.url:
        tmp = f.local.with_suffix(f.local.suffix + ".part")
        urllib.request.urlretrieve(f.url, tmp)
        os.replace(tmp, f.local)
        return
    # Download into a per-repo staging tree (so huggingface_hub can resume),
    # then move the finished file to its flat location on the same drive.
    staged = hf_hub_download(f.repo, f.path, local_dir=STAGING / f.repo.replace("/", "__"))
    os.replace(staged, f.local)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[], help="groups (e.g. image, video/wan21) or file keys")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    files = selected(args.only)
    todo = [f for f in files if not f.local.exists()]
    need = sum(f.size for f in todo)

    if args.list:
        for f in files:
            mark = "ok " if f.local.exists() else "-- "
            print(f"{mark}{gb(f.size):>9}  {f.key:<32} {f.local}")
        print(f"\nmissing {len(todo)} files, {gb(need)}")
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(MODELS_DIR).free
    print(f"{len(todo)} files to fetch, {gb(need)}; {gb(free)} free on {MODELS_DIR.anchor}", flush=True)
    if need > free:
        print("Not enough free space.", file=sys.stderr)
        return 1

    failed = []
    done = 0
    for i, f in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {f.key} ({gb(f.size)}) <- {f.repo or f.url}", flush=True)
        try:
            fetch(f)
        except Exception as e:  # keep going; report at the end
            print(f"   FAILED: {e}", flush=True)
            failed.append(f.key)
            continue
        done += f.size
        dt = max(time.time() - t0, 1e-6)
        print(f"   done in {dt:.0f}s ({f.size / dt / 1e6:.1f} MB/s); {gb(done)} of {gb(need)}", flush=True)

    if not args.only or any(o.startswith("voice") for o in args.only):
        for repo, pattern, group in VOICE_DIRS:
            dest = MODELS_DIR / group
            print(f"voices: {repo}/{pattern} -> {dest}", flush=True)
            snapshot_download(repo, allow_patterns=[pattern], local_dir=STAGING / repo.replace("/", "__"))
            dest.mkdir(parents=True, exist_ok=True)
            for p in (STAGING / repo.replace("/", "__")).glob(pattern):
                shutil.copy2(p, dest / p.name)

    if failed:
        print(f"\nFailed: {', '.join(failed)}. Re-run to retry.", flush=True)
        return 1
    print("\nAll downloads complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
