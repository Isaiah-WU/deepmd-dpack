#!/usr/bin/env python3
"""Emit a per-variant manifest fragment after a build (used by nightly CI).

Each matrix build job runs this once. It finds the installer(s) it just built
under dist/<variant>/, computes the sha256 of the FULL installer (reassembling
split parts), builds the GitHub Release download URLs, and writes a small JSON
fragment to frag/<variant>.json. A later job merges all fragments into
assets/manifest.json (see merge_manifest.py).

Env / args:
  REPO       owner/repo, e.g. Isaiah-WU/deepmd-dpack            (default from git or env)
  CUDA       cuda version like "12.9", or "" for the CPU build
  VERSION    deepmd-kit version (default: read assets/version.txt)
"""
import glob
import hashlib
import json
import os
import sys

repo = os.environ.get("REPO", "Isaiah-WU/deepmd-dpack")
cuda = os.environ.get("CUDA", "").strip()
version = os.environ.get("VERSION", "").strip()
if not version:
    with open("assets/version.txt", encoding="utf-8") as fh:
        version = fh.read().strip()

# Rolling nightly channel: ONE release tag holds the latest dated build of every
# variant (PyTorch-style /whl/nightly). The deepmd version lives in the asset
# filename + manifest "version" field, NOT the tag. Override with TAG if needed.
tag = os.environ.get("TAG", "nightly")
subdir = f"cuda{cuda.replace('.', '')}" if cuda else "cpu"
base = f"https://github.com/{repo}/releases/download/{tag}"


def sha256_of(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


# Match only the final installer(s); the constructor cache holds unrelated
# *.sh scripts. Installers are named deepmd-kit-<ver>-<variant>-Linux-x86_64.sh.
parts = sorted(glob.glob(f"dist/{subdir}/deepmd-kit-*.sh.[0-9]*"))
singles = glob.glob(f"dist/{subdir}/deepmd-kit-*.sh")

# backend/note are env-driven so Mode A (cpu/cuda129) keeps tf+jax+torch while
# Mode C (cuda126/128/130, set by the verify lane) reports pytorch + its own note.
entry = {
    "type": "gpu" if cuda else "cpu",
    "cuda": cuda or None,
    "backend": os.environ.get("BACKEND", "tf+jax+torch"),
}
_note = os.environ.get("NOTE", "").strip()
if cuda:
    entry["note"] = _note or "Covers the CUDA 12.x ~ 13.0 driver line via NVIDIA minor-version compatibility."

if parts:
    entry["parts"] = [f"{base}/{os.path.basename(p)}" for p in parts]
    entry["sha256"] = sha256_of(parts)
elif singles:
    f = singles[0]
    entry["url"] = f"{base}/{os.path.basename(f)}"
    entry["sha256"] = sha256_of([f])
else:
    sys.exit(f"gen_manifest_fragment: no installer found in dist/{subdir}/")

os.makedirs("frag", exist_ok=True)
out = {"variant": subdir, "entry": entry}
with open(f"frag/{subdir}.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
