#!/usr/bin/env bash
# Stage-2 build script for the DeePMD-kit offline installer (conda constructor).
#
# Design goal (per mentor): the agent NEVER writes build commands itself. It only
# calls this script with parameters. Everything error-prone is frozen here so a
# weak model produces the same result as a strong one.
#
# Two ways to select WHAT deepmd-kit goes in the installer:
#   A. By RELEASED VERSION (default): pulls a pre-built conda package by version.
#   B. By GIT COMMIT: first run scripts/build_pkg_from_commit.sh to build the
#      commit into a LOCAL conda channel, then pass --from-commit-channel <dir>
#      here. constructor then bundles that exact commit build.
#
# Usage:
#   bash scripts/build.sh [options]
#
# Options (ALL optional; zero-arg run = CPU build of the default version):
#   -v, --version <ver>          deepmd-kit version          (default: $VERSION or 3.1.3)
#   -c, --cuda <ver>             CUDA version, e.g. 12.9;
#                                empty/omitted = CPU build    (default: $CUDA_VERSION or "")
#   --torch-version <ver>        pin PyTorch version in the installer (default: none = no torch)
#   --backend <name>             ML backends to bundle: all (default), tensorflow, pytorch, jax
#   --glibc <ver>                target system GLIBC version; also sets CONDA_OVERRIDE_GLIBC
#   --example <name>            download this deepmd-kit example alongside the installer (dpa4, se_e2_a)
#                               so verify_offline.sh can run real training+inference offline
#   --from-commit-channel <dir>  local channel from build_pkg_from_commit.sh;
#                                reads <dir>/COMMIT_BUILD.env to pin version+build+cuda+python
#   -r, --recipe-dir <dir>       dir containing construct.yaml (default: bundled assets/)
#   -o, --output-dir <dir>       where to write the installer  (default: ./dist)
#   --split <N>                  split each produced .sh into N parts (GitHub 2GiB cap;
#                                GPU installers are multi-GB). Reassemble with `cat`.
#   -h, --help                   show this help
#
# Recommended CUDA version is 12.9 (matches upstream deepmd-kit-recipes/installer CI).
# The recipe (construct.yaml + pre/post_install.sh + LICENSE) is bundled in assets/
# and used BY DEFAULT, so a release build is self-contained on a clean machine.

set -euo pipefail

# --- locate the skill root so the bundled recipe is found from any cwd --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_RECIPE="$SKILL_ROOT/assets"

# --- defaults -----------------------------------------------------------------
VERSION="${VERSION:-3.1.3}"
CUDA_VERSION="${CUDA_VERSION:-}"
RECIPE_DIR="$DEFAULT_RECIPE"
OUTPUT_DIR="$(pwd)/dist"
COMMIT_CHANNEL=""
SPLIT_PARTS=""
DEEPMD_BUILD="${DEEPMD_BUILD:-}"
DEEPMD_LOCAL_CHANNEL="${DEEPMD_LOCAL_CHANNEL:-}"
DEEPMD_PY_VERSION="${DEEPMD_PY_VERSION:-}"
DEEPMD_COMMIT=""
# New: version+hardware parameterization (mentor requirement)
DP_BACKEND="${DP_BACKEND:-all}"
TORCH_VERSION="${TORCH_VERSION:-}"
TARGET_GLIBC="${TARGET_GLIBC:-}"
EXAMPLE="${EXAMPLE:-}"   # which example to bundle for offline verify (dpa4, se_e2_a, or none)

usage() { awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "${BASH_SOURCE[0]}"; }
fail() { echo "BUILD FAILED: $*" >&2; exit 1; }

# --- parse args (also accepts two positional args for backward compatibility) -
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--version)             VERSION="$2"; shift 2 ;;
    -c|--cuda)                CUDA_VERSION="$2"; shift 2 ;;
    --torch-version)          TORCH_VERSION="$2"; shift 2 ;;
    --backend)                DP_BACKEND="$2"; shift 2 ;;
    --glibc)                  TARGET_GLIBC="$2"; shift 2 ;;
    --example)                EXAMPLE="$2"; shift 2 ;;
    --from-commit-channel)    COMMIT_CHANNEL="$2"; shift 2 ;;
    -r|--recipe-dir)          RECIPE_DIR="$2"; shift 2 ;;
    -o|--output-dir)          OUTPUT_DIR="$2"; shift 2 ;;
    --split)                  SPLIT_PARTS="$2"; shift 2 ;;
    -h|--help)                usage; exit 0 ;;
    -*)                       echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
    *)                        POSITIONAL+=("$1"); shift ;;
  esac
done
# Backward compat: build.sh <recipe_dir> <version> [cuda]
if [[ ${#POSITIONAL[@]} -ge 1 ]]; then RECIPE_DIR="${POSITIONAL[0]}"; fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then VERSION="${POSITIONAL[1]}"; fi
if [[ ${#POSITIONAL[@]} -ge 3 ]]; then CUDA_VERSION="${POSITIONAL[2]}"; fi

# --- commit mode: pin version/build/cuda/python/channel from Stage-1 output ----
# Parse COMMIT_BUILD.env with a SAFE key=value reader (do NOT `source` it).
if [[ -n "$COMMIT_CHANNEL" ]]; then
  [[ -d "$COMMIT_CHANNEL" ]] || fail "commit channel dir not found: $COMMIT_CHANNEL"
  ENV_FILE="$COMMIT_CHANNEL/COMMIT_BUILD.env"
  [[ -f "$ENV_FILE" ]] || fail "no COMMIT_BUILD.env in $COMMIT_CHANNEL (run build_pkg_from_commit.sh first)"
  read_env() { grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//'; }
  DEEPMD_COMMIT="$(read_env DEEPMD_COMMIT)"
  VERSION="$(read_env DEEPMD_PKG_VERSION)"; [[ -n "$VERSION" ]] || fail "COMMIT_BUILD.env missing DEEPMD_PKG_VERSION"
  DEEPMD_BUILD="$(read_env DEEPMD_PKG_BUILD)"
  CUDA_VERSION="$(read_env DEEPMD_CUDA_VERSION)"
  DEEPMD_PY_VERSION="$(read_env DEEPMD_PY_VERSION)"
  abs_channel="$(cd "$COMMIT_CHANNEL" && pwd)"
  DEEPMD_LOCAL_CHANNEL="file://$abs_channel"
  echo "==> Commit mode: building installer for deepmd-kit commit ${DEEPMD_COMMIT:-?}"
  echo "    pinned version=$VERSION build=${DEEPMD_BUILD:-<variant>} py=${DEEPMD_PY_VERSION:-?} cuda=${CUDA_VERSION:-CPU}"
fi

export VERSION CUDA_VERSION DEEPMD_BUILD DEEPMD_LOCAL_CHANNEL DEEPMD_PY_VERSION DP_BACKEND TORCH_VERSION TARGET_GLIBC

# --- GPU build needs virtual-package overrides so a GPU-LESS build node can
#     solve+download the CUDA variant (matches upstream installer CI) ----------
if [[ -n "$CUDA_VERSION" ]]; then
  export CONDA_OVERRIDE_CUDA="$CUDA_VERSION"
  export CONDA_OVERRIDE_GLIBC="${CONDA_OVERRIDE_GLIBC:-${TARGET_GLIBC:-2.28}}"
  export CONDA_SOLVER="${CONDA_SOLVER:-libmamba}"
fi

# --- preflight checks ---------------------------------------------------------
echo "==> Preflight checks"
command -v conda >/dev/null 2>&1 || fail "conda not found on PATH. Install Miniconda/Mambaforge first."

if ! command -v constructor >/dev/null 2>&1; then
  echo "    constructor not found, installing into the current conda env..."
  conda install constructor -y >/dev/null || fail "could not install constructor"
fi
# The heavy CUDA dependency graph effectively requires the libmamba solver.
if [[ -n "$CUDA_VERSION" ]] && ! conda list -n base 2>/dev/null | grep -q conda-libmamba-solver; then
  echo "    installing conda-libmamba-solver (recommended for CUDA solves)..."
  conda install -n base conda-libmamba-solver -y >/dev/null || \
    echo "    WARNING: could not install libmamba solver; CUDA solve may be slow/fail" >&2
fi
constructor --version

[[ -d "$RECIPE_DIR" ]] || fail "recipe dir does not exist: $RECIPE_DIR"
[[ -f "$RECIPE_DIR/construct.yaml" ]] || fail "no construct.yaml in recipe dir: $RECIPE_DIR"

# construct.yaml references ../LICENSE relative to the recipe dir; make sure it exists.
if [[ ! -f "$RECIPE_DIR/../LICENSE" && ! -f "$RECIPE_DIR/LICENSE" ]]; then
  fail "LICENSE referenced by construct.yaml (license_file: ../LICENSE) is missing"
fi

# Validate --split early (GNU split; suffix length derived from part count).
if [[ -n "$SPLIT_PARTS" ]]; then
  [[ "$SPLIT_PARTS" =~ ^[0-9]+$ && "$SPLIT_PARTS" -ge 1 ]] || fail "--split must be a positive integer"
fi
if [[ -n "$CUDA_VERSION" && -z "$SPLIT_PARTS" ]]; then
  echo "    NOTE: GPU installers are often >2GiB (GitHub's per-asset limit). To ship via" >&2
  echo "          GitHub Releases, re-run with --split 3 (local verify works unsplit)." >&2
fi

mkdir -p "$OUTPUT_DIR"

echo "==> Build parameters"
echo "    recipe dir : $RECIPE_DIR"
echo "    version    : $VERSION"
echo "    variant    : ${CUDA_VERSION:+CUDA $CUDA_VERSION}${CUDA_VERSION:-CPU}"
if [[ -n "$DEEPMD_BUILD" ]];         then echo "    pinned build : $DEEPMD_BUILD"; fi
if [[ -n "$DEEPMD_PY_VERSION" ]];    then echo "    pinned python: $DEEPMD_PY_VERSION"; fi
if [[ -n "$DEEPMD_LOCAL_CHANNEL" ]]; then echo "    local channel: $DEEPMD_LOCAL_CHANNEL"; fi
if [[ -n "$CUDA_VERSION" ]];         then echo "    overrides  : CONDA_OVERRIDE_CUDA=$CONDA_OVERRIDE_CUDA CONDA_OVERRIDE_GLIBC=$CONDA_OVERRIDE_GLIBC solver=$CONDA_SOLVER"; fi
echo "    output dir : $OUTPUT_DIR"

# --- build --------------------------------------------------------------------
BUILD_LOG="$OUTPUT_DIR/build-${VERSION}.log"
echo "==> Running constructor (full log: $BUILD_LOG)"
# Run in a subshell so we don't leave the caller in the recipe dir. pipefail
# ensures a constructor failure (not tee's success) fails the build.
( cd "$RECIPE_DIR" && constructor . --output-dir "$OUTPUT_DIR" --cache-dir "${CONSTRUCTOR_CACHE_DIR:-$OUTPUT_DIR/cache}" ) 2>&1 | tee "$BUILD_LOG"

# --- post-build: confirm a CUDA build actually bundled cuda deepmd-kit --------
# The manifest's "variant" reflects INTENT; this checks the produced contents so
# a silent CPU-for-GPU artifact does not pass as a GPU build.
if [[ -n "$CUDA_VERSION" ]]; then
  if grep -qiE 'deepmd-kit-[^[:space:]]*cuda' "$BUILD_LOG"; then
    echo "    OK: build log shows a cuda deepmd-kit package was bundled."
  else
    echo "    WARNING: requested CUDA but could not confirm a cuda deepmd-kit build in the log." >&2
    echo "             Inspect $BUILD_LOG and the offline verify before trusting this as a GPU build." >&2
  fi
fi

# --- verify artifact & emit manifest -----------------------------------------
# --- download verification example (while we still have internet) ----------------
if [[ -n "$EXAMPLE" ]]; then
  EXAMPLE_DIR="$OUTPUT_DIR/examples/water"
  echo "==> Downloading deepmd-kit example: $EXAMPLE (for offline verify)"
  rm -rf "$EXAMPLE_DIR" 2>/dev/null || true
  mkdir -p "$EXAMPLE_DIR"
  if command -v git >/dev/null 2>&1; then
    git clone --filter=blob:none --no-checkout --branch master https://github.com/deepmodeling/deepmd-kit.git "$EXAMPLE_DIR/repo" 2>/dev/null || true
    if [[ -d "$EXAMPLE_DIR/repo" ]]; then
      ( cd "$EXAMPLE_DIR/repo" && git sparse-checkout init --cone && git sparse-checkout set examples/water/ && git checkout master ) || true
      if [[ -d "$EXAMPLE_DIR/repo/examples/water" ]]; then
        cp -r "$EXAMPLE_DIR/repo/examples/water/"* "$EXAMPLE_DIR/"
      fi
      rm -rf "$EXAMPLE_DIR/repo"
    fi
  fi
  if [[ -f "$EXAMPLE_DIR/se_e2_a/input.json" ]]; then
    echo "    example downloaded OK (~$(du -sh "$EXAMPLE_DIR" | cut -f1))"
  else
    echo "    WARNING: could not download example; verify will fall back to synthetic data" >&2
  fi
fi

echo "==> Collecting produced installer(s)"
shopt -s nullglob
INSTALLERS=("$OUTPUT_DIR"/*.sh)
shopt -u nullglob
[[ ${#INSTALLERS[@]} -gt 0 ]] || fail "constructor produced no .sh installer in $OUTPUT_DIR"

MANIFEST="$OUTPUT_DIR/manifest-${VERSION}.txt"
{
  echo "deepmd-kit offline installer build manifest"
  echo "version       : $VERSION"
  echo "variant       : ${CUDA_VERSION:+cuda$CUDA_VERSION}${CUDA_VERSION:-cpu}"
  echo "pinned_build  : ${DEEPMD_BUILD:-(by variant selector)}"
  echo "pinned_python : ${DEEPMD_PY_VERSION:-(unpinned)}"
  echo "source_commit : ${DEEPMD_COMMIT:-(released version)}"
  echo "recipe_dir    : $RECIPE_DIR"
  echo "constructor   : $(constructor --version 2>&1)"
  echo "build_log     : $BUILD_LOG"
  echo "artifacts     :"
  for f in "${INSTALLERS[@]}"; do
    abs="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
    size="$(du -h "$f" | cut -f1)"
    sha="$(sha256sum "$f" | cut -d' ' -f1)"
    echo "  - path   : $abs"
    echo "    size   : $size"
    echo "    sha256 : $sha"
  done
} | tee "$MANIFEST"

# --- optional split for GitHub (GPU installers exceed the 2GiB asset limit) ---
if [[ -n "$SPLIT_PARTS" ]]; then
  suffix_len="${#SPLIT_PARTS}"   # so >=10 parts get wide-enough numeric suffixes
  for f in "${INSTALLERS[@]}"; do
    echo "==> Splitting $(basename "$f") into $SPLIT_PARTS parts (GNU split; original kept)"
    ( cd "$(dirname "$f")" && split -a "$suffix_len" -d -n "$SPLIT_PARTS" "$(basename "$f")" "$(basename "$f")." )
    echo "    reassemble on target: cat $(basename "$f").* > $(basename "$f")"
  done
fi

echo ""
echo "BUILD OK. Manifest: $MANIFEST"
echo "Next: verify it really installs OFFLINE with:"
echo "  bash $SKILL_ROOT/scripts/verify_offline.sh \"${INSTALLERS[0]}\" \"$VERSION\""
if [[ -n "$CUDA_VERSION" ]]; then
  echo "  (GPU build: run verify on a node WITH an NVIDIA GPU + driver)"
fi
exit 0
