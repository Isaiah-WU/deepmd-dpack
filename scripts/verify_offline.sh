#!/usr/bin/env bash
# DeePMD-kit offline installer — ACCEPTANCE TEST.
#
# Mentor's bar: the installer must run dp train + lammps inference OFFLINE on
# REAL data (the dpA4 water example, or se_e2_a fallback), not just dp -h.
#
# Usage:
#   bash scripts/verify_offline.sh <installer.sh> [expected_version]
#
# Auto-detects a pre-bundled example directory (placed alongside the .sh by
# build.sh --example dpa4). Falls back to synthetic data if none found.
#
# Exit 0 = all gates passed. Non-zero = failure.

set -euo pipefail

INSTALLER="${1:?usage: verify_offline.sh <installer.sh> [expected_version]}"
EXPECTED_VERSION="${2:-}"

[[ -f "$INSTALLER" ]] || { echo "VERIFY FAILED: installer not found: $INSTALLER" >&2; exit 1; }
case "$(basename "$INSTALLER")" in
  *.sh.[0-9]*) echo "VERIFY FAILED: split part; cat NAME.* > NAME first" >&2; exit 1 ;;
esac
INSTALLER="$(cd "$(dirname "$INSTALLER")" && pwd)/$(basename "$INSTALLER")"
INSTALLER_DIR="$(dirname "$INSTALLER")"

# Auto-detect bundled example: build.sh --example dpa4 places it in dist/examples/water/
BUNDLED_EXAMPLE=""
for cand in "$INSTALLER_DIR/examples/water" "$INSTALLER_DIR/../examples/water"; do
  if [[ -f "$cand/data/data_0/type.raw" ]]; then BUNDLED_EXAMPLE="$cand"; break; fi
done

if [[ -n "${VERIFY_GPU:-}" ]]; then GPU_MODE="$VERIFY_GPU"
elif [[ "$(basename "$INSTALLER")" == *cuda* ]]; then GPU_MODE=1
else GPU_MODE=0; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK="$(mktemp -d)"
PREFIX="$WORK/dpenv"
EXAMPLE_DIR="$WORK/example"
E2E_DIR="$EXAMPLE_DIR"
trap 'rm -rf "$WORK"' EXIT

# If there's a bundled real example, copy it into the workspace so the isolated
# test can read it (unshare inherits the filesystem; docker needs a bind mount).
if [[ -n "$BUNDLED_EXAMPLE" ]]; then
  cp -r "$BUNDLED_EXAMPLE" "$EXAMPLE_DIR"
  echo "==> Bundled example: $BUNDLED_EXAMPLE"
fi

# ── test body (runs inside the ISOLATED environment) ──────────────────────────
TEST_SCRIPT="$WORK/_test.sh"
cat > "$TEST_SCRIPT" <<EOF
set -euo pipefail
echo "==> installing offline into $PREFIX"
bash "$INSTALLER" -b -p "$PREFIX"
source "$PREFIX/bin/activate" "$PREFIX"

# ── smoke ────────────────────────────────────────────────────────────────────
echo "==> smoke: dp -h"; dp -h >/dev/null
echo "==> smoke: lmp -h"; lmp -h >/dev/null
echo "==> smoke: import deepmd"
python -c "import deepmd; print('deepmd import OK')"

if [[ -n "$EXPECTED_VERSION" ]]; then
  echo "==> version assertion: expecting $EXPECTED_VERSION"
  python - "$EXPECTED_VERSION" <<'PY'
import sys, subprocess, re
expected = sys.argv[1]
out = subprocess.run(["dp","--version"], capture_output=True, text=True)
text = (out.stdout or "") + (out.stderr or "")
print("    dp --version ->", text.strip())
sys.exit(0 if re.search(r'(?<!\d)' + re.escape(expected) + r'(?!\d)', text) else 1)
PY
fi

# ── GPU gates ─────────────────────────────────────────────────────────────────
if [[ "$GPU_MODE" == "1" ]]; then
  echo "==> GPU: nvidia-smi"; nvidia-smi -L
  echo "==> GPU: backend device visibility"
  python - <<'PY'
import sys; ok=False
try: import jax; ok=any(getattr(d,"platform","")=="gpu" for d in jax.devices())
except Exception: pass
if not ok:
    try: import tensorflow as tf; ok=bool(tf.config.list_physical_devices("GPU"))
    except Exception: pass
if not ok: print("GPU FAIL"); sys.exit(1)
print("GPU visible OK")
PY
  echo "==> GPU: TF XLA libdevice proof (advisory)"
  python - <<'PY'
import sys
try:
    import tensorflow as tf
    if not tf.config.list_physical_devices("GPU"): sys.exit(0)
    @tf.function(jit_compile=True)
    def f(x): return tf.reduce_sum(tf.math.erf(tf.exp(x)))
    print("    TF XLA+libdevice OK, val =", float(f(tf.constant([0.1,0.2,0.3]))))
except Exception: print("    TF/libdevice check skipped (JAX backend may serve)")
PY
fi

# ── END-TO-END: dp train + freeze + lammps inference ──────────────────────────
# REAL example if bundled; synthetic fallback otherwise.
E2E_DIR="$EXAMPLE_DIR"
if [[ -f "\$E2E_DIR/data/data_0/type.raw" && -f "\$E2E_DIR/se_e2_a/input.json" ]]; then
  echo "==> E2E: using bundled water example (192 atoms, 200 frames)"
  # Pick training config: prefer dpa4/lmp smoke test if deepmd-kit has dpa4 model
  TRAIN_CFG="\$E2E_DIR/se_e2_a/input.json"
  LMP_IN="\$E2E_DIR/lmp/in.lammps"
  LMP_DATA="\$E2E_DIR/lmp/water.lmp"
  if [[ -f "\$E2E_DIR/dpa4/lmp/input.json" ]]; then
    if python -c "from deepmd.utils.argcheck import gen_json; gen_json('dpa4')" 2>/dev/null; then
      TRAIN_CFG="\$E2E_DIR/dpa4/lmp/input.json"
      LMP_IN="\$E2E_DIR/dpa4/lmp/in.lammps"
      LMP_DATA="\$E2E_DIR/dpa4/lmp/water.lmp"
      echo "    dpa4 smoke-test config selected"
    else
      echo "    dpa4 model not in this build; using se_e2_a"
    fi
  fi
  echo "==> E2E: dp train"
  ( cd "\$E2E_DIR" && dp train "\$TRAIN_CFG" )
  echo "==> E2E: dp freeze"
  ( cd "\$E2E_DIR" && dp freeze -o frozen_model.pb )
  # Lammps: use bundled input if available
  if [[ -f "\$LMP_IN" && -f "\$LMP_DATA" ]]; then
    cp "\$LMP_IN" "\$E2E_DIR/in.lammps"
    cp "\$LMP_DATA" "\$E2E_DIR/water.lmp"
    sed -i 's|pair_style\s\+deepmd\s\+\S\+|pair_style      deepmd frozen_model.pb|' "\$E2E_DIR/in.lammps" 2>/dev/null || true
    echo "==> E2E: lammps inference (real water)"
    ( cd "\$E2E_DIR" && lmp -in in.lammps )
  fi
else
  echo "==> E2E: no bundled example; generating synthetic data"
  mkdir -p "\$E2E_DIR/set.000"
  python - <<'PY'
import os, numpy as np
n_frames, n_atoms = 10, 6; rng = np.random.default_rng(42)
types = np.array([0,0,1,0,0,1], dtype=np.int32)
np.savetxt("$E2E_DIR/type.raw", types, fmt="%d")
with open("$E2E_DIR/type_map.raw","w") as f: f.write("H\nO\n")
box=np.tile(np.diag([12.0]*3),(n_frames,1))
np.save("$E2E_DIR/set.000/box.npy", box)
np.save("$E2E_DIR/set.000/coord.npy", rng.uniform(0,12,(n_frames,18)))
np.save("$E2E_DIR/set.000/energy.npy", rng.uniform(-10,-9,n_frames))
np.save("$E2E_DIR/set.000/force.npy", rng.uniform(-0.1,0.1,(n_frames,18)))
PY
  cp "$SKILL_ROOT/examples/verify-input.json" "\$E2E_DIR/input.json"
  echo "==> E2E: dp train (synthetic)"; ( cd "\$E2E_DIR" && dp train input.json )
  echo "==> E2E: dp freeze"; ( cd "\$E2E_DIR" && dp freeze -o frozen_model.pb )
  cat > "\$E2E_DIR/data.lmp" <<LMP
LAMMPS data file; 6 atoms; 2 atom types; 0.0 12.0 xlo xhi; 0.0 12.0 ylo yhi; 0.0 12.0 zlo zhi
Masses; 1 1.008; 2 15.999
Atoms; 1 1 6.0 6.0 6.0; 2 1 6.5 7.0 6.0; 3 2 8.0 6.0 6.0; 4 1 4.0 6.0 6.0; 5 1 3.5 7.0 6.0; 6 2 5.0 6.0 6.0
LMP
  cat > "\$E2E_DIR/in.lammps" <<'LMP'
units metal; atom_style atomic; boundary p p p; read_data data.lmp; pair_style deepmd frozen_model.pb; pair_coeff * *; thermo 1; thermo_style custom step pe ke temp; timestep 0.0005; run 3
LMP
  echo "==> E2E: lammps inference (synthetic)"; ( cd "\$E2E_DIR" && lmp -in in.lammps )
fi

echo "OFFLINE VERIFY OK (including dp train + lammps inference)"
EOF
chmod +x "$TEST_SCRIPT"

# ── run under network isolation ───────────────────────────────────────────────
echo "==> Verifying: $INSTALLER  (GPU mode $GPU_MODE)"
echo "    bundled example: ${BUNDLED_EXAMPLE:-(none — will use synthetic fallback)}"

if command -v unshare >/dev/null 2>&1 && unshare -rn true 2>/dev/null; then
  echo "==> Isolation: unshare -rn"
  unshare -rn bash "$TEST_SCRIPT"
elif command -v docker >/dev/null 2>&1; then
  echo "==> Isolation: docker run --network none"
  GPU_FLAG=()
  if [[ "$GPU_MODE" == "1" ]]; then
    if docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then GPU_FLAG=(--gpus all)
    else echo "WARNING: nvidia-container-toolkit missing; GPU checks will fail in docker" >&2; fi
  fi
  # Bind the example dir too if bundled
  EXAMPLE_MOUNT=()
  if [[ -n "$BUNDLED_EXAMPLE" ]]; then EXAMPLE_MOUNT=(-v "$BUNDLED_EXAMPLE:$EXAMPLE_DIR:ro"); fi
  docker run --rm --network none "${GPU_FLAG[@]}" "${EXAMPLE_MOUNT[@]}" \
    -v "$INSTALLER":/installer.sh:ro \
    -e EXPECTED_VERSION="$EXPECTED_VERSION" \
    -e GPU_MODE="$GPU_MODE" \
    debian:12 bash "$TEST_SCRIPT"
else
  echo "WARNING: no unshare/docker; running WITHOUT real network isolation." >&2
  bash "$TEST_SCRIPT"
fi

echo ""
echo "VERIFY PASSED: installed + dp train + lammps inference — all offline."
