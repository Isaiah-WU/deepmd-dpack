# Handoff — how to verify this skill is mentor-grade

This skill builds a DeePMD-kit **offline installer** (`.sh`) with conda
`constructor`, then proves it installs on an air-gapped machine. Your mentor's
bar: **every run succeeds, on any model** (`pass_rate = 100%`, variance = 0).

## What changed (and why it matters for "every run succeeds")
| Fix | Why |
| --- | --- |
| `build.sh` now defaults to the **bundled** `assets/` recipe | Old script required an external repo dir → failed on clean machines → model improvised. Now self-contained. |
| Added repo-root `LICENSE` | `construct.yaml` references `../LICENSE`; it was missing, so even the bundled recipe couldn't build. |
| `build.sh` hardened (`set -euo pipefail`, arg + construct.yaml checks, manifest with abs path + sha256) | Deterministic success/fail signal instead of a bare `ls`. |
| `construct.yaml` parameterized + RC-label warning + `freeze.sh` | Floating specs were the "sometimes fails" root cause. Now pinnable; freeze captures exact versions. |
| New `scripts/verify_offline.sh` | The missing acceptance test: installs the `.sh` with the network cut off and asserts `dp`/`lmp`/version. |
| `SKILL.md` is orchestration-only; raw `constructor` moved to `references/notes.md`; removed `#ler` junk | One path, not two → less model variance. |
| New `evals/` harness | Turns "run it many times, all pass" into a measurable pass_rate + variance. |

### Round 2 (mentor: GPU variant + commit-keyed packaging)
| Fix | Why |
| --- | --- |
| `build.sh` exports `CONDA_OVERRIDE_CUDA`/`CONDA_OVERRIDE_GLIBC=2.28` + libmamba when `--cuda` set | A GPU-less build node otherwise can't resolve the CUDA variant (missing `__cuda`). Matches upstream installer CI. |
| `build.sh --split N` | GPU `.sh` is multi-GB > GitHub's 2GiB asset limit; split + `cat` reassembly. |
| `verify_offline.sh` GPU mode (auto from `*cuda*` filename) | Adds `nvidia-smi`, TF GPU visibility, and a `jit_compile=True` XLA op that proves the libdevice-hack works. Must run on a GPU node. |
| New `scripts/build_pkg_from_commit.sh` (Stage 1) + `construct.yaml` `DEEPMD_LOCAL_CHANNEL`/`DEEPMD_BUILD` + `build.sh --from-commit-channel` (Stage 2) | Commit packaging: no per-commit conda package exists, so build the commit from source into a local channel first, then constructor bundles that exact build. |

### Round 3 (mentor: one-line install + nightly + multi-CUDA)
| Fix | Why |
| --- | --- |
| `dpack` package manager + `install.sh` bootstrap | One-line install like dp1s/pixi: `dpack install dp` (online, auto-detect CUDA, download split parts → reassemble → sha256 → install) or `--file` (offline). Installs to user dir, no root. |
| `.github/workflows/nightly.yml` | PyTorch-style daily CI building CPU + CUDA variants → Release. |
| Version-aware CUDA guard in `build.sh` + libdevice fix (pin `py_5` + `cuda-nvvm` + post_install symlink self-heal) | Resolved the TF-backend libdevice JIT failure; guard fails fast when a requested `cudaXXX` build doesn't exist on conda-forge. |
| Multi-CUDA resolved | conda-forge ships one `cudaXXX` build per deepmd release; the 3.2.0b0 `cuda129` build covers the whole CUDA 12.x + 13.0 driver line via NVIDIA minor-version compatibility. 12.8/13.1 were never published upstream. Full evidence in `references/verification-log.md`. |

## The acceptance flow (run in this order)

### 1. On a Linux build machine WITH internet
```bash
cd deepmd-dpack
bash scripts/build.sh --version 3.1.3            # CPU
bash scripts/verify_offline.sh dist/*.sh 3.1.3   # proves offline install works
bash scripts/freeze.sh dist/*.sh                 # capture exact pins -> dist/*.lock.txt
```
Then paste the key versions from the freeze output into `assets/construct.yaml`.

### 1b. GPU variant (build anywhere; VERIFY on a GPU node)
```bash
bash scripts/build.sh --version 3.1.3 --cuda 12.9          # add --split 3 to ship via GitHub
# on a Bohrium GPU node with an NVIDIA driver:
bash scripts/verify_offline.sh dist/deepmd-kit-3.1.3-cuda129-Linux-x86_64.sh 3.1.3
```

### 1c. Commit-keyed build (Mode B — two stages, Linux + Docker)
```bash
# list configs once, then pick one matching your cuda+python:
bash scripts/build_pkg_from_commit.sh --commit <sha> --cuda 12.9 \
     --config linux_64_cuda_compiler_version12.9mpimpichpython3.11.____cpython
bash scripts/build.sh --from-commit-channel ./local-channel
bash scripts/verify_offline.sh dist/*.sh
```

### 2. Deterministic stability (Layer 1)
```bash
RUNS=3 VERSION=3.1.3 bash evals/run_build_verify.sh
```
Want: `pass_rate: 3/3 = 100%` and `build_variance: ZERO`.

### 3. Agent stability (Layer 2 — the mentor's actual test)
```bash
RUNS=5 AGENT_CMD='claude -p' bash evals/run_agent_benchmark.sh evals/prompts.md
# repeat with a second model / Codex to prove model-agnostic
```
Want: `agent pass_rate = 100%` on ≥2 models.

## Sign-off checklist (show the mentor)
- [ ] Layer 1: pass_rate 100%, build_variance ZERO
- [ ] Layer 2: pass_rate 100% across ≥2 models, ≥5 runs each
- [ ] versions pinned in `construct.yaml` (from `freeze.sh`); not building off `deepmd-kit_rc`
- [ ] `dist/*.lock.txt` committed as the reproducibility record

## Verification status
The full pipeline has been **run end-to-end on Bohrium** (Tesla V100, driver
CUDA 13.0): dpack online + offline install, CPU + GPU builds, PyTorch and
TensorFlow backends, dp train → freeze → lammps, across ubuntu22.04 and
ubuntu24.04 images, plus the multi-CUDA (12.6 + 12.9) GPU checks. Detailed
records, commands, and findings are in `references/verification-log.md`.

## Known limits / honest gaps
- Builds are `Linux-x86_64` only; the constructor build + offline verify must run
  on Linux (a build can run on a GPU-less node, but GPU verify needs a GPU host).
- CUDA offline-verify needs a GPU host with an NVIDIA driver; on a CPU-only box
  the GPU smoke tests fail by design — run them on a Bohrium GPU node.
- GLIBC ≥ 2.17 is required on any target machine.
- **Commit mode (Stage 1) needs validation on Linux against the live feedstock.**
  `build_pkg_from_commit.sh` drives `conda-forge/deepmd-kit-feedstock`: it rewrites
  the recipe `source:` to a git source at the commit, sets the version label, and
  builds a `.ci_support` variant via `CI=1 python build-locally.py <config>`
  (Docker). Verified against the live feedstock: source is a YAML list with a
  lammps second source (preserved), configs are `cuda_compiler_version{12.9,None}`
  × `python{3.10,3.11,3.12}`, and build-locally.py takes a positional config. The
  residual risk is that a git-source build of an arbitrary modern commit may need
  recipe dep pins to match — confirm on first run; the script fails loudly if the
  source rewrite doesn't produce a `git_rev`. CUDA builds need Docker, tens of
  minutes to >1h.
- The COMPILER cuda in the `.ci_support` config name is distinct from the RUNTIME
  `--cuda` pinned into the installer; keep them consistent (both 12.9 here).
- Pin a specific `--feedstock-ref` for reproducible commit builds rather than `main`.
- Strongly consider asking jinzhe/njzjz to publish per-commit conda packages
  upstream — that collapses all of Stage 1 to "point constructor at a channel".
