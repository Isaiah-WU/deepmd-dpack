---
name: deepmd-offline-installer
description: >
  Build a DeePMD-kit offline installer (.sh) locally using conda constructor.
  Produces a self-contained installer that installs deepmd-kit, lammps and all
  dependencies on machines without internet access.
  USE WHEN the user wants to build, make, or create a DeePMD-kit offline
  installer package locally, package deepmd-kit for offline install, or
  reproduce the installer build outside of CI.
compatibility: Requires conda. Internet access needed at build time. Builds CPU or CUDA variants.
license: LGPL-3.0-or-later
metadata:
  author: Isaiah-WU
  version: '1.0'
---

# DeePMD-kit Offline Installer (local build)

Build a self-contained `.sh` offline installer for DeePMD-kit using conda
`constructor`, locally instead of on CI.

## Quick Start

One-click build with the bundled script:

```bash
bash scripts/build.sh <recipe_dir> <version> [cuda_version]
bash scripts/build.sh /path/to/deepmd-kit-installer/deepmd-kit 3.1.3
```

Or run manually:

```bash
conda install constructor -y
cd <recipe_dir>
export VERSION=3.1.3
export CUDA_VERSION=
constructor .
```

## Agent responsibilities

1. Confirm conda is available (`conda --version`).
2. Confirm `constructor` is installed; if not, install it.
3. Confirm the recipe directory (with `construct.yaml`) is available.
4. Collect build parameters: target version, and CPU or CUDA variant.
5. Run constructor and confirm the `.sh` installer is produced.
6. Report the output file name and size.

## Workflow

### Step 1: Install constructor

```bash
conda install constructor -y
constructor --version
```

### Step 2: Get the recipe

The recipe lives in the deepmd-kit-installer repo, folder `deepmd-kit/`,
containing `construct.yaml`, `pre_install.sh`, `post_install.sh`.

```bash
cd /path/to/deepmd-kit-installer/deepmd-kit
ls
```

### Step 3: Set build parameters

```bash
export VERSION=3.1.3
export CUDA_VERSION=
```

### Step 4: Build the installer

```bash
constructor .
```

### Step 5: Verify output

```bash
ls -lh *.sh
```

## Key parameters

| Variable | Meaning | Example |
| --- | --- | --- |
| VERSION | deepmd-kit version | 3.1.3 |
| CUDA_VERSION | empty for CPU; string for CUDA | "" or 12.1 |

## Agent checklist

- [ ] conda available
- [ ] constructor installed
- [ ] recipe directory with construct.yaml present
- [ ] VERSION and CUDA_VERSION set
- [ ] constructor completes without errors
- [ ] `.sh` installer produced and size reported

#ler
