# dpack — DeepModeling 包管理器

[中文版](#中文版) | [English Version](#english-version)

> 一行命令，在任何机器（包括断网超算）装上 DeePMD-kit 全套。
> 自动识别 GPU/CUDA → 选版本 → 下载 → 安装。对标 pixi / brew / conda。

---

## 中文版

### 1 · 安装 dpack

装到用户目录，**不需要 root**：

```bash
curl -fsSL https://raw.githubusercontent.com/Isaiah-WU/deepmd-dpack/main/install.sh | bash
```

如提示 `dpack` 找不到，执行 `export PATH="$HOME/.local/bin:$PATH"` 或重开终端。

### 2 · 用 dpack 装工具

```bash
# 有网机器：自动检测 GPU/CUDA，下载对应版本并安装
dpack install dp

# 断网机器（超算）：先手动把离线包拷过去，再指向本地文件
dpack install dp --file ./deepmd-kit-3.2.0b0-20260624-a1b2c3d-cuda129-Linux-x86_64.sh

# 装完按提示激活即可
source <安装路径>/bin/activate <安装路径>
dp --version
```

> GPU 用户只装这一个包就行：它覆盖**整个 CUDA 12.x ~ 13.0 驱动**，不用对版本
> （靠 [NVIDIA 向后兼容](https://docs.nvidia.com/deploy/cuda-compatibility/)）。

### 3 · 命令一览

| 命令 | 作用 |
|------|------|
| `dpack install dp` | 在线安装：自动选版本 → 下载 → 装 |
| `dpack install dp --file <pkg.sh>` | 离线安装：从本地离线包安装 |
| `dpack install dp --prefix <dir>` | 指定安装目录 |
| `dpack list` | 查看已安装的工具 |

#### 离线安装与分片包

GPU 离线包很大（约 6 GB），超过 GitHub 单文件 2 GB 上限，所以被切成 3 片：
`...sh.0`、`...sh.1`、`...sh.2`。**你不用手动合并——`--file` 指向去掉 `.0` 的"基础文件名"，dpack 会自动把同目录下的分片拼好再安装。**

```bash
# 情况一：单个完整 .sh（CPU 包，没切片）—— 直接指它
dpack install dp --file ./deepmd-kit-3.2.0b0-20260624-8f40cf6-cpu-Linux-x86_64.sh

# 情况二：分片包（GPU 包）—— 把 3 片放同一个目录：
#   deepmd-kit-...-cuda129-Linux-x86_64.sh.0
#   deepmd-kit-...-cuda129-Linux-x86_64.sh.1
#   deepmd-kit-...-cuda129-Linux-x86_64.sh.2
# 然后 --file 指向【不带 .0 的基础名】，dpack 自动 cat 合并再装：
dpack install dp --file ./deepmd-kit-3.2.0b0-20260624-8f40cf6-cuda129-Linux-x86_64.sh
#                              ↑ 注意：结尾是 .sh，不是 .sh.0
```

> 超算场景：在有网机器从 [Release](https://github.com/Isaiah-WU/deepmd-dpack/releases) 下载 3 片 →
> U 盘拷到断网机器同一目录 → 上面那条命令一行装好（无需联网、无需 root）。
> 也可以手动合并：`cat *.sh.0 *.sh.1 *.sh.2 > full.sh && bash full.sh -b -p /opt/deepmd`。

### 4 · 支持的工具

| 工具 | 说明 | 状态 |
|------|------|------|
| `dp` | DeePMD-kit + LAMMPS + TF/JAX/PyTorch | 可用 |
| `dpgen` | 自动数据生成 | 规划中 |
| 更多 | 采样 / 蒸馏 / … | 规划中 |

目标：**一个 `dpack` 装整个 DeepModeling 生态**（`dpack install dpgen`、`dpack upgrade dp` 等逐步支持）。

### 5 · 装完能用什么

| 组件 | 用途 |
|------|------|
| `dp` | 训练 / 冻结 / 测试神经网络势函数 |
| `lmp` | LAMMPS 分子动力学推理 |
| `dp_ipi` | i-PI 接口 |
| `mpirun` / `horovod` | 多节点并行训练 |
| Python `deepmd` / `dpdata` / `pylammps` | 脚本化调用 |

### 6 · 环境要求

| 条件 | 说明 |
|------|------|
| 操作系统 | Linux x86_64 |
| 联网 | 在线安装需要；离线 `--file` 模式不需要 |
| 权限 | 无需 root（装到用户目录） |
| GLIBC | ≥ 2.17 |
| GPU 版 | 需兼容的 NVIDIA 驱动（CUDA 12.x ~ 13.0） |

### 它怎么工作（选读）

```
每天自动构建离线包  →  发布到 GitHub Release  →  dpack 按你的机器自动选版本、下载、安装
   nightly CI            （附 manifest 清单）         （或你手动下载 → --file 安装）
```

科学计算集群通常**断网**，`conda install` / `pip install` 用不了。dpack 把 deepmd-kit +
LAMMPS + TF/JAX/PyTorch + MPI + 全部依赖打包成单个自包含文件——U 盘拷过去一行命令就装好，
无需联网、无需 root、无需手动设 CUDA_HOME。

---

## 给维护者：构建与发布离线包

> 以下是**内部维护**内容——普通用户不需要看。这个 repo 除了 dpack，还附带一套离线包
> 构建脚本、自动发布的 nightly workflow，和一个 Claude Code skill。

### 仓库结构

```
deepmd-dpack/
├─ dpack                  包管理器（用户入口）
├─ install.sh             dpack 引导安装脚本（curl | bash）
│
├─ .github/workflows/
│  └─ nightly.yml         每日自动构建 + 发布（对标 PyTorch nightly）
│
├─ scripts/               固化构建/验收脚本
│  ├─ build.sh            构建离线安装包
│  ├─ verify_offline.sh   断网全流程验收（dp train + lammps）
│  ├─ gen_manifest_fragment.py / merge_manifest.py  manifest 自动生成
│  ├─ build_pkg_from_commit.sh  Git commit → conda 包（高级）
│  └─ freeze.sh           锁定版本 → 可复现构建
│
├─ assets/                constructor 配方 + 工具清单
│  ├─ construct.yaml      Jinja2 模板（定义装什么）
│  ├─ manifest.json       工具清单（dpack 读它拿下载链接，由 CI 自动更新）
│  ├─ version.txt         版本号单一来源
│  └─ pre/post_install.sh 安装前后钩子
│
├─ SKILL.md               附带的 Claude Code skill（Agent 编排构建）
├─ HANDOFF.md             开发交接 + 验收清单（内部）
├─ references/            内部文档（用户无需阅读）
│  ├─ notes.md            构建/排查参考、GPU/commit 流程
│  └─ verification-log.md 验证记录（Bohrium 实测结果与踩坑）
└─ examples/ evals/       验证数据 / 质量评测
```

### 自动发布闭环

release `v3.2.0b0` 现有 5 个变体：`cpu + cuda126 + cuda128 + cuda129 + cuda130`（`cuda126/128/130` 的 GPU 计算走 PyTorch）。

**流程：**

```
GitHub Actions（每天）               玻尔 GPU（cron）
  构建 cu126/128/130 → 传 Release      下当天的包 → GPU 验证（train/freeze/lammps）
                                       → 通过才写进 manifest.json
```

GitHub 负责**构建**，玻尔负责 **GPU 验证**；`manifest.json` 只收录验证通过的变体，dpack 据此为用户选包。

**怎么用：**
- 手动触发构建：`Actions → Nightly Build (Mode C) → Run workflow`（选 `cu126` / `cu128` / `cu130` / `all`）。
- 手动跑验证并发布（在玻尔 GPU 节点）：`export GH_PAT=<细粒度 PAT> && bash scripts/verify_and_publish.sh`。
- 自动：GitHub 每天构建；玻尔 cron 每天验证并发布。

### 手动单独构建一个包

```bash
# Mode A：cpu / cuda129（conda constructor）
bash scripts/build.sh                 # CPU
bash scripts/build.sh --cuda 12.9     # cuda129

# Mode C：cuda126 / 128 / 130（需 Linux GPU 节点）
bash scripts/build_modec.sh cu126     # 或 cu128 / cu130
```

- **Mode B**：conda-forge 没有的未发布 commit，先源码编译到本地 channel 再打包，需 Docker。详见 [references/notes.md](references/notes.md)。

### License

LGPL-3.0-or-later

---

## English Version

### 1 · Install dpack

Installs to your user directory, **no root needed**:

```bash
curl -fsSL https://raw.githubusercontent.com/Isaiah-WU/deepmd-dpack/main/install.sh | bash
```

If `dpack` isn't found, run `export PATH="$HOME/.local/bin:$PATH"` or open a new shell.

### 2 · Install tools with dpack

```bash
# Online: auto-detect GPU/CUDA, download + install the right build
dpack install dp

# Air-gapped (HPC): copy the offline package over, then point at the local file
dpack install dp --file ./deepmd-kit-3.2.0b0-20260624-a1b2c3d-cuda129-Linux-x86_64.sh

# Activate as printed, then:
source <prefix>/bin/activate <prefix>
dp --version
```

> GPU users install just one package: it covers the entire **CUDA 12.x ~ 13.0 driver line** —
> no version-matching ([NVIDIA backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)).

### 3 · Commands

| Command | What it does |
|---------|--------------|
| `dpack install dp` | Online install: auto-pick variant → download → install |
| `dpack install dp --file <pkg.sh>` | Offline install from a local package |
| `dpack install dp --prefix <dir>` | Install to a specific directory |
| `dpack list` | List installed tools |

#### Offline install & split packages

A GPU package is large (~6 GB) and exceeds GitHub's 2 GB per-file limit, so it ships as 3 parts:
`...sh.0`, `...sh.1`, `...sh.2`. **You don't merge them by hand — point `--file` at the base name
(without `.0`) and dpack reassembles the parts in that directory before installing.**

```bash
# Case 1: a single complete .sh (CPU package, not split) — point at it directly
dpack install dp --file ./deepmd-kit-3.2.0b0-20260624-8f40cf6-cpu-Linux-x86_64.sh

# Case 2: a split package (GPU) — put the 3 parts in one directory:
#   deepmd-kit-...-cuda129-Linux-x86_64.sh.0 / .sh.1 / .sh.2
# then point --file at the BASE name (no .0); dpack cats them and installs:
dpack install dp --file ./deepmd-kit-3.2.0b0-20260624-8f40cf6-cuda129-Linux-x86_64.sh
#                              ↑ note: ends in .sh, not .sh.0
```

> HPC flow: download the 3 parts from the [Release](https://github.com/Isaiah-WU/deepmd-dpack/releases)
> on a networked machine → copy them into one directory on the air-gapped machine → run the command
> above (no network, no root). Or merge manually: `cat *.sh.0 *.sh.1 *.sh.2 > full.sh && bash full.sh -b -p /opt/deepmd`.

### 4 · Supported Tools

| Tool | Description | Status |
|------|-------------|--------|
| `dp` | DeePMD-kit + LAMMPS + TF/JAX/PyTorch | available |
| `dpgen` | Automated data generation | planned |
| more | sampling / distillation / … | planned |

Goal: **one `dpack` for the whole DeepModeling ecosystem** (`dpack install dpgen`, `dpack upgrade dp`, …).

### 5 · What You Get

| Component | Purpose |
|-----------|---------|
| `dp` | train / freeze / test neural-network potentials |
| `lmp` | LAMMPS molecular dynamics inference |
| `dp_ipi` | i-PI interface |
| `mpirun` / `horovod` | multi-node parallel training |
| Python `deepmd` / `dpdata` / `pylammps` | scripting |

### 6 · Requirements

| Requirement | Detail |
|-------------|--------|
| OS | Linux x86_64 |
| Network | needed for online install; not for offline `--file` |
| Privileges | none (installs to user dir) |
| GLIBC | ≥ 2.17 |
| GPU variant | compatible NVIDIA driver (CUDA 12.x ~ 13.0) |

### How It Works (optional)

```
nightly CI builds offline packages  →  publishes to a GitHub Release  →  dpack auto-picks
   (daily)                              (with a manifest)                 the build for your
                                                                          machine, downloads,
                                                                          installs
```

HPC clusters are usually disconnected — `conda install` / `pip install` don't work. dpack packs
deepmd-kit + LAMMPS + TF/JAX/PyTorch + MPI + every dependency into one self-extracting archive:
copy it over a USB stick and one command installs it. No network, no root, no manual CUDA_HOME.

---

## For Maintainers

> Internal content — regular users can skip this. Besides dpack, this repo bundles the offline-package
> build scripts, the nightly auto-publish workflow, and a Claude Code skill.

```bash
git clone https://github.com/Isaiah-WU/deepmd-dpack.git
cd deepmd-dpack

bash scripts/build.sh                 # CPU build (~10-15 min)
bash scripts/build.sh --cuda 12.9     # GPU build (covers 12.x~13.0 drivers)

# Offline acceptance: cut network → install → dp train → freeze → lammps
bash scripts/verify_offline.sh dist/*/*.sh $(cat assets/version.txt)
```

The nightly pipeline (`nightly.yml`) runs **daily at 02:17 UTC** (and on pushes touching
`construct.yaml` / `version.txt` / `build.sh`): it builds cpu + cuda129 → creates/updates a GitHub Release →
uploads installers (GPU auto-split) → regenerates `assets/manifest.json` → commits it back. Trigger
manually via `Actions → Nightly Build & Publish → Run workflow` (`variant`: cpu / cuda129 / all).

Build flags: `--version`, `--cuda`, `--backend {all,tensorflow,pytorch,jax}`, `--torch-version`,
`--glibc`, `--example {dpa4,se_e2_a}`, `--split N`, `--from-commit-channel`. **Mode A** packs released
conda packages (default); **Mode B** builds a git commit from source into a local channel first (Docker).
GPU ships only `cuda129` — conda-forge publishes one CUDA build per deepmd release, covering the whole
12.x driver line via minor-version compatibility. Details:
[references/notes.md](references/notes.md), [references/verification-log.md](references/verification-log.md),
[HANDOFF.md](HANDOFF.md).

### License

LGPL-3.0-or-later
