# DeePMD-kit Offline Installer Skill

[中文版](#中文版) | [English Version](#english-version)

---

## 中文版

### 这是什么

DeepModeling 社区的分发体系——**对标 PyTorch 的自动化构建 + 离线安装 + 包管理器**。

**有网机器**
```bash
dpack install dp                # 自动检测 GPU，下载 + 安装
dpack install dp --cuda 12.8    # 指定 CUDA 版本
```

**断网机器（超算）**
```bash
dpack install dp --file ./xxx.sh   # 指向本地包，一行安装，无需 CUDA Toolkit
```

**后台**：GitHub Actions 每天自动构建 CPU + 多个 CUDA 版本 → 上传 Release → manifest 自动更新。

四个组件：

| 组件 | 文件 | 对标 |
|------|------|------|
| 包管理器 | `dpack` + `install.sh` | pixi / brew / dp1s |
| 自动构建 | `.github/workflows/nightly.yml` | PyTorch nightly |
| 离线包构建 | `scripts/build.sh` + `assets/construct.yaml` | conda constructor |
| 断网验收 | `scripts/verify_offline.sh` | dp train + lammps |

### 解决的问题

科学计算集群通常断网，`conda install` / `pip install` 不可用。这个项目把 deepmd-kit + LAMMPS + TF/JAX/PyTorch + MPI + 全部依赖打包成自包含文件，通过 U 盘搬运即可。以后逐步加入 dpgen、采样、蒸馏等 DeepModeling 工具——一个 `dpack` 装所有。

### 工作流程

| 阶段 | 在哪做 | 做什么 | 产出 |
|------|--------|--------|------|
| **① 自动化** | GitHub Actions | nightly CI 每天构建多 CUDA 版本 | 上传 Release |
| **② 安装** | 用户机器 | `dpack install dp` 或 `bash xxx.sh` | deepmd-kit 环境 |
| **③ 验收** | 任意机器 | `verify_offline.sh` 切网 → 安装 → dp train → lammps | ✅ 或 ❌ |

> 💡 **一句话**：CI 自动打包 → 用户 `dpack install dp` → 断网也能装。

### 仓库结构

```
deepmd-offline-installer-skill/
│
├─ ⭐ dpack                包管理器入口 — curl | bash 一键安装
├─ 📁 .github/workflows/
│  └─    nightly.yml       自动化 nightly 构建（对飙 PyTorch CI）
│
├─ 📄 README.md            中英文使用文档
├─ 📄 SKILL.md              Agent 操作手册（Agent 阅读，只做编排）
├─ 📄 LICENSE              LGPL-3.0
│
├─ 📁 scripts/             固化脚本（Agent 不可即兴写命令）
│  ├─ ⭐ build.sh          构建入口 — 打包离线安装器
│  ├─ ⭐ verify_offline.sh 验收入口 — 断网全流程验证
│  ├─    build_pkg_        Git commit → conda 包（高级）
│  │     from_commit.sh
│  └─    freeze.sh         锁定版本 → 可复现构建
│
├─ 📁 assets/              constructor 配方
│  ├─ ⭐ construct.yaml    Jinja2 模板 → 定义装什么
│  ├─ ⭐ manifest.json     工具清单 → dpack 读取下载链接
│  ├─    version.txt       版本号单一来源
│  ├─    pre_install.sh    安装前提示
│  └─    post_install.sh   安装后提示
│
├─ 📁 examples/            验证数据
│  └─    verify-input.json 最小训练配置（v2 格式）
│
├─ 📁 evals/               质量评测
│  ├─    README.md
│  ├─    run_build_verify  脚本稳定性
│  └─    run_agent_benchmark Agent 稳定性
│
└─ 📁 references/
   └─    notes.md           排查手册 & GPU / commit 流程
```

### 一键安装（推荐）

```bash
# 1. 装 dpack（装到用户目录，不需要 root）
curl -fsSL https://raw.githubusercontent.com/Isaiah-WU/deepmd-offline-installer-skill/main/install.sh | bash

# 2a. 在线安装：自动检测 GPU/CUDA → 下载 → 安装
dpack install dp

# 2b. 离线安装（超算无网）：指向已拷贝到本地的 .sh 包
#     分片包也行——给主文件名，dpack 自动 cat 合并 .0 .1 .2
dpack install dp --file ./deepmd-kit-3.2.0b0-cuda129-Linux-x86_64.sh

# 查看已装工具
dpack list
```

> **为超算设计**：dpack 装在用户目录（无需 root），离线 `--file` 模式不需要网络。对标 dp1s 的一行安装、pixi 的体验。以后 `dpack install dpgen`、`dpack upgrade dp` 逐步支持。

### 构建自己的离线包

如果你需要自建离线包（而非用 dpack 下载预编译的），见下方「构建」章节。

### 前置要求

| 条件 | 说明 |
|------|------|
| 操作系统 | Linux x86_64（构建机和目标机均需） |
| conda | 任意 Miniconda / Mambaforge |
| 构建时 | **必须联网**（constructor 从 conda-forge 下载包） |
| 构建内存 | 推荐 ≥ 8 GB |
| 目标机 | 无需联网；GLIBC ≥ 2.17；GPU 版需 NVIDIA 驱动 |
| 磁盘（构建机） | 约 5 GB 临时空间 + 1.5 GB（CPU）/ 3+ GB（GPU）输出 |

### 快速开始

```bash
# 1. 获取代码
git clone https://github.com/Isaiah-WU/deepmd-offline-installer-skill.git
cd deepmd-offline-installer-skill

# 2. 基础版本号在 assets/version.txt（单一来源），构建时自动读取
cat assets/version.txt   # → 3.2.0b0

# 3. 构建 CPU 版（约 10-15 分钟）
bash scripts/build.sh

# 4. 产物在 dist/cpu/ 下（按 CUDA 版本分目录，对标 PyTorch /whl/{cuXXX}/）
ls -lh dist/cpu/

# 5. 断网验收
bash scripts/verify_offline.sh dist/cpu/*.sh $(cat assets/version.txt)

# 6. 拿到断网机器上安装
bash dist/cpu/*.sh -b -p /opt/deepmd
source /opt/deepmd/bin/activate /opt/deepmd
dp --version
```

### 选择你的版本（对标 PyTorch get-started）

第一步：`nvidia-smi` 看你的 CUDA 驱动版本，选对应的包下载。如果没 GPU，选 CPU。

| 你的机器 | 下载 |
|----------|------|
| CPU（无 GPU） | `bash dist/cpu/*.sh -b -p /opt/deepmd` |
| CUDA 12.6 驱动 | `bash dist/cuda126/*.sh -b -p /opt/deepmd` |
| CUDA 12.8 驱动 | `bash dist/cuda128/*.sh -b -p /opt/deepmd` |
| CUDA 13.1 驱动 | `bash dist/cuda131/*.sh -b -p /opt/deepmd` |

第二步：选的包是一行命令安装——不需要手动装 CUDA Toolkit 或设 CUDA_HOME。

### 常用命令

#### 构建不同版本

```bash
# 产物自动按 CUDA 分目录：dist/cpu/  dist/cuda128/  dist/cuda131/
bash scripts/build.sh --cuda 12.6
bash scripts/build.sh --cuda 12.8
bash scripts/build.sh --cuda 13.1

# CPU 版
bash scripts/build.sh

# 指定后端（TF / PyTorch / JAX / 全都要）
bash scripts/build.sh --backend pytorch --torch-version ">=2.5"

# 打包时附带真实训练数据（验证时用真实数据跑 dp train）
bash scripts/build.sh --example dpa4
```

#### 验证

```bash
# 基础验收（合成数据训练 + lammps 推理）
bash scripts/verify_offline.sh dist/*.sh <期望版本号>

# GPU 验收（需 GPU 节点，自动检测 cuda 文件名）
bash scripts/verify_offline.sh dist/deepmd-kit-3.1.3-cuda129-Linux-x86_64.sh 3.1.3
```

### 已验证的组合

以下组合已在 Bohrium 平台通过端到端离线验证（`unshare -rn` 切网 → 安装 → `dp train` → `dp freeze` → `lammps` 推理）：

| deepmd-kit | CPU/GPU | 后端 | 训练数据 | 训练时间 | LAMMPS | 状态 |
|---|---|---|---|---|---|---|
| 3.1.3 | CPU | TF + JAX | 合成 6 原子 | 3.85s | 123ms | ✅ |
| 3.2.0b0 | CPU | TF + JAX + PyTorch | **dpa4 真实 192 原子** | — | — | ✅ |
| 3.2.0b0 | GPU | TF + JAX + PyTorch | 合成 6 原子 | 2.65s | 12ms | ✅ |
| 3.2.0b0 | GPU | TF + JAX + PyTorch | dpa4 真实 192 原子 | — | — | ✅ |

> GPU 节点：4× Tesla V100-SXM2-16GB，CUDA 12.9。LAMMPS 推理 GPU 比 CPU 快 **10 倍**。
>
> `--cuda` 参数支持任意值，默认 CUDA 12.9，通过 NVIDIA minor-version compatibility 兼容多数 12.x 驱动。

### 完整参数列表

| 标志 | 含义 | 默认值 |
|------|------|--------|
| `--version` | deepmd-kit 版本 | `3.1.3` |
| `--cuda` | CUDA 版本，空 = CPU | `""` |
| `--backend` | ML 后端：`all` / `tensorflow` / `pytorch` / `jax` | `all` |
| `--torch-version` | PyTorch 版本 pin | 无（不装 PyTorch） |
| `--glibc` | 目标系统 GLIBC 版本 | `2.28`（仅 GPU） |
| `--example` | 附带下载的 example：`dpa4` / `se_e2_a` | 无 |
| `--split` | 切割输出文件为 N 份（GitHub 2GiB 限制） | 不切割 |
| `--from-commit-channel` | 从本地 channel 打包（Mode B，见下方） | 无 |
| `--output-dir` | 输出目录 | `./dist` |

环境变量：`TF_VERSION`、`LAMMPS_VERSION`、`VERSION`、`CUDA_VERSION`。

### 两种打包模式

#### Mode A：发行版打包（默认，常用）

直接从 conda-forge 已发布的预编译包打包。适合 v3.1.3、v3.2.0b0 等有 conda 包的版本。

```bash
bash scripts/build.sh --version 3.1.3 [--cuda 12.9] [--backend ...]
```

#### Mode B：Git Commit 打包（高级）

当 conda-forge 没有你要的版本时（例如某个未发布 commit），先从源码编译 deepmd-kit，再打包。需要 Docker。

```bash
# Stage 1: 从 commit 编译到本地 channel（需 Docker，30 分钟-1 小时）
bash scripts/build_pkg_from_commit.sh --commit <sha> --cuda 12.9 --config <配置名>

# Stage 2: 打包
bash scripts/build.sh --from-commit-channel ./local-channel
```

> 注：v3.2.0b0 已有 conda 包，用 Mode A 即可跑 dpa4。Mode B 仅在需要真正未发布的 commit 时使用。

### 验收流程

验证脚本执行 6 个步骤，任一失败即退出：

| 步骤 | 操作 | 说明 |
|------|------|------|
| ① | 切网 | `unshare -rn` 断开网络 |
| ② | 安装 | `.sh` 安装到临时目录 |
| ③ | 冒烟 | `dp -h` `lmp -h` `import deepmd` 版本号校验 |
| ③-GPU | GPU 检测 | `nvidia-smi` · TF/JAX GPU · XLA libdevice |
| ④ | 训练 | 有 example → 192 原子真实 DFT 数据；无 → 6 原子合成数据 |
| ⑤ | 冻结 | `dp freeze` 生成可部署模型 |
| ⑥ | 推理 | `lmp` 加载模型跑 MD |

全部通过输出 `VERIFY PASSED` 并 `exit 0`。

### 安装后的能力

安装离线包后，用户获得：

| 组件 | 用途 |
|------|------|
| `dp` | 训练 / 冻结 / 测试神经网络势函数 |
| `lmp` | LAMMPS 分子动力学推理 |
| `dp_ipi` | i-PI 接口 |
| `mpirun` / `horovod` | 多节点并行训练 |
| Python `deepmd` / `dpdata` / `pylammps` | 脚本化调用 |

### 注意事项

- 构建时**必须联网**，安装时**无需联网**
- GPU 构建产物可能超 2 GB，用 `--split 3` 切割后用 `cat` 合并
- 目标机 GLIBC 需 ≥ 2.17，GPU 版需兼容的 NVIDIA 驱动
- Bohrium 等共享平台注意构造缓存位置（可能触发配额）

### License

LGPL-3.0-or-later

---

## English Version

### What This Is

A distribution system for the DeepModeling ecosystem — **PyTorch-style automated nightly builds + offline installer + package manager**.

**Online machine**
```bash
dpack install dp                # auto-detect GPU, download + install
dpack install dp --cuda 12.8    # explicit CUDA version
```

**Air-gapped machine (HPC)**
```bash
dpack install dp --file ./xxx.sh   # install from a local package, no network
```

**Infrastructure**: GitHub Actions builds CPU + multiple CUDA variants daily → uploads to Release → manifest auto-updated.

Four components:

| Component | File | Modeled After |
|-----------|------|---------------|
| Package manager | `dpack` + `install.sh` | pixi / brew / dp1s |
| Nightly CI | `.github/workflows/nightly.yml` | PyTorch nightly |
| Offline builder | `scripts/build.sh` + `assets/construct.yaml` | conda constructor |
| Acceptance test | `scripts/verify_offline.sh` | dp train + lammps |

### Problem Solved

HPC clusters are disconnected from the internet. This project packages deepmd-kit + LAMMPS + TF/JAX/PyTorch + MPI + all dependencies into a single self-extracting archive. Future: `dpack install dpgen`, sampling tools, distillation tools — one package manager for the entire DeepModeling ecosystem.

### How It Works

| Phase | Where | What | Output |
|-------|-------|------|--------|
| **① Build** | GitHub Actions | Nightly CI builds multi-CUDA variants | Release .sh files |
| **② Install** | Any Linux | `dpack install dp` or `bash xxx.sh` | deepmd-kit environment |
| **③ Verify** | Any Linux | `verify_offline.sh` cut net → install → train → lammps | ✅ or ❌ |

> 💡 **TL;DR**: CI auto-builds → user runs `dpack install dp` → works offline.

### Repository Layout

```
deepmd-offline-installer-skill/
│
├─ ⭐ dpack              Package manager entry — curl | bash install
├─ 📁 .github/workflows/
│  └─    nightly.yml     Nightly CI (PyTorch-style automated builds)
│
├─ 📄 README.md          User guide (CN + EN)
├─ 📄 SKILL.md           Agent manual (orchestration only)
├─ 📄 LICENSE            LGPL-3.0
│
├─ 📁 scripts/           Frozen scripts (never hand-write commands)
│  ├─ ⭐ build.sh        Build entry point
│  ├─ ⭐ verify_offline.sh Acceptance entry point
│  ├─    build_pkg_      Git commit → conda package (advanced)
│  │     from_commit.sh
│  └─    freeze.sh       Pin versions → reproducible builds
│
├─ 📁 assets/            constructor recipe
│  ├─ ⭐ construct.yaml  Jinja2 template → bill of materials
│  ├─ ⭐ manifest.json   Tool catalog → dpack reads download URLs
│  ├─    version.txt     Single source of truth for base version
│  ├─    pre_install.sh  Pre-install notes
│  └─    post_install.sh Post-install guidance
│
├─ 📁 examples/          Verification data
│  └─    verify-input.json Minimal training config (v2 format)
│
├─ 📁 evals/             Quality benchmarks
│  ├─    README.md
│  ├─    run_build_verify  Script-level stability
│  └─    run_agent_benchmark Agent-level stability
│
└─ 📁 references/
   └─    notes.md        Deep-dive & troubleshooting
```

### One-Line Install (Recommended)

```bash
# 1. Install dpack (into your user dir — no root)
curl -fsSL https://raw.githubusercontent.com/Isaiah-WU/deepmd-offline-installer-skill/main/install.sh | bash

# 2a. Online: auto-detect GPU/CUDA -> download -> install
dpack install dp

# 2b. Offline (air-gapped HPC): install from a local .sh you copied over
#     Split parts work too — give the main name, dpack reassembles .0 .1 .2
dpack install dp --file ./deepmd-kit-3.2.0b0-cuda129-Linux-x86_64.sh

# List installed tools
dpack list
```

> **Built for HPC**: dpack installs to your user dir (no root), and `--file` mode needs no network. Like dp1s's one-line install + pixi UX. Future: `dpack install dpgen`, `dpack upgrade dp`, etc.

### Build Your Own Offline Packages

If you need to build offline installers yourself (instead of using dpack's pre-built releases), see the Build section below.

### Prerequisites

| Requirement | Detail |
|-------------|--------|
| OS | Linux x86_64 (both build and target) |
| conda | Any Miniconda / Mambaforge |
| Build time | **Internet required** (constructor downloads from conda-forge) |
| Build RAM | ≥ 8 GB recommended |
| Target machine | No internet needed; GLIBC ≥ 2.17; GPU variant needs NVIDIA driver |
| Disk (build) | ~5 GB temp + 1.5 GB (CPU) / 3+ GB (GPU) output |

### Quick Start

```bash
# 1. Clone
git clone https://github.com/Isaiah-WU/deepmd-offline-installer-skill.git
cd deepmd-offline-installer-skill

# 2. Build CPU variant (~10-15 min)
bash scripts/build.sh --version 3.1.3

# 3. Verify offline (~2 min)
bash scripts/verify_offline.sh dist/deepmd-kit-3.1.3-cpu-Linux-x86_64.sh 3.1.3

# 4. Ship to air-gapped machine and install
bash dist/deepmd-kit-3.1.3-cpu-Linux-x86_64.sh -b -p /opt/deepmd
source /opt/deepmd/bin/activate /opt/deepmd
dp --version    # → DeePMD-kit v3.1.3
```

### Common Usage

```bash
# GPU variant
bash scripts/build.sh --version 3.1.3 --cuda 12.9

# v3.2.0b0 + PyTorch (for dpa4 model support)
bash scripts/build.sh --version 3.2.0b0 --backend pytorch --torch-version ">=2.5"

# Specify backend + target GLIBC
bash scripts/build.sh --version 3.1.3 --backend all --torch-version ">=2.5" --glibc 2.28

# Bundle real training data for verification
bash scripts/build.sh --version 3.1.3 --example dpa4

# Verify (auto-detects GPU from filename)
bash scripts/verify_offline.sh dist/*.sh <expected_version>
```

### Verified Combinations

All end-to-end verified on Bohrium (`unshare -rn` isolation → install → `dp train` → `dp freeze` → `lammps` inference):

| deepmd-kit | CPU/GPU | Backends | Training Data | Train Time | LAMMPS | Status |
|---|---|---|---|---|---|---|
| 3.1.3 | CPU | TF + JAX | Synthetic 6-atom | 3.85s | 123ms | ✅ |
| 3.2.0b0 | CPU | TF + JAX + PyTorch | **dpa4 real 192-atom** | — | — | ✅ |
| 3.2.0b0 | GPU | TF + JAX + PyTorch | Synthetic 6-atom | 2.65s | 12ms | ✅ |
| 3.2.0b0 | GPU | TF + JAX + PyTorch | dpa4 real 192-atom | — | — | ✅ |

> GPU node: 4× Tesla V100-SXM2-16GB, CUDA 12.9. LAMMPS inference: GPU 10× faster than CPU.
>
> `--cuda` accepts any value. Default CUDA 12.9 is compatible with most 12.x drivers via NVIDIA minor-version compatibility.

### Full Parameter Reference

| Flag | Meaning | Default |
|------|---------|---------|
| `--version` | deepmd-kit version | `3.1.3` |
| `--cuda` | CUDA version (empty = CPU) | `""` |
| `--backend` | ML backends: `all` / `tensorflow` / `pytorch` / `jax` | `all` |
| `--torch-version` | Pin PyTorch version | none (no PyTorch) |
| `--glibc` | Target system GLIBC version | `2.28` (for GPU) |
| `--example` | Download example data: `dpa4` / `se_e2_a` | none |
| `--split` | Split output into N parts (GitHub 2GiB limit) | off |
| `--from-commit-channel` | Pack from local channel (Mode B, see below) | none |
| `--output-dir` | Output directory | `./dist` |

### Two Packaging Modes

#### Mode A — Released version (default, recommended)

Packages a pre-built conda package from conda-forge. Works for v3.1.3, v3.2.0b0, etc.

```bash
bash scripts/build.sh --version 3.1.3 [--cuda 12.9] [--backend ...]
```

#### Mode B — Git commit (advanced)

For commits without a published conda package: builds deepmd-kit from source first, then packages. Requires Docker, 30–60 min.

```bash
# Stage 1: build commit to local channel
bash scripts/build_pkg_from_commit.sh --commit <sha> --cuda 12.9 --config <config_stem>

# Stage 2: package
bash scripts/build.sh --from-commit-channel ./local-channel
```

> Note: v3.2.0b0 already has a conda package — use Mode A for dpa4. Mode B is only needed for genuinely unreleased commits.

### Acceptance Test

The verification script runs 6 gates. Any failure exits immediately:

| Step | Action | Detail |
|------|--------|--------|
| ① | Isolate | `unshare -rn` cuts network |
| ② | Install | `.sh` to temp prefix |
| ③ | Smoke | `dp -h` `lmp -h` `import deepmd` version check |
| ③-GPU | GPU check | `nvidia-smi` · TF/JAX GPU · XLA libdevice |
| ④ | Train | Bundled example → 192-atom DFT data; no example → 6-atom synthetic |
| ⑤ | Freeze | `dp freeze` → deployable model |
| ⑥ | Inference | `lmp` loads frozen model, runs MD |

All pass → `VERIFY PASSED`, exit 0.

### Post-Install Capabilities

| Component | Capability |
|-----------|-----------|
| `dp` | Train / freeze / test neural network potentials |
| `lmp` | LAMMPS molecular dynamics inference |
| `dp_ipi` | i-PI interface |
| `mpirun` / `horovod` | Multi-node parallel training |
| Python `deepmd` / `dpdata` / `pylammps` | Scripted workflows |

### Notes

- **Internet is required at build time**; none needed at install time
- GPU installers may exceed 2 GB — use `--split 3` and reassemble with `cat`
- Target machine must have GLIBC ≥ 2.17; GPU variant needs a compatible NVIDIA driver
- On shared platforms like Bohrium, watch for disk quotas on constructor cache

### License

LGPL-3.0-or-later
