# DeePMD-kit Offline Installer Skill

[中文版](#中文版) | [English Version](#english-version)

---

## 中文版

### 这是什么

一个 [Claude Code Skill](https://docs.anthropic.com/en/docs/claude-code/skills)，用于**在一台联网 Linux 机器上**构建 DeePMD-kit 的离线安装包（自解压 `.sh` 文件），然后将该安装包拿到**任意断网机器上**一键安装，安装后即可运行 `dp train`（训练势函数）和 `lmp`（LAMMPS 分子动力学推理）。

### 解决的问题

科学计算集群通常断开互联网。传统的安装方式（`conda install` / `pip install`）在断网环境不可用。这个 skill 把 deepmd-kit + LAMMPS + TensorFlow/JAX/PyTorch + MPI + 全部数百个依赖打包成一个自包含文件，通过 U 盘、内网等任意方式搬运即可。

### 整体架构

```mermaid
flowchart TD
    A["🔧 build.sh<br/>接收参数<br/>───<br/>版本 · CPU/GPU · 后端 · 硬件"] --> B["📋 construct.yaml<br/>购物清单"]
    A --> C["📦 constructor<br/>下载 + 打包"]
    A --> D["📂 example 数据<br/>下载（可选）"]
    B --> C
    C --> E["💾 dist/<br/>├── xxx.sh (1.4~3 GB)<br/>└── manifest (sha256)"]
    D --> E
    E -->|"搬运到断网机器"| F["✅ verify_offline.sh<br/>───<br/>① 切网<br/>② 安装<br/>③ 冒烟测试<br/>④ dp train<br/>⑤ dp freeze<br/>⑥ lammps 推理<br/>───<br/>exit 0 = 通过"]
```

### 仓库结构

```
deepmd-offline-installer-skill/
│
├─ 📄 README.md          中英文使用文档
├─ 📄 SKILL.md            Agent 操作手册（Agent 阅读，只做编排）
├─ 📄 LICENSE             LGPL-3.0
├─ 📄 HANDOFF.md          交接 & 验收清单
│
├─ 📁 scripts/            固化脚本（Agent 不可即兴写命令）
│  ├─ ⭐ build.sh         构建入口 — 打包离线安装器
│  ├─ ⭐ verify_offline   验收入口 — 断网全流程验证
│  ├─    build_pkg_       Git commit → conda 包（高级）
│  │     from_commit.sh
│  └─    freeze.sh        锁定版本 → 可复现构建
│
├─ 📁 assets/             constructor 配方
│  ├─ ⭐ construct.yaml   Jinja2 模板 → 定义装什么
│  ├─    pre_install.sh   安装前提示
│  └─    post_install.sh  安装后提示
│
├─ 📁 examples/           验证数据
│  └─    verify-input     最小训练配置（v2 格式）
│        .json
│
├─ 📁 evals/              质量评测
│  ├─    README / prompts / assertions
│  ├─    run_build_       脚本稳定性（反复构建 N 次）
│  │     verify.sh
│  └─    run_agent_        Agent 稳定性（跨模型 N 次）
│        benchmark.sh
│
└─ 📁 references/
   └─    notes.md          排查手册 & GPU / commit 流程
```

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

# 2. 构建 CPU 版（约 10-15 分钟）
bash scripts/build.sh --version 3.1.3

# 3. 断网验收（约 2 分钟）
bash scripts/verify_offline.sh dist/deepmd-kit-3.1.3-cpu-Linux-x86_64.sh 3.1.3

# 4. 拿到断网机器上安装
bash dist/deepmd-kit-3.1.3-cpu-Linux-x86_64.sh -b -p /opt/deepmd
source /opt/deepmd/bin/activate /opt/deepmd
dp --version    # → DeePMD-kit v3.1.3
```

### 常用命令

#### 构建不同版本

```bash
# GPU 版（CUDA 12.9；构建机无需 GPU，验证需 GPU 节点）
bash scripts/build.sh --version 3.1.3 --cuda 12.9

# v3.2.0b0 + PyTorch（用于 dpa4 模型）
bash scripts/build.sh --version 3.2.0b0 --backend pytorch --torch-version ">=2.5"

# 指定后端 + 目标 GLIBC
bash scripts/build.sh --version 3.1.3 --backend all --torch-version ">=2.5" --glibc 2.28

# 打包时附带真实训练数据（验证时用真实数据跑 dp train）
bash scripts/build.sh --version 3.1.3 --example dpa4
```

#### 验证

```bash
# 基础验收（合成数据训练 + lammps 推理）
bash scripts/verify_offline.sh dist/*.sh <期望版本号>

# GPU 验收（需 GPU 节点，自动检测 cuda 文件名）
bash scripts/verify_offline.sh dist/deepmd-kit-3.1.3-cuda129-Linux-x86_64.sh 3.1.3
```

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

### 验证流程

```mermaid
flowchart TD
    A["📥 输入：.sh 安装包 + 版本号"] --> B["🔌 ① 网络隔离<br/>unshare -rn"]
    B --> C["📦 ② 离线安装<br/>安装到临时目录"]
    C --> D["✅ ③ 冒烟测试<br/>dp -h · lmp -h · import deepmd · 版本号<br/>GPU: nvidia-smi · GPU visible · XLA"]
    D --> E{"有 bundled example?"}
    E -->|"是"| F["🧪 ④ 真实训练<br/>192 原子水分子<br/>× 200 帧 DFT 数据"]
    E -->|"否"| G["🧪 ④ 合成训练<br/>6 原子 × 10 帧<br/>随机生成"]
    F --> H["❄️ ⑤ dp freeze<br/>冻结 → .pb 模型"]
    G --> H
    H --> I["🚀 ⑥ lammps 推理<br/>加载模型 → MD 模拟"]
    I --> J["🎉 VERIFY PASSED<br/>exit 0"]
```

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

A [Claude Code Skill](https://docs.anthropic.com/en/docs/claude-code/skills) that builds a **self-contained offline installer** (`.sh` file) for DeePMD-kit on an **internet-connected Linux machine**. The resulting installer can be transferred to any **air-gapped machine** and installed with a single bash command, providing the full deepmd-kit stack including `dp train` and `lmp` inference.

### Problem Solved

HPC clusters are typically disconnected from the internet. Standard `conda install` / `pip install` workflows do not work. This skill packages deepmd-kit + LAMMPS + TensorFlow/JAX/PyTorch + MPI + hundreds of dependencies into a single self-extracting archive, transferable via USB, internal network, or any other means.

### Architecture

```mermaid
flowchart TD
    A["🔧 build.sh<br/>Receives params<br/>───<br/>version · CPU/GPU · backend · hardware"] --> B["📋 construct.yaml<br/>Bill of materials"]
    A --> C["📦 constructor<br/>Fetch + package"]
    A --> D["📂 Example data<br/>fetch (optional)"]
    B --> C
    C --> E["💾 dist/<br/>├── xxx.sh (1.4–3 GB)<br/>└── manifest (sha256)"]
    D --> E
    E -->|"Ship to air-gapped node"| F["✅ verify_offline.sh<br/>───<br/>① Isolate network<br/>② Install<br/>③ Smoke tests<br/>④ dp train<br/>⑤ dp freeze<br/>⑥ LAMMPS inference<br/>───<br/>exit 0 = passed"]
```

### Repository Layout

```
deepmd-offline-installer-skill/
│
├─ 📄 README.md          User guide (CN + EN)
├─ 📄 SKILL.md           Agent manual (orchestration only)
├─ 📄 LICENSE            LGPL-3.0
├─ 📄 HANDOFF.md         Sign-off checklist
│
├─ 📁 scripts/           Frozen scripts (never hand-write commands)
│  ├─ ⭐ build.sh        Build entry point
│  ├─ ⭐ verify_offline  Acceptance entry point
│  ├─    build_pkg_      Git commit → conda package (advanced)
│  │     from_commit.sh
│  └─    freeze.sh       Pin versions → reproducible builds
│
├─ 📁 assets/            constructor recipe
│  ├─ ⭐ construct.yaml  Jinja2 template → bill of materials
│  ├─    pre_install.sh  Pre-install notes
│  └─    post_install.sh Post-install guidance
│
├─ 📁 examples/          Verification data
│  └─    verify-input    Minimal training config (v2 format)
│        .json
│
├─ 📁 evals/             Quality benchmarks
│  ├─    README / prompts / assertions
│  ├─    run_build_      Script-level stability
│  │     verify.sh
│  └─    run_agent_      Agent-level stability
│        benchmark.sh
│
└─ 📁 references/
   └─    notes.md        Deep-dive & troubleshooting
```

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

### Verification Flow

```mermaid
flowchart TD
    A["📥 Input: .sh installer + version"] --> B["🔌 ① Network isolation<br/>unshare -rn"]
    B --> C["📦 ② Offline install<br/>Install to temp prefix"]
    C --> D["✅ ③ Smoke tests<br/>dp -h · lmp -h · import deepmd · version<br/>GPU: nvidia-smi · GPU visible · XLA"]
    D --> E{"Bundled example?"}
    E -->|"Yes"| F["🧪 ④ Real training<br/>192 atoms × 200 frames<br/>DFT data"]
    E -->|"No"| G["🧪 ④ Synthetic training<br/>6 atoms × 10 frames<br/>random data"]
    F --> H["❄️ ⑤ dp freeze<br/>Freeze → .pb model"]
    G --> H
    H --> I["🚀 ⑥ LAMMPS inference<br/>Load model → MD simulation"]
    I --> J["🎉 VERIFY PASSED<br/>exit 0"]
```

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
