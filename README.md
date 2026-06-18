# DeePMD-kit Offline Installer Skill

[中文版](#中文版) | [English Version](#english-version)

---

## 中文版

### 这是什么

一个 [Claude Code Skill](https://docs.anthropic.com/en/docs/claude-code/skills)，用于**在一台联网 Linux 机器上**构建 DeePMD-kit 的离线安装包（自解压 `.sh` 文件），然后将该安装包拿到**任意断网机器上**一键安装，安装后即可运行 `dp train`（训练势函数）和 `lmp`（LAMMPS 分子动力学推理）。

### 解决的问题

科学计算集群通常断开互联网。传统的安装方式（`conda install` / `pip install`）在断网环境不可用。这个 skill 把 deepmd-kit + LAMMPS + TensorFlow/JAX/PyTorch + MPI + 全部数百个依赖打包成一个自包含文件，通过 U 盘、内网等任意方式搬运即可。

### 整体架构

```
                        ┌─────────────────────────────┐
                        │  assets/construct.yaml      │
                        │  (conda 包清单，Jinja2 模板)  │
                        │  定义：装什么包、什么版本      │
                        └──────────────┬──────────────┘
                                       │
┌──────────────┐              ┌────────▼────────┐
│  用户/Agent  │ ──传参──▶    │ scripts/build.sh │  ← ★ 核心
│  指定版本、   │              │ 编排整个构建流程   │
│  后端、硬件   │              └────────┬────────┘
└──────────────┘                       │
                          ┌────────────▼────────────┐
                          │  conda constructor      │
                          │  下载包 → 校验 → 打包    │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  dist/xxx.sh (1.4+ GB)  │
                          │  + manifest (sha256)     │
                          └────────────┬────────────┘
                                       │ 搬运到断网机器
                          ┌────────────▼────────────┐
                          │  scripts/verify_offline │  ← ★ 验收
                          │  .sh                    │
                          │  断网安装 → dp train    │
                          │  → dp freeze → lammps   │
                          └─────────────────────────┘
```

### 仓库结构

```
deepmd-offline-installer-skill/
├── README.md                     ← 本文件
├── SKILL.md                      ← Claude Code Skill 定义（Agent 阅读）
├── LICENSE                       ← LGPL-3.0
├── HANDOFF.md                    ← 交接/验收清单
│
├── scripts/                      ← 可执行脚本（固化的，Agent 不写命令只调脚本）
│   ├── build.sh                  ★ 构建离线安装包（唯一入口）
│   ├── verify_offline.sh         ★ 断网验收（安装→训练→推理）
│   ├── build_pkg_from_commit.sh   从 Git commit 编译 deepmd-kit（高级）
│   └── freeze.sh                  锁定包版本（可复现构建）
│
├── assets/                       ← conda constructor 配方
│   ├── construct.yaml            ★ Jinja2 模板：定义安装包内容
│   ├── pre_install.sh            安装前提示
│   └── post_install.sh           安装后提示
│
├── examples/                     ← 验证用数据
│   └── verify-input.json         最小化训练配置（v2 格式）
│
├── evals/                        ← 质量评测
│   ├── README.md
│   ├── prompts.md                测试 prompt
│   ├── assertions.md             客观断言
│   ├── run_build_verify.sh       脚本层稳定性
│   └── run_agent_benchmark.sh    Agent 层稳定性
│
└── references/
    └── notes.md                  详细技术文档
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

### 验证流程（verify_offline.sh 做了什么）

```
1. 切网（unshare -rn 或 docker --network none）
2. 安装 .sh 到临时目录
3. 冒烟测试：dp -h / lmp -h / import deepmd / 版本校验
4. GPU（自动识别）：nvidia-smi / TF-GPU / JAX-GPU / XLA libdevice 证明
5. 端到端：
   ├── 有 bundled example → 用真实 192 原子水分子训练 + lammps 推理
   └── 无 bundled example → 生成合成数据，训练 10 步 + lammps 推理
6. 全部通过 → exit 0
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

```
                        ┌─────────────────────────────┐
                        │  assets/construct.yaml      │
                        │  (Jinja2 template — what     │
                        │   goes into the installer)   │
                        └──────────────┬──────────────┘
                                       │
┌──────────────┐              ┌────────▼────────┐
│  User/Agent  │ ──params──▶  │ scripts/build.sh │  ← ★ Core
│  version,    │              │ orchestrates build │
│  backend,    │              └────────┬────────┘
│  hardware    │                       │
└──────────────┘          ┌────────────▼────────────┐
                          │  conda constructor      │
                          │  download → verify →    │
                          │  package into .sh       │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  dist/xxx.sh (1.4+ GB)  │
                          │  + manifest (sha256)     │
                          └────────────┬────────────┘
                                       │ ship to air-gapped node
                          ┌────────────▼────────────┐
                          │  scripts/verify_offline │  ← ★ Acceptance
                          │  .sh                    │
                          │  offline install →      │
                          │  dp train → freeze →    │
                          │  lammps inference       │
                          └─────────────────────────┘
```

### Repository Layout

```
deepmd-offline-installer-skill/
├── README.md                     ← This file
├── SKILL.md                      ← Claude Code Skill definition
├── LICENSE                       ← LGPL-3.0
├── HANDOFF.md                    ← Handoff and sign-off checklist
│
├── scripts/                      ← Frozen scripts (agent never writes commands)
│   ├── build.sh                  ★ Build the offline installer
│   ├── verify_offline.sh         ★ Offline acceptance test
│   ├── build_pkg_from_commit.sh   Build from a git commit (advanced)
│   └── freeze.sh                  Pin package versions (reproducible builds)
│
├── assets/                       ← conda constructor recipe
│   ├── construct.yaml            ★ Jinja2 template defining installer contents
│   ├── pre_install.sh            Runtime pre-install notes
│   └── post_install.sh           Post-install guidance
│
├── examples/                     ← Verification data
│   └── verify-input.json         Minimal training config (v2 format)
│
├── evals/                        ← Quality benchmarks
│   ├── README.md
│   ├── prompts.md
│   ├── assertions.md
│   ├── run_build_verify.sh
│   └── run_agent_benchmark.sh
│
└── references/
    └── notes.md                  Detailed technical docs
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

```
1. Isolate network (unshare -rn or docker --network none)
2. Install .sh into throwaway prefix
3. Smoke: dp -h / lmp -h / import deepmd / version check
4. GPU (auto): nvidia-smi / TF-GPU / JAX-GPU / XLA libdevice proof
5. End-to-end:
   ├── Bundled example → real 192-atom water training + lammps inference
   └── No example → synthetic data, 10-step training + lammps inference
6. All gates pass → exit 0
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
