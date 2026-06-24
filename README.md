# dpack — DeepModeling 包管理器

[中文版](#中文版) | [English Version](#english-version)

> 一行命令，在任何机器（包括断网超算）装上 DeePMD-kit 全套。
> 自动识别 GPU/CUDA → 选版本 → 下载 → 安装。对标 pixi / brew / conda。

---

## 中文版

### 这是什么

**dpack** 是 DeepModeling 生态的包管理器。用户不用关心 conda、CUDA、依赖——一行命令装好 DeePMD-kit + LAMMPS + TF/JAX/PyTorch + MPI：

```bash
# 1. 装 dpack（用户目录，无需 root）
curl -fsSL https://raw.githubusercontent.com/Isaiah-WU/dpack/main/install.sh | bash

# 2. 装 deepmd-kit —— 自动检测 GPU/CUDA，下载对应版本并安装
dpack install dp

# 断网机器（超算）：先手动拷贝离线包，再指向本地文件
dpack install dp --file ./deepmd-kit-3.2.0b0-cuda129-Linux-x86_64.sh

# 查看已装工具
dpack list
```

GPU 版一个包覆盖**整个 CUDA 12.x ~ 13.0 驱动**——不用对版本（靠 [NVIDIA 向后兼容](https://docs.nvidia.com/deploy/cuda-compatibility/)）。

### 它怎么工作

```
┌─────────────────┐   每天    ┌─────────────┐   读取    ┌──────────────┐
│  nightly CI 自动 │ ───────▶ │  发布到平台  │ ◀─────── │    dpack     │
│  构建多版本离线包 │  发布     │ GitHub Release│  manifest │  包管理器     │
└─────────────────┘          └─────────────┘          └──────┬───────┘
                                                              │ 自动选版本
                                                              ▼
                                              ┌───────────────────────────┐
                                              │  用户机器：检测 GPU/CUDA →   │
                                              │  下载对应包 → 一行安装        │
                                              │  （或用户手动下载 → 安装）    │
                                              └───────────────────────────┘
```

1. **构建**：`nightly.yml` 每天用 conda `constructor` 把 deepmd-kit 全套打成自包含的 `.sh` 离线安装包（CPU + 各 CUDA 版本）。
2. **发布**：自动上传到 GitHub Release，并更新 `assets/manifest.json`（版本、下载链接、sha256）。
3. **安装**：`dpack` 读 manifest，按用户机器自动选版本、下载、安装。也支持用户先手动下载再 `--file` 安装。

### 为什么需要它

科学计算集群通常**断网**，`conda install` / `pip install` 用不了。dpack 把所有依赖打包成单个自包含文件，U 盘拷过去一行命令就装好，无需联网、无需 root、无需手动设 CUDA_HOME。

以后逐步纳入 dpgen、采样、蒸馏等 DeepModeling 工具——**一个 `dpack` 装整个生态**：

```bash
dpack install dp          # 现在可用
dpack install dpgen       # 规划中
dpack upgrade dp          # 规划中
```

### 支持的工具

| 工具 | 说明 | 状态 |
|------|------|------|
| `dp` | DeePMD-kit + LAMMPS + TF/JAX/PyTorch | ✅ 可用 |
| `dpgen` | 自动数据生成 | 🚧 规划中 |
| 更多 | 采样 / 蒸馏 / … | 🚧 规划中 |

### 安装后能用什么

| 组件 | 用途 |
|------|------|
| `dp` | 训练 / 冻结 / 测试神经网络势函数 |
| `lmp` | LAMMPS 分子动力学推理 |
| `dp_ipi` | i-PI 接口 |
| `mpirun` / `horovod` | 多节点并行训练 |
| Python `deepmd` / `dpdata` / `pylammps` | 脚本化调用 |

### 用户前置要求

| 条件 | 说明 |
|------|------|
| 操作系统 | Linux x86_64 |
| 联网 | 在线安装需要；离线 `--file` 模式不需要 |
| 权限 | 无需 root（装到用户目录） |
| GLIBC | ≥ 2.17 |
| GPU 版 | 需兼容的 NVIDIA 驱动（CUDA 12.x ~ 13.0） |

---

## 给维护者：构建与发布离线包

> 以下是**内部维护**内容——普通用户不需要看。这个 repo 除了 dpack，还附带一套
> 离线包构建脚本和一个 Claude Code skill，用来生产 dpack 分发的离线安装包。

### 仓库结构

```
dpack/
├─ ⭐ dpack                  包管理器（用户入口）
├─ ⭐ install.sh             dpack 引导安装脚本（curl | bash）
│
├─ 📁 .github/workflows/
│  └─    nightly.yml         每日自动构建 + 发布（对标 PyTorch nightly）
│
├─ 📁 scripts/               固化构建/验收脚本
│  ├─    build.sh            构建离线安装包
│  ├─    verify_offline.sh   断网全流程验收（dp train + lammps）
│  ├─    build_pkg_from_commit.sh  Git commit → conda 包（高级）
│  └─    freeze.sh           锁定版本 → 可复现构建
│
├─ 📁 assets/                constructor 配方 + 工具清单
│  ├─    construct.yaml      Jinja2 模板（定义装什么）
│  ├─    manifest.json       工具清单（dpack 读它拿下载链接）
│  ├─    version.txt         版本号单一来源
│  └─    pre/post_install.sh 安装前后钩子
│
├─ 📄 SKILL.md               附带的 Claude Code skill（Agent 编排构建）
├─ 📄 HANDOFF.md             开发交接 + 验收清单（内部）
├─ 📁 references/            内部文档（用户无需阅读）
│  ├─    notes.md            构建/排查参考、GPU/commit 流程
│  └─    verification-log.md 验证记录（Bohrium 实测结果与踩坑）
└─ 📁 examples/ evals/       验证数据 / 质量评测
```

### 构建一个离线包

```bash
git clone https://github.com/Isaiah-WU/dpack.git
cd dpack

bash scripts/build.sh                 # CPU 版（约 10-15 分钟）
bash scripts/build.sh --cuda 12.9     # GPU 版（覆盖 12.x~13.0 驱动）

# 断网验收：切网 → 安装 → dp train → freeze → lammps 推理
bash scripts/verify_offline.sh dist/*/*.sh $(cat assets/version.txt)
```

> GPU 版只有 `cuda129`：conda-forge 对每个 deepmd 版本只发一个 CUDA build，靠 minor
> 兼容覆盖整条 12.x 驱动线。细节与多 CUDA 验证见
> [references/verification-log.md](references/verification-log.md)。

### 构建参数

| 标志 | 含义 | 默认值 |
|------|------|--------|
| `--version` | deepmd-kit 版本 | 读 `assets/version.txt` |
| `--cuda` | CUDA 版本，空 = CPU | `""` |
| `--backend` | `all` / `tensorflow` / `pytorch` / `jax` | `all` |
| `--torch-version` | PyTorch 版本 pin | 无 |
| `--glibc` | 目标系统 GLIBC | `2.28`（仅 GPU） |
| `--example` | 附带 example 数据：`dpa4` / `se_e2_a` | 无 |
| `--split` | 切割输出为 N 份（GitHub 2GiB 限制） | 不切割 |
| `--from-commit-channel` | 从本地 channel 打包（Mode B） | 无 |

### 两种打包模式

- **Mode A（默认）**：直接从 conda-forge 已发布的预编译包打包。有 conda 包的版本（如 3.2.0b0）用这个。
- **Mode B（高级）**：conda-forge 没有的版本（某个未发布 commit），先从源码编译到本地 channel，再打包。需 Docker。详见 [references/notes.md](references/notes.md)。

### 断网验收的 6 步

`verify_offline.sh` 依次执行，任一失败即退出：① `unshare -rn` 切网 → ② 安装 `.sh` 到临时目录 → ③ 冒烟（`dp -h` / `lmp -h` / import / 版本号）→ ③-GPU 检测（`nvidia-smi` / TF·JAX GPU / XLA libdevice）→ ④ `dp train` → ⑤ `dp freeze` → ⑥ `lmp` 推理。全过输出 `VERIFY PASSED`。

### License

LGPL-3.0-or-later

---

## English Version

### What This Is

**dpack** is the package manager for the DeepModeling ecosystem. One line installs the
full DeePMD-kit stack (DeePMD-kit + LAMMPS + TF/JAX/PyTorch + MPI) on any machine,
including air-gapped HPC — no conda, CUDA, or dependency wrangling needed.

```bash
# 1. Install dpack (user dir, no root)
curl -fsSL https://raw.githubusercontent.com/Isaiah-WU/dpack/main/install.sh | bash

# 2. Install deepmd-kit — auto-detects GPU/CUDA, downloads + installs the right build
dpack install dp

# Air-gapped machine: copy the offline package over, then point at the local file
dpack install dp --file ./deepmd-kit-3.2.0b0-cuda129-Linux-x86_64.sh

# List installed tools
dpack list
```

One GPU package covers the entire **CUDA 12.x ~ 13.0 driver line** — no version-matching
needed ([NVIDIA backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)).

### How It Works

```
[nightly CI builds offline packages daily]
      → [publishes to a platform: GitHub Release + manifest.json]
            → [dpack reads the manifest, auto-selects the build for your machine,
               downloads + installs]   (or: user downloads manually, then --file)
```

1. **Build** — `nightly.yml` uses conda `constructor` to pack the whole stack into a
   self-extracting `.sh` (CPU + CUDA variants) every day.
2. **Publish** — uploads to GitHub Release and updates `assets/manifest.json`
   (version, URLs, sha256).
3. **Install** — `dpack` reads the manifest, picks the right build for the user's
   machine, downloads, and installs. Manual download + `--file` also works.

### Why

HPC clusters are usually disconnected — `conda install` / `pip install` don't work.
dpack packs every dependency into one self-extracting archive: copy it over a USB
stick and one command installs it. No network, no root, no manual CUDA_HOME.

Future: `dpack install dpgen`, sampling, distillation — one package manager for the
whole ecosystem.

### Supported Tools

| Tool | Description | Status |
|------|-------------|--------|
| `dp` | DeePMD-kit + LAMMPS + TF/JAX/PyTorch | ✅ available |
| `dpgen` | Automated data generation | 🚧 planned |
| more | sampling / distillation / … | 🚧 planned |

### User Prerequisites

| Requirement | Detail |
|-------------|--------|
| OS | Linux x86_64 |
| Network | needed for online install; not for offline `--file` |
| Privileges | none (installs to user dir) |
| GLIBC | ≥ 2.17 |
| GPU variant | compatible NVIDIA driver (CUDA 12.x ~ 13.0) |

---

## For Maintainers: building & publishing offline packages

> Internal maintenance content — regular users can skip this. Besides dpack, this repo
> bundles the offline-package build scripts and a Claude Code skill used to produce the
> packages dpack distributes.

```bash
git clone https://github.com/Isaiah-WU/dpack.git
cd dpack

bash scripts/build.sh                 # CPU build (~10-15 min)
bash scripts/build.sh --cuda 12.9     # GPU build (covers 12.x~13.0 drivers)

# Offline acceptance: cut network → install → dp train → freeze → lammps
bash scripts/verify_offline.sh dist/*/*.sh $(cat assets/version.txt)
```

Build flags: `--version`, `--cuda`, `--backend {all,tensorflow,pytorch,jax}`,
`--torch-version`, `--glibc`, `--example {dpa4,se_e2_a}`, `--split N`,
`--from-commit-channel`. **Mode A** packs released conda packages (default);
**Mode B** builds a git commit from source into a local channel first (Docker).

GPU ships only `cuda129` — conda-forge publishes one CUDA build per deepmd release,
covering the whole 12.x driver line via minor-version compatibility. Build/verification
details: [references/notes.md](references/notes.md),
[references/verification-log.md](references/verification-log.md). Developer handoff +
sign-off: [HANDOFF.md](HANDOFF.md).

### License

LGPL-3.0-or-later
