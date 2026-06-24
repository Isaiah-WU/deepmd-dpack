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
| `dpack install dp --file <pkg.sh>` | 离线安装：从本地离线包安装（支持分片，自动合并） |
| `dpack install dp --prefix <dir>` | 指定安装目录 |
| `dpack list` | 查看已安装的工具 |

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

### 自动发布闭环（nightly.yml）

**每天 UTC 02:17（北京时间 10:17）自动运行**（也会在 `construct.yaml` / `version.txt` / `build.sh` 改动推送时触发）：

```
构建(cpu + cuda129) → 创建/更新 GitHub Release → 上传安装包（GPU 自动分片）
                    → 重生 assets/manifest.json → 提交回 main
```

手动触发：`Actions → Nightly Build & Publish → Run workflow`，`variant` 可选 `cpu` / `cuda129` / `all`。

### 手动构建一个离线包

```bash
git clone https://github.com/Isaiah-WU/deepmd-dpack.git
cd deepmd-dpack

bash scripts/build.sh                 # CPU 版（约 10-15 分钟）
bash scripts/build.sh --cuda 12.9     # GPU 版（覆盖 12.x~13.0 驱动）

# 断网验收：切网 → 安装 → dp train → freeze → lammps 推理
bash scripts/verify_offline.sh dist/*/*.sh $(cat assets/version.txt)
```

构建参数：`--version`、`--cuda`、`--backend {all,tensorflow,pytorch,jax}`、`--torch-version`、
`--glibc`、`--example {dpa4,se_e2_a}`、`--split N`、`--from-commit-channel`。

- **Mode A（默认）**：从 conda-forge 已发布的预编译包打包。有 conda 包的版本（如 3.2.0b0）用这个。
- **Mode B（高级）**：conda-forge 没有的版本（某个未发布 commit），先源码编译到本地 channel 再打包，需 Docker。详见 [references/notes.md](references/notes.md)。

> GPU 当前只有 `cuda129`：conda-forge 对每个 deepmd 版本只发一个 CUDA build，靠 minor 兼容
> 覆盖整条 12.x 驱动线。多 CUDA 的查证与实测见 [references/verification-log.md](references/verification-log.md)。

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
| `dpack install dp --file <pkg.sh>` | Offline install from a local package (split parts auto-reassembled) |
| `dpack install dp --prefix <dir>` | Install to a specific directory |
| `dpack list` | List installed tools |

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
