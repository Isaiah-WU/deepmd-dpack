# 内部维护文档 — 构建 / 发布 / 验证（deepmd-dpack）

> 面向**维护者**。普通用户看 [README.md](../README.md) 即可。本文件是构建流水线、三种构建模式、
> 变体矩阵、验证策略与实测记录的权威说明。当前 deepmd-kit 版本：**3.2.0b0**
> （单一来源 [assets/version.txt](../assets/version.txt)）。

## 1 · 架构总览

```
Nightly CI（GitHub Actions，每天；检测到 deepmd 新版才动）
  ├─ 全矩阵构建（无 GPU 的 CPU runner，对标 PyTorch nightly）
  │    cpu / cuda129     → Mode A（conda constructor）
  │    cuda126 / cuda128 → Mode C（pip torch + conda-pack）
  │    cuda131           → cuda128 同字节改名（13.1 驱动）
  ├─ 每个变体：无 GPU 冒烟测试（装上 → dp --version → import deepmd）
  ├─ 传到 per-version Release（tag = v<version>）
  └─ manifest job：汇总片段 → 直接提交 assets/manifest.json
                    ↓
           dpack 读 manifest → 按 nvidia-smi 选变体 → 下载 → 校验 sha256 → 安装
```

**关键设计决定：**
- **每版本一个 Release**（`v<version>`），非滚动 `nightly` tag。
- **manifest.json 是 dpack 唯一索引**，由 nightly CI 的 `manifest` job 提交。

## 2 · 三种构建模式（为什么要分）

**根因：conda-forge 每个 deepmd 版本只发布一个 CUDA 编译包**（3.2.0b0 是 cuda129，硬钉 `cuda-version>=12.9,<13`）。围绕“要打包的包从哪来”分出三种模式：

| | Mode A | Mode B | Mode C |
|---|---|---|---|
| 脚本 | `build.sh` | `build_pkg_from_commit.sh` → `build.sh` | `build_modec.sh` |
| 工具 | conda **constructor** | 源码编译 → 本地 channel → constructor | **pip + conda-pack** |
| 包来源 | conda-forge 已发布的包 | 从 git commit 自编 | PyPI 的 torch cuXXX wheel + pip deepmd |
| 产出变体 | **cpu、cuda129** | 未发布的 commit | **cuda126、cuda128** |
| 后端 | TF + JAX + PyTorch（全） | 全 | PyTorch(GPU) + CPU-only TF |
| 何时用 | 上游有现成的 | 打包未发布 commit（Docker，重） | 上游没有的 CUDA 版，只能自己 pip 造 |

**Mode A/C 的分裂不是主动设计，是被上游“有什么/没什么”逼出来的**：上游有的用 constructor，上游没有的 GPU 版只能 pip 造。所以“全矩阵 nightly”必然 A+C 两套一起跑（都在无 GPU 的 runner 上完成）。

## 3 · 变体矩阵（3.2.0b0）

| 变体 | CUDA | 模式 | 后端 | 说明 |
|---|---|---|---|---|
| `cpu` | — | A | tf+jax+torch | conda constructor |
| `cuda126` | 12.6 | C | pytorch | pip torch cu126 |
| `cuda128` | 12.8 | C | pytorch | pip torch cu128 |
| `cuda129` | 12.9 | A | tf+jax+torch | conda-forge 唯一的 CUDA 包 |
| `cuda131` | 13.1 | C（别名） | pytorch | = cuda128 同字节改名，面向 13.1 驱动 |

Mode C 配方（见 [build_modec.sh](../scripts/build_modec.sh)）：`torch==2.11.*`（`download.pytorch.org/whl/cuXXX`）+ `deepmd-kit` + `e3nn` + `mpich` + `tensorflow-cpu==2.21.0`（仅为加载 lmp 插件）+ `lammps[mpi]~=2025.7.22.2.0` → `conda-pack` 成自解压 `.sh`。

## 4 · Nightly CI 流水线（[nightly.yml](../.github/workflows/nightly.yml)）

- **触发**：每天 02:17 UTC（cron）或手动 `workflow_dispatch`（可选单个变体）。
- **变了才构建**：`detect_latest_deepmd.py` 比对 PyPI 最新版 vs manifest 已发布版；没变则 `build` 整个跳过，零成本（`workflow_dispatch` 永远强制构建）。
- **job 依赖**：`setup`(选矩阵) → `resolve`(查版本) → `prepare`(建 release) → `build`(全矩阵并行) → `manifest`(汇总提交) + `prune`(清旧资产) → `pass`(alls-green)。
- **build 每个变体**：按 `mode` 分支（A→build.sh，C→build_modec.sh）→ 无 GPU 冒烟测试 → >2GiB 切 3 片 → 传 release → 生成 `frag/<variant>.json`（cuda128 额外生成 cuda131 片段）。
- **manifest job**：下载所有片段 → `merge_manifest.py` 合并（保留未重建的变体）→ commit + push `assets/manifest.json`。此 job **不跑第三方 pip**，故写 token 安全。
- **prune**：默认 dry-run；确认无误后把仓库变量 `PRUNE_APPLY` 设 `true` 才真删（永不删 manifest 引用的资产）。

## 5 · 验证策略（导师模型）+ 如何做一次性验证

**日常**：CI 的**无 GPU 冒烟测试**（装上 + `dp --version` + `import deepmd`）证明“打包能装、能导入”，通过即写 manifest。**不做每晚 GPU 实跑。**

**一次性 / 需要时的完整 GPU 端到端验证**（在有 GPU 的节点，如玻尔）：
```bash
# Mode C 变体(cuda126/128/131)：装好 → source .../bin/activate →
bash scripts/verify_offline_modec.sh          # torch.cuda + dp train + freeze + lammps MD
# Mode A 变体(cpu/cuda129)
bash scripts/verify_offline.sh dist/<variant>/*.sh 3.2.0b0
```
这两个脚本是“打包流程验证一次”用的真门；`nightly.yml` 不调用它们。

## 6 · 验证记录（实测）

测试平台：Bohrium，Tesla V100 / T4，驱动 CUDA 12.9 / 13.0 / 13.1。

| 变体 | 机器 / 驱动 | 完整 train → freeze → lammps |
|---|---|---|
| `cpu` | 任意 Linux | ✅ Mode A 基线，dp/lmp 可用 |
| `cuda126` | GPU 节点 | ✅ |
| `cuda128`（= cuda131 配方） | T4 / CUDA 13.1 驱动 | ✅ **13.1/T4 实测跑通** |
| `cuda129` | V100 / CUDA 12.9 | ✅ train + freeze + lammps 完成 |
| `cuda131` | 13.1 精确匹配 | ✅ = cuda128 同一份字节 |
| SPIN 能力 | cuda126 包内探测 | ✅ LAMMPS 带 SPIN package，`pair_style deepspin` 存在（支持 spin/dpa4 自旋模型） |

> ⚠️ 历史更正：早前“5 变体全部完整验证”对 `cuda130` 是过度宣称（当时 LAMMPS 未真跑）。经 13.1/T4 严格测试 `cuda130` 崩溃，已弃。本表只标实际跑通的结果。

## 7 · 发布模型 & manifest & dpack 选包

- **每版本一个 Release**：tag `v<version>`（如 `v3.2.0b0`），存放该版本所有变体的分片。deepmd 出新版 → CI 建新 release `v<新版本>`，manifest top-level version 随最新片段前移。
- **manifest.json**：dpack 从 `raw.githubusercontent.com/<repo>/main/assets/manifest.json` 读。每变体记 `parts`（或 `url`）+ `sha256` + `backend` + `note`。dpack 装完校验 sha256。
- **GPU 包切 3 片**（GitHub 单资产 2GiB 上限）；dpack `--file` 指基础名自动拼片。

dpack 选包（`pick_variant`：精确匹配 → 否则 ≤ 驱动的最高可用 → cpu）：

| 机器驱动 CUDA（nvidia-smi 报的）| dpack 装的变体包 |
|---|---|
| 13.1 | `cuda131`（= cuda128 字节）|
| 13.0 | `cuda129`（向后兼容；无 cuda130 包）|
| 12.9 | `cuda129` |
| 12.8 | `cuda128` |
| 12.6 | `cuda126` |

## 8 · 维护者常用命令

```bash
# 手动单独构建一个变体
bash scripts/build.sh --version 3.2.0b0                 # cpu     (Mode A)
bash scripts/build.sh --version 3.2.0b0 --cuda 12.9     # cuda129 (Mode A)
bash scripts/build_modec.sh cu126                       # cuda126 (Mode C；或 cu128)

# 手动触发 CI：Actions → Nightly Build (all variants) → Run workflow（选 all / cpu / cuda129 / cu126 / cu128）

# 清理 release 旧资产（先 dry-run）
TAG=v3.2.0b0 KEEP_BUILDS=2 python scripts/prune_release_assets.py
TAG=v3.2.0b0 GH_TOKEN=<PAT> python scripts/prune_release_assets.py --apply
```

- **Mode B**（未发布 commit）：先 `build_pkg_from_commit.sh --commit <sha> [--cuda 12.9] [--config <stem>]`（Docker，重），再 `build.sh --from-commit-channel ./local-channel`。细节见 [notes.md](notes.md)。

## 9 · 仓库结构

```
deepmd-dpack/
├─ dpack                  包管理器（用户入口）
├─ install.sh             dpack 引导安装（curl | bash）
├─ .github/workflows/
│  └─ nightly.yml         每日全矩阵构建 + 冒烟 + 直接写 manifest
├─ scripts/
│  ├─ build.sh            Mode A：constructor 构建（cpu/cuda129）
│  ├─ build_modec.sh      Mode C：pip torch + conda-pack（cuda126/128）
│  ├─ build_pkg_from_commit.sh  Mode B：commit → 本地 channel（Docker）
│  ├─ verify_offline.sh / verify_offline_modec.sh  一次性 GPU 端到端验证
│  ├─ detect_latest_deepmd.py   查 PyPI 最新版（变了才构建）
│  ├─ gen_manifest_fragment.py / merge_manifest.py  manifest 片段生成/合并
│  ├─ prune_release_assets.py   清 release 旧资产（受 manifest 保护）
│  └─ freeze.sh           锁定精确版本 → 可复现
├─ assets/
│  ├─ construct.yaml      Mode A 的 constructor 配方（Jinja2）
│  ├─ manifest.json       dpack 索引（由 nightly CI 提交）
│  ├─ version.txt         版本单一来源
│  └─ pre/post_install.sh 安装钩子
├─ SKILL.md               Claude Code skill（Agent 编排构建）
├─ HANDOFF.md             早期验收记录（历史）
├─ references/
│  ├─ notes.md            构建/排查参考、GPU/commit 流程
│  └─ verification-log.md 本文件（内部维护权威说明）
└─ examples/ evals/       验证数据 / 质量评测
```
