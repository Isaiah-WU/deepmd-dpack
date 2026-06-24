# Verification Log — 开发验证记录（内部）

> 这是**开发记录**，不是用户文档。记录在 Bohrium 上做过的验证、踩过的坑、技术发现。
> 用户文档见 [README.md](../README.md)；构建/排查参考见 [notes.md](./notes.md)。

测试平台：Bohrium，节点 4× / 1× Tesla V100-SXM2-16GB，驱动 580.105（CUDA 13.0），
镜像 ubuntu22.04-py3.10 与 ubuntu24.04-py3.12。

---

## 1. 安装方式

| 方式 | 命令 | 状态 |
|---|---|---|
| dpack 引导安装（用户目录，无 root） | `curl install.sh \| bash` | ✅ |
| dpack 离线安装（无网，本地包） | `dpack install dp --file ./xxx.sh` | ✅ |
| dpack 在线安装（自动下载 3 片 → 合并 → sha256 校验 → 装） | `dpack install dp` | ✅ |

dpack 在线安装的分片下载用 `curl --retry 5 -C -`（断点续传），扛住了一次 GitHub 503
和一次 SSL EOF；前面下好的分片缓存复用。

## 2. 跨镜像

| 基础镜像 | 安装方式 | 状态 |
|---|---|---|
| ubuntu22.04-py3.10 | dpack 在线 + 离线 | ✅ |
| ubuntu24.04-py3.12 | dpack 在线（下载 → 合并 → 校验 → 装） | ✅ |

## 3. 端到端流程（`unshare -rn` 切网 → 安装 → train → freeze → lammps）

| deepmd-kit | 变体 | 后端 | 训练数据 | 状态 |
|---|---|---|---|---|
| 3.1.3 | CPU | TF + JAX | 合成 6 原子 | ✅ |
| 3.2.0b0 | CPU | TF/JAX/PyTorch | dpa4 真实 192 原子 | ✅ |
| 3.2.0b0 | GPU cuda129 | **PyTorch**（dpa4 真实场景） | 真实 + 合成 | ✅ |
| 3.2.0b0 | GPU cuda129 | **TensorFlow** | 合成 6 原子 | ✅ |

GPU LAMMPS 推理比 CPU 快约 10 倍（12ms vs 123ms）。

## 4. 多 CUDA 版本

### 4.1 为什么 3.2.0b0 只能是 cuda129（依赖图实证）

deepmd 3.2.0b0 同时依赖 **TensorFlow 2.19** 和 **PyTorch 2.11**（feedstock
`conda_build_config.yaml` 钉死）。这两个在 conda-forge 上的 CUDA 构建：

```
TensorFlow 2.19：cuda128, cuda129
PyTorch    2.11：cuda129, cuda130
两者交集       ：cuda129   ← deepmd 3.2.0b0 唯一能成立的 CUDA 版本
```

实测命令（任意有 conda 的机器）：
```bash
conda search -c conda-forge "pytorch=2.11"    | grep -oE 'cuda[0-9]+' | sort -u  # cuda129 cuda130
conda search -c conda-forge "tensorflow=2.19" | grep -oE 'cuda[0-9]+' | sort -u  # cuda128 cuda129
```

→ **cuda126 / cuda128 / cuda13x 对 3.2.0b0 都做不出来**（缺 TF 或缺 PyTorch），
连源码编（Mode B）也救不了——不是我们没编，是上游 TF/PyTorch 的 CUDA 构建不重合。
这由 Google/Meta 发布的 CUDA 版本决定。PyTorch 已有 cuda130（CUDA 13），但 TF 还没有，
所以等 TF 发 CUDA-13，deepmd 才能编 cuda130。

### 4.2 源码编 cuda128 的尝试与实证（Mode B，已验证不可行）

在 32 核节点用 conda-build + 3.2.0b0 的 feedstock 配方（commit `5bd8a7e`）尝试编 cuda128，
dry-run solve 直接证明 PyTorch 2.11 没有 cuda128：

```bash
CONDA_OVERRIDE_CUDA=12.8 conda create -n probe128 --dry-run \
  -c conda-forge/label/deepmd-kit_rc -c conda-forge \
  "cuda-version=12.8" "tensorflow=2.19.*=*cuda*" "pytorch=2.11.*=*cuda*"
# → pytorch-2.11.0-cuda129_generic requires cuda-version >=12.9,<13  ⊥ cuda-version 12.8
#   （PyTorch 2.11 只有 cuda129/cuda130，没有 cuda128）
```

### 4.3 conda-forge deepmd-kit 历史 CUDA 矩阵（`conda search` 实测）

```
2.0.3 ~ 2.1.4   cuda102 / cuda110 / cuda111 / cuda112
2.2.6 ~ 2.2.11  cuda112 / cuda118 / cuda120
3.0.0           cuda120
3.0.1 ~ 3.1.0   cuda126
3.1.1           cuda126 + cuda129
3.1.2 ~ 3.2.0b0 cuda129
```

注：旧版本（3.1.1）能有 cuda126，是因为它配的是 TF 2.18（TF 2.18 有 cuda126）。
3.2.0b0 配 TF 2.19，所以只有 cuda129。**不能为了凑 12.6 用旧版 deepmd——旧版没有 dpa4。**

### 4.4 跨 CUDA 环境的独立验证（dpack 安装 + 完整 train/freeze/lammps）

由于 3.2.0b0 只能是 cuda129 这一个包，"多 CUDA 验证" = 在**不同 CUDA 环境**的节点上，
各自用 dpack 装这同一个 cuda129 包，独立跑通完整 train + freeze + lammps。靠 NVIDIA
向后兼容，cuda129 包在所有 CUDA 12.x ~ 13.x 环境都能跑。

实测于 Bohrium（驱动统一 13.0）：

| 节点 CUDA 环境 | 怎么造出该环境 | GPU | 安装方式 | train+freeze+lammps | 状态 |
|---|---|---|---|---|---|
| 无 host toolkit | ubuntu24.04-py3.12 镜像 | V100 | dpack 在线 | 全流程 | ✅ |
| CUDA 12.1 | ubuntu22.04-cuda12.1 镜像自带 | T4 | dpack 在线 | 全流程 | ✅ |
| **CUDA 12.6** | `conda install cuda-toolkit=12.6` | T4 | dpack（缓存包）| 全流程 | ✅ |
| **CUDA 12.8** | `conda install cuda-toolkit=12.8` | T4 | dpack（缓存包）| 全流程 | ✅ |
| **CUDA 13.1** | `conda install cuda-toolkit=13.1` | T4 | dpack（缓存包）| 全流程 | ✅ |

复现一个环境（以 12.6 为例）：
```bash
conda create -p /tmp/cuda126 -c conda-forge cuda-toolkit=12.6 -y
export PATH="/tmp/cuda126/bin:$PATH"; nvcc --version | grep release   # → 12.6
bash scripts/verify_offline.sh ~/.dpack/cache/dp-cuda129.sh 3.2.0b0    # → VERIFY PASSED
```

> **重要说明（机制）**：Bohrium 所有节点的 **驱动统一是 13.0**（宿主机注入，选任何镜像都不变）；
> 上表变的是**容器里的 CUDA toolkit**（镜像自带或 `conda install cuda-toolkit=X` 装出来）。
> 我们的包**自带 cuda129 运行时**，不用宿主机 toolkit——所以这组验证证明的是：**宿主机无论是
> 什么 CUDA toolkit 环境（老到 12.1、新到 13.1，甚至没装），我们的包都能独立装、独立跑通全流程**
> （自包含）。
>
> 要测真正不同的**驱动版本**（真 12.6 / 12.8 驱动），Bohrium 给不了（驱动固定 13.0），需 AWS
> 等能选驱动的云或物理机。但 cuda129 在不同驱动上的可用性由 NVIDIA 官方的向后/minor 兼容保证
> （PyTorch 的 cuXXX wheel 依赖的是同一套机制），且 13.0 驱动已实测跑通。

## 5. TF 后端 libdevice 修复

**根因**：libmamba solver 损坏被迫退回 classic solver → `libdevice-hack-for-tensorflow`
的符号链接断裂 → TF>=2.19 的 XLA JIT 报 `libdevice not found`。只影响 TF 后端，
dpa4/PyTorch/JAX/CPU 不受影响。

**修复**：
- construct.yaml：pin `libdevice-hack-for-tensorflow=*=py_5` + 加 `cuda-nvvm`（保证真实目标存在）
- post_install.sh：安装后自动把 `libdevice.10.bc` 重新软链到 `$PREFIX/nvvm/libdevice/`，
  并写 activate hook 设 `XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX`
- 构建务必用 **libmamba** solver；用 /opt/mamba 的老 conda（22.11）不认 libmamba 插件，
  需装新 Miniforge（conda 26.x）

**回归验证**：用 libmamba 重新构建 cuda129 后，TF 后端 train + freeze + lammps 全跑通，
零 libdevice 错误。

## 6. 平台踩坑（Bohrium）

- **磁盘**：GPU 安装包解压需 ~44 GB 临时空间。系统盘默认 40G 不够，要手动设 ≥ 100 GB。
- **NAS 不能解压**：装到 `/personal`（阿里云 NFS）时 constructor 解压报 `InvalidArchiveError`
  / BrokenProcessPool。必须装到本地盘。
- **NAS 配额**：`/personal` 有配额，堆测试文件会 `Disk quota exceeded`。
- **构建内存**：GPU 解 CUDA 依赖图吃内存，3.8 GB 节点会 OOM Killed，要 ≥ 8 GB（实际用 32G）。
- **清华源 403**：Bohrium 上要去掉清华 channel，用 defaults + conda-forge。
