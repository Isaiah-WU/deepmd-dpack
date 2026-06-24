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

### 4.1 3.2.0b0 只能是 cuda129

deepmd 3.2.0b0 依赖 TF 2.19 + PyTorch 2.11（feedstock 钉死），两者 CUDA 构建交集只有 cuda129：

```
TF 2.19     : cuda128, cuda129
PyTorch 2.11: cuda129, cuda130
交集        : cuda129
```
```bash
conda search -c conda-forge "tensorflow=2.19" | grep -oE 'cuda[0-9]+' | sort -u  # cuda128 cuda129
conda search -c conda-forge "pytorch=2.11"    | grep -oE 'cuda[0-9]+' | sort -u  # cuda129 cuda130
```

cuda126/128/13x 对 3.2.0b0 都做不出（缺 TF 或缺 PyTorch），源码编也不行——上游构建不重合。

### 4.2 cuda128 源码编实证不可行

feedstock（commit `5bd8a7e`）+ conda-build，dry-run 即证 PyTorch 2.11 无 cuda128：
```bash
CONDA_OVERRIDE_CUDA=12.8 conda create -n probe128 --dry-run \
  -c conda-forge/label/deepmd-kit_rc -c conda-forge \
  "cuda-version=12.8" "tensorflow=2.19.*=*cuda*" "pytorch=2.11.*=*cuda*"
# → pytorch 2.11 requires cuda-version >=12.9  ⊥  cuda-version 12.8
```

### 4.3 conda-forge deepmd CUDA 矩阵（`conda search` 实测）

```
2.0.3~2.1.4    cuda102/110/111/112
2.2.6~2.2.11   cuda112/118/120
3.0.1~3.1.0    cuda126
3.1.1          cuda126 + cuda129
3.1.2~3.2.0b0  cuda129
```
3.1.1 有 cuda126 因配 TF 2.18；3.2.0b0 配 TF 2.19 故只有 cuda129。旧版无 dpa4，不能用。

### 4.4 跨 CUDA 环境独立验证

同一个 cuda129 包，dpack 安装 + 完整 train/freeze/lammps。Bohrium 实测（驱动统一 13.0，变 toolkit）：

| CUDA 环境 | 造法 | GPU | 全流程 | 状态 |
|---|---|---|---|---|
| 无 toolkit | ubuntu24.04 镜像 | V100 | train+freeze+lammps | ✅ |
| 12.1 | ubuntu22.04-cuda12.1 镜像 | T4 | 同上 | ✅ |
| 12.6 | `conda install cuda-toolkit=12.6` | T4 | 同上 | ✅ |
| 12.8 | `conda install cuda-toolkit=12.8` | T4 | 同上 | ✅ |
| 13.0 | toolkit 13.0 + 驱动本就是 13.0 | T4 | 同上 | ✅ |
| 13.1 | `conda install cuda-toolkit=13.1` | T4 | 同上 | ✅ |

复现（以 12.6 为例）：
```bash
conda create -p /tmp/cuda126 -c conda-forge cuda-toolkit=12.6 -y
export PATH="/tmp/cuda126/bin:$PATH"; nvcc --version | grep release
bash scripts/verify_offline.sh ~/.dpack/cache/dp-cuda129.sh 3.2.0b0   # → VERIFY PASSED
```

机制：包自带 cuda129 运行时，不用宿主机 toolkit → 验证的是"宿主机任何 CUDA toolkit（12.1~13.1）
下包都能独立装跑"。13.0 那行是真驱动层面（`nvidia-smi`=13.0）。换真驱动版本需非 Bohrium 机器，
但 cuda129 跨驱动可用由 NVIDIA 向后/minor 兼容保证。

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
