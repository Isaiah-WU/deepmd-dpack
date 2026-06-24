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

## 4. 多 CUDA 版本验证

**结论**：在 Bohrium 上无法测不同 CUDA *驱动*（所有节点驱动统一 13.0，宿主机注入，
选任何镜像都不变）。但靠 NVIDIA 向后兼容，13.0 驱动能跑任何更老的 CUDA 运行时。所以
验证方式 = 造出针对不同 CUDA 编译的包，全装到同一台 13.0 节点，验证各自 GPU 可用。

| CUDA 编译版本 | 来源 | 验证 | 状态 |
|---|---|---|---|
| CUDA 12.6 | deepmd 3.1.1 `cuda126` | `torch.version.cuda=12.6` + `torch.cuda.is_available()=True` | ✅ |
| CUDA 12.9 | deepmd 3.2.0b0 `cuda129` | train + freeze + lammps 全流程 | ✅ |
| CUDA 12.8 / 13.1 | — | conda-forge 任何 deepmd 版本都未发布 | ⛔ 需源码编 |

### conda-forge deepmd-kit 的 CUDA 矩阵（`conda search` 实测）

```
2.0.3 ~ 2.1.4   cuda102 / cuda110 / cuda111 / cuda112
2.2.6 ~ 2.2.11  cuda112 / cuda118 / cuda120
3.0.0           cuda120
3.0.1 ~ 3.1.0   cuda126
3.1.1           cuda126 + cuda129   ← 唯一同时有两个的版本
3.1.2 ~ 3.2.0b0 cuda129
```

→ **12.8 和 13.x 从未为任何 deepmd 版本发布过。** 每个版本一个 cudaXXX build。

### 关键发现：cuda129 build 硬锁 cuda-version

`deepmd-kit-3.2.0b0-cuda129*` 带 run constraint `cuda-version >=12.9,<13`。在 construct.yaml
里设 `cuda-version 12.6` 会让 constructor solve **直接失败**（LibMambaUnsatisfiableError），
不是静默错配。所以用现成包只能造 cuda129 一个 GPU 安装包。

复现（dry-run，几秒出 UnsatisfiableError）：
```bash
conda create -n probe126 --dry-run \
  -c conda-forge/label/deepmd-kit_rc -c conda-forge \
  "deepmd-kit=3.2.0b0=cuda*" "cuda-version=12.6"
# → deepmd-kit cuda129 requires cuda-version >=12.9,<13 ⊥ cuda-version==12.6
```

### 验证 cuda126 包 GPU 可用（不走 constructor，直接建 conda 环境）

注意：**不能强制 TF/PyTorch 版本**——TF 2.19 只有 cuda128/cuda129 build，会和 cuda126 冲突。
让 solver 自己挑（会拉到 TF 2.18 + PyTorch 2.6 的 cuda126 build）。

```bash
conda create -p /tmp/cu126 -c conda-forge "deepmd-kit=3.1.1=cuda126*" lammps -y
conda run -p /tmp/cu126 python -c \
  "import torch; print(torch.version.cuda, torch.cuda.is_available())"
# → 12.6 True
```

有趣细节：cuda126 *编译*的 deepmd/torch/TF，运行时 CUDA 库被解成 `cuda-version 12.9`
（minor 兼容），最终跑在 13.0 驱动上——三层都靠 minor/向后兼容咬合。这正是单包覆盖
12.x 驱动线的底层原理。

### 给 3.2.0b0 / dpa4 的结论

dpa4 需要 3.2.0b0，而它只有 cuda129。靠 minor 兼容，这一个包覆盖整个 12.x 驱动线 + 13.0，
用户不用关心自己的 CUDA 版本。要真正的 per-minor 编译版（cuda126/cuda128）只能走
feedstock 源码编（Mode B，见 [notes.md](./notes.md)），但 minor 兼容使其无实际收益。

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
