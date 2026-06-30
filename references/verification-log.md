# Verification Log（内部验证记录）

> 测试平台：Bohrium，Tesla V100 / T4，驱动 CUDA 13.0 / 13.1。deepmd-kit `3.2.0b0`。

## 关键事实：deepmd-kit 3.2.0b0 的 GPU/LAMMPS 是 **CUDA 12** 编译的
deepmd 的 LAMMPS 插件运行时硬 dlopen `libcudart.so.12`（CUDA 12）。PyPI 只有一个通用 wheel = CUDA 12；conda-forge 也是 cuda-version 12.9。**deepmd 3.2.0b0 没有任何 CUDA 13 构建，PyTorch 也无 cu131。** 由此：

- **deepmd 不像 PyTorch 那样为每个 CUDA 版本单独编译**——它只有一个 CUDA 12 构建；我们的 GPU 包差异只在搭配的 torch 的 CUDA 小版本。
- **CUDA 12 的包覆盖 12.x ~ 13.x 所有 GPU 机器**（NVIDIA 驱动向后兼容）。
- **CUDA 13 的包（cuda130）已弃**：CUDA-13 torch + CUDA-12 deepmd 同进程，torch 的 JIT 融合内核崩溃（13.1/T4 实测，不可靠）。
- **13.1 机器的可靠包 = cuda128**（13.1/T4 实测 train + freeze + LAMMPS 跑通）。

## 变体与后端
| 变体 | torch CUDA | 覆盖 / 匹配 | GPU 后端 | 说明 |
|---|---|---|---|---|
| `cpu`     | —    | 全部（CPU）        | TF + JAX + PyTorch | Mode A |
| `cuda126` | 12.6 | ≥ 12.6（含 13.x）  | PyTorch            | Mode C |
| `cuda128` | 12.8 | ≥ 12.8（含 13.x）  | PyTorch            | Mode C |
| `cuda129` | 12.9 | ≥ 12.9（含 13.x）  | TF + JAX + PyTorch | Mode A |
| `cuda131` | 12.8 | 13.1 机器（精确匹配）| PyTorch          | **= cuda128 别名**；deepmd 无原生 CUDA 13，靠驱动向后兼容在 13.1 跑 |

> `cuda130` 已从构建与 manifest 移除（CUDA 13 torch 与 CUDA 12 deepmd 冲突，LAMMPS 崩溃）。

## 验证状态
| 变体 | 安装 + GPU 可见 | 完整 train → freeze → lammps |
|---|:---:|---|
| `cpu`     | ✅ | ✅ Mode A 基线 |
| `cuda126` | ✅ | ✅ |
| `cuda128` | ✅ | ✅ **13.1 / T4 实测跑通** |
| `cuda131` | ✅ | ✅ = cuda128（13.1 精确匹配，同一份字节）|
| `cuda129` | ✅ | ⏳ Mode A，新 GPU 节点 LAMMPS 待严格验证（dpack 对 13.0 / 12.9 选它）|

> ⚠️ 更正：早前"5 变体全部完整验证"对 `cuda130` 是过度宣称（当时 LAMMPS 未真跑）。
> 经 13.1/T4 严格测试，`cuda130` 的 LAMMPS 崩溃，已弃。本表只标实际跑通的结果。

## dpack 选包（删 cuda130、加 cuda131 后）
| 机器 CUDA | 选到 |
|---|---|
| 13.1 | `cuda131`（= cuda128 字节，已验证）|
| 13.0 | `cuda129`（向后兼容；LAMMPS 待验）|
| 12.9 | `cuda129` |
| 12.8 | `cuda128` |
| 12.6 | `cuda126` |

## 安装方式
| 方式 | 命令 |
|---|---|
| dpack 引导（用户目录、无 root） | `curl install.sh \| bash` |
| dpack 在线（自动选版本 → 下载 → 校验 → 装） | `dpack install dp` |
| dpack 离线（无网） | `dpack install dp --file <pkg.sh> [--sha256 <hex>]` |
