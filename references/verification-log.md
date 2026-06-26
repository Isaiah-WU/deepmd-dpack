# Verification Log（内部验证记录）

> 测试平台：Bohrium，Tesla V100 / T4，驱动 580.105（CUDA 13.0）。deepmd-kit `3.2.0b0`。

## 变体与后端

| 变体 | CUDA | GPU 计算后端 | 构建方式 |
|---|---|---|---|
| `cpu`     | —    | TF + JAX + PyTorch（CPU） | Mode A |
| `cuda129` | 12.9 | TF + JAX + PyTorch        | Mode A |
| `cuda126` | 12.6 | PyTorch                   | Mode C |
| `cuda128` | 12.8 | PyTorch                   | Mode C |
| `cuda130` | 13.0 | PyTorch                   | Mode C |

- **Mode A**（`cpu` / `cuda129`）：conda-forge 预编译，带完整 **TF + JAX + PyTorch**。
- **Mode C**（`cuda126` / `cuda128` / `cuda130`）：**GPU 计算只走 PyTorch**；另带 CPU 版 TensorFlow，仅用于加载 LAMMPS 的 deepmd 插件，**不做 GPU 计算、不含 JAX**。

## 验证状态

| 变体 | 安装 + GPU 可见 | 完整 train → freeze → lammps |
|---|:---:|:---:|
| `cpu`     | ✅ | ✅ |
| `cuda129` | ✅ | ✅ |
| `cuda128` | ✅ | ✅ |
| `cuda126` | ✅ | ✅ |
| `cuda130` | ✅ | ✅ |

> 5 个变体均已通过验证（安装 + GPU + 完整 train→freeze→lammps）并收录进 `manifest.json`，dpack 据此为用户选包。
> 后续自动发布道（玻尔 cron）保持"`manifest.json` 只收录验证通过的变体"这一不变量。

## 安装方式（均已验证）

| 方式 | 命令 |
|---|---|
| dpack 引导（用户目录、无 root） | `curl install.sh \| bash` |
| dpack 在线（自动选版本 → 下载 → 校验 → 装） | `dpack install dp` |
| dpack 离线（无网） | `dpack install dp --file <pkg.sh>` |

跨镜像验证通过：`ubuntu22.04-py3.10`、`ubuntu24.04-py3.12`、`ubuntu22.04-cuda12.1`。
