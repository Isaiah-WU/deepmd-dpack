# Verification Log — 开发验证记录(内部)

> 开发记录,非用户文档(用户文档见 [README.md](../README.md))。
> 测试平台:Bohrium,Tesla V100-16GB / T4,驱动 580.105(**CUDA 13.0**),镜像 ubuntu22.04-py3.10 / ubuntu24.04-py3.12 / ubuntu22.04-cuda12.1。deepmd-kit **3.2.0b0**。

## 1. 离线包变体与后端

release `v3.2.0b0` 共 **5 个变体**:

| 变体 | CUDA | GPU 计算后端 | 其它后端 | 构建方式 |
|---|---|---|---|---|
| `cpu`     | —    | —              | TF + JAX + PyTorch(CPU) | Mode A(conda-forge constructor) |
| `cuda129` | 12.9 | TF + JAX + PyTorch | —                   | Mode A(conda-forge constructor) |
| `cuda126` | 12.6 | **PyTorch**        | TF-CPU(仅插件)、无 JAX | Mode C(pip torch cu126 + conda-pack) |
| `cuda128` | 12.8 | **PyTorch**        | TF-CPU(仅插件)、无 JAX | Mode C(pip torch cu128 + conda-pack) |
| `cuda130` | 13.0 | **PyTorch**        | TF-CPU(仅插件)、无 JAX | Mode C(pip torch cu130 + conda-pack) |

**后端说明(重要,别误读):**
- **cpu / cuda129(Mode A)**:来自 conda-forge 预编译包,带完整 TF + JAX + PyTorch;cuda129 三个后端都能做 GPU 计算。
- **cuda126 / 128 / 130(Mode C)**:**GPU 计算只走 PyTorch**(torch 对应 cuXXX)。捆绑的 `tensorflow-cpu` **仅用于加载 LAMMPS 的 deepmd 插件**(插件入口硬 `import tensorflow`),**不参与 GPU 计算**;**不含 JAX**。
- **为什么分两套**:conda-forge 每个 deepmd 发布只编一个 CUDA(3.2.0b0 = cuda129),要多 CUDA 必须自编;自编走 PyTorch 官方 pip 源取 torch(cu126/128/130 都有),而 TF 只有 `cu12`(CPU)能搭,所以 Mode C 以 PyTorch 为主。

## 2. 各变体验证状态(诚实标注)

| 变体 | 离线装 + dp/lmp 在位 + GPU 可见 | 完整 train→freeze→lammps |
|---|---|---|
| `cpu`     | ✅ | ✅ Mode A 基线(早期 Bohrium 实测,含 dpa4 真实 192 原子) |
| `cuda129` | ✅ | ✅ Mode A 基线(PyTorch dpa4 + TensorFlow;GPU lammps 比 CPU 快约 10×) |
| `cuda128` | ✅(GPU True 12.8) | ✅ 本会话实测:train + freeze + lammps MD;conda-pack 异地解压后仍跑通;`dpack install dp --cuda 12.8` 全新节点装通 |
| `cuda126` | ✅(GPU True 12.6) | ⏳ 待验证道(玻尔 cron)补完整 MD |
| `cuda130` | ✅(GPU True 13.0,13.0 原生) | ⏳ 待验证道(玻尔 cron)补完整 MD |

> **manifest 只收录"完整验过"的变体——这是 dpack 选包的唯一闸门。** cuda126/130 目前是"装得上 + dp/lmp 在位 + GPU 可见"级别;完整 train→freeze→lammps 端到端由自动验证道持续补齐(见 README 自动发布闭环)。

## 3. 安装方式(均已验证)

- **dpack 引导**:`curl install.sh | bash`,装到用户目录、无需 root ✅
- **dpack 在线**:自动读 `nvidia-smi` 检测驱动 CUDA → `pick_variant` 选变体 → 下分片 → `cat` 拼接 → sha256 校验 → 安装 ✅(分片下载用 `curl --retry 5 -C -` 断点续传,扛过 GitHub 503 / SSL EOF / 大文件瞬断)
- **dpack 离线**:`--file` 指本地包,无网可装 ✅
- **跨镜像**:ubuntu22.04-py3.10、ubuntu24.04-py3.12、ubuntu22.04-cuda12.1 均通过

## 4. Mode C 关键技术点(踩坑实录)

- **TF 版本必须 `2.21.0`**:deepmd 3.2.0b0 的 pip wheel 实际对着 TF 2.21.0 编译(其 metadata 误标 2.18.1);装错版本 → `libdeepmd_op.so: undefined symbol: ...absl...MixingHashState...`。
- **`e3nn` 必装**:3.2.0b0 的 PyTorch 描述符(sezm)`import e3nn`,缺了任何 `dp --pt` 命令在 import 阶段就崩。
- **LAMMPS**:`lmp` 来自 PyPI `lammps[mpi]~=2025.7.22.2.0`;`pair_style deepmd` 来自 deepmd wheel 自带的 `libdeepmd_lmp.so`,靠 `LAMMPS_PLUGIN_PATH` 自动加载;插件入口硬 `import tensorflow`(故 Mode C 必带 `tensorflow-cpu`)。**不要用 conda-forge 的 lammps**(它的 deepmd 插件按 conda-forge 的 libdeepmd/libtorch 编,与 pip 的 cu128/torch 不是同一 ABI,dlopen 必失败)。
- **conda-pack 打包**:env 含软链接目录,GNU tar 会对少数目录报 `Directory renamed before its status could be extracted`(**良性**:文件已解全,只是没设目录时间戳,退码非 0)→ 自解压脚本里 `tail … | tar -xz || true` + 校验 `bin/conda-unpack` 存在;**别加 `--delay-directory-restore`**(反而让所有目录都报)。
- **relocation(异地可用)**:激活钩子用 `$CONDA_PREFIX` **动态**算 `LAMMPS_PLUGIN_PATH` + `LD_LIBRARY_PATH`(后者指向 torch 自带的 `nvidia-*-cu12` 库),**绝不硬编码绝对路径**;`conda-unpack` 一次性(跑完别再移动该目录)。已实测:解压到全新目录 → `source bin/activate`(裸 source,无 conda)→ `conda-unpack` → lmp MD 跑通。
- **NVIDIA 兼容**:cuXXX 包靠向后兼容跑在更新的驱动上——cu126 / cu128 / cu130 在 T4 / 13.0 驱动上 GPU 均可见可用(`torch.cuda.is_available()=True`)。
- **已知 bug**:`dp test`(Python 推理路径)在 3.2.0b0 有 TorchScript JIT bug(`ModelOutputDef is not found`),**不影响 LAMMPS(C++ 路径)与 train/freeze**;待报 deepmd upstream。

## 5. 平台踩坑(Bohrium)

- 安装/解压必须用**本地盘**,别装到 `/personal`(阿里云 NFS,constructor/解压报 `InvalidArchiveError`)。
- 大文件下载偏慢(~9 MB/s)、首次可能瞬时中断;dpack 的 `-C -` 续传重跑即可接上。
- 旧节点 `git` 到 github.com 偶发 GnuTLS TLS 错,但 `pip` / `curl` 走 HTTPS 正常(release 资产上传/下载、api.github.com 均通)。
