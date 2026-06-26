#!/usr/bin/env bash
# build_modec.sh — Mode C：为某个 CUDA 自编一个离线 .sh 安装器(固化已验证的配方)
#
# 用法:
#   bash scripts/build_modec.sh cu126        # 产出 cuda126 包
#   bash scripts/build_modec.sh cu128        # cuda128
#   bash scripts/build_modec.sh cu130        # cuda130(13.0)
#
# 产出: /tmp/deepmd-kit-<VER>-<DATE>-<HASH>-cuda<XXX>-Linux-x86_64.sh  (+ sha256)
#
# 环境变量(CI 用):
#   SKIP_GPU_CHECK=1   跳过需要 GPU 的自检(GitHub runner 无 GPU);仍保留 import/打包完整性校验。
#                      真正的 GPU 验证(torch.cuda + train/freeze/lammps)在玻尔做(见 verify_and_publish.sh)。
#   BUILD_HASH=<hash>  覆盖文件名里的短哈希(CI 用 git rev-parse --short HEAD)。
#   BUILD_DATE=<YYYYMMDD>  覆盖日期。
#
# 说明:换 CUDA 只改 torch 的 cuXXX;tensorflow-cpu 是 CPU 版(仅为加载 lmp 插件,不做 GPU 计算),
#       所以 13.0 也用同一套配方(GPU 后端 = torch cu130)。
set -euo pipefail

CU="${1:?用法: bash build_modec.sh cu126|cu128|cu130}"
NUM="${CU#cu}"                                   # cu126 -> 126
VARIANT="cuda${NUM}"                             # -> cuda126
VER="${DEEPMD_VER:-3.2.0b0}"
DATE="${BUILD_DATE:-$(date +%Y%m%d)}"
HASH="${BUILD_HASH:-$(git rev-parse --short HEAD 2>/dev/null || echo manual)}"
TF_VER="${TF_VER:-2.21.0}"                       # 必须匹配 deepmd wheel 编译的 TF
TORCH_SPEC="${TORCH_SPEC:-torch==2.11.*}"
LAMMPS_SPEC="${LAMMPS_SPEC:-lammps[mpi]~=2025.7.22.2.0}"
ENV="/tmp/build-$VARIANT"
TAR="/tmp/${VARIANT}.tar.gz"
OUTSH="/tmp/deepmd-kit-${VER}-${DATE}-${HASH}-${VARIANT}-Linux-x86_64.sh"

echo "==================================================================="
echo " build_modec: $VARIANT   (torch $CU + tensorflow-cpu $TF_VER)"
echo " 产出: $OUTSH   SKIP_GPU_CHECK=${SKIP_GPU_CHECK:-0}"
echo "==================================================================="

command -v conda >/dev/null || { echo "ERROR: 需要 conda"; exit 1; }

echo "==> [1/7] 建环境 $ENV"
rm -rf "$ENV"; conda create -y -p "$ENV" python=3.11 >/dev/null

echo "==> [2/7] 装 torch($CU) + deepmd $VER + e3nn + mpich"
"$ENV/bin/pip" install $TORCH_SPEC --index-url "https://download.pytorch.org/whl/$CU"
"$ENV/bin/pip" install --pre "deepmd-kit==$VER" e3nn mpich

echo "==> [3/7] 装 tensorflow-cpu==$TF_VER(加载 lmp 插件必须)+ lammps wheel"
"$ENV/bin/pip" install "tensorflow-cpu==$TF_VER"
"$ENV/bin/pip" install --no-deps "$LAMMPS_SPEC"
"$ENV/bin/pip" cache purge >/dev/null 2>&1 || true   # 回收 ~数 GB pip 缓存,降磁盘峰值

echo "==> [4/7] 自检"
if [ -z "${SKIP_GPU_CHECK:-}" ]; then
  # 本地/玻尔:完整 GPU 自检
  "$ENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "GPU 不可见(这台没 GPU 或驱动/CUDA 不匹配)"
print("   torch", torch.__version__, "| cuda(built)", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0))
from deepmd.pt.utils import env as e; print("   deepmd-pt DEVICE", e.DEVICE)
import deepmd.lmp; print("   lmp plugin dir", deepmd.lmp.get_op_dir())
PY
else
  # CI(无 GPU):只验装包完整性,不碰 torch.cuda;真 GPU 验证在玻尔
  echo "   SKIP_GPU_CHECK=1 → 只做 import/打包完整性校验(GPU 验证留给玻尔验证道)"
  "$ENV/bin/python" - <<'PY'
import torch
print("   torch", torch.__version__, "| cuda(built)", torch.version.cuda)
import deepmd
from deepmd.pt.utils import env  # 触发 PT 后端加载(纯 import,不需 GPU)
import deepmd.lmp, os
d = deepmd.lmp.get_op_dir()
assert d and os.path.isdir(d), f"lmp 插件目录不存在: {d!r}"
print("   deepmd", getattr(deepmd, "__version__", "?"), "| lmp plugin dir", d)
PY
fi

echo "==> [5/7] 写 relocation-safe 激活钩子(用 \$CONDA_PREFIX 动态算路径)"
mkdir -p "$ENV/etc/conda/activate.d"
cat > "$ENV/etc/conda/activate.d/zz-deepmd-lmp.sh" <<'HOOK'
_PY="$CONDA_PREFIX/bin/python"
export LAMMPS_PLUGIN_PATH="$("$_PY" -c 'import deepmd,os;print(os.path.join(os.path.dirname(deepmd.__file__),"lib"))')${LAMMPS_PLUGIN_PATH:+:$LAMMPS_PLUGIN_PATH}"
export LD_LIBRARY_PATH="$("$_PY" -c 'import os,glob,torch;b=os.path.dirname(torch.__file__);base=os.path.dirname(b);print(":".join([os.path.join(b,"lib")]+glob.glob(os.path.join(base,"nvidia","*","lib"))))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _PY
HOOK

echo "==> [6/7] conda-pack 打包 -> $TAR,然后立刻删环境降磁盘峰值"
"$ENV/bin/pip" install conda-pack >/dev/null
rm -f "$TAR"; "$ENV/bin/conda-pack" -p "$ENV" -o "$TAR" --ignore-missing-files
rm -rf "$ENV"                                         # 环境已打进 tar,删掉(峰值从 ~28G 降到 ~14G)

echo "==> [7/7] 包成自解压 .sh -> $OUTSH,然后删 tar"
cat > "$OUTSH" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
PREFIX=""
while [ $# -gt 0 ]; do case "$1" in
  -p) PREFIX="$2"; shift 2;;
  -b) shift;;
  *) shift;;
esac; done
[ -n "$PREFIX" ] || { echo "用法: bash $0 -b -p <安装目录>"; exit 1; }
mkdir -p "$PREFIX"
echo "解压到 $PREFIX ..."
ARCHIVE_LINE=$(awk '/^__ARCHIVE_BELOW__$/{print NR+1; exit}' "$0")
tail -n +"$ARCHIVE_LINE" "$0" | tar -xz -C "$PREFIX" || true
[ -x "$PREFIX/bin/conda-unpack" ] || { echo "解压不完整(缺 conda-unpack),安装失败"; exit 1; }
echo "conda-unpack(一次性) ..."
"$PREFIX/bin/conda-unpack"
echo "完成。激活: source $PREFIX/bin/activate ; dp --version ; lmp -h"
exit 0
__ARCHIVE_BELOW__
STUB
cat "$TAR" >> "$OUTSH"
chmod +x "$OUTSH"
rm -f "$TAR"

echo ""
echo "================== 完成 =================="
echo "OUTSH=$OUTSH"
ls -lh "$OUTSH"
sha256sum "$OUTSH"
