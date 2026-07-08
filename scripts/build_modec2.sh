#!/usr/bin/env bash
# build_modec2.sh — Mode C2:完全 CONDA-FREE 的离线包【原型】(先支持 cu126)。
#
# ⚠️⚠️ 原型 / 未验证:此脚本只在纸面按第一性原理设计,尚未在真机跑通。
#     请先用 scripts/verify_modec2.sh 在一台 glibc>=2.28 的机器上跑通 7 项验证,再谈推广。
#     它【不替换】现有 build_modec.sh / build.sh —— 是并列的新路子。
#
# 第一性原理:Mode C 里 conda 只干 3 件机械活,没一件非它不可,逐一换成 conda-free 等价物:
#   (1) conda create   → 提供 python 环境  → python-build-standalone(PBS)便携 CPython
#   (2) conda-pack     → 打成可搬的 tar     → 普通 tar -czf
#   (3) conda-unpack   → 搬家后修库路径     → 不需要:wheel 自带 $ORIGIN RPATH,本就可搬
#       + bin/activate → 激活              → 一个自定位的 env.sh(纯 shell,不碰 conda)
# pip 那一层(torch / deepmd / e3nn / mpich / tensorflow-cpu / lammps wheel)与 build_modec.sh
# 【逐字一致】—— 这层本就是 conda-free 的,不动。
#
# 用法:  bash scripts/build_modec2.sh cu126        # (或 cu128)
# 产出:  /tmp/deepmd-kit-<VER>-<DATE>-<HASH>-cuda<XXX>-nocondan-Linux-x86_64.sh
set -euo pipefail

CU="${1:?用法: bash build_modec2.sh cu126|cu128}"
NUM="${CU#cu}"; VARIANT="cuda${NUM}"
VER="${DEEPMD_VER:-3.2.0b0}"
DATE="${BUILD_DATE:-$(date +%Y%m%d)}"
HASH="${BUILD_HASH:-$(git rev-parse --short HEAD 2>/dev/null || echo manual)}"
TF_VER="${TF_VER:-2.21.0}"                       # 必须匹配 deepmd wheel 编译的 TF
TORCH_SPEC="${TORCH_SPEC:-torch==2.11.*}"
LAMMPS_SPEC="${LAMMPS_SPEC:-lammps[mpi]~=2025.7.22.2.0}"

# python-build-standalone(astral):钉死版本以可复现。cp311 与现配方的 wheel ABI 一致。
# 注意:PBS 每期 release 只上架【当期最新】小版本;PBS_PYVER 与 PBS_DATE 必须配对存在
# (3.11.9+20250612 不存在 → 404;20250612 期上架的是 3.11.13,已实测 HTTP 200)。
PBS_PYVER="${PBS_PYVER:-3.11.13}"
PBS_DATE="${PBS_DATE:-20250612}"
PBS_URL="${PBS_URL:-https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_DATE}/cpython-${PBS_PYVER}+${PBS_DATE}-x86_64-unknown-linux-gnu-install_only.tar.gz}"

ENV="/tmp/build-${VARIANT}-nc"
TAR="/tmp/${VARIANT}-nc.tar.gz"
OUTSH="/tmp/deepmd-kit-${VER}-${DATE}-${HASH}-${VARIANT}-nocondan-Linux-x86_64.sh"

echo "======================================================================"
echo " build_modec2 (CONDA-FREE 原型): $VARIANT"
echo "   torch $CU + tensorflow-cpu $TF_VER + PBS CPython $PBS_PYVER"
echo "   产出: $OUTSH"
echo "======================================================================"

# ── [1/7] 便携 CPython(替代 conda create;全程不碰 conda)────────────────────
echo "==> [1/7] 下载 python-build-standalone → $ENV/python"
rm -rf "$ENV"; mkdir -p "$ENV"
curl -fsSL "$PBS_URL" | tar -xz -C "$ENV"        # install_only:解出 $ENV/python/{bin,lib}/...
PY="$ENV/python/bin/python3"
[ -x "$PY" ] || PY="$ENV/python/bin/python${PBS_PYVER%.*}"
[ -x "$PY" ] || { echo "❌ 没找到 PBS python(布局可能变了,查 $ENV/python/bin)"; ls -l "$ENV/python/bin" || true; exit 1; }
"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$PY" -m pip install -q -U pip
# 保证 `python` 这个名字存在(verify_modec2.sh / verify_offline_modec.sh 调用 python 而非 python3;
# PBS install_only 布局不一定带 python 符号名,缺了会让 GPU 端到端项报 python: command not found)
[ -e "$ENV/python/bin/python" ] || ln -sf python3 "$ENV/python/bin/python"

# ── [2-3/7] pip 装栈(与 build_modec.sh 逐字一致,只是解释器换成 PBS)──────────
echo "==> [2/7] pip 装 torch($CU) + deepmd $VER + e3nn + mpich"
"$PY" -m pip install "$TORCH_SPEC" --index-url "https://download.pytorch.org/whl/$CU"
"$PY" -m pip install --pre "deepmd-kit==$VER" e3nn mpich
echo "==> [3/7] pip 装 tensorflow-cpu==$TF_VER(加载 lmp 插件必须)+ lammps wheel"
"$PY" -m pip install "tensorflow-cpu==$TF_VER"
"$PY" -m pip install --no-deps "$LAMMPS_SPEC"
"$PY" -m pip cache purge >/dev/null 2>&1 || true

# ── [4/7] import 自检(不需 GPU)—— 这是本方案最大的未知点:PBS 吃不吃得下这些 wheel ──
echo "==> [4/7] import 自检(PBS 解释器 + 这些 wheel 能否装上并导入)"
"$PY" - <<'PYEOF'
import torch; print("   torch", torch.__version__, "| cuda(built)", torch.version.cuda)
import deepmd; print("   deepmd", getattr(deepmd, "__version__", "?"))
import deepmd.lmp, os
d = deepmd.lmp.get_op_dir(); assert d and os.path.isdir(d), f"lmp 插件目录不存在: {d!r}"
print("   lmp plugin dir", d)
import tensorflow as tf; print("   tensorflow", tf.__version__)
print("   ✅ import 自检通过")
PYEOF

# ── [5/7] 写自定位 env.sh(替代 conda activate;纯 shell,source 时不跑 python)──
# 用【单引号 heredoc】写字面内容:构建时不做任何展开;所有路径在 source 时由 $BASH_SOURCE 自算。
echo "==> [5/7] 生成 conda-free 激活脚本 env.sh"
cat > "$ENV/env.sh" <<'ENVSH'
# conda-free 激活:  source <prefix>/env.sh
# 自定位到本文件所在目录,设 PATH + 原生库路径 —— 全程不调用、不依赖、不修改任何 conda。
_DP="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd)"
export PATH="$_DP/python/bin:$PATH"
_SP="$(echo "$_DP"/python/lib/python*/site-packages)"
# LD_LIBRARY_PATH:torch 自带 + nvidia/* CUDA 运行时 + tensorflow + mpich(lammps 靠 $ORIGIN 自解析)
_LD="$_SP/torch/lib"
for _d in "$_SP"/nvidia/*/lib; do [ -d "$_d" ] && _LD="$_LD:$_d"; done
[ -d "$_SP/tensorflow" ] && _LD="$_LD:$_SP/tensorflow"
for _d in "$_SP"/mpich/lib; do [ -d "$_d" ] && _LD="$_LD:$_d"; done
export LD_LIBRARY_PATH="$_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# deepmd 的 LAMMPS 插件 .so 所在目录
export LAMMPS_PLUGIN_PATH="$_SP/deepmd/lib${LAMMPS_PLUGIN_PATH:+:$LAMMPS_PLUGIN_PATH}"
unset _DP _SP _LD _d
ENVSH

# ── [6/7] 重写 console-script 的 shebang → 可重定位 ─────────────────────────
# pip 装出来的 dp/lmp/pip 首行是构建时的绝对路径;改成 env-python,配合 env.sh 把 python/bin
# 放到 PATH 最前(所以用户【必须先 source env.sh】再用 dp/lmp)。只改 shebang 里含 python 的脚本,
# 跳过二进制/符号链接(python3 等)。
echo "==> [6/7] 重写 shebang + 断言"
for f in "$ENV"/python/bin/*; do
  [ -f "$f" ] || continue
  IFS= read -r first < "$f" 2>/dev/null || continue
  case "$first" in
    "#!"*python*) sed -i '1s|^#!.*|#!/usr/bin/env python3|' "$f" ;;
  esac
done
for s in dp lmp pip; do
  f="$ENV/python/bin/$s"
  [ -e "$f" ] || { echo "❌ 缺 $s(未随 wheel 装上?)"; exit 1; }
  first="$(head -1 "$f" 2>/dev/null || true)"
  case "$first" in
    "#!"*python3) : ;;                                       # python 脚本:shebang 已改 env python3 ✓
    "#!"*)        echo "❌ $s 是脚本但 shebang 未改成 env python3(首行: $first)"; exit 1 ;;
    *)            echo "   $s 非脚本(ELF 二进制),靠 env.sh LD_LIBRARY_PATH 重定位 ✓" ;;
  esac
done

# ── [7/7] 普通 tar(无 conda-pack)→ 自解压 .sh(无 conda-unpack)────────────
echo "==> [7/7] tar + 自解压 stub → $OUTSH"
rm -f "$TAR"; tar -czf "$TAR" -C "$ENV" .
cat > "$OUTSH" <<'STUB'
#!/usr/bin/env bash
# conda-free 离线安装器:解压到 <prefix>,然后 `source <prefix>/env.sh`。无需 conda。
set -euo pipefail
PREFIX=""
while [ $# -gt 0 ]; do case "$1" in
  -p) PREFIX="$2"; shift 2;; -b) shift;; *) shift;;
esac; done
[ -n "$PREFIX" ] || { echo "用法: bash $0 -b -p <安装目录>"; exit 1; }
# glibc 预检(manylinux wheel 要求 >=2.28)
gv="$(ldd --version 2>/dev/null | sed -n '1s/.* //p')"
case "$gv" in
  ''|2.1[0-9]|2.2[0-7]) echo "⚠ 本机 glibc=${gv:-未知},低于 2.28 —— 可能装不上/跑不起(本包需 glibc>=2.28)";;
esac
mkdir -p "$PREFIX"
echo "解压到 $PREFIX ..."
LINE=$(awk '/^__ARCHIVE_BELOW__$/{print NR+1; exit}' "$0")
tail -n +"$LINE" "$0" | tar -xz -C "$PREFIX"
[ -x "$PREFIX/python/bin/dp" ] || { echo "❌ 解压不完整(缺 python/bin/dp)"; exit 1; }
echo "完成(conda-free)。激活并使用:"
echo "  source $PREFIX/env.sh"
echo "  dp --version ; lmp -h"
exit 0
__ARCHIVE_BELOW__
STUB
cat "$TAR" >> "$OUTSH"; chmod +x "$OUTSH"; rm -f "$TAR"

echo ""; echo "================== 完成(原型)=================="
echo "OUTSH=$OUTSH"; ls -lh "$OUTSH"; sha256sum "$OUTSH"
echo "下一步:在 glibc>=2.28 的机器上跑  bash scripts/verify_modec2.sh $OUTSH"
