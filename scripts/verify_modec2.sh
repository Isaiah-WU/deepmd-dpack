#!/usr/bin/env bash
# verify_modec2.sh — 验证 conda-free 原型(build_modec2.sh 的产物)的 7 项关键点。
#
# 这 7 项是"只能实测、不能靠推理"的:每一项对应对抗验证列出的一个未知/风险。
# 用法:  bash scripts/verify_modec2.sh /tmp/deepmd-kit-...-cuda126-nocondan-Linux-x86_64.sh
#   - 第 4、5 项(GPU 端到端 / MPI 多进程)需要在带 NVIDIA GPU 的节点上跑;无 GPU 会跳过。
#   - 第 6 项(和用户 conda 不冲突)最好在一台【已装 conda】的机器上跑。
#   - 第 7 项(离线)需要 unshare(需要用户命名空间权限)。
# 不改任何东西,只做检查;末尾给 PASS/FAIL 汇总,全过 exit 0。
set -uo pipefail

SH="${1:?用法: bash scripts/verify_modec2.sh <installer.sh>}"
SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$SH" ] || { echo "找不到安装包: $SH"; exit 1; }
pass=0; fail=0; skip=0
ok(){ echo "   ✅ $*"; pass=$((pass+1)); }
no(){ echo "   ❌ $*"; fail=$((fail+1)); }
sk(){ echo "   ⏭  $*"; skip=$((skip+1)); }

echo "============================================================"
echo " verify_modec2 —— conda-free 原型 7 项验证"
echo "   安装包: $SH"
echo "   本机 glibc: $(ldd --version 2>/dev/null | sed -n '1s/.* //p')"
echo "============================================================"

# ── [1] 安装 + import:PBS 解释器吃得下 torch/deepmd/lammps/tf 吗(#1 未知)──────
echo "== [1] 安装 + import(PBS + 这些 wheel)=="
rm -rf /tmp/dp1
if bash "$SH" -b -p /tmp/dp1 >/tmp/dp1.install.log 2>&1; then
  if /tmp/dp1/python/bin/python -c \
     "import torch,deepmd,deepmd.lmp,tensorflow;print('import OK', torch.__version__)"; then
    ok "安装 + import 全过"
  else no "import 失败(PBS 吃不下某个 wheel?见上)"; fi
else no "安装(自解压)失败,见 /tmp/dp1.install.log"; tail -15 /tmp/dp1.install.log; fi

# ── [2] 重定位:换个不同目录还能跑(shebang 重写 + $ORIGIN 是否真可搬)──────────
echo "== [2] 重定位到另一个目录 =="
rm -rf /tmp/dp2
if bash "$SH" -b -p /tmp/dp2 >/dev/null 2>&1; then
  echo "   dp2 的 dp shebang: $(head -1 /tmp/dp2/python/bin/dp)"
  if ( set +u; source /tmp/dp2/env.sh; set -u
       dp --version >/dev/null 2>&1 && python -c "import deepmd" >/dev/null 2>&1 ); then
    ok "在 dp2 目录可跑(可重定位)"
  else no "dp2 目录跑不起来(shebang/重定位有问题)"; fi
else no "dp2 安装失败"; fi

# ── [3] env.sh 正确性:source 不报错、库目录真存在、不跑 python ─────────────────
echo "== [3] env.sh 正确性 =="
if ( set -e; source /tmp/dp1/env.sh
     [ -n "${LD_LIBRARY_PATH:-}" ] && [ -d "${LAMMPS_PLUGIN_PATH%%:*}" ] ); then
  ok "env.sh source 无错,LAMMPS_PLUGIN_PATH 目录存在"
  ( source /tmp/dp1/env.sh
    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | while read -r d; do
      [ -z "$d" ] || [ -d "$d" ] || echo "   ⚠ LD_LIBRARY_PATH 里不存在的目录: $d"; done )
else no "env.sh source 出错或路径不对"; fi

# ── [4] GPU 端到端:train → freeze → lammps(需 GPU 节点)──────────────────────
echo "== [4] GPU 端到端(需 GPU)=="
if command -v nvidia-smi >/dev/null 2>&1; then
  if ( set +u; source /tmp/dp1/env.sh; set -u; bash "$SKILL/scripts/verify_offline_modec.sh" ); then
    ok "train → freeze → lammps 通过"
  else no "GPU 端到端失败"; fi
else sk "无 GPU,跳过(务必在 GPU 节点重跑这一项)"; fi

# ── [5] 打包内 MPICH 多进程:用【包内】mpirun,别用系统的 ───────────────────────
echo "== [5] bundled MPICH 多进程 =="
if command -v nvidia-smi >/dev/null 2>&1 && [ -x /tmp/dp1/python/bin/mpirun ]; then
  printf 'units lj\natom_style atomic\nlattice fcc 0.8442\nregion b block 0 4 0 4 0 4\ncreate_box 1 b\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nrun 0\n' > /tmp/in.mini
  if ( set +u; source /tmp/dp1/env.sh; set -u
       /tmp/dp1/python/bin/mpirun -n 2 lmp -in /tmp/in.mini ) >/tmp/dp1.mpi.log 2>&1; then
    ok "包内 mpirun -n 2 lmp 通过"
  else no "多进程 MPI 失败,见 /tmp/dp1.mpi.log"; tail -15 /tmp/dp1.mpi.log; fi
else sk "无 GPU 或包内无 mpirun,跳过"; fi

# ── [6] 和用户已有 conda 不冲突(最关键的解耦证据)────────────────────────────
echo "== [6] 用户 conda 激活时不冲突 =="
if command -v conda >/dev/null 2>&1; then
  if ( set +u
       eval "$(conda shell.bash hook)" 2>/dev/null; conda activate base 2>/dev/null
       before="${CONDA_PREFIX:-}"
       source /tmp/dp1/env.sh
       [ "${CONDA_PREFIX:-}" = "$before" ] || { echo "   CONDA_PREFIX 被改了"; exit 1; }
       case "$(command -v dp)" in /tmp/dp1/*) : ;; *) echo "   dp 未指向我们的包: $(command -v dp)"; exit 1;; esac
       dp --version >/dev/null 2>&1 ); then
    ok "用户 conda 激活着也不冲突,dp 指向本包"
  else no "和用户 conda 有冲突"; fi
else sk "本机无 conda —— 这条最能证明解耦,请在装了 conda 的机器上重跑"; fi

# ── [7] 全程离线(装 + 激活不联网)───────────────────────────────────────────
echo "== [7] 断网下装 + 激活 =="
if command -v unshare >/dev/null 2>&1; then
  rm -rf /tmp/dp3
  if unshare -rn bash -c \
     'bash "'"$SH"'" -b -p /tmp/dp3 >/dev/null 2>&1 && set +u && source /tmp/dp3/env.sh && dp --version >/dev/null 2>&1'; then
    ok "断网下装 + 激活 + dp 可跑"
  else no "疑似安装/激活需要联网"; fi
else sk "无 unshare,跳过"; fi

echo ""
echo "============================================================"
echo " 结果:PASS=$pass  FAIL=$fail  SKIP=$skip"
echo "============================================================"
[ "$fail" -eq 0 ]
