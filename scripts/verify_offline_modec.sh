#!/usr/bin/env bash
# verify_offline_modec.sh — 在【已激活】的 Mode C/C2 环境里做端到端验证。
#   要求:torch 看得见 GPU(CPU 模式除外)→ dp --pt train(合成数据,10 步)→ dp --pt freeze → lammps MD 跑完。
#   全过 exit 0;任一步失败 exit 非 0。作为一次性端到端验证手动运行(GPU 包在 GPU 节点)。
# 用法:先 `source <prefix>/bin/activate`(C2 包用 `source <prefix>/env.sh`),再 `bash verify_offline_modec.sh`。
#   CPU 包:DP_DEVICE=cpu bash verify_offline_modec.sh(跳过 GPU 断言,任意机器可跑)。
set -euo pipefail

command -v dp  >/dev/null || { echo "dp 不在 PATH(环境没激活?)"; exit 1; }
command -v lmp >/dev/null || { echo "lmp 不在 PATH"; exit 1; }

if [ "${DP_DEVICE:-gpu}" = "cpu" ]; then
python - <<'PY'
import torch
print("CPU 模式 | torch", torch.__version__)
PY
else
python - <<'PY'
import torch
assert torch.cuda.is_available(), "GPU 不可见(驱动/CUDA 不匹配,或不在 GPU 节点)"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cuda", torch.version.cuda)
PY
fi

D=$(mktemp -d); cd "$D"
python - <<'PY'
import numpy as np, os, json
s=os.path.join(os.getcwd(),"sys"); os.makedirs(s+"/set.000",exist_ok=True)
open(s+"/type.raw","w").write("0 0 1 1 1 1")
open(s+"/type_map.raw","w").write("O\nH\n"); open(s+"/nopbc","w").close()
r=np.random.default_rng(0)
np.save(s+"/set.000/coord.npy",  r.uniform(0,5,(5,18)).astype("f8"))
np.save(s+"/set.000/energy.npy", r.uniform(-1,1,(5,)).astype("f8"))
np.save(s+"/set.000/force.npy",  r.uniform(-1,1,(5,18)).astype("f8"))
inp={"model":{"type_map":["O","H"],
      "descriptor":{"type":"se_e2_a","sel":[16,16],"rcut_smth":0.5,"rcut":6.0,"neuron":[25,50,100],"axis_neuron":16,"seed":1},
      "fitting_net":{"neuron":[120,120,120],"seed":1}},
     "learning_rate":{"type":"exp","start_lr":1e-3,"stop_lr":1e-8,"decay_steps":5000},
     "loss":{"type":"ener","start_pref_e":0.02,"limit_pref_e":1,"start_pref_f":1000,"limit_pref_f":1},
     "training":{"training_data":{"systems":["./sys"],"batch_size":1},"numb_steps":10,"seed":10,"disp_freq":5,"save_freq":10}}
json.dump(inp,open("in.json","w"),indent=2)
PY

dp --pt train in.json >train.log 2>&1 || { echo "❌ train 失败"; tail -20 train.log; exit 1; }
dp --pt freeze -o frozen.pth >/dev/null 2>&1 || { echo "❌ freeze 失败"; exit 1; }

printf 'LAMMPS data file\n\n6 atoms\n2 atom types\n0 12 xlo xhi\n0 12 ylo yhi\n0 12 zlo zhi\n\nMasses\n\n1 1.008\n2 15.999\n\nAtoms\n\n1 1 6 6 6\n2 1 6.5 7 6\n3 2 8 6 6\n4 1 4 6 6\n5 1 3.5 7 6\n6 2 5 6 6\n' > data.lmp
printf 'units metal\natom_style atomic\nboundary p p p\nread_data data.lmp\npair_style deepmd frozen.pth\npair_coeff * * H O\nthermo 1\nrun 3\n' > in.lammps
# 不只看退出码(deepmd 在 GPU 崩溃时退出码可能仍是 0,如 cuda130 的 MPI_Abort),看实际产出判定:
lmp -in in.lammps >lmp.log 2>&1 || true
# ① 命中明确错误标志 → 判失败(即使退出码 0)。
#    "Cannot find libcudart" 不会误伤成功时的 "Successfully load libcudart"(子串不同)。
#    CPU 模式例外:CPU 版 torch 本就没有 libcudart,deepmd 的 dlopen 探测会打印这句后
#    正常走 CPU(设计如此),不是错误标志——只在 GPU 模式下把它列入黑名单。
ERRPAT='Unknown pair style|MPI_Abort|terminate called'
[ "${DP_DEVICE:-gpu}" = "cpu" ] || ERRPAT="Cannot find libcudart|$ERRPAT"
if grep -qiE "$ERRPAT" lmp.log; then
  echo "❌ lammps 出现错误标志(见下)"; tail -25 lmp.log; exit 1
fi
# ② 必须真跑完(崩在半路不会打印这行)——核心成功信号。
grep -qE 'Total wall time' lmp.log || { echo "❌ lammps 未跑完(无 Total wall time,疑似中途崩)"; tail -25 lmp.log; exit 1; }
# ③ deepmd 是否参与:Mode A 可能编进 lmp、Mode C 走插件,措辞不同;只告警不判失败,避免误伤。
grep -qiE 'deepmd' lmp.log || echo "⚠ 日志未见 'deepmd' 字样(已跑完;通常只是措辞差异)"

echo "✅ 端到端验证通过:torch(${DP_DEVICE:-gpu}) + dp train + dp freeze + lammps MD(跑完、无错误标志)"
