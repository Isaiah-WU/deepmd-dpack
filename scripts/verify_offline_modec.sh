#!/usr/bin/env bash
# verify_offline_modec.sh — 在【已激活】的 Mode C 环境里做 GPU 端到端验证。
#   要求:torch 看得见 GPU → dp --pt train(合成数据,10 步)→ dp --pt freeze → lammps MD 跑完。
#   全过 exit 0;任一步失败 exit 非 0。供 verify_and_publish.sh(玻尔 cron)调用。
# 用法:先 `source <prefix>/bin/activate`,再 `bash verify_offline_modec.sh`。
set -euo pipefail

command -v dp  >/dev/null || { echo "dp 不在 PATH(环境没激活?)"; exit 1; }
command -v lmp >/dev/null || { echo "lmp 不在 PATH"; exit 1; }

python - <<'PY'
import torch
assert torch.cuda.is_available(), "GPU 不可见(驱动/CUDA 不匹配,或不在 GPU 节点)"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cuda", torch.version.cuda)
PY

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
lmp -in in.lammps >lmp.log 2>&1 || { echo "❌ lmp 运行失败"; tail -20 lmp.log; exit 1; }
grep -qE 'Loading plugin.*[Dd]eepmd' lmp.log || { echo "❌ lmp 未加载 deepmd 插件"; tail -20 lmp.log; exit 1; }
grep -qE 'Total wall time' lmp.log        || { echo "❌ lmp run 未完成"; tail -20 lmp.log; exit 1; }

echo "✅ GPU 端到端验证通过:torch GPU + dp train + dp freeze + lammps MD"
