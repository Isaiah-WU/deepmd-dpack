#!/usr/bin/env bash
# verify_and_publish.sh — Mode C 的"验证道",在玻尔 GPU 节点上 cron 运行。
#
# 对每个 cuda 变体:从 release 下载 CI 当天构建的【最新】分片 → 拼接 → GPU 端到端验证
# (verify_offline_modec.sh:torch.cuda + dp train + dp freeze + lammps)→ 通过才把该变体写进
# manifest.json 并 push。**这是 dpack 的唯一放行闸:manifest 只收录 GPU 验证过的变体。**
#
# 安全/凭证(重要):
#   - GH_PAT = 细粒度 PAT,仅对本 repo 的 Contents: write。短期、用完即弃(玻尔节点回收)。
#   - 该凭证别写进脚本、别留在被下载 .sh 能读到的目录。它是本方案最高价值的秘密。
#   - sha256 在【GPU 验证的这同一份字节】上重算,作为写入 manifest 的哈希(不信任 CI 算的)。
#
# 用法(玻尔 cron 或手动):
#   export GH_PAT=github_pat_xxx
#   export VARIANTS="cuda126 cuda128 cuda130"     # 可选,默认这三个
#   bash verify_and_publish.sh
set -euo pipefail

REPO="${REPO:-Isaiah-WU/deepmd-dpack}"
TAG="${TAG:-v$(cat assets/version.txt 2>/dev/null || echo 3.2.0b0)}"
VARIANTS="${VARIANTS:-cuda126 cuda128 cuda130}"
WORK="${WORK:-/tmp/verify-publish}"
: "${GH_PAT:?需要细粒度 PAT(对 $REPO Contents:write);别写进脚本/别留在节点}"
command -v conda >/dev/null || { echo "需要 conda"; exit 1; }

rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
gh_get() { curl -fsSL -H "Authorization: Bearer $GH_PAT" "$@"; }

echo "==> clone repo(用于改 manifest + push)"
git clone "https://x-access-token:${GH_PAT}@github.com/${REPO}.git" repo
SKILL="$WORK/repo"

echo "==> 取 release 资产清单"
ASSETS=$(gh_get "https://api.github.com/repos/${REPO}/releases/tags/${TAG}")

PASS=()
for V in $VARIANTS; do
  echo "===================== 验证 $V ====================="
  # 找该变体【日期最大】的一组分片的基名(deepmd-kit-...-<V>-Linux-x86_64.sh)
  BASE=$(echo "$ASSETS" | V="$V" python -c "
import sys,json,re,os
v=os.environ['V']
names=[a['name'] for a in json.load(sys.stdin).get('assets',[])]
bases=sorted({re.sub(r'\.sh\.[0-9]+$','.sh',n) for n in names if re.search(rf'-{v}-Linux-x86_64\.sh\.[0-9]+$',n)})
print(bases[-1] if bases else '')
")
  [ -n "$BASE" ] || { echo "  release 上没有 $V 的分片,跳过"; continue; }
  echo "  最新构建: $BASE"

  d="$WORK/$V"; rm -rf "$d"; mkdir -p "$d"; cd "$d"
  for i in 0 1 2; do
    gh_get -o "$BASE.$i" "https://github.com/${REPO}/releases/download/${TAG}/${BASE}.$i"
  done
  cat "$BASE".[0-9]* > "$BASE"
  SHA=$(sha256sum "$BASE" | cut -d' ' -f1)         # 在将要 GPU 验证的这份字节上重算
  echo "  sha256=$SHA"

  echo "  -- 安装 + GPU 端到端验证 --"
  rm -rf "$d/inst"; bash "$BASE" -b -p "$d/inst"
  set +e
  ( source "$d/inst/bin/activate" && bash "$SKILL/scripts/verify_offline_modec.sh" )
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "  ❌ $V GPU 验证失败(rc=$rc),不入 manifest,保留上一份已验证条目不动"
    cd "$WORK"; continue
  fi

  echo "  -- 生成 manifest 片段(用 GPU 验过的字节 + Mode C 后端)--"
  cd "$SKILL"
  num="${V#cuda}"; cuda_dot="${num:0:2}.${num:2}"   # cuda126 -> 12.6
  mkdir -p "dist/$V"; cp "$d/$BASE".[0-9]* "dist/$V/"
  REPO="$REPO" CUDA="$cuda_dot" VERSION="${TAG#v}" \
  BACKEND="pytorch" \
  NOTE="Mode C (conda-pack self-extracting). PyTorch GPU backend; CPU-only TensorFlow bundled to load the LAMMPS plugin." \
    python scripts/gen_manifest_fragment.py
  # 防漂移:核对片段里的 sha 与本地重算一致
  grep -q "$SHA" frag/*.json || { echo "  ❌ 片段 sha 与重算不符,放弃 $V"; cd "$WORK"; continue; }
  PASS+=("$V")
  cd "$WORK"
done

if [ ${#PASS[@]} -eq 0 ]; then
  echo "本轮无变体通过 GPU 验证,manifest 未变"; exit 0
fi

echo "==> 合并通过的片段 → manifest.json → commit + push"
cd "$SKILL"
python scripts/merge_manifest.py
git config user.name  "bohrium-verify[bot]"
git config user.email "bohrium-verify@users.noreply.github.com"
git add assets/manifest.json
if git diff --cached --quiet; then
  echo "manifest 无变化"
else
  git commit -m "verify: 发布 GPU 验证通过的变体 ${PASS[*]} 到 manifest"
  git push
  echo "已发布(GPU 验过):${PASS[*]}"
fi
