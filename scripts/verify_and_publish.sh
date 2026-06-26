#!/usr/bin/env bash
# verify_and_publish.sh — Mode C 验证道(玻尔 GPU cron)。
#
# 流程:从 release 自动发现各变体当天最新的分片 → 匿名下载 → 断言大小后拼接 → GPU 端到端验证
#       → 在验过的字节上重算 sha256 → 重生成 manifest 片段(BACKEND=pytorch)→ 合并 → 只在通过后 push。
# manifest 是 dpack 唯一索引,且只由本脚本写 = 真门。
#
# 安全模型(关键):
#   - 公开 repo + 公开 release:clone 与下载全部【匿名】,GH_PAT 不落磁盘(不进 .git/config)。
#   - 运行下载来的(第三方)安装包时用 `env -u GH_PAT` 擦掉密钥 → 不可信代码读不到 PAT(环境/磁盘都读不到)。
#   - GH_PAT 只在最后一步 push 时用(此时不可信代码早已跑完)。
#   - 分片数与变体列表都【从 release 资产清单推导】,不写死;拼接后断言总大小=各分片之和才算 sha。
#
# 用法:
#   export GH_PAT=github_pat_xxx              # 细粒度 PAT,仅对本 repo Contents:write
#   export VARIANTS="cuda126 cuda128 cuda130" # 可选;留空=自动发现 release 上所有变体
#   bash verify_and_publish.sh
set -euo pipefail

REPO="${REPO:-Isaiah-WU/deepmd-dpack}"
VARIANTS="${VARIANTS:-}"            # 空=自动发现 release 上所有 cudaXXX/cpu 变体
WORK="${WORK:-/tmp/verify-publish}"
DRY_RUN="${DRY_RUN:-0}"            # 1=只下载+GPU验证+算 manifest 差异,不提交不 push(无需 PAT)
# DRY_RUN 时全程匿名,不需要 token;只有真发布才需要 GH_PAT。
[ "$DRY_RUN" = "1" ] || : "${GH_PAT:?需要细粒度 PAT(对 $REPO Contents:write);只在最后 push 用。或设 DRY_RUN=1 先空跑}"
PY="$(command -v python3 || command -v python || true)"; : "${PY:?需要 python}"
command -v conda >/dev/null || { echo "需要 conda"; exit 1; }

rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"

echo "==> 匿名 clone(token 不落磁盘)"
git clone "https://github.com/${REPO}.git" repo
SKILL="$WORK/repo"

# 单一滚动通道:固定 tag=nightly;deepmd 版本取自 version.txt(写进 manifest 的 version 字段)。
TAG="${TAG:-nightly}"
VER="$(cat "$SKILL/assets/version.txt" 2>/dev/null || echo 3.2.0b0)"
echo "==> 目标 release: $TAG (deepmd $VER)"

echo "==> 取 release 资产清单(匿名)"
ASSETS="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/tags/${TAG}")"

# 自动发现变体
if [ -z "$VARIANTS" ]; then
  VARIANTS="$(printf '%s' "$ASSETS" | "$PY" - <<'PY'
import sys, json, re
data = json.load(sys.stdin)
vs = set()
for a in data.get("assets", []):
    m = re.search(r"-(cuda\d+|cpu)-Linux-x86_64\.sh\.[0-9]+$", a["name"])
    if m:
        vs.add(m.group(1))
print(" ".join(sorted(vs)))
PY
)"
fi
echo "   要验证的变体: ${VARIANTS:-（无）}"

PASS=()
for V in $VARIANTS; do
  echo "===================== 验证 $V ====================="
  # 取该变体【日期最新】一组分片:第1行 base、第2行 按数字序的分片名、第3行 期望总字节数
  INFO="$(printf '%s' "$ASSETS" | V="$V" "$PY" - <<'PY'
import sys, json, re, os
V = os.environ["V"]
data = json.load(sys.stdin)
sizes = {a["name"]: a.get("size", 0) for a in data.get("assets", [])}
pat = re.compile(r"^(?P<base>deepmd-kit-.*-" + re.escape(V) + r"-Linux-x86_64\.sh)\.(?P<idx>[0-9]+)$")
groups = {}
for n in sizes:
    m = pat.match(n)
    if m:
        groups.setdefault(m.group("base"), []).append((int(m.group("idx")), n, sizes[n]))
if groups:
    def datekey(b):
        m = re.search(r"-(\d{8})-", b)
        return m.group(1) if m else "0"
    base = sorted(groups, key=datekey)[-1]          # 取日期最新的一组
    parts = sorted(groups[base], key=lambda t: t[0]) # 数字序(10+ 片也正确)
    print(base)
    print(" ".join(p[1] for p in parts))
    print(sum(p[2] for p in parts))
PY
)"
  BASE="$(printf '%s\n' "$INFO" | sed -n 1p)"
  NAMES="$(printf '%s\n' "$INFO" | sed -n 2p)"
  TOTAL="$(printf '%s\n' "$INFO" | sed -n 3p)"
  [ -n "$BASE" ] || { echo "  release 上没有 $V 的分片,跳过"; continue; }
  echo "  最新构建: $BASE  ($(printf '%s' "$NAMES" | wc -w) 片, 期望 ${TOTAL}B)"

  # 变了才验:若该变体在 release 上的最新构建,已经是 manifest 当前发布的那一份,
  # 就跳过——不下载、不安装、不碰 GPU。(所有变体都跳 → 本轮根本不开 GPU。)
  pub="$(BASE="$BASE" MV="$V" "$PY" - "$SKILL/assets/manifest.json" <<'PY'
import json, os, sys
base, v = os.environ["BASE"], os.environ["MV"]
try:
    m = json.load(open(sys.argv[1]))
except Exception:
    print("no"); sys.exit()
ent = m.get("tools", {}).get("dp", {}).get("variants", {}).get(v, {})
urls = ent.get("parts") or ([ent["url"]] if ent.get("url") else [])
names = [u.rsplit("/", 1)[-1] for u in urls]
print("yes" if any(n == base or n.startswith(base + ".") for n in names) else "no")
PY
)"
  if [ "$pub" = "yes" ]; then
    echo "  $V 已是 manifest 当前发布版,跳过(无需 GPU)"
    continue
  fi

  d="$WORK/$V"; rm -rf "$d"; mkdir -p "$d"; cd "$d"
  for name in $NAMES; do
    curl -fsSL -o "$name" "https://github.com/${REPO}/releases/download/${TAG}/${name}"
  done
  cat $NAMES > "$BASE"                                  # NAMES 已是数字序
  ACTUAL=$(stat -c%s "$BASE")
  [ "$ACTUAL" -eq "$TOTAL" ] || { echo "  ❌ 拼接大小 $ACTUAL != 期望 $TOTAL(分片缺失/损坏),$V 跳过"; cd "$WORK"; continue; }
  SHA=$(sha256sum "$BASE" | cut -d' ' -f1)
  echo "  sha256=$SHA"

  echo "  -- 安装 + GPU 端到端验证(运行第三方安装包时擦掉 GH_PAT)--"
  rm -rf "$d/inst"; bash "$BASE" -b -p "$d/inst"
  set +e
  env -u GH_PAT bash -c 'source "$1/bin/activate" && bash "$2/scripts/verify_offline_modec.sh"' _ "$d/inst" "$SKILL"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "  ❌ $V GPU 验证失败(rc=$rc),不入 manifest,保留上一份已验证条目"
    cd "$WORK"; continue
  fi

  echo "  -- 生成 manifest 片段(用 GPU 验过的字节 + Mode C 后端)--"
  cd "$SKILL"
  if [ "$V" = "cpu" ]; then CUDA=""; else num="${V#cuda}"; CUDA="${num:0:2}.${num:2}"; fi
  rm -rf "dist/$V"; mkdir -p "dist/$V"
  for name in $NAMES; do cp "$d/$name" "dist/$V/"; done
  REPO="$REPO" CUDA="$CUDA" VERSION="$VER" TAG="$TAG" \
  BACKEND="pytorch" \
  NOTE="Mode C (conda-pack self-extracting). PyTorch GPU backend; CPU-only TensorFlow bundled to load the LAMMPS plugin." \
    "$PY" scripts/gen_manifest_fragment.py
  # 防漂移:片段里的 sha 必须等于我们在验过的字节上重算的 sha
  grep -q "$SHA" "frag/$V.json" || { echo "  ❌ 片段 sha 与重算不符,放弃 $V"; cd "$WORK"; continue; }
  PASS+=("$V")
  cd "$WORK"
done

if [ ${#PASS[@]} -eq 0 ]; then
  echo "本轮没有需要发布的变体(已发布的跳过 / 未通过的保留原条目),manifest 未变"; exit 0
fi

echo "==> 合并通过的片段 → manifest.json"
cd "$SKILL"
"$PY" scripts/merge_manifest.py

if [ "$DRY_RUN" = "1" ]; then
  echo ""
  echo "===== DRY_RUN:不提交、不 push。manifest 将会变成(diff)====="
  git --no-pager diff -- assets/manifest.json || true
  echo "===== DRY_RUN 结束;GPU 验证通过的变体:${PASS[*]} ====="
  exit 0
fi

echo "==> commit → push(仅此步用 token)"
git config user.name  "bohrium-verify[bot]"
git config user.email "bohrium-verify@users.noreply.github.com"
git add assets/manifest.json
if git diff --cached --quiet; then
  echo "manifest 无变化"
else
  git commit -m "verify: 发布 GPU 验证通过的变体 ${PASS[*]} 到 manifest"
  git pull --rebase origin main                         # 先并入并发提交(匿名读)
  git push "https://x-access-token:${GH_PAT}@github.com/${REPO}.git" HEAD:main
  echo "已发布(GPU 验过):${PASS[*]}"
fi
