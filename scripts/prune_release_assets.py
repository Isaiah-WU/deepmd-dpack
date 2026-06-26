#!/usr/bin/env python3
"""清理滚动 nightly release 里堆积的旧构建。

对照 PyTorch:它的 nightly 索引只保留最近 ~60 个版本、且【从不删 S3 上的 wheel 文件】。
GitHub Releases 相反——资产持久、单文件 2GiB 上限、会无限堆积,所以【必须我们自己删】。

策略(单一 nightly tag 内):
  - 每个变体只保留最近 KEEP_BUILDS 个【日期构建】;
  - 外加【live manifest.json 仍在引用的任何构建,绝不删】(= dpack 当前在发的,对应 PyTorch
    从不删被索引的 wheel);
  - 其余删除。
  - 默认 DRY-RUN(只打印);加 --apply 才真删。
  - 加载 manifest 失败时直接中止(宁可不删,也不误删被引用的)。

Env / args:
  REPO         owner/repo(默认 Isaiah-WU/deepmd-dpack)
  TAG          release tag(默认 nightly)
  KEEP_BUILDS  每变体保留的最近日期构建数(默认 2)
  MANIFEST     live manifest 的 URL 或本地路径(默认拉 raw main)
  GH_TOKEN     --apply 时删除资产需要(列资产可匿名)
  --apply      真删(否则 dry-run)
"""
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict

REPO = os.environ.get("REPO", "Isaiah-WU/deepmd-dpack")
TAG = os.environ.get("TAG", "nightly")
KEEP = int(os.environ.get("KEEP_BUILDS", "2"))
APPLY = "--apply" in sys.argv
TOKEN = os.environ.get("GH_TOKEN", "")
MANIFEST = os.environ.get(
    "MANIFEST", f"https://raw.githubusercontent.com/{REPO}/main/assets/manifest.json"
)


def api(path, method="GET"):
    req = urllib.request.Request(f"https://api.github.com{path}", method=method)
    req.add_header("Accept", "application/vnd.github+json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        return r.status, (json.loads(body) if body else None)


# ── live manifest: 收集被引用的资产 basename(受保护,绝不删)──────────────────
protected = set()
try:
    if MANIFEST.startswith("http"):
        with urllib.request.urlopen(MANIFEST, timeout=30) as r:
            m = json.load(r)
    else:
        with open(MANIFEST, encoding="utf-8") as fh:
            m = json.load(fh)
    for tool in m.get("tools", {}).values():
        for v in tool.get("variants", {}).values():
            urls = v.get("parts") or ([v["url"]] if v.get("url") else [])
            for u in urls:
                protected.add(u.rsplit("/", 1)[-1])
except Exception as e:
    sys.exit(f"prune: 读不到 manifest({MANIFEST}):{e} —— 为防误删被引用资产,中止。")

# ── 列出 release 的全部资产(分页)────────────────────────────────────────────
try:
    _, rel = api(f"/repos/{REPO}/releases/tags/{TAG}")
except Exception as e:
    sys.exit(f"prune: 读不到 release {TAG}:{e}")
rid = rel["id"]
assets = []
page = 1
while True:
    _, batch = api(f"/repos/{REPO}/releases/{rid}/assets?per_page=100&page={page}")
    if not batch:
        break
    assets.extend(batch)
    if len(batch) < 100:
        break
    page += 1

# ── 分组:variant -> base(.sh)-> [(name, id)];记录每个 base 的日期 ───────────
pat = re.compile(
    r"^(?P<base>deepmd-kit-.+?-(?P<date>\d{8})-[^-]+-(?P<variant>cuda\d+|cpu)-Linux-x86_64\.sh)\.(?P<idx>\d+)$"
)
builds = defaultdict(lambda: defaultdict(list))
dates = {}
unknown = []
for a in assets:
    mm = pat.match(a["name"])
    if not mm:
        unknown.append(a["name"])
        continue
    v, base = mm.group("variant"), mm.group("base")
    builds[v][base].append((a["name"], a["id"]))
    dates[(v, base)] = mm.group("date")

to_delete = []
print(f"== prune {REPO}@{TAG}: {len(assets)} 资产, {len(builds)} 变体, KEEP_BUILDS={KEEP}, apply={APPLY} ==")
for v in sorted(builds):
    ordered = sorted(builds[v], key=lambda b: dates[(v, b)], reverse=True)  # 新→旧
    keep = set(ordered[:KEEP])
    print(f"  [{v}] {len(ordered)} 个日期构建")
    for base in ordered:
        names = [n for n, _ in builds[v][base]]
        is_prot = any(n in protected for n in names)
        if base in keep or is_prot:
            tag_ = "keep" + ("+manifest" if is_prot else "")
            print(f"     {tag_:14} {dates[(v, base)]}  {base}")
        else:
            print(f"     {'DELETE':14} {dates[(v, base)]}  {base}")
            to_delete.extend(builds[v][base])

if unknown:
    print(f"  (跳过 {len(unknown)} 个不识别命名的资产)")
print(f"  受 manifest 保护: {len(protected)} 个 basename")

if not to_delete:
    print("无需删除。")
    sys.exit(0)

print(f"\n{'真删' if APPLY else 'dry-run, 不删'}:{len(to_delete)} 个分片")
for name, aid in to_delete:
    if APPLY:
        if not TOKEN:
            sys.exit("prune: --apply 需要 GH_TOKEN")
        api(f"/repos/{REPO}/releases/assets/{aid}", method="DELETE")
        print("  deleted", name)
    else:
        print("  [dry-run] would delete", name)
