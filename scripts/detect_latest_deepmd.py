#!/usr/bin/env python3
"""检测 PyPI 上最新可装的 deepmd-kit 版本,并对配方 pin 兼容性给出告警。

对照 PyTorch:PyTorch nightly "每晚打最新" 是因为它掌控自己的源码 + 依赖。我们不同——
deepmd 是第三方包,且 build_modec.sh 里 deepmd 与 tensorflow-cpu / torch / lammps 是【版本死锁
配方】。所以这里只"检测最新 + 判断 pin 是否还兼容 + 报警",真正放行交给玻尔 GPU 验证门:
版本即使被 bump,GPU 验证不过就进不了 manifest,用户拿不到。

策略:
  PRERELEASE=1(默认)  连预发布(b/rc)一起算最新
  PRERELEASE=0          只算正式版
安全:
  - 只选【有 linux cp311 wheel】的版本(避免 bump 到只有 sdist、pip --pre 装不上的版本)
  - 新版本声明的 torch / tensorflow 依赖若和我们的 pin(TORCH_PIN / TF_PIN)冲突 → 打 WARNING,
    提示维护者先核对 build_modec.sh 的配方,别让流水线悄悄构建错栈。
  - 写 GitHub Actions 输出 version= / changed= / compat_warn=

Env:
  PRERELEASE  "1"/"0"(默认 "1")
  WRITE       "1" 时在有变化时改写 assets/version.txt(默认 "0" = 只检测)
  TORCH_PIN   默认 "2.11"        TF_PIN  默认 "2.21"   —— 仅用于兼容告警
  GITHUB_OUTPUT  Actions 注入,接收 version= / changed= / compat_warn=
"""
import json
import os
import sys
import urllib.request

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

PKG = "deepmd-kit"
PRERELEASE = os.environ.get("PRERELEASE", "1") == "1"
WRITE = os.environ.get("WRITE", "0") == "1"
TORCH_PIN = os.environ.get("TORCH_PIN", "2.11")
TF_PIN = os.environ.get("TF_PIN", "2.21")


def pypi(path):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{path}/json", timeout=30) as r:
        return json.load(r)


def has_linux_x86_wheel(files):
    """是否有 linux x86_64、且 python 标签兼容 3.11 的 wheel。

    deepmd-kit 发的是 py37 通用 wheel(如 deepmd_kit-3.2.0b0-py37-none-
    manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl),不是 cp311 专用 wheel,
    所以这里按 'manylinux x86_64 + py3 通用/cp311/abi3' 来判。
    """
    for f in files:
        if f.get("yanked"):
            continue
        n = f.get("filename", "").lower()
        if not n.endswith(".whl"):
            continue
        if "x86_64" not in n or "aarch64" in n or "manylinux" not in n:
            continue
        if "-py3" in n or "-cp311-" in n or ("-cp3" in n and "abi3" in n):
            return True
    return False


data = pypi(PKG)
cands = []
for ver, files in data.get("releases", {}).items():
    if not files or all(f.get("yanked") for f in files):
        continue
    try:
        V = Version(ver)
    except InvalidVersion:
        continue
    if V.is_prerelease and not PRERELEASE:
        continue
    if not has_linux_x86_wheel(files):
        continue
    cands.append(V)

if not cands:
    sys.exit("detect_latest_deepmd: 找不到任何有 linux x86_64 wheel 的可装版本")

latest = str(max(cands))

cur = ""
try:
    with open("assets/version.txt", encoding="utf-8") as fh:
        cur = fh.read().strip()
except FileNotFoundError:
    pass

# 不降级:若检测到的"最新"<= 当前(例如 PRERELEASE=0 选到比当前预发布更旧的正式版),保持当前。
if cur:
    try:
        if Version(latest) <= Version(cur):
            latest = cur
    except InvalidVersion:
        pass
changed = latest != cur

# 兼容告警:看选中版本声明的 torch / tensorflow 依赖
warns = []
try:
    reqs = pypi(f"{PKG}/{latest}").get("info", {}).get("requires_dist") or []
except Exception:
    reqs = []
for raw in reqs:
    try:
        req = Requirement(raw)
    except Exception:
        continue
    name = req.name.lower()
    if name == "torch" and req.specifier and Version(TORCH_PIN + ".0") not in req.specifier:
        warns.append(f"deepmd {latest} 要求 torch {req.specifier};我们 pin 的是 torch=={TORCH_PIN}.* —— 核对 build_modec.sh")
    if name in ("tensorflow", "tensorflow-cpu") and req.specifier and Version(TF_PIN + ".0") not in req.specifier:
        warns.append(f"deepmd {latest} 要求 {name} {req.specifier};我们 pin 的是 {TF_PIN}.* —— 核对 build_modec.sh")
warns = list(dict.fromkeys(warns))  # 去重(requires_dist 常有同名多 marker 条目)

print(f"最新可装 deepmd-kit(prerelease={PRERELEASE}): {latest}")
print(f"当前 version.txt: {cur or '(无)'}   changed={changed}")
for w in warns:
    print("WARNING:", w)

out = os.environ.get("GITHUB_OUTPUT")
if out:
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(f"version={latest}\n")
        fh.write(f"changed={'true' if changed else 'false'}\n")
        fh.write(f"compat_warn={'true' if warns else 'false'}\n")

if WRITE and changed:
    with open("assets/version.txt", "w", encoding="utf-8") as fh:
        fh.write(latest + "\n")
    print(f"version.txt -> {latest}")
