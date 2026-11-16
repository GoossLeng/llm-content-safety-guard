#!/usr/bin/env bash
# 下载公开补充集原始文件到 work/raw/（HuggingFace 不可达时用 GitHub raw）
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import os, urllib.request
RAW = "https://raw.githubusercontent.com"
TARGETS = [
    ("safety_prompts_typical.json", f"{RAW}/thu-coai/Safety-Prompts/main/typical_safety_scenarios.json"),
    ("safety_prompts_instruction.json", f"{RAW}/thu-coai/Safety-Prompts/main/instruction_attack_scenarios.json"),
    ("flames_1k_zh.jsonl", f"{RAW}/AI45Lab/Flames/main/Flames_1k_Chinese.jsonl"),
    ("xstest_prompts.csv", f"{RAW}/paul-rottger/xstest/main/xstest_prompts.csv"),
]
out = os.path.join("..", "work", "raw")
os.makedirs(out, exist_ok=True)
for name, url in TARGETS:
    dest = os.path.join(out, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print("skip", name); continue
    print("fetch", url)
    urllib.request.urlretrieve(url, dest)
print("done")
PY
