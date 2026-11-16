"""训练/校准数据构造。

训练用核心请求与评测集严格分离：默认从公开安全语料（Safety-Prompts）抽取核心请求，
并剔除与 620 条评测集重复的样本；赛方 50 条全程留作测试，不参与训练。
如需按旧口径从赛方样本取核心，可显式传 --core-source official（仅用于内部对照，
其结果不能与赛方 50 条上的评测混用）。

输入：work/raw/safety_prompts_typical.json、work/raw/safety_prompts_instruction.json
      work/public_eval.jsonl（用于排除评测样本）
      solution/data/benign_zh.jsonl、solution/data/augment_templates.json

产出：work/train_harmful.jsonl、work/train_benign.jsonl、solution/reports/train_stats.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "solution", "src"))

from normalize import normalize  # noqa: E402

RAW = os.path.join(ROOT, "work", "raw")
XLSX = os.path.join(ROOT, "安全防护赛道样本数据集.xlsx")
PUBLIC = os.path.join(ROOT, "work", "public_eval.jsonl")
TEMPLATES = os.path.join(ROOT, "solution", "data", "augment_templates.json")
BENIGN = os.path.join(ROOT, "solution", "data", "benign_zh.jsonl")
OUT_HARMFUL = os.path.join(ROOT, "work", "train_harmful.jsonl")
OUT_BENIGN = os.path.join(ROOT, "work", "train_benign.jsonl")
OUT_STATS = os.path.join(ROOT, "solution", "reports", "train_stats.json")

BENIGN_FRAMES = (
    "我在准备一份材料，想请教一下：{core}",
    "顺便问个实际的问题：{core}",
)
INTENT_RE = re.compile(r"(如何|怎么|怎样|方法|步骤|教程|指南|教我|告诉我|请提供|请给出|请写|生成|撰写|制作|策划|详细说明)")

# Safety-Prompts 类目 → 赛题七类场景
SCENE_MAP = {
    "Insult": "仇恨辱骂",
    "Unfairness_And_Discrimination": "偏见歧视",
    "Crimes_And_Illegal_Activities": "违法犯罪",
    "Physical_Harm": "身心健康",
    "Mental_Health": "身心健康",
    "Privacy_And_Property": "隐私财产",
    "Ethics_And_Morality": "伦理道德",
    "Goal_Hijacking": "伦理道德",
    "Prompt_Leaking": "隐私财产",
    "Role_Play_Instruction": "伦理道德",
    "Unsafe_Instruction_Topic": "违法犯罪",
    "Inquiry_With_Unsafe_Opinion": "偏见歧视",
    "Reverse_Exposure": "谣言错误",
}


def load_eval_texts() -> set[str]:
    texts = set()
    if os.path.exists(PUBLIC):
        with open(PUBLIC, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    texts.add(json.loads(line)["text"])
    return texts


def public_cores(n_per_scene: int, seed: int) -> list[dict]:
    """从公开语料按场景抽核心请求，排除评测集文本。"""
    eval_texts = load_eval_texts()
    rng = random.Random(seed)
    pool: dict[str, list[str]] = {}
    for name in ("safety_prompts_typical.json", "safety_prompts_instruction.json"):
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            continue
        data = json.load(open(path, encoding="utf-8"))
        for category, rows in data.items():
            scene = SCENE_MAP.get(category)
            if not scene:
                continue
            for row in rows:
                text = re.sub(r"\s+", " ", str(row.get("prompt", ""))).strip()
                if not (15 <= len(text) <= 200):
                    continue
                if not INTENT_RE.search(text):
                    continue
                if text in eval_texts:
                    continue
                pool.setdefault(scene, []).append(text)

    cores = []
    for scene, texts in pool.items():
        uniq = sorted(set(texts))
        take = min(n_per_scene, len(uniq))
        for text in rng.sample(uniq, take):
            cores.append({"scene": scene, "core": text, "origin": "public"})
    return cores


def official_cores() -> list[dict]:
    import pandas as pd

    df = pd.read_excel(XLSX)
    cores = []
    for row in df.itertuples():
        text = str(row.prompt)
        core = normalize(text, level=2).normalized
        if 4 <= len(core) <= 200:
            cores.append({"scene": row.scene, "core": core, "origin": "official", "source_id": int(row.id)})
    return cores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-source", choices=["public", "official"], default="public")
    parser.add_argument("--n-per-scene", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    templates = json.load(open(TEMPLATES, encoding="utf-8"))

    cores = public_cores(args.n_per_scene, args.seed) if args.core_source == "public" else official_cores()
    if not cores:
        raise SystemExit("核心请求为空，检查 work/raw 是否已下载")

    harmful = []
    for index, item in enumerate(cores, start=1):
        for family, spec in templates.items():
            if family.startswith("_"):
                continue
            for variant, template in enumerate(spec["templates"], start=1):
                if spec.get("mode") == "base64":
                    payload = base64.b64encode(item["core"].encode("utf-8")).decode("ascii")
                    text = template.format(payload=payload)
                else:
                    text = template.format(core=item["core"])
                harmful.append(
                    {
                        "id": f"H-{index:03d}-{family}-{variant}",
                        "scene": item["scene"],
                        "family": family,
                        "text": text,
                        "core_origin": item["origin"],
                    }
                )

    benign = []
    with open(BENIGN, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            benign.append({"id": row["id"] + "-v0", "category": row["category"], "text": row["prompt"], "variant": 0})
            for index, frame in enumerate(BENIGN_FRAMES, start=1):
                benign.append(
                    {
                        "id": f"{row['id']}-v{index}",
                        "category": row["category"],
                        "text": frame.format(core=row["prompt"]),
                        "variant": index,
                    }
                )
    random.Random(args.seed).shuffle(benign)

    os.makedirs(os.path.dirname(OUT_HARMFUL), exist_ok=True)
    with open(OUT_HARMFUL, "w", encoding="utf-8") as fh:
        for item in harmful:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(OUT_BENIGN, "w", encoding="utf-8") as fh:
        for item in benign:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_scene: dict[str, int] = {}
    for item in harmful:
        by_scene[item["scene"]] = by_scene.get(item["scene"], 0) + 1

    stats = {
        "core_source": args.core_source,
        "n_cores": len(cores),
        "cores_by_scene": {scene: sum(1 for c in cores if c["scene"] == scene) for scene in sorted({c["scene"] for c in cores})},
        "families": [k for k in templates if not k.startswith("_")],
        "variants_per_family": 6,
        "harmful_total": len(harmful),
        "benign_total": len(benign),
        "harmful_by_scene": by_scene,
        "train_test_separation": "核心请求来自公开语料并已排除 620 条评测集文本；赛方 50 条不参与训练",
    }
    with open(OUT_STATS, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
