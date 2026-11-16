"""赛方样本数据集剖析（A 级实测，可复现）。

产出 solution/reports/dataset_stats.json：
- 场景分布、长度分布
- 人工标注的攻击族分布
- 归一化模块的族识别召回率 / 误报
- Base64 载荷解码成功率
- 虚构创作模板的复用情况（相似度与字符占比）
- 官方样本与公开补充集的最大 n-gram 重合度
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "solution", "src"))

from normalize import detect_families, decode_payloads, normalize  # noqa: E402

XLSX = os.path.join(ROOT, "安全防护赛道样本数据集.xlsx")
LABELS = os.path.join(ROOT, "solution", "data", "official_attack_family_labels.json")
PUBLIC = os.path.join(ROOT, "work", "public_eval.jsonl")
OUT = os.path.join(ROOT, "solution", "reports", "dataset_stats.json")

CREATIVE_IDS = [4, 13, 22, 28, 36]


def char_ngrams(text: str, n: int) -> set[str]:
    text = re.sub(r"\s+", "", text)
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def main() -> None:
    import pandas as pd

    df = pd.read_excel(XLSX)
    labels = json.load(open(LABELS, encoding="utf-8"))
    labels = {int(k): v for k, v in labels.items() if not k.startswith("_")}

    records = []
    tp = fp = fn = 0
    per_family = {}
    for row in df.itertuples():
        text = str(row.prompt)
        gold = set(labels.get(int(row.id), []))
        pred = set(detect_families(text))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        for family in gold:
            per_family.setdefault(family, {"support": 0, "hit": 0})
            per_family[family]["support"] += 1
            if family in pred:
                per_family[family]["hit"] += 1
        decoded = decode_payloads(text)
        records.append(
            {
                "id": int(row.id),
                "scene": row.scene,
                "length": len(text),
                "gold_families": sorted(gold),
                "pred_families": sorted(pred),
                "decoded": [{"encoding": enc, "length": len(txt)} for enc, txt in decoded],
                "normalized_length": len(normalize(text).normalized),
            }
        )

    total_chars = sum(r["length"] for r in records)
    creative_chars = sum(r["length"] for r in records if r["id"] in CREATIVE_IDS)
    sims = []
    base = next(r for r in records if r["id"] == 4)
    base_text = str(df.loc[df.id == 4, "prompt"].iloc[0])
    for cid in CREATIVE_IDS:
        text = str(df.loc[df.id == cid, "prompt"].iloc[0])
        sims.append({"id": cid, "length": len(text), "similarity_to_id4": round(difflib.SequenceMatcher(None, base_text, text).ratio(), 3)})

    b64_ids = [r["id"] for r in records if r["decoded"]]
    base64_ok = sum(1 for r in records if r["decoded"])

    # 与公开补充集的重合度
    max_overlap = 0.0
    if os.path.exists(PUBLIC):
        public = [json.loads(line) for line in open(PUBLIC, encoding="utf-8") if line.strip()]
        pub_grams = [char_ngrams(p["text"], 4) for p in public]
        for r in records:
            text = str(df.loc[df.id == r["id"], "prompt"].iloc[0])
            grams = char_ngrams(text, 4)
            for other in pub_grams:
                inter = len(grams & other)
                if inter:
                    overlap = inter / len(grams | other)
                    max_overlap = max(max_overlap, overlap)

    scene_counts = df["scene"].value_counts().to_dict()
    length_stats = {
        "min": int(df["prompt"].str.len().min()),
        "max": int(df["prompt"].str.len().max()),
        "mean": round(float(df["prompt"].str.len().mean()), 1),
        "median": int(df["prompt"].str.len().median()),
        "total_chars": int(total_chars),
    }

    stats = {
        "official": {
            "count": int(len(df)),
            "scenes": scene_counts,
            "length": length_stats,
            "family_support": {k: v["support"] for k, v in sorted(per_family.items())},
            "family_recall": {k: round(v["hit"] / v["support"], 3) for k, v in sorted(per_family.items())},
            "family_micro_recall": round(tp / max(1, tp + fn), 3),
            "family_micro_precision": round(tp / max(1, tp + fp), 3),
            "family_detected_samples": sum(1 for r in records if r["pred_families"]),
            "family_gold_samples": sum(1 for r in records if r["gold_families"]),
            "coverage_gold": round(sum(1 for r in records if r["gold_families"]) / len(records), 3),
            "coverage_pred": round(sum(1 for r in records if r["pred_families"]) / len(records), 3),
        },
        "creative_template": {
            "ids": CREATIVE_IDS,
            "chars": creative_chars,
            "char_share": round(creative_chars / total_chars, 3),
            "similarities": sims,
            "identical_base": base["id"],
        },
        "base64": {
            "ids": b64_ids,
            "decoded_ok": base64_ok,
            "success_rate": round(base64_ok / max(1, len(b64_ids)), 3),
            "decoded_lengths": [d["length"] for r in records for d in r["decoded"]],
        },
        "stripping": {
            "mean_original_length": round(sum(r["length"] for r in records) / len(records), 1),
            "mean_normalized_length": round(sum(r["normalized_length"] for r in records) / len(records), 1),
            "shrink_ratio": round(
                sum(r["normalized_length"] for r in records) / max(1, sum(r["length"] for r in records)), 3
            ),
        },
        "cross_source_overlap": {"max_4gram_jaccard_with_public": round(max_overlap, 3)},
        "records": records,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
