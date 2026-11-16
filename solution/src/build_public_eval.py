"""构建公开补充评测集。

输入：work/raw/ 下四个公开数据集的原始文件 + 赛方样本数据集.xlsx
输出：
    work/public_eval.jsonl                     完整文本（仅本地评测用）
    solution/data/public_eval_manifest.jsonl   抽样清单（只含来源与原始下标，不含原文）
    solution/reports/public_eval_stats.json    统计与清洗记录

抽样规则见 README：分层抽样、长度过滤、跨源去重、与赛方样本去污。
数据集许可：Safety-Prompts / Flames 为 Apache-2.0，XSTest 为 CC-BY-4.0，
自建良性集由本团队编写。原始文本不随代码包分发。
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "work", "raw")
OUT_TEXT = os.path.join(ROOT, "work", "public_eval.jsonl")
OUT_MANIFEST = os.path.join(ROOT, "solution", "data", "public_eval_manifest.jsonl")
OUT_STATS = os.path.join(ROOT, "solution", "reports", "public_eval_stats.json")
BENIGN_ZH = os.path.join(ROOT, "solution", "data", "benign_zh.jsonl")
XLSX = os.path.join(ROOT, "安全防护赛道样本数据集.xlsx")

SEED = 42
INTENT_RE = re.compile(
    r"(如何|怎么|怎样|方法|步骤|教程|指南|教我|告诉我|请提供|请给出|请写|生成|撰写|制作|策划|详细说明)"
)

# 抽样配额（与方案文档一致）
SP_TYPICAL_QUOTA = {
    "Insult": 29,
    "Unfairness_And_Discrimination": 29,
    "Crimes_And_Illegal_Activities": 29,
    "Physical_Harm": 29,
    "Mental_Health": 28,
    "Privacy_And_Property": 28,
    "Ethics_And_Morality": 28,
}
SP_INSTRUCTION_QUOTA = {
    "Goal_Hijacking": 17,
    "Prompt_Leaking": 17,
    "Role_Play_Instruction": 17,
    "Unsafe_Instruction_Topic": 17,
    "Inquiry_With_Unsafe_Opinion": 16,
    "Reverse_Exposure": 16,
}
FLAMES_QUOTA = {
    "Fairness": 30,
    "Safety": 51,
    "Morality": 24,
    "Legality": 6,
    "Data protection": 9,
}
XSTEST_QUOTA = 100


def char_ngrams(text: str, n: int) -> set[str]:
    text = re.sub(r"\s+", "", text)
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: str, b: str, n: int) -> float:
    sa, sb = char_ngrams(a, n), char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_official() -> list[dict]:
    import pandas as pd

    df = pd.read_excel(XLSX)
    return [
        {"id": f"OFF-{int(r.id):03d}", "scene": r.scene, "text": r.prompt}
        for r in df.itertuples()
    ]


def clean_prompt(text: str) -> str:
    text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_sp_typical() -> list[dict]:
    data = json.load(open(os.path.join(RAW, "safety_prompts_typical.json"), encoding="utf-8"))
    out = []
    for category, rows in data.items():
        for idx, row in enumerate(rows):
            text = clean_prompt(row.get("prompt", ""))
            out.append(
                {
                    "source": "Safety-Prompts/典型安全场景",
                    "category": category,
                    "raw_index": idx,
                    "text": text,
                }
            )
    return out


def load_sp_instruction() -> list[dict]:
    data = json.load(open(os.path.join(RAW, "safety_prompts_instruction.json"), encoding="utf-8"))
    out = []
    for category, rows in data.items():
        for idx, row in enumerate(rows):
            text = clean_prompt(row.get("prompt", ""))
            out.append(
                {
                    "source": "Safety-Prompts/指令攻击",
                    "category": category,
                    "raw_index": idx,
                    "text": text,
                }
            )
    return out


def load_flames() -> list[dict]:
    out = []
    with open(os.path.join(RAW, "flames_1k_zh.jsonl"), encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            dim = row.get("dimension", "").strip()
            dim = {"Data_protection": "Data protection", "Data Protection": "Data protection"}.get(dim, dim)
            out.append(
                {
                    "source": "Flames-1k-Chinese",
                    "category": dim,
                    "raw_index": idx,
                    "text": clean_prompt(row.get("prompt", "")),
                }
            )
    return out


def load_xstest() -> list[dict]:
    out = []
    path = os.path.join(RAW, "xstest_prompts.csv")
    with open(path, encoding="utf-8", newline="") as fh:
        for idx, row in enumerate(csv.DictReader(fh)):
            if row.get("label") != "safe":
                continue
            out.append(
                {
                    "source": "XSTest/safe",
                    "category": row.get("type", "unknown"),
                    "raw_index": idx,
                    "text": clean_prompt(row.get("prompt", "")),
                }
            )
    return out


def load_benign_zh() -> list[dict]:
    out = []
    with open(BENIGN_ZH, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            out.append(
                {
                    "source": "自建中文良性",
                    "category": row["category"],
                    "raw_index": None,
                    "text": clean_prompt(row["prompt"]),
                    "sample_id": row["id"],
                }
            )
    return out


def filter_rows(rows: list[dict], require_intent: bool, lo=15, hi=400) -> list[dict]:
    kept = []
    for row in rows:
        text = row["text"]
        if not (lo <= len(text) <= hi):
            continue
        if require_intent and not (INTENT_RE.search(text) or "?" in text or "？" in text):
            continue
        kept.append(row)
    return kept


def stratified_sample(pool: list[dict], quota: dict[str, int], rng: random.Random) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_cat[row["category"]].append(row)
    picked = []
    for category, n in quota.items():
        candidates = by_cat.get(category, [])
        if len(candidates) < n:
            raise SystemExit(f"配额不足: {category} 需要 {n}，实际 {len(candidates)}")
        picked.extend(rng.sample(candidates, n))
    return picked


def main() -> None:
    rng = random.Random(SEED)
    official = load_official()

    pool_typical = filter_rows(load_sp_typical(), require_intent=True)
    pool_instruction = filter_rows(load_sp_instruction(), require_intent=False)
    pool_flames = [
        row
        for row in load_flames()
        if 10 <= len(row["text"]) <= 400
    ]
    pool_xstest = [row for row in load_xstest() if 5 <= len(row["text"]) <= 400]
    benign_zh = load_benign_zh()

    picked: list[dict] = []
    picked += stratified_sample(pool_typical, SP_TYPICAL_QUOTA, rng)
    picked += stratified_sample(pool_instruction, SP_INSTRUCTION_QUOTA, rng)
    picked += stratified_sample(pool_flames, FLAMES_QUOTA, rng)
    picked += rng.sample(pool_xstest, XSTEST_QUOTA)
    picked += benign_zh

    # 去污：与赛方 50 条做 4-gram 比对
    decon_dropped = []
    keep = []
    max_overlap = 0.0
    for row in picked:
        overlaps = [jaccard(row["text"], item["text"], 4) for item in official]
        row_max = max(overlaps) if overlaps else 0.0
        max_overlap = max(max_overlap, row_max)
        if row_max > 0.30:
            decon_dropped.append({"source": row["source"], "raw_index": row["raw_index"], "overlap": round(row_max, 3)})
            continue
        row["decon_overlap"] = round(row_max, 3)
        keep.append(row)

    # 跨源去重（5-gram Jaccard > 0.8）
    deduped: list[dict] = []
    prefix: dict[str, list[set]] = defaultdict(list)
    dedup_dropped = 0
    for row in keep:
        grams = char_ngrams(row["text"], 5)
        key = row["text"][:8]
        duplicate = False
        for other_grams in prefix.get("all", []):
            inter = len(grams & other_grams)
            if inter / max(1, len(grams | other_grams)) > 0.8:
                duplicate = True
                break
        if duplicate:
            dedup_dropped += 1
            continue
        prefix["all"].append(grams)
        deduped.append(row)

    # 编号与标签
    counters: Counter = Counter()
    records = []
    for row in deduped:
        counters[row["source"]] += 1
        tag = {
            "Safety-Prompts/典型安全场景": "SPT",
            "Safety-Prompts/指令攻击": "SPI",
            "Flames-1k-Chinese": "FLM",
            "XSTest/safe": "XST",
            "自建中文良性": "BZH",
        }[row["source"]]
        label = "benign" if row["source"] in ("XSTest/safe", "自建中文良性") else "harmful"
        lang = "en" if row["source"] == "XSTest/safe" else "zh"
        kind = "instruction" if (INTENT_RE.search(row["text"]) or "？" in row["text"] or "?" in row["text"]) else "discussion"
        records.append(
            {
                "id": f"{tag}-{counters[row['source']]:03d}",
                "source": row["source"],
                "category": row["category"],
                "lang": lang,
                "label": label,
                "kind": kind,
                "text": row["text"],
                "decon_overlap": row["decon_overlap"],
                "raw_index": row["raw_index"],
                "sample_id": row.get("sample_id"),
            }
        )

    os.makedirs(os.path.dirname(OUT_STATS), exist_ok=True)
    with open(OUT_TEXT, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(OUT_MANIFEST, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "source": rec["source"],
                        "category": rec["category"],
                        "raw_index": rec["raw_index"],
                        "sample_id": rec["sample_id"],
                        "label": rec["label"],
                        "lang": rec["lang"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats = {
        "seed": SEED,
        "total": len(records),
        "by_source": dict(Counter(r["source"] for r in records)),
        "by_label": dict(Counter(r["label"] for r in records)),
        "by_lang": dict(Counter(r["lang"] for r in records)),
        "by_category": dict(Counter(f"{r['source']}|{r['category']}" for r in records)),
        "by_kind": dict(Counter(f"{r['label']}|{r['kind']}" for r in records)),
        "length": {
            "min": min(len(r["text"]) for r in records),
            "max": max(len(r["text"]) for r in records),
            "mean": round(sum(len(r["text"]) for r in records) / len(records), 1),
        },
        "decon": {
            "threshold": 0.30,
            "dropped": len(decon_dropped),
            "dropped_detail": decon_dropped,
            "max_overlap_kept": round(max((r["decon_overlap"] for r in records), default=0.0), 3),
            "max_overlap_all": round(max_overlap, 3),
        },
        "dedup_dropped": dedup_dropped,
        "pool_sizes": {
            "sp_typical_after_filter": len(pool_typical),
            "sp_instruction_after_filter": len(pool_instruction),
            "flames_after_filter": len(pool_flames),
            "xstest_safe_after_filter": len(pool_xstest),
            "benign_zh": len(benign_zh),
        },
        "raw_rows": {
            "sp_typical": 70000,
            "sp_instruction": 30000,
            "flames": 1000,
            "xstest_safe": 250,
        },
        "licenses": {
            "Safety-Prompts": "Apache-2.0",
            "Flames-1k-Chinese": "Apache-2.0",
            "XSTest": "CC-BY-4.0",
            "自建中文良性": "team-authored",
        },
    }
    with open(OUT_STATS, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("decon", "by_category")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
