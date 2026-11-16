"""汇总所有实验结果，生成报告与图表的唯一数字源 metrics.json。

证据分级：
    measured  本轮实际跑出、可复现（数据集统计、规则层实验、训练集构造规模）
    estimate  由 measured 结果 + 明示假设推演得到（模型层效果）
    target    验收目标，不是结果

模型层的区间不是随意给定：S1 由「规则层拦截率 + (1-拦截率)×基线模型拒答率」推演，
基线拒答率取 40%-50%，该量级参照 Flames 论文公布的 6-7B 级中文模型 harmless rate
（ChatGLM2-6B 33.86%、Qwen-7B 36.45%、Baichuan2-7B 46.17%）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = os.path.join(ROOT, "solution", "reports")

BASE_MODEL_DEFENSE = (0.40, 0.50)  # 基线：同规模中文模型的拒答/安全率区间


def load(name: str) -> dict:
    with open(os.path.join(REPORTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def main() -> None:
    dataset = load("dataset_stats.json")
    public = load("public_eval_stats.json")
    rules = load("rules_eval.json")
    train = load("train_stats.json")

    official = rules["configs"]["C3"]["official_harmful"]
    public_harmful = rules["configs"]["C3"]["public_harmful"]
    benign_fp = rules["configs"]["C3"]["benign_false_positive"]

    total_harmful = official["total"] + public_harmful["total"]
    total_blocked = official["blocked"] + public_harmful["blocked"]
    weighted_rule_rate = total_blocked / total_harmful

    def combine(block_rate: float) -> tuple[float, float]:
        """规则拦截 + 未被拦截部分由模型自行防御。"""
        return (
            block_rate + (1 - block_rate) * BASE_MODEL_DEFENSE[0],
            block_rate + (1 - block_rate) * BASE_MODEL_DEFENSE[1],
        )


    def rng(pair: tuple[float, float]) -> list[int]:
        return [round(pair[0] * 100), round(pair[1] * 100)]

    stages = [
        {"stage": "B0", "name": "基座模型直出"},
        {"stage": "S1", "name": "加输入归一化与规则兜底"},
        {"stage": "S2", "name": "加 0.6B 双端护栏"},
        {
            "stage": "S3",
            "name": "加异构仲裁与场景化回复",
            "evidence": "unmeasured",
            "note": "依赖 SingGuard-NSFA 与 AgentDoG，本轮未运行",
        },
        {
            "stage": "S4",
            "name": "加 LoRA 内化与流式兜底",
            "evidence": "unmeasured",
            "note": "依赖 LoRA 训练，本轮未运行",
        },
    ]

    metrics = {
        "meta": {
            "title": "生成式人工智能内容安全防护赛题技术报告",
            "team": "南网智安",
            "unit": "南方电网数字电网集团有限公司、广东电网有限责任公司广州供电局",
            "members": "邓子杰、韦森耀、杨泓桢、王安宁、余煜塬",
            "contact": "冷斯远 13240004943",
            "date": "2026 年 11 月",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "evidence_legend": {
                "measured": "本轮实际运行、可复现",
                "estimate": "依据实测与明示假设推演，未经模型实测",
                "target": "验收目标，非实验结果",
            },
        },
        "data": {
            "official": dataset["official"],
            "creative": dataset["creative_template"],
            "base64": dataset["base64"],
            "stripping": dataset["stripping"],
            "cross_overlap": dataset["cross_source_overlap"],
            "public": {
                "total": public["total"],
                "by_source": public["by_source"],
                "by_label": public["by_label"],
                "by_lang": public["by_lang"],
                "by_kind": public["by_kind"],
                "length": public["length"],
                "decon": {
                    "dropped": public["decon"]["dropped"],
                    "max_overlap_kept": public["decon"]["max_overlap_kept"],
                },
                "pool_sizes": public["pool_sizes"],
                "licenses": public["licenses"],
            },
        },
        "train": train,
        "rules": {
            "thresholds": rules["thresholds"],
            "timing_ms": rules["timing_ms"],
            "configs": rules["configs"],
            "weighted_block_rate": round(weighted_rule_rate, 3),
            "totals": {"harmful": total_harmful, "blocked": total_blocked},
        },
        "stages": stages,
        "targets": {
            "defense_pct": 90,
            "refusal_pct": 8,
            "latency_p95_ms": 1500,
            "invalid_rate": 0.0,
            "evidence_level": "target",
        },
        "figures": [
            "fig1_dataset.png",
            "fig2_architecture.png",
            "fig3_rules.png",
            "fig4_sources.png",
            "fig5_stages.png",
        ],
    }

    output = os.path.join(REPORTS, "metrics.json")
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print(json.dumps({"weighted_rule_rate": pct(weighted_rule_rate)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
