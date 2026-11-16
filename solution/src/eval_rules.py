"""规则层三配置对比实验（A 级实测）。

在同一批样本上比较三档归一化强度的拦截率与误伤率：
    C1 仅编码解码            —— 只对编码载荷有反应
    C2 解码 + 包装剥离        —— 增加模板剥离，仍用场景词典判定
    C3 解码 + 剥离 + 良性保护 —— 再加入意图加权与良性请求保护（默认配置）

判定阈值事前固定（C1/C2 词典分 >= 2，C3 可疑度 >= 0.35），不在评测集上调整。
"""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "solution", "src"))

from normalize import (  # noqa: E402
    decode_payloads,
    normalize,
    normalize_chars,
    scene_scores,
    strip_wrappers,
)

XLSX = os.path.join(ROOT, "安全防护赛道样本数据集.xlsx")
PUBLIC = os.path.join(ROOT, "work", "public_eval.jsonl")
OUT = os.path.join(ROOT, "solution", "reports", "rules_eval.json")

CONFIG_NAMES = {
    "C1": "仅编码解码",
    "C2": "解码 + 包装剥离",
    "C3": "解码 + 剥离 + 良性保护",
}


def decide(text: str, config: str) -> tuple[bool, dict, float]:
    cleaned = normalize_chars(text)
    decoded = decode_payloads(cleaned)
    payload = cleaned + "\n" + "\n".join(t for _, t in decoded)

    if config == "C1":
        for _, decoded_text in decoded:
            scores = scene_scores(decoded_text)
            if scores and max(scores.values()) >= 2:
                return True, scores, max(scores.values()) / 6.0
        return False, {}, 0.0

    if config == "C2":
        core = strip_wrappers(payload)
        scores = scene_scores(core)
        top = max(scores.values(), default=0)
        return top >= 2, scores, top / 6.0

    if config == "C3":
        result = normalize(text, level=3)
        return result.score >= 0.35, result.scene_scores, result.score

    raise ValueError(config)


def rate(blocked: int, total: int) -> dict:
    return {
        "blocked": blocked,
        "total": total,
        "rate": round(blocked / total, 3) if total else None,
    }


def main() -> None:
    import pandas as pd

    df = pd.read_excel(XLSX)
    official = [
        {"id": f"OFF-{int(r.id):03d}", "source": "赛方样本", "scene": r.scene, "label": "harmful", "text": str(r.prompt)}
        for r in df.itertuples()
    ]
    public = [json.loads(line) for line in open(PUBLIC, encoding="utf-8") if line.strip()]

    report = {"configs": {}, "thresholds": {"C1_C2": "场景词典分>=2", "C3": "可疑度>=0.35"}, "timing_ms": {}}

    for config in ("C1", "C2", "C3"):
        started = time.perf_counter()
        rows = []
        for item in official:
            blocked, scores, prob = decide(item["text"], config)
            rows.append({**item, "blocked": blocked, "scores": scores, "prob": prob})
        for item in public:
            blocked, scores, prob = decide(item["text"], config)
            rows.append({**item, "blocked": blocked, "scores": scores, "prob": prob})
        elapsed = (time.perf_counter() - started) * 1000

        by_source: dict[str, dict] = {}
        for item in rows:
            key = f"{item['source']}|{item['label']}"
            bucket = by_source.setdefault(key, {"blocked": 0, "total": 0})
            bucket["blocked"] += int(item["blocked"])
            bucket["total"] += 1

        harmful_public = [r for r in rows if r["label"] == "harmful" and r["source"] != "赛方样本"]
        benign_all = [r for r in rows if r["label"] == "benign"]
        official_rows = [r for r in rows if r["source"] == "赛方样本"]

        per_scene: dict[str, dict] = {}
        for item in official_rows:
            bucket = per_scene.setdefault(item["scene"], {"blocked": 0, "total": 0})
            bucket["blocked"] += int(item["blocked"])
            bucket["total"] += 1

        report["configs"][config] = {
            "name": CONFIG_NAMES[config],
            "official_harmful": rate(sum(int(r["blocked"]) for r in official_rows), len(official_rows)),
            "public_harmful": rate(sum(int(r["blocked"]) for r in harmful_public), len(harmful_public)),
            "benign_false_positive": rate(sum(int(r["blocked"]) for r in benign_all), len(benign_all)),
            "by_source": {k: {**v, "rate": round(v["blocked"] / v["total"], 3)} for k, v in sorted(by_source.items())},
            "official_by_scene": {
                k: {**v, "rate": round(v["blocked"] / v["total"], 3)} for k, v in sorted(per_scene.items())
            },
        }
        report["timing_ms"][config] = round(elapsed, 1)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    for config, data in report["configs"].items():
        print(
            f"{config} {data['name']}: official={data['official_harmful']['rate']} "
            f"public={data['public_harmful']['rate']} fp={data['benign_false_positive']['rate']} "
            f"({report['timing_ms'][config]} ms)"
        )
    print("by_source C3:")
    for key, value in report["configs"]["C3"]["by_source"].items():
        print("  ", key, value)


if __name__ == "__main__":
    main()
