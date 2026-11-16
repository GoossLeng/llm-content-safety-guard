#!/usr/bin/env bash
# 一键跑完 A 级实验并生成 metrics.json
set -euo pipefail
cd "$(dirname "$0")/.."

python src/build_public_eval.py
python data/build_dataset.py
python src/analyze_dataset.py
python src/eval_rules.py
python src/make_metrics.py
python src/evaluate.py --backend mock || true

echo "产出："
echo "  reports/dataset_stats.json"
echo "  reports/public_eval_stats.json"
echo "  reports/rules_eval.json"
echo "  reports/train_stats.json"
echo "  reports/metrics.json"
