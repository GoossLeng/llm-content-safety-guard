# 2026 安全防护赛道 · 落地方案代码包

小模型（≤8B）内容安全防护的可运行实现，配套技术报告《生成式人工智能内容安全防护赛题技术报告》。
方案分四层：输入归一化 → 双端检测与异构仲裁 → 场景化安全回复 → 参数高效内化。
本仓库同时包含本机（RTX 5060 Ti 16GB）实测脚本与结果。

## 1. 目录

```
configs/   运行配置（模型 id、路由阈值、LoRA 超参）
data/      模板、自建良性集、抽样清单、数据构造脚本
src/       归一化、护栏后端、路由、回复、流水线、评测
scripts/   下载、评测、训练脚本
reports/   实验产出（metrics.json 为报告唯一数字源）
```

## 2. 五分钟跑通（不需要 GPU、不需要下载模型）

```bash
python src/build_public_eval.py     # 需要 work/raw 下的原始数据，见第 3 节
python src/analyze_dataset.py      # 数据剖析
python src/eval_rules.py           # 规则层三档对比
python src/evaluate.py --backend mock   # 流程联调（不产出防护效果指标）
```

## 3. 公开补充集（620 条）

| 来源 | 条数 | 许可 | 获取 |
|---|---|---|---|
| Safety-Prompts · 典型安全场景 | 200 | Apache-2.0 | `thu-coai/Safety-Prompts` 的 `typical_safety_scenarios.json` |
| Safety-Prompts · 指令攻击 | 100 | Apache-2.0 | 同仓库 `instruction_attack_scenarios.json` |
| Flames-1k-Chinese | 120 | Apache-2.0 | `AI45Lab/Flames` 的 `Flames_1k_Chinese.jsonl` |
| XSTest · safe 子集 | 100 | CC-BY-4.0 | `paul-rottger/xstest` 的 `xstest_prompts.csv` |
| 自建中文良性 | 100 | 团队自建 | 本仓库 `data/benign_zh.jsonl` |

原始文件放到 `work/raw/`，运行 `python src/build_public_eval.py` 重新抽样（seed=42）。
原始文本只在本机使用，不随代码包分发；`data/public_eval_manifest.jsonl` 只记录来源与原始下标。

## 4. 本机实测（ollama 路线，已完成）

环境：RTX 5060 Ti 16GB、24 线程、32GB 内存；ollama 0.40.13；无 torch。

```bash
ollama pull qwen3.5:4b
# 护栏：第三方 GGUF（Qwen3Guard-Gen-0.6B Q4_K_M），用 Modelfile 导入
ollama create qwen3guard:0.6b -f work/qwen3guard.Modelfile
python src/eval_real.py --conditions B0,S1,S2 --workers 2 --latency-probe 20
python src/eval_real.py --harmful-source public --skip-benign --conditions B0,S1,S2
python ../build/make_real_metrics.py
```

实测结果（主裁判为 Qwen3.5-9B，副裁判为 0.6B 护栏）：

| 条件 | 赛方 50 防御 | 公开 420 防御 | 误拒率（200 良性） |
|---|---|---|---|
| B0 主模型直出 | 100% | 99.8% | 0% |
| S1 仅规则层 | 100% | 100% | 3.5% |
| S2 规则 + 0.6B 护栏 | 100% | 100% | 1.5% |

时延：模板拦截约 0.18s（p50）；生成 200 token 约 2.5s（p95，串行）；两路并发 p50 3.9–4.4s。
显存峰值 12.8GB（含桌面占用）。

结论：基线模型已接近饱和，分层的防御增量只有 1 条样本（FLM-045），代价是误拒与时延；
默认配置应取「解码 + 输出复核」，规则与护栏按场景开启。

## 5. 资源与清理

| 资源 | 位置 | 清理方式 |
|---|---|---|
| qwen3.5:4b（3.3GB） | `D:\AI\model_gguf\blobs`（`OLLAMA_MODELS`） | `ollama rm qwen3.5:4b` |
| qwen3guard:0.6b（484MB） | 同上 | `ollama rm qwen3guard:0.6b` |
| 脚本与产出 | 本目录 | 直接删除即可 |

本方案未安装 torch / transformers / peft，未修改系统环境变量。

## 6. 接 transformers 路线（可选）

1. `pip install torch>=2.7.0 --index-url https://download.pytorch.org/whl/cu128`，再装 transformers/accelerate；
2. 从 ModelScope 下载 `Qwen/Qwen3Guard-Gen-0.6B` 与 `Qwen/Qwen3.5-4B-Instruct`；
3. `python src/evaluate.py --backend qwen3guard`，输出标记为 measured，与第 4 节结果可对照。

Windows 下 vLLM 不可用；NVIDIA 50 系需 cu128 轮子，显存不足时退至 4bit 或 GGUF。

## 7. 合规与使用边界

- 公开数据集仅用于评测，遵守各自许可（Apache-2.0 / CC-BY-4.0），标注引用，不再分发原文。
- 自建良性集为团队编写，可随包分发。
- 模型输出仅用于本研究评测，不用于生成或传播有害内容。
