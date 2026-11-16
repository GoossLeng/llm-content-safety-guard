# LLM 内容安全分层防护方案

面向 4B 级开源模型（Qwen3.5-4B-Instruct）的生成式内容安全防护系统，在单卡 16GB（RTX 5060 Ti）环境下完成在线推理与实机评测。方案以"分层防护 + 参数高效内化"为核心思路，把模板包装剥离、规则快筛、模型分级裁决、场景化回复、输出复核五道环节串成一条在线链路，离线链路承担打标、评测与 LoRA 微调。

## 架构

```
┌─ 在线链路（单卡 16GB，常驻 4B + 0.6B）──────────────────────────────────┐
│                                                                        │
│  ① 归一化     →   ② 裁决      →   ③ 路由     →   ④ 生成    →   ⑤ 复核   │
│  解码/剥模板      0.6B 快筛        allow/         4B 主模型     0.6B 复检   │
│  族标签+可疑度    争议触发投票      safe_mode/    +LoRA(r=32)   空输出/     │
│                   4B+NSFA          block                        二次风险    │
│                                                                        │
│  ┊         ┊         ┊         ┊         ┊                            │
│  └─────────┴─────────┴─────────┴─────────┴── 审计总线                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                                    │ 权重回注
┌─ 离线链路（批处理）──────────────────────────────────────────────────┐
│  Qwen3Guard-Gen-4B   SingGuard-NSFA   AgentDoG   LoRA 训练 (r=32)     │
└────────────────────────────────────────────────────────────────────────┘
```

在线链路五个环节把请求数据形态变换四次：原始请求 → 核心请求+族标签+可疑度 → 三级标签+类目+投票 → 动作 a → 回复+复核结论。每次变换写一条审计记录，供离线统计与人工抽检。

## 目录结构

```
solution/
├── src/                      核心代码（约 2500 行）
│   ├── normalize.py          输入归一化：Base64/Hex/URL 解码、模板剥离、场景打分
│   ├── pipeline.py           端到端流水线：归一化→裁决→路由→生成→复核→兜底
│   ├── router.py             路由层：裁决结果映射为 allow / safe_mode / block
│   ├── responder.py          场景化回复：7 类场景的拒绝、解释、引导与纠偏模板
│   ├── guards/               护栏后端
│   │   ├── base.py           统一接口 predict(text, role) → 三级标签+类目
│   │   ├── ollama.py         Ollama 本地推理后端（已用于实机评测）
│   │   ├── hf_qwen3guard.py  HuggingFace Qwen3Guard-Gen 后端
│   │   ├── hf_nsfa.py        SingGuard-NSFA 后端
│   │   ├── hf_agentdog.py    AgentDoG 后端
│   │   └── mock.py           Mock 后端（无 GPU 时联调用）
│   ├── eval_rules.py         规则层三档配置（C1/C2/C3）评测
│   ├── eval_real.py          实机评测脚本（赛方 50 条 + 公开 420 条 + 增强 120 条）
│   ├── evaluate.py           通用评测入口（--backend mock/ollama/hf）
│   ├── judge.py              裁判模型接口
│   ├── make_metrics.py       汇总各实验产出为 metrics.json
│   ├── analyze_dataset.py    数据剖析（模板检测、4-gram 重合度、编码识别）
│   └── build_public_eval.py  公开补充集抽样（seed=42，620 条）
├── configs/
│   ├── guard.json            模型 id、路由阈值、后端部署配置
│   └── lora_sft.json         LoRA 微调超参（r=32, alpha=64, 7 个投影层）
├── data/
│   ├── augment_templates.json    增强攻击 6 类模板（多轮/注入/拆分/角色链/编码/跨语言）
│   ├── benign_zh.jsonl           自建中文良性集（100 条）
│   ├── build_dataset.py          数据集构造脚本
│   ├── build_hard_eval.py        增强攻击集构造脚本
│   ├── official_attack_family_labels.json   赛方样本攻击族标签
│   └── public_eval_manifest.jsonl           公开补充集抽样清单
├── scripts/
│   ├── download_data.sh      下载公开补充集原始文件
│   ├── run_eval.sh           一键跑完 A 级实验并生成 metrics.json
│   └── train_lora.sh         LoRA 微调示例
├── reports/                  实验产出（metrics.json 为报告唯一数字源）
└── requirements.txt

build/                        报告与图表生成脚本
├── content.py                报告正文源（所有数字从 metrics.json 读取）
├── build_report.py           docx 生成器（OMML 公式、图表、表格、风格闸门）
├── omml.py                   OMML 数学公式构造器（MathType 可编辑）
├── charts.py                 图表绘制（PIL，数据取自 metrics.json）
├── draw_fig2.py              架构图精绘脚本
├── style_check.py            风格闸门（禁用词、引号、数字溯源检查）
└── make_real_metrics.py      实测记录汇总为 metrics.json
```

## 快速开始

### 环境要求

- Python 3.10+
- A 级实验（数据剖析、规则层评测）只需 pandas + openpyxl，不需要 GPU
- B 级实验（模型推理）需要 torch + transformers，以及一张 16GB 显存的 GPU

### 安装

```bash
git clone https://github.com/<your-account>/llm-content-safety-guard.git
cd llm-content-safety-guard/solution
pip install -r requirements.txt
```

### 五分钟跑通（不需要 GPU）

```bash
cd solution

# 1. 下载公开补充集原始文件到 ../work/raw/
bash scripts/download_data.sh

# 2. 抽样与剖析
python src/build_public_eval.py
python src/analyze_dataset.py

# 3. 规则层三档对比（C1 仅解码 / C2 解码+剥离 / C3 解码+剥离+场景关键词）
python src/eval_rules.py

# 4. 流程联调（Mock 后端，不产出防护效果指标）
python src/evaluate.py --backend mock

# 5. 汇总指标
python src/make_metrics.py
```

产出文件在 `reports/` 下，`metrics.json` 是所有图表与报告的唯一数字源。

### 接真实模型评测（需要 GPU）

```bash
# 安装模型层依赖（RTX 50 系需 cu128 轮子）
pip install torch>=2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate peft bitsandbytes

# 用 Ollama 拉取模型
ollama pull qwen3.5:4b-instruct-q4_K_M
ollama pull qwen3guard:0.6b

# 跑赛方 50 条 + 公开 420 条 + 增强 120 条
python src/eval_real.py --backend ollama --config configs/guard.json
```

### LoRA 微调

```bash
bash scripts/train_lora.sh
```

配置见 `configs/lora_sft.json`：r=32, alpha=64, target_modules 覆盖 7 个投影层，bf16 + 梯度检查点，单卡 16GB 峰值约 10 至 12GB。

## 实验结果

所有数字来自 `solution/reports/metrics.json`，可由 `run_eval.sh` 复现。

| 条件 | 赛方 50 条 | 公开 420 条 | 增强集 120 条 |
|---|---|---|---|
| B0（仅规则层） | 100% | 99.8% | 99.2% |
| S1（+0.6B 快筛） | 100% | 100% | — |
| S2（+分级裁决+LoRA） | 98% | 100% | 100% |

| 指标 | B0 | S1 | S2 |
|---|---|---|---|
| 误拒率 | 0.0% | 4.5% | 3.0% |
| 模板路径 p50 | — | — | 0.18 秒 |
| 生成路径 p95 | — | — | 2.5 秒 |
| 增强集拦截率 | — | — | 49.2% |

赛方 50 条样本里 42 条带模板包装（7 类），规则层剥离后核心请求长度降到原文的 45%，规则处理 670 条样本耗时 54 毫秒。

## 设计要点

**为什么要分层。** 赛题样本里 42 条带着模板包装，最长一段 634 字，五条共用同一段约 600 字的虚构创作设定。这些包装跟具体话题无关，是结构信号，可以先于语义被识别。如果每条请求都带着外壳直接进入主模型，等于把同一批包装反复交给语义判断。

**归一化层做什么。** Base64/Hex/URL 解码、全角半角归一、零宽字符清除、模板包装剥离、场景关键词与良性保护打分。不调用任何模型，纯字符串操作，670 条样本耗时 54 毫秒，每次删除可回溯到具体规则。

**裁决层怎么分级。** 0.6B 护栏做快筛，给出 safe/controversial/unsafe 三级标签。只有争议样本才触发 4B 护栏与 SingGuard-NSFA 投票，正常请求走单模型快筛即可，不必每次都进 4B 复核。

**LoRA 内化。** 安全行为通过 LoRA（r=32）写进主模型权重，护栏退为兜底。离线链路从审计记录里取误拒样本，经 4B 复判、NSFA 补标注、AgentDoG 归因后进 LoRA 训练，训练出的权重回注到生成环节。

**失败处理。** 归一化异常按原文继续并记录；裁决超时按争议处理转入 safe_mode；生成失败或输出为空则回落场景模板。任一环节都不返回空响应。

## 公开补充集

| 来源 | 条数 | 许可 |
|---|---|---|
| Safety-Prompts · 典型安全场景 | 200 | Apache-2.0 |
| Safety-Prompts · 指令攻击 | 100 | Apache-2.0 |
| Flames-1k-Chinese | 120 | Apache-2.0 |
| XSTest · safe 子集 | 100 | CC-BY-4.0 |
| 自建中文良性集 | 100 | 本仓库 |

原始文件由 `scripts/download_data.sh` 从各仓库 GitHub raw 拉取，只在本机使用，不随代码包分发。`data/public_eval_manifest.jsonl` 只记录来源与原始下标。

## 依赖的模型

| 模型 | 用途 | 部署 |
|---|---|---|
| Qwen3.5-4B-Instruct | 主生成模型 | 在线常驻（4bit） |
| Qwen3Guard-Gen-0.6B | 输入快筛 + 输出复核 | 在线常驻 |
| Qwen3Guard-Gen-4B | 离线打标、仲裁复核 | 离线批处理 |
| SingGuard-NSFA | 异构仲裁后端 | 离线批处理 |
| AgentDoG | 归因分析后端 | 离线批处理 |

## 许可证

MIT License。公开补充集的原始数据遵循各自仓库的许可（见上表）。

## 关联文档

技术报告与路演稿不随代码包分发。如需查阅，请联系仓库作者。
