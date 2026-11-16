#!/usr/bin/env bash
# LoRA 微调示例（需要 torch / transformers / peft / accelerate）
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
import json
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

cfg = json.load(open("configs/lora_sft.json", encoding="utf-8"))
model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])

lora = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["lora_alpha"],
    lora_dropout=cfg["lora"]["lora_dropout"],
    target_modules=cfg["lora"]["target_modules"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.gradient_checkpointing_enable()

# 训练数据的组织方式：有害样本使用「安全回复模板」，无害样本使用「正常回答」。
# 具体对话模板随底座模型变化，此处只给出骨架，落地时按模型卡调整。
def encode(row):
    messages = [
        {"role": "system", "content": "你是内容安全助手，回答需符合法律法规与公序良俗。"},
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": row["target"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer(text, truncation=True, max_length=cfg["training"]["max_seq_length"])

args = TrainingArguments(
    output_dir="work/lora_out",
    learning_rate=cfg["training"]["learning_rate"],
    num_train_epochs=cfg["training"]["num_train_epochs"],
    per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
    warmup_ratio=cfg["training"]["warmup_ratio"],
    bf16=cfg["training"]["bf16"],
    logging_steps=cfg["training"]["logging_steps"],
    save_steps=cfg["training"]["save_steps"],
)
print("配置就绪：", json.dumps(cfg["training"], ensure_ascii=False))
print("实际训练需先把 train_harmful/train_benign 补上 target 字段并加载为 Dataset")
PY
