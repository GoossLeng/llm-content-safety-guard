"""Qwen3Guard-Gen 后端。

用法：
    from guards.hf_qwen3guard import Qwen3GuardBackend
    guard = Qwen3GuardBackend("Qwen/Qwen3Guard-Gen-0.6B")
    guard.predict(prompt, role="prompt")
    guard.predict(response, role="response", prompt=prompt)

说明：
- 输出格式按 Qwen3Guard 模型卡的约定解析（Safety / Categories / Refusal 字段），
  若上游模型卡调整字段，只需改 parse_output。
- transformers 与 torch 采用延迟导入，未安装依赖时本模块仍可被 import，
  便于在无 GPU 环境下做单元测试与流程联调。
"""

from __future__ import annotations

import re

from guards.base import GuardResult, clamp_level, map_categories

SAFETY_RE = re.compile(r"Safety\s*[:：]\s*(\w+)", re.IGNORECASE)
CATEGORY_RE = re.compile(r"Categories?\s*[:：]\s*(.+)", re.IGNORECASE)
REFUSAL_RE = re.compile(r"Refusal\s*[:：]\s*(\w+)", re.IGNORECASE)


class Qwen3GuardBackend:
    name = "qwen3guard-gen"

    def __init__(self, model_id: str = "Qwen/Qwen3Guard-Gen-0.6B", device: str = "cuda", max_new_tokens: int = 128):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return self._model, self._tokenizer
        import torch  # noqa: F401  (延迟导入)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self._model.eval()
        return self._model, self._tokenizer

    def _messages(self, text: str, role: str, prompt: str | None) -> list[dict]:
        if role == "prompt":
            return [{"role": "user", "content": text}]
        # 回复审核：把用户请求与模型回复一并给护栏模型
        user = prompt if prompt is not None else ""
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": text},
        ]

    def predict(self, text: str, role: str = "prompt", prompt: str | None = None) -> GuardResult:
        model, tokenizer = self._load()
        messages = self._messages(text, role, prompt)
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([chat], return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        return self.parse_output(raw)

    def parse_output(self, raw: str) -> GuardResult:
        safety = SAFETY_RE.search(raw)
        category = CATEGORY_RE.search(raw)
        refusal = REFUSAL_RE.search(raw)
        level = clamp_level(safety.group(1)) if safety else "safe"
        categories = map_categories(category.group(1)) if category else []
        return GuardResult(
            level=level,
            categories=categories,
            score=1.0 if level == "unsafe" else (0.5 if level == "controversial" else 0.0),
            backend=self.name,
            raw=raw.strip(),
            refusal=(refusal.group(1).lower().startswith("yes") if refusal else None),
        )
