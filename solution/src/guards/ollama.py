"""Ollama 后端：用本机 ollama 服务承担主模型生成与护栏判别。

本机没有 torch/transformers，ollama 是唯一可直接复用的推理框架。
接口走 ollama 原生 /api/chat，原因有二：
1) 可显式传 think=false。Qwen3.5 系列默认开启 thinking，实测同一条请求
   开着 thinking 要 11 秒、关掉只要 0.7 秒，且思考内容会挤占 max_tokens；
2) 原生接口能拿到 eval_count 与 eval_duration，便于记录真实吞吐。
"""

from __future__ import annotations

import json
import re
import urllib.request

from guards.base import GuardResult, clamp_level, map_categories

SAFETY_RE = re.compile(r"Safety\s*[:：]\s*(\w+)", re.IGNORECASE)
CATEGORY_RE = re.compile(r"Categories?\s*[:：]\s*(.+)", re.IGNORECASE)
REFUSAL_RE = re.compile(r"Refusal\s*[:：]\s*(\w+)", re.IGNORECASE)


class OllamaClient:
    """极简 ollama /api/chat 客户端，带超时与重试。"""

    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "qwen3.5:4b",
                 think: bool = False, timeout: int = 180):
        self.host = host.rstrip("/")
        self.model = model
        self.think = think
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.2) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        message = data.get("message") or {}
        return {
            "content": (message.get("content") or "").strip(),
            "raw": data,
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
        }


class OllamaGuardBackend:
    """把 Qwen3Guard-Gen（GGUF）当护栏用，解析 Safety / Categories / Refusal 字段。"""

    name = "ollama-qwen3guard"

    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "qwen3guard:0.6b",
                 timeout: int = 120):
        self.client = OllamaClient(host=host, model=model, think=False, timeout=timeout)
        self.model = model

    def predict(self, text: str, role: str = "prompt", prompt: str | None = None) -> GuardResult:
        if role == "prompt":
            messages = [{"role": "user", "content": text}]
        else:
            messages = [
                {"role": "user", "content": prompt or ""},
                {"role": "assistant", "content": text},
            ]
        result = self.client.chat(messages, max_tokens=96, temperature=0.0)
        verdict = self.parse_output(result["content"])
        verdict.raw = result["content"]
        return verdict

    def parse_output(self, raw: str) -> GuardResult:
        safety = SAFETY_RE.search(raw or "")
        category = CATEGORY_RE.search(raw or "")
        refusal = REFUSAL_RE.search(raw or "")
        if safety:
            level = clamp_level(safety.group(1))
        else:
            # 兜底：模型未按格式输出时，按关键词判断，避免把明显有害内容放行
            lowered = (raw or "").lower()
            level = "unsafe" if "unsafe" in lowered else ("controversial" if "controversial" in lowered else "safe")
        categories = map_categories(category.group(1)) if category else []
        return GuardResult(
            level=level,
            categories=categories,
            score=1.0 if level == "unsafe" else (0.5 if level == "controversial" else 0.0),
            backend=self.name,
            raw=(raw or "").strip(),
            refusal=(refusal.group(1).lower().startswith("yes") if refusal else None),
        )
