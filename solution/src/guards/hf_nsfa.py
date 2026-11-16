"""SingGuard-NSFA 后端（蚂蚁 AI 安全实验室）。

NSFA 提供双模推理：生成式推理用于离线审计，分类头用于在线拦截（官方报告约 50ms）。
本实现默认走 OpenAI 兼容接口（vLLM / SGLang 部署均可），也支持本地 transformers 调用。
调用格式以模型卡为准；若上游调整字段，修改 build_prompt 与 parse_output 即可。
"""

from __future__ import annotations

import json
import re

from guards.base import GuardResult, clamp_level, map_categories

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class SingGuardNSFABackend:
    name = "singguard-nsfa"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001/v1",
        model: str = "inclusionAI/SingGuard-NSFA-2B",
        api_key: str = "EMPTY",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def build_prompt(self, text: str, role: str, prompt: str | None) -> str:
        task = "用户请求" if role == "prompt" else "模型回复"
        context = f"\n【对应的用户请求】{prompt}" if prompt else ""
        return (
            "你是内容安全检测器。请判断下面的" + task + "是否包含安全风险，"
            "风险类别从下列七类中选择：仇恨辱骂、偏见歧视、伦理道德、身心健康、违法犯罪、谣言错误、隐私财产。"
            '只输出 JSON：{"safety":"safe|controversial|unsafe","categories":["..."],"reason":"一句话"}。'
            f"{context}\n【{task}】{text}"
        )

    def predict(self, text: str, role: str = "prompt", prompt: str | None = None) -> GuardResult:
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": self.build_prompt(text, role, prompt)}],
                "temperature": 0.0,
                "max_tokens": 256,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        raw = payload["choices"][0]["message"]["content"]
        return self.parse_output(raw)

    def parse_output(self, raw: str) -> GuardResult:
        match = JSON_RE.search(raw or "")
        level, categories = "safe", []
        if match:
            try:
                data = json.loads(match.group(0))
                level = clamp_level(str(data.get("safety", "safe")))
                categories = data.get("categories") or map_categories(str(data.get("reason", "")))
            except json.JSONDecodeError:
                level = clamp_level(raw.split("\n")[0])
        return GuardResult(
            level=level,
            categories=list(categories),
            score=1.0 if level == "unsafe" else (0.5 if level == "controversial" else 0.0),
            backend=self.name,
            raw=(raw or "").strip(),
        )
