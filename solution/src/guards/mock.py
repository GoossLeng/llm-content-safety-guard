"""联调用模拟后端。

仅用于在没有模型权重时跑通「归一化 → 路由 → 回复 → 输出校验」的整条链路，
其判定来自 normalize 的词典规则与攻击族标签，不代表任何模型的真实能力。
输出的 GuardResult 带 backend="mock"，evaluate.py 会把这类结果标记为 simulated。
"""

from __future__ import annotations

from guards.base import GuardResult
from normalize import FAMILIES, normalize


class MockGuardBackend:
    name = "mock"

    def __init__(self, strictness: float = 0.35):
        self.strictness = strictness

    def predict(self, text: str, role: str = "prompt", prompt: str | None = None) -> GuardResult:
        target = text if role == "prompt" else text
        result = normalize(target, level=3)
        level = "unsafe" if result.score >= self.strictness else ("controversial" if result.score >= 0.2 else "safe")
        categories = [result.top_scene] if result.scene_scores else []
        family_note = "、".join(f for f in result.families if f in FAMILIES)
        return GuardResult(
            level=level,
            categories=categories,
            score=result.score,
            backend=self.name,
            raw=f"families={family_note}; top={result.top_scene}; score={result.score}",
        )
