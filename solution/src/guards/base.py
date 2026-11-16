"""护栏后端统一接口。

所有后端都返回 GuardResult：level 取 safe / controversial / unsafe，
categories 为命中的风险类目（尽量映射到赛题七类场景），
detail 保留原始输出，便于审计与归因。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

SCENE_LABELS = ("仇恨辱骂", "偏见歧视", "伦理道德", "身心健康", "违法犯罪", "谣言错误", "隐私财产")


@dataclass
class GuardResult:
    level: str = "safe"
    categories: list[str] = field(default_factory=list)
    score: float = 0.0
    backend: str = ""
    raw: str = ""
    refusal: bool | None = None

    @property
    def blocked(self) -> bool:
        return self.level == "unsafe"

    @property
    def needs_review(self) -> bool:
        return self.level in ("unsafe", "controversial")


class GuardBackend(Protocol):
    name: str

    def predict(self, text: str, role: str = "prompt") -> GuardResult: ...


def clamp_level(value: str) -> str:
    value = (value or "").strip().lower()
    if value.startswith("unsafe"):
        return "unsafe"
    if value.startswith("controversial"):
        return "controversial"
    return "safe"


def map_categories(raw: str) -> list[str]:
    """把护栏模型给出的英文/中文类目尽量映射到赛题七类场景。"""
    text = (raw or "").lower()
    mapping = {
        "仇恨辱骂": ("hate", "harassment", "insult", "abuse", "辱骂", "仇恨"),
        "偏见歧视": ("bias", "discrimin", "stereotype", "unfair", "歧视", "偏见"),
        "伦理道德": ("ethic", "moral", "academic", "cheat", "伦理", "道德", "作弊"),
        "身心健康": ("self-harm", "suicide", "mental", "health", "drug", "自杀", "自残", "健康"),
        "违法犯罪": ("crime", "illegal", "weapon", "violent", "cyber", "犯罪", "违法", "暴力"),
        "谣言错误": ("misinformation", "rumor", "false", "conspiracy", "谣言", "虚假"),
        "隐私财产": ("privacy", "property", "pii", "credential", "phishing", "隐私", "财产", "钓鱼"),
    }
    hits = []
    for scene, keys in mapping.items():
        if any(key in text for key in keys):
            hits.append(scene)
    return hits
