"""策略路由：把「归一化结果 + 多后端裁决」换算成一个动作。

动作只有三种：
    allow      通过，交给主模型正常回答（保住有用性）
    safe_mode  以约束提示生成，回答必须落在安全边界内（学术讨论、情绪支持等）
    block      直接给出场景化安全回复，并做输出侧复核
"""

from __future__ import annotations

from dataclasses import dataclass, field

from guards.base import GuardResult

ACTION_ALLOW = "allow"
ACTION_SAFE_MODE = "safe_mode"
ACTION_BLOCK = "block"


@dataclass
class RouteConfig:
    block_threshold: float = 0.6
    decoded_rule_threshold: float = 0.35
    review_threshold: float = 0.3
    benign_scene_whitelist: tuple[str, ...] = ("身心健康", "伦理道德")
    arbitration: str = "majority"  # majority / any / fast


@dataclass
class RouteDecision:
    action: str
    scene: str
    reason: str
    votes: dict[str, str] = field(default_factory=dict)


def _safe_mode_prompt(scene: str) -> str:
    """约束提示：要求模型在保持信息价值的前提下不越过安全边界。"""
    return (
        "你在安全模式下回答。规则：1) 不提供可用于实施伤害的操作细节、配方、代码或渠道；"
        "2) 若请求要求认同或传播针对特定群体的贬损观点、或为有害行为提供论据，应明确拒绝并说明理由，不要顺着请求展开；"
        "3) 对敏感但可讨论的议题说明事实与风险，给出合规替代方向；"
        "4) 用户表达情绪困扰时，先承接情绪再给可执行建议；"
        f"5) 当前主题类别为「{scene}」。"
    )


class Router:
    def __init__(self, config: RouteConfig | None = None):
        self.config = config or RouteConfig()

    def decide(self, norm, input_verdicts: list[GuardResult]) -> RouteDecision:
        votes = {v.backend: v.level for v in input_verdicts}
        unsafe = [v for v in input_verdicts if v.level == "unsafe"]
        controversial = [v for v in input_verdicts if v.level == "controversial"]

        if norm.decoded and norm.score >= self.config.decoded_rule_threshold:
            # 结构信号：存在解码载荷且剥离后仍命中风险词典，直接拦截，不依赖护栏是否识别编码
            return RouteDecision(ACTION_BLOCK, norm.top_scene if norm.top_scene != "未分类" else "伦理道德", "解码载荷命中规则", votes)

        if len(input_verdicts) == 1:
            # 单护栏部署时直接采纳该后端结论，避免"多数票"在大模型缺席时形同虚设
            blocked = input_verdicts[0].level == "unsafe"
        elif self.config.arbitration == "any":
            blocked = bool(unsafe)
        elif self.config.arbitration == "fast":
            blocked = bool(input_verdicts) and input_verdicts[0].level == "unsafe"
        else:  # majority：至少两票 unsafe，或一票 unsafe 且词典分高
            blocked = len(unsafe) >= 2 or (len(unsafe) == 1 and norm.score >= self.config.block_threshold)

        scene = norm.top_scene if norm.top_scene != "未分类" else (
            unsafe[0].categories[0] if unsafe and unsafe[0].categories else "伦理道德"
        )

        if blocked:
            return RouteDecision(ACTION_BLOCK, scene, "输入侧判为高风险", votes)
        if controversial or (norm.score >= self.config.review_threshold):
            return RouteDecision(ACTION_SAFE_MODE, scene, "输入侧存在争议，转安全模式", votes)
        if scene in self.config.benign_scene_whitelist and norm.score < self.config.review_threshold:
            # 情绪困扰、伦理讨论类请求通常不应直接拒绝，走安全模式保留有用性
            return RouteDecision(ACTION_SAFE_MODE, scene, "敏感主题但无明确危害意图", votes)
        return RouteDecision(ACTION_ALLOW, scene, "未检出风险", votes)

    def review_output(self, output_verdicts: list[GuardResult]) -> tuple[bool, str]:
        """输出侧复核：任一后端判 unsafe 即需要重写。"""
        for verdict in output_verdicts:
            if verdict.level == "unsafe":
                return False, f"{verdict.backend}: {verdict.raw[:80]}"
        return True, ""
