"""端到端流水线：归一化 → 输入裁决 → 路由 → 生成 → 输出复核 → 兜底。

设计约束（对应赛题注意事项）：
- 任何异常、超时、空输出都必须被兜底成一条非空的安全回复；
- 不允许对无害请求一律拒答，allow / safe_mode 两条路径都要保证信息量；
- 输出侧复核不通过时先重写，重写仍不通过才回落模板。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from normalize import normalize
from responder import Reply, render
from router import ACTION_ALLOW, ACTION_BLOCK, ACTION_SAFE_MODE, RouteConfig, Router, _safe_mode_prompt


@dataclass
class PipelineResult:
    prompt: str
    action: str
    scene: str
    text: str
    latency_ms: float
    input_votes: dict[str, str] = field(default_factory=dict)
    output_votes: dict[str, str] = field(default_factory=dict)
    families: list[str] = field(default_factory=list)
    fallback: bool = False
    notes: str = ""


class SafePipeline:
    def __init__(
        self,
        input_guards: list,
        output_guards: list | None = None,
        generator: Callable[[str, str], str] | None = None,
        config: RouteConfig | None = None,
        timeout_s: float = 8.0,
    ):
        self.input_guards = input_guards
        self.output_guards = output_guards if output_guards is not None else input_guards[:1]
        self.generator = generator or (lambda system, user: "")
        self.router = Router(config)
        self.timeout_s = timeout_s

    def _guarded_generate(self, system_prompt: str, user_prompt: str) -> str:
        started = time.perf_counter()
        text = self.generator(system_prompt, user_prompt)
        if (time.perf_counter() - started) > self.timeout_s:
            raise TimeoutError("生成超时")
        return (text or "").strip()

    def run(self, prompt: str) -> PipelineResult:
        started = time.perf_counter()
        norm = normalize(prompt, level=3)

        # 设计上护栏应看到"解码/剥离后的核心请求"，否则 Base64 一类包装会整条漏判。
        # 同时保留原文，避免剥离过度导致语义丢失。
        guard_input = prompt
        if norm.normalized and norm.normalized != prompt:
            guard_input = f"{norm.normalized}\n---\n{prompt}"

        verdicts = []
        for guard in self.input_guards:
            try:
                verdicts.append(guard.predict(guard_input, role="prompt"))
            except Exception:  # 单后端故障不应阻断整条链路
                continue
        decision = self.router.decide(norm, verdicts)

        text, fallback, notes = "", False, ""
        if decision.action == ACTION_ALLOW:
            try:
                text = self._guarded_generate("", prompt)
            except Exception as exc:  # noqa: BLE001
                notes = f"生成失败: {type(exc).__name__}"
        elif decision.action == ACTION_SAFE_MODE:
            try:
                text = self._guarded_generate(_safe_mode_prompt(decision.scene), prompt)
            except Exception as exc:  # noqa: BLE001
                notes = f"生成失败: {type(exc).__name__}"

        if decision.action == ACTION_BLOCK or not text:
            reply: Reply = render(decision.scene, prompt, decision.reason)
            text = reply.text
            fallback = True

        # 输出侧复核：只对模型生成的文本做，模板兜底本身已按场景对齐
        output_votes = {}
        if not fallback:
            verdicts_out = []
            for guard in self.output_guards:
                try:
                    verdict = guard.predict(text, role="response", prompt=prompt)
                except Exception:  # noqa: BLE001
                    continue
                verdicts_out.append(verdict)
                output_votes[verdict.backend] = verdict.level
            ok, reason = self.router.review_output(verdicts_out)
            if not ok:
                text = render(decision.scene, prompt, reason).text
                fallback = True
                notes = (notes + " 输出复核未通过").strip()

        if not text.strip():
            text = render(decision.scene, prompt, "空输出兜底").text
            fallback = True

        latency = (time.perf_counter() - started) * 1000
        return PipelineResult(
            prompt=prompt,
            action=decision.action,
            scene=decision.scene,
            text=text,
            latency_ms=round(latency, 1),
            input_votes=decision.votes,
            output_votes=output_votes,
            families=norm.families,
            fallback=fallback,
            notes=notes,
        )
