"""护栏后端注册表：按配置构造后端实例，未安装依赖时给出明确提示。"""

from __future__ import annotations

from guards.base import GuardBackend, GuardResult  # noqa: F401
from guards.mock import MockGuardBackend


def build_backend(name: str, **kwargs) -> GuardBackend:
    name = (name or "mock").lower()
    if name in ("mock", "simulate"):
        return MockGuardBackend(**kwargs)
    if name in ("ollama", "ollama-guard", "qwen3guard-ollama"):
        from guards.ollama import OllamaGuardBackend

        return OllamaGuardBackend(**kwargs)
    if name in ("qwen3guard", "hf_qwen3guard"):
        from guards.hf_qwen3guard import Qwen3GuardBackend

        return Qwen3GuardBackend(**kwargs)
    if name in ("nsfa", "singguard"):
        from guards.hf_nsfa import SingGuardNSFABackend

        return SingGuardNSFABackend(**kwargs)
    if name in ("agentdog", "agentdog15"):
        from guards.hf_agentdog import AgentDoGBackend

        return AgentDoGBackend(**kwargs)
    raise ValueError(f"未知后端: {name}")
