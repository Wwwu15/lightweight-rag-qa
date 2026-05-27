"""OpenAI-compatible chat model helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os

from langchain_openai import ChatOpenAI


@dataclass(slots=True)
class LLMConfig:
    """外部 LLM API 配置。"""

    base_url: str
    model: str
    api_key: str
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量读取 LLM 配置。"""
        return cls(
            base_url=os.getenv("LLM_BASE_URL", ""),
            model=os.getenv("LLM_MODEL", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        )


def build_chat_model(config: LLMConfig) -> ChatOpenAI:
    """Create a ChatOpenAI client for OpenAI-compatible APIs."""
    return ChatOpenAI(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
    )
