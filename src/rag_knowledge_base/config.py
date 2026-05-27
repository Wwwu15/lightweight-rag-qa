from __future__ import annotations

"""项目配置读取模块，统一管理 .env 和默认参数。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _env(name: str, default: str, *aliases: str) -> str:
    """读取环境变量，支持新旧变量名兼容。"""
    for candidate in (name, *aliases):
        value = os.getenv(candidate)
        if value not in (None, ""):
            return value
    return default


def _env_int(name: str, default: int) -> int:
    """读取整数型环境变量。"""
    aliases = {
        "RAG_CHUNK_SIZE": ("CHUNK_SIZE",),
        "RAG_CHUNK_OVERLAP": ("CHUNK_OVERLAP",),
        "RAG_TOP_K": ("TOP_K",),
    }
    value = None
    for candidate in (name, *aliases.get(name, ())):
        value = os.getenv(candidate)
        if value not in (None, ""):
            break
    if value is None or value == "":
        return default
    return int(value)


@dataclass(slots=True)
class Settings:
    """应用运行配置。"""

    raw_dir: Path = field(default_factory=lambda: Path(_env("RAG_RAW_DIR", "data/raw", "RAW_DATA_DIR")))
    chroma_dir: Path = field(
        default_factory=lambda: Path(_env("RAG_CHROMA_DIR", "data/chroma", "CHROMA_PERSIST_DIR"))
    )
    ollama_model: str = field(
        default_factory=lambda: _env("RAG_OLLAMA_MODEL", "nomic-embed-text:v1.5", "OLLAMA_EMBED_MODEL")
    )
    ollama_base_url: str = field(
        default_factory=lambda: _env("RAG_OLLAMA_BASE_URL", "http://localhost:11434", "OLLAMA_BASE_URL")
    )
    llm_base_url: str = field(default_factory=lambda: _env("RAG_LLM_BASE_URL", "", "LLM_BASE_URL"))
    llm_model: str = field(default_factory=lambda: _env("RAG_LLM_MODEL", "", "LLM_MODEL"))
    llm_api_key: str = field(default_factory=lambda: _env("RAG_LLM_API_KEY", "", "LLM_API_KEY"))
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 200))
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 4))

    def __post_init__(self) -> None:
        """加载 .env 后重新覆盖默认配置。"""
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        self.raw_dir = Path(_env("RAG_RAW_DIR", str(self.raw_dir), "RAW_DATA_DIR"))
        self.chroma_dir = Path(_env("RAG_CHROMA_DIR", str(self.chroma_dir), "CHROMA_PERSIST_DIR"))
        self.ollama_model = _env("RAG_OLLAMA_MODEL", self.ollama_model, "OLLAMA_EMBED_MODEL")
        self.ollama_base_url = _env("RAG_OLLAMA_BASE_URL", self.ollama_base_url, "OLLAMA_BASE_URL")
        self.llm_base_url = _env("RAG_LLM_BASE_URL", self.llm_base_url, "LLM_BASE_URL")
        self.llm_model = _env("RAG_LLM_MODEL", self.llm_model, "LLM_MODEL")
        self.llm_api_key = _env("RAG_LLM_API_KEY", self.llm_api_key, "LLM_API_KEY")
        self.chunk_size = _env_int("RAG_CHUNK_SIZE", self.chunk_size)
        self.chunk_overlap = _env_int("RAG_CHUNK_OVERLAP", self.chunk_overlap)
        self.top_k = _env_int("RAG_TOP_K", self.top_k)

    @classmethod
    def from_env(cls) -> "Settings":
        """从当前环境构建配置对象。"""
        return cls()

    @property
    def raw_data_dir(self) -> Path:
        """兼容旧代码中的 raw_data_dir 命名。"""
        return self.raw_dir

    @property
    def chroma_persist_dir(self) -> Path:
        """兼容旧代码中的 chroma_persist_dir 命名。"""
        return self.chroma_dir
