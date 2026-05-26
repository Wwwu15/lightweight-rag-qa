"""Lightweight local RAG knowledge base package."""

from rag_knowledge_base.llm import LLMConfig, build_chat_model
from rag_knowledge_base.rag import RAGConfig, RAGKnowledgeBase

__version__ = "0.1.0"

__all__ = [
    "LLMConfig",
    "RAGConfig",
    "RAGKnowledgeBase",
    "__version__",
    "build_chat_model",
]
