"""Prompt construction for retrieval-augmented question answering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def build_rag_prompt(question: str, documents: Sequence[Mapping[str, str]]) -> str:
    """Build a grounded QA prompt with numbered source context."""
    context_blocks = []
    for index, document in enumerate(documents, start=1):
        # 每个检索片段都带编号，方便模型在答案中引用来源。
        source = document.get("source") or "unknown"
        chunk_id = document.get("chunk_id") or ""
        content = document.get("content") or ""
        chunk_suffix = f" ({chunk_id})" if chunk_id else ""
        context_blocks.append(f"[{index}] Source: {source}{chunk_suffix}\n{content}")

    context = "\n\n".join(context_blocks) or "No context was retrieved."
    return (
        "Use only the context below to answer the question. "
        "If the context is insufficient, say you do not know. "
        "Answer in Chinese. "
        "Cite sources by their bracketed numbers when relevant.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
