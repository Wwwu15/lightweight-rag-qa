"""Modular RAG core for document ingestion, retrieval, and answering."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib import request

from langchain_core.documents import Document

from rag_knowledge_base.llm import LLMConfig, build_chat_model
from rag_knowledge_base.prompts import build_rag_prompt


@dataclass(slots=True)
class RAGConfig:
    persist_directory: Path | str
    embedding_model: str = "nomic-embed-text:v1.5"
    ollama_base_url: str = "http://localhost:11434"
    collection_name: str = "rag_knowledge_base"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "data/chroma"),
            embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("TOP_K", "4")),
        )

    @classmethod
    def from_settings(cls, settings: Any) -> "RAGConfig":
        return cls(
            persist_directory=getattr(settings, "chroma_persist_dir", getattr(settings, "chroma_dir")),
            embedding_model=getattr(settings, "ollama_model"),
            ollama_base_url=getattr(settings, "ollama_base_url"),
            chunk_size=getattr(settings, "chunk_size"),
            chunk_overlap=getattr(settings, "chunk_overlap"),
            top_k=getattr(settings, "top_k"),
        )


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    sources: list[dict[str, Any]]


class RAGKnowledgeBase:
    """Small injectable RAG pipeline suitable for app code and tests."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        *,
        embeddings: Any | None = None,
        llm: Any | None = None,
        vector_store: Any | None = None,
        vector_store_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or RAGConfig.from_env()
        self.embeddings = embeddings or build_ollama_embeddings(self.config)
        self.llm = llm
        self._vector_store_factory = vector_store_factory or build_chroma_vector_store
        self.vector_store = vector_store or self._vector_store_factory(
            persist_directory=str(self.config.persist_directory),
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
        )
        self._splitter = build_text_splitter(self.config)

    def add_documents(self, documents: Sequence[Mapping[str, Any] | Document]) -> list[str]:
        """Split and ingest documents with deterministic chunk metadata."""
        chunks: list[Document] = []
        for document in documents:
            text, source, metadata = _normalize_input_document(document)
            document_id = str(metadata.get("document_id") or source or _content_id(text))
            base_metadata = {**metadata, "source": source, "document_id": document_id}

            for chunk_index, chunk_text in enumerate(self._split_text(text)):
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": chunk_index,
                    "chunk_id": f"{document_id}:{chunk_index}",
                }
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        if not chunks:
            return []
        return list(self.vector_store.add_documents(chunks))

    def ingest_documents(self, documents: Sequence[Mapping[str, Any] | Document]) -> list[str]:
        return self.add_documents(documents)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve the most similar chunks for a query."""
        k = top_k or self.config.top_k
        return list(self.vector_store.similarity_search(query, k=k))

    def delete_documents(self, *, source: str | None = None, file_name: str | None = None) -> int:
        """Delete chunks matching a raw document source or file name."""
        if not source and not file_name:
            return 0

        matching_ids = self._matching_vector_ids(source=source, file_name=file_name)
        if not matching_ids:
            return 0

        if hasattr(self.vector_store, "delete"):
            self.vector_store.delete(ids=matching_ids)
        elif hasattr(self.vector_store, "delete_by_ids"):
            self.vector_store.delete_by_ids(matching_ids)
        else:
            raise NotImplementedError("Vector store does not support deleting chunks")
        return len(matching_ids)

    def clear(self) -> int:
        """Remove all chunks from the current vector store collection."""
        deleted = len(self._all_documents())
        if hasattr(self.vector_store, "reset_collection"):
            self.vector_store.reset_collection()
        elif hasattr(self.vector_store, "delete_collection"):
            self.vector_store.delete_collection()
            self.vector_store = self._vector_store_factory(
                persist_directory=str(self.config.persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name,
            )
        elif hasattr(self.vector_store, "clear"):
            self.vector_store.clear()
        else:
            raise NotImplementedError("Vector store does not support clearing chunks")
        return deleted

    def answer(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """Answer a question and return source citations for retrieved chunks."""
        retrieved = self.retrieve(question, top_k=top_k)
        prompt_documents = [
            {
                "content": document.page_content,
                "source": str(document.metadata.get("source", "unknown")),
                "chunk_id": str(document.metadata.get("chunk_id", "")),
            }
            for document in retrieved
        ]
        prompt = build_rag_prompt(question=question, documents=prompt_documents)
        llm = self.llm or build_chat_model(LLMConfig.from_env())
        response = llm.invoke(prompt)
        answer_text = getattr(response, "content", response)

        return {
            "answer": str(answer_text),
            "sources": [
                {
                    "source": str(document.metadata.get("source", "unknown")),
                    "chunk_id": str(document.metadata.get("chunk_id", "")),
                }
                for document in retrieved
            ],
        }

    def _split_text(self, text: str) -> list[str]:
        return [chunk for chunk in self._splitter.split_text(text) if chunk.strip()]

    def _all_documents(self) -> list[Document]:
        if hasattr(self.vector_store, "documents"):
            return list(self.vector_store.documents)

        collection = getattr(self.vector_store, "_collection", None)
        if collection is not None:
            payload = collection.get(include=["metadatas", "documents"])
            documents = payload.get("documents") or []
            metadatas = payload.get("metadatas") or []
            return [
                Document(page_content=content or "", metadata=metadata or {})
                for content, metadata in zip(documents, metadatas)
            ]
        return []

    def _matching_vector_ids(self, *, source: str | None, file_name: str | None) -> list[str]:
        collection = getattr(self.vector_store, "_collection", None)
        if collection is not None:
            payload = collection.get(include=["metadatas"])
            ids = payload.get("ids") or []
            metadatas = payload.get("metadatas") or []
            matching_ids = []
            for vector_id, metadata in zip(ids, metadatas):
                metadata = metadata or {}
                source_matches = source is not None and metadata.get("source") == source
                file_matches = file_name is not None and metadata.get("file_name") == file_name
                if source_matches or file_matches:
                    matching_ids.append(str(vector_id))
            return matching_ids

        matching_ids = []
        for document in self._all_documents():
            metadata = document.metadata
            source_matches = source is not None and metadata.get("source") == source
            file_matches = file_name is not None and metadata.get("file_name") == file_name
            if source_matches or file_matches:
                chunk_id = metadata.get("chunk_id")
                if chunk_id:
                    matching_ids.append(str(chunk_id))
        return matching_ids


def build_text_splitter(config: RAGConfig) -> Any:
    """Create a RecursiveCharacterTextSplitter from the installed LangChain package."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            RecursiveCharacterTextSplitter = _FallbackRecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def build_ollama_embeddings(config: RAGConfig) -> Any:
    """Create Ollama embeddings using the configured nomic model by default."""
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        return SimpleOllamaEmbeddings(model=config.embedding_model, base_url=config.ollama_base_url)

    return OllamaEmbeddings(model=config.embedding_model, base_url=config.ollama_base_url)


def build_chroma_vector_store(
    *,
    persist_directory: str,
    embedding_function: Any,
    collection_name: str,
) -> Any:
    """Create a persistent Chroma vector store."""
    try:
        from langchain_chroma import Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            return JsonVectorStore(
                persist_directory=persist_directory,
                embedding_function=embedding_function,
                collection_name=collection_name,
            )

    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )


def _normalize_input_document(document: Mapping[str, Any] | Document) -> tuple[str, str, dict[str, Any]]:
    if isinstance(document, Document):
        metadata = dict(document.metadata)
        text = document.page_content
        source = str(metadata.get("source", "unknown"))
        return text, source, metadata

    text = str(document.get("text") or document.get("page_content") or "")
    metadata = dict(document.get("metadata") or {})
    source = str(document.get("source") or metadata.get("source") or "unknown")
    return text, source, metadata


def _content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _source_metadata(document: Document) -> dict[str, Any]:
    metadata = dict(document.metadata)
    return {
        "source": str(metadata.get("source", "unknown")),
        "file_name": metadata.get("file_name") or Path(str(metadata.get("source", "unknown"))).name,
        "page": metadata.get("page"),
        "chunk_index": metadata.get("chunk_index"),
        "chunk_id": str(metadata.get("chunk_id", "")),
    }


class RagPipeline(RAGKnowledgeBase):
    @classmethod
    def from_settings(cls, settings: Any) -> "RagPipeline":
        return cls(config=RAGConfig.from_settings(settings))

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        retrieved = self.retrieve(question, top_k=top_k)
        prompt_documents = [
            {
                "content": document.page_content,
                "source": str(document.metadata.get("source", "unknown")),
                "chunk_id": str(document.metadata.get("chunk_id", "")),
            }
            for document in retrieved
        ]
        prompt = build_rag_prompt(question=question, documents=prompt_documents)
        llm = self.llm or build_chat_model(LLMConfig.from_env())
        response = llm.invoke(prompt)
        answer_text = getattr(response, "content", response)
        return RAGAnswer(
            answer=str(answer_text),
            sources=[_source_metadata(document) for document in retrieved],
        )


class _FallbackRecursiveCharacterTextSplitter:
    """Small fallback used only when LangChain's splitter package is unavailable."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - self.chunk_overlap
        return chunks


class SimpleOllamaEmbeddings:
    """Minimal Ollama embeddings client compatible with LangChain vector stores."""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Ollama embedding response did not include embedding: {data}")
        return [float(value) for value in embedding]


class JsonVectorStore:
    """Small persistent cosine vector store used when Chroma is unavailable."""

    def __init__(
        self,
        *,
        persist_directory: str,
        embedding_function: Any,
        collection_name: str,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.store_path = self.persist_directory / f"{collection_name}.json"
        self.documents: list[Document] = []
        self.vectors: list[list[float]] = []
        self._load()

    def add_documents(self, documents: Sequence[Document]) -> list[str]:
        sanitized_documents = [_sanitize_document(document) for document in documents]
        texts = [document.page_content for document in sanitized_documents]
        vectors = self.embedding_function.embed_documents(texts)
        ids = []
        for document, vector in zip(sanitized_documents, vectors):
            chunk_id = str(document.metadata.get("chunk_id") or _content_id(document.page_content))
            ids.append(chunk_id)
            self.documents.append(document)
            self.vectors.append([float(value) for value in vector])
        self._persist()
        return ids

    def similarity_search(self, query: str, k: int) -> list[Document]:
        if not self.documents:
            return []
        query_vector = self.embedding_function.embed_query(query)
        scored = [
            (_cosine_similarity(query_vector, vector), index)
            for index, vector in enumerate(self.vectors)
        ]
        scored.sort(reverse=True)
        return [self.documents[index] for _, index in scored[:k]]

    def delete(self, ids: Sequence[str] | None = None, where: Mapping[str, Any] | None = None) -> None:
        id_set = set(ids or [])

        def should_delete(document: Document) -> bool:
            if id_set and str(document.metadata.get("chunk_id")) in id_set:
                return True
            if where:
                return all(document.metadata.get(key) == value for key, value in where.items())
            return False

        kept_documents: list[Document] = []
        kept_vectors: list[list[float]] = []
        for document, vector in zip(self.documents, self.vectors):
            if not should_delete(document):
                kept_documents.append(document)
                kept_vectors.append(vector)
        self.documents = kept_documents
        self.vectors = kept_vectors
        self._persist()

    def reset_collection(self) -> None:
        self.documents = []
        self.vectors = []
        self._persist()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        raw_payload = self.store_path.read_text(encoding="utf-8")
        if not raw_payload.strip():
            self.reset_collection()
            return
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            self.reset_collection()
            return
        self.documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in payload.get("documents", [])
        ]
        self.vectors = payload.get("vectors", [])

    def _persist(self) -> None:
        payload = {
            "documents": [
                {"page_content": document.page_content, "metadata": document.metadata}
                for document in self.documents
            ],
            "vectors": self.vectors,
        }
        self.store_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _sanitize_document(document: Document) -> Document:
    return Document(
        page_content=_strip_surrogates(document.page_content),
        metadata=_sanitize_json_value(dict(document.metadata)),
    )


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_surrogates(value)
    if isinstance(value, dict):
        return {
            _strip_surrogates(str(key)): _sanitize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    return value


def _strip_surrogates(value: str) -> str:
    return "".join(char for char in value if not 0xD800 <= ord(char) <= 0xDFFF)
