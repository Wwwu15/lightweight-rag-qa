from pathlib import Path

import pytest

from rag_knowledge_base.llm import LLMConfig, build_chat_model
from rag_knowledge_base.prompts import build_rag_prompt
from langchain_core.documents import Document

from rag_knowledge_base.rag import JsonVectorStore, RAGConfig, RAGKnowledgeBase


class FakeEmbeddings:
    def __init__(self):
        self.documents = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[float(len(text))] for text in texts]

    def embed_query(self, text):
        return [float(len(text))]


class FakeVectorStore:
    def __init__(self, persist_directory, embedding_function, collection_name):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)
        return [doc.metadata["chunk_id"] for doc in documents]

    def similarity_search(self, query, k):
        return self.documents[:k]

    def delete(self, ids=None, where=None):
        if ids:
            id_set = set(ids)
            self.documents = [
                doc for doc in self.documents if doc.metadata.get("chunk_id") not in id_set
            ]
            return

        if where:
            def matches(document):
                return all(document.metadata.get(key) == value for key, value in where.items())

            self.documents = [doc for doc in self.documents if not matches(doc)]

    def reset_collection(self):
        self.documents = []


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        class Response:
            content = "Paris is the capital."

        return Response()


def test_add_documents_splits_text_and_attaches_stable_metadata(tmp_path):
    rag = RAGKnowledgeBase(
        RAGConfig(
            persist_directory=tmp_path / "chroma",
            chunk_size=30,
            chunk_overlap=5,
            collection_name="test",
        ),
        embeddings=FakeEmbeddings(),
        vector_store_factory=FakeVectorStore,
    )

    ids = rag.add_documents(
        [
            {
                "text": "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda.",
                "source": "notes.txt",
                "metadata": {"page": 2},
            }
        ]
    )

    assert ids
    stored = rag.vector_store.documents
    assert len(stored) > 1
    assert stored[0].page_content
    assert stored[0].metadata == {
        "source": "notes.txt",
        "page": 2,
        "document_id": "notes.txt",
        "chunk_index": 0,
        "chunk_id": "notes.txt:0",
    }
    assert stored[1].metadata["chunk_id"] == "notes.txt:1"


def test_retrieve_uses_similarity_search_top_k(tmp_path):
    rag = RAGKnowledgeBase(
        RAGConfig(persist_directory=tmp_path / "chroma", chunk_size=20, chunk_overlap=0),
        embeddings=FakeEmbeddings(),
        vector_store_factory=FakeVectorStore,
    )
    rag.add_documents(
        [
            {"text": "First source text", "source": "one.md"},
            {"text": "Second source text", "source": "two.md"},
        ]
    )

    results = rag.retrieve("capital city?", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "one.md"


def test_answer_builds_prompt_calls_llm_and_returns_citations(tmp_path):
    fake_llm = FakeLLM()
    rag = RAGKnowledgeBase(
        RAGConfig(persist_directory=tmp_path / "chroma", chunk_size=40, chunk_overlap=0),
        embeddings=FakeEmbeddings(),
        llm=fake_llm,
        vector_store_factory=FakeVectorStore,
    )
    rag.add_documents(
        [
            {"text": "Paris is the capital of France.", "source": "france.md"},
            {"text": "Berlin is the capital of Germany.", "source": "germany.md"},
        ]
    )

    result = rag.answer("What is the capital of France?", top_k=2)

    assert result["answer"] == "Paris is the capital."
    assert result["sources"] == [
        {"source": "france.md", "chunk_id": "france.md:0"},
        {"source": "germany.md", "chunk_id": "germany.md:0"},
    ]
    assert "Question:\nWhat is the capital of France?" in fake_llm.prompts[0]
    assert "[1] Source: france.md" in fake_llm.prompts[0]
    assert "Paris is the capital of France." in fake_llm.prompts[0]


def test_prompt_includes_context_and_source_instructions():
    prompt = build_rag_prompt(
        question="What changed?",
        documents=[
            {
                "content": "The API now returns citations.",
                "source": "design.md",
                "chunk_id": "design.md:0",
            }
        ],
    )

    assert "Use only the context below" in prompt
    assert "Answer in Chinese." in prompt
    assert "Question:\nWhat changed?" in prompt
    assert "[1] Source: design.md" in prompt
    assert "The API now returns citations." in prompt


def test_build_chat_model_uses_openai_compatible_configuration(monkeypatch):
    calls = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr("rag_knowledge_base.llm.ChatOpenAI", FakeChatOpenAI)

    model = build_chat_model(
        LLMConfig(
            base_url="https://example.test/v1",
            model="gpt-compatible",
            api_key="secret",
            temperature=0.1,
        )
    )

    assert isinstance(model, FakeChatOpenAI)
    assert calls == {
        "base_url": "https://example.test/v1",
        "model": "gpt-compatible",
        "api_key": "secret",
        "temperature": 0.1,
    }


def test_default_config_uses_expected_models(tmp_path):
    config = RAGConfig(persist_directory=tmp_path)

    assert config.embedding_model == "nomic-embed-text:v1.5"
    assert Path(config.persist_directory) == tmp_path


def test_delete_documents_removes_chunks_by_source_and_file_name(tmp_path):
    rag = RAGKnowledgeBase(
        RAGConfig(persist_directory=tmp_path / "chroma", chunk_size=50, chunk_overlap=0),
        embeddings=FakeEmbeddings(),
        vector_store_factory=FakeVectorStore,
    )
    rag.add_documents(
        [
            {
                "text": "Alpha source content",
                "source": str(tmp_path / "alpha.pdf"),
                "metadata": {"file_name": "alpha.pdf"},
            },
            {
                "text": "Beta source content",
                "source": str(tmp_path / "beta.docx"),
                "metadata": {"file_name": "beta.docx"},
            },
        ]
    )

    deleted = rag.delete_documents(source=str(tmp_path / "alpha.pdf"), file_name="alpha.pdf")

    assert deleted == 1
    assert [doc.metadata["file_name"] for doc in rag.vector_store.documents] == ["beta.docx"]


def test_clear_vector_store_removes_all_chunks(tmp_path):
    rag = RAGKnowledgeBase(
        RAGConfig(persist_directory=tmp_path / "chroma", chunk_size=50, chunk_overlap=0),
        embeddings=FakeEmbeddings(),
        vector_store_factory=FakeVectorStore,
    )
    rag.add_documents([{"text": "Alpha", "source": "alpha.pdf"}])

    deleted = rag.clear()

    assert deleted == 1
    assert rag.vector_store.documents == []


def test_json_vector_store_strips_invalid_surrogate_text_before_persisting(tmp_path):
    store = JsonVectorStore(
        persist_directory=str(tmp_path),
        embedding_function=FakeEmbeddings(),
        collection_name="test",
    )

    ids = store.add_documents(
        [
            Document(
                page_content="bad \ud835 text",
                metadata={"source": "bad.pdf", "file_name": "bad\ud835.pdf", "chunk_id": "bad:0"},
            )
        ]
    )

    assert ids == ["bad:0"]
    content = (tmp_path / "test.json").read_text(encoding="utf-8")
    assert "\\ud835" not in content
    reloaded = JsonVectorStore(
        persist_directory=str(tmp_path),
        embedding_function=FakeEmbeddings(),
        collection_name="test",
    )
    assert reloaded.documents[0].page_content == "bad  text"
    assert reloaded.documents[0].metadata["file_name"] == "bad.pdf"


def test_json_vector_store_recovers_from_empty_or_corrupt_store_file(tmp_path):
    store_path = tmp_path / "test.json"
    store_path.write_text("", encoding="utf-8")

    empty_store = JsonVectorStore(
        persist_directory=str(tmp_path),
        embedding_function=FakeEmbeddings(),
        collection_name="test",
    )

    assert empty_store.documents == []
    assert empty_store.vectors == []
    assert store_path.read_text(encoding="utf-8") == '{"documents": [], "vectors": []}'

    store_path.write_text("{not valid json", encoding="utf-8")

    corrupt_store = JsonVectorStore(
        persist_directory=str(tmp_path),
        embedding_function=FakeEmbeddings(),
        collection_name="test",
    )

    assert corrupt_store.documents == []
    assert corrupt_store.vectors == []
    assert store_path.read_text(encoding="utf-8") == '{"documents": [], "vectors": []}'
