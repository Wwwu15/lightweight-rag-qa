from pathlib import Path

import pytest

from rag_knowledge_base.config import Settings


@pytest.fixture(autouse=True)
def clear_rag_env(monkeypatch):
    for name in [
        "RAG_RAW_DIR",
        "RAG_CHROMA_DIR",
        "RAG_OLLAMA_MODEL",
        "RAG_OLLAMA_BASE_URL",
        "RAG_LLM_BASE_URL",
        "RAG_LLM_MODEL",
        "RAG_LLM_API_KEY",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_TOP_K",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_settings_use_defaults_and_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "RAG_RAW_DIR=uploads/raw",
                "RAG_CHROMA_DIR=stores/chroma",
                "RAG_OLLAMA_MODEL=nomic-embed-text",
                "RAG_OLLAMA_BASE_URL=http://ollama:11434",
                "RAG_LLM_BASE_URL=http://llm:8000/v1",
                "RAG_LLM_MODEL=tiny-llm",
                "RAG_LLM_API_KEY=secret-key",
                "RAG_CHUNK_SIZE=512",
                "RAG_CHUNK_OVERLAP=64",
                "RAG_TOP_K=7",
            ]
        )
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.raw_dir == Path("uploads/raw")
    assert settings.chroma_dir == Path("stores/chroma")
    assert settings.ollama_model == "nomic-embed-text"
    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.llm_base_url == "http://llm:8000/v1"
    assert settings.llm_model == "tiny-llm"
    assert settings.llm_api_key == "secret-key"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.top_k == 7


def test_settings_defaults_when_env_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.raw_dir == Path("data/raw")
    assert settings.chroma_dir == Path("data/chroma")
    assert settings.chunk_size > settings.chunk_overlap >= 0
    assert settings.top_k > 0
