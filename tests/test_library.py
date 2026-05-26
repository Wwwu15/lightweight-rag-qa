from pathlib import Path

from rag_knowledge_base.library import (
    clear_knowledge_base,
    clear_raw_documents_by_suffix,
    delete_documents,
    group_raw_documents_by_type,
    list_raw_documents,
)


class FakePipeline:
    def __init__(self) -> None:
        self.deleted = []
        self.cleared = False

    def delete_documents(self, *, source=None, file_name=None):
        self.deleted.append({"source": source, "file_name": file_name})
        return 3

    def clear(self):
        self.cleared = True
        return 9


class FakeLockedPath:
    suffix = ".pdf"
    name = "locked.pdf"

    def __init__(self) -> None:
        self.unlinked = False

    def exists(self):
        return True

    def is_file(self):
        return True

    def unlink(self):
        raise PermissionError("locked")

    def resolve(self):
        return self

    def relative_to(self, _parent):
        return self

    def __str__(self):
        return "locked.pdf"


def test_list_raw_documents_returns_supported_files_only(tmp_path):
    (tmp_path / "alpha.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "beta.docx").write_text("docx", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("txt", encoding="utf-8")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")

    documents = list_raw_documents(tmp_path)

    assert [doc.name for doc in documents] == ["alpha.pdf", "beta.docx"]


def test_group_raw_documents_by_type_splits_word_and_pdf(tmp_path):
    (tmp_path / "alpha.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "beta.docx").write_text("docx", encoding="utf-8")
    (tmp_path / "gamma.PDF").write_text("pdf", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("txt", encoding="utf-8")

    groups = group_raw_documents_by_type(tmp_path)

    assert [path.name for path in groups["word"]] == ["beta.docx"]
    assert [path.name for path in groups["pdf"]] == ["alpha.pdf", "gamma.PDF"]


def test_delete_documents_removes_raw_files_and_vector_chunks(tmp_path):
    alpha = tmp_path / "alpha.pdf"
    beta = tmp_path / "beta.docx"
    alpha.write_text("pdf", encoding="utf-8")
    beta.write_text("docx", encoding="utf-8")
    pipeline = FakePipeline()

    result = delete_documents(tmp_path, pipeline, ["alpha.pdf", "beta.docx"])

    assert not alpha.exists()
    assert not beta.exists()
    assert result.deleted_files == ["alpha.pdf", "beta.docx"]
    assert result.deleted_chunks == 6
    assert pipeline.deleted == [
        {"source": str(alpha), "file_name": "alpha.pdf"},
        {"source": str(beta), "file_name": "beta.docx"},
    ]


def test_delete_documents_reports_locked_file_without_deleting_vectors(tmp_path, monkeypatch):
    pipeline = FakePipeline()
    locked_path = FakeLockedPath()
    monkeypatch.setattr("rag_knowledge_base.library._safe_child", lambda _raw, _name: locked_path)

    result = delete_documents(tmp_path, pipeline, ["locked.pdf"])

    assert result.deleted_files == []
    assert result.deleted_chunks == 0
    assert result.failed_files == ["locked.pdf"]
    assert pipeline.deleted == []


def test_delete_documents_ignores_path_traversal(tmp_path):
    outside = tmp_path.parent / "outside.pdf"
    outside.write_text("keep", encoding="utf-8")

    result = delete_documents(tmp_path, FakePipeline(), ["../outside.pdf"])

    assert outside.exists()
    assert result.deleted_files == []
    assert result.deleted_chunks == 0


def test_clear_knowledge_base_clears_raw_and_vector_dirs(tmp_path):
    raw_dir = tmp_path / "raw"
    chroma_dir = tmp_path / "chroma"
    raw_dir.mkdir()
    chroma_dir.mkdir()
    (raw_dir / "alpha.pdf").write_text("pdf", encoding="utf-8")
    (raw_dir / ".gitkeep").write_text("", encoding="utf-8")
    (chroma_dir / "store.json").write_text("{}", encoding="utf-8")
    (chroma_dir / ".gitkeep").write_text("", encoding="utf-8")
    pipeline = FakePipeline()

    result = clear_knowledge_base(raw_dir, chroma_dir, pipeline)

    assert result.deleted_files == ["alpha.pdf"]
    assert result.deleted_chunks == 9
    assert not (raw_dir / "alpha.pdf").exists()
    assert (raw_dir / ".gitkeep").exists()
    assert not (chroma_dir / "store.json").exists()
    assert (chroma_dir / ".gitkeep").exists()
    assert pipeline.cleared is True


def test_clear_raw_documents_by_suffix_deletes_only_matching_type(tmp_path):
    word = tmp_path / "alpha.docx"
    pdf = tmp_path / "beta.pdf"
    other_pdf = tmp_path / "gamma.pdf"
    word.write_text("docx", encoding="utf-8")
    pdf.write_text("pdf", encoding="utf-8")
    other_pdf.write_text("pdf", encoding="utf-8")
    pipeline = FakePipeline()

    result = clear_raw_documents_by_suffix(tmp_path, pipeline, ".pdf")

    assert word.exists()
    assert not pdf.exists()
    assert not other_pdf.exists()
    assert result.deleted_files == ["beta.pdf", "gamma.pdf"]
    assert result.deleted_chunks == 6
