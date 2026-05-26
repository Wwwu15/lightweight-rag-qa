from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_RAW_SUFFIXES = {".pdf", ".docx"}
KEEP_FILES = {".gitkeep"}


@dataclass(slots=True)
class DeleteResult:
    deleted_files: list[str]
    deleted_chunks: int
    failed_files: list[str] | None = None


def list_raw_documents(raw_dir: str | Path) -> list[Path]:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []
    return sorted(
        [
            path
            for path in raw_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_RAW_SUFFIXES
        ],
        key=lambda path: path.name.lower(),
    )


def group_raw_documents_by_type(raw_dir: str | Path) -> dict[str, list[Path]]:
    documents = list_raw_documents(raw_dir)
    return {
        "word": [path for path in documents if path.suffix.lower() == ".docx"],
        "pdf": [path for path in documents if path.suffix.lower() == ".pdf"],
    }


def delete_documents(
    raw_dir: str | Path,
    pipeline: Any,
    file_names: Iterable[str],
) -> DeleteResult:
    raw_path = Path(raw_dir)
    deleted_files: list[str] = []
    failed_files: list[str] = []
    deleted_chunks = 0

    for file_name in file_names:
        document_path = _safe_child(raw_path, file_name)
        if (
            document_path is None
            or not document_path.exists()
            or not document_path.is_file()
            or not _is_supported_raw_document(document_path)
        ):
            continue

        try:
            document_path.unlink()
        except OSError:
            failed_files.append(document_path.name)
            continue

        # Delete vectors only after the raw file is gone, keeping file and index state consistent.
        deleted_files.append(document_path.name)
        deleted_chunks += int(
            pipeline.delete_documents(source=str(document_path), file_name=document_path.name)
        )

    return DeleteResult(
        deleted_files=deleted_files,
        deleted_chunks=deleted_chunks,
        failed_files=failed_files,
    )


def clear_knowledge_base(raw_dir: str | Path, chroma_dir: str | Path, pipeline: Any) -> DeleteResult:
    raw_path = Path(raw_dir)
    chroma_path = Path(chroma_dir)
    deleted_files = [path.name for path in list_raw_documents(raw_path)]
    deleted_chunks = int(pipeline.clear())

    _clear_directory(raw_path)
    _clear_directory(chroma_path)
    return DeleteResult(deleted_files=deleted_files, deleted_chunks=deleted_chunks)


def clear_raw_documents_by_suffix(
    raw_dir: str | Path,
    pipeline: Any,
    suffix: str,
) -> DeleteResult:
    normalized_suffix = suffix.lower()
    if normalized_suffix not in SUPPORTED_RAW_SUFFIXES:
        return DeleteResult(deleted_files=[], deleted_chunks=0)

    names = [
        path.name
        for path in list_raw_documents(raw_dir)
        if path.suffix.lower() == normalized_suffix
    ]
    return delete_documents(raw_dir, pipeline, names)


def _safe_child(parent: Path, file_name: str) -> Path | None:
    try:
        # Resolve against the raw directory and discard path traversal attempts.
        parent_resolved = parent.resolve()
        child = (parent_resolved / Path(file_name).name).resolve()
        child.relative_to(parent_resolved)
        return child
    except ValueError:
        return None


def _is_supported_raw_document(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_RAW_SUFFIXES


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name in KEEP_FILES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
