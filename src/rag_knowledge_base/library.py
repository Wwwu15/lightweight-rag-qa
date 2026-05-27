from __future__ import annotations

"""本地知识库文档管理模块。"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_RAW_SUFFIXES = {".pdf", ".docx"}
KEEP_FILES = {".gitkeep"}


@dataclass(slots=True)
class DeleteResult:
    """文档删除操作结果。"""

    deleted_files: list[str]
    deleted_chunks: int
    failed_files: list[str] | None = None


def list_raw_documents(raw_dir: str | Path) -> list[Path]:
    """列出 raw 目录中支持管理的文档。"""
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
    """按 Word 和 PDF 类型分组文档。"""
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
    """删除指定 raw 文件，并同步删除对应向量分片。"""
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
    """清空 raw 文档和向量库数据。"""
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
    """按文件类型清空 raw 文档和对应向量分片。"""
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
    """只允许删除 raw 目录下的直接子文件。"""
    try:
        parent_resolved = parent.resolve()
        child = (parent_resolved / Path(file_name).name).resolve()
        child.relative_to(parent_resolved)
        return child
    except ValueError:
        return None


def _is_supported_raw_document(path: Path) -> bool:
    """判断文件是否属于可管理文档类型。"""
    return path.suffix.lower() in SUPPORTED_RAW_SUFFIXES


def _clear_directory(path: Path) -> None:
    """清空目录内容，但保留 .gitkeep。"""
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name in KEEP_FILES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
