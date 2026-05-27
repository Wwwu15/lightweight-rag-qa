from __future__ import annotations

"""Streamlit 页面入口，负责文档入库、文档管理和问答交互。"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_knowledge_base.config import Settings
from rag_knowledge_base.documents import load_document, save_uploaded_file
from rag_knowledge_base.library import (
    DeleteResult,
    clear_knowledge_base,
    clear_raw_documents_by_suffix,
    delete_documents,
    group_raw_documents_by_type,
)
from rag_knowledge_base.rag import RagPipeline


st.set_page_config(page_title="本地 RAG 知识库", page_icon="RAG", layout="wide")


def hide_streamlit_builtin_chrome() -> None:
    """隐藏 Streamlit 自带的英文菜单、部署按钮和页脚。"""
    st.markdown(
        """
        <style>
        #MainMenu,
        footer,
        [data-testid="stMainMenu"],
        [data-testid="stMainMenuButton"],
        [data-testid="stDeployButton"],
        [data-testid="stAppDeployButton"],
        [data-testid="stBaseButton-header"] {
            display: none !important;
            visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    """读取并缓存运行配置。"""
    return Settings.from_env()


def get_pipeline(_settings: Settings) -> RagPipeline:
    """创建 RAG 流水线实例。"""
    return RagPipeline.from_settings(_settings)


def show_delete_result(result: DeleteResult) -> None:
    """根据删除结果显示成功或失败提示。"""
    if result.deleted_files:
        st.success(
            f"已删除 {len(result.deleted_files)} 个文件，"
            f"同步移除 {result.deleted_chunks} 个向量分片。"
        )
    if result.failed_files:
        failed = "、".join(result.failed_files)
        st.error(
            f"以下文件删除失败：{failed}。"
            "文件可能正在被系统、浏览器、PDF 阅读器或向量库进程占用，"
            "请关闭占用该文件的程序后重试。"
        )


def rerun_after_success(result: DeleteResult) -> None:
    """删除完全成功后刷新页面，让文档列表立即更新。"""
    if not result.failed_files:
        st.rerun()


def render_sources(sources: list[dict[str, str]]) -> None:
    """渲染回答引用来源。"""
    if not sources:
        st.caption("暂无引用来源。")
        return

    st.markdown("**引用来源**")
    for index, source in enumerate(sources, start=1):
        file_name = source.get("file_name") or source.get("source") or "未知文档"
        page = source.get("page")
        chunk = source.get("chunk_index")
        details = []
        if page not in (None, ""):
            details.append(f"第 {page} 页")
        if chunk not in (None, ""):
            details.append(f"分片 {chunk}")
        suffix = f" ({', '.join(details)})" if details else ""
        st.caption(f"[{index}] {file_name}{suffix}")


def render_document_group(
    *,
    label: str,
    suffix: str,
    documents: list[Path],
    settings: Settings,
    pipeline: RagPipeline,
    delete_label: str,
    clear_label: str,
) -> None:
    """渲染某一类文档的列表、勾选删除和清空按钮。"""
    with st.expander(f"{label}（{len(documents)}）", expanded=False):
        if not documents:
            st.caption(f"暂无 {label}。")
            return

        selected_names: list[str] = []
        for document in documents:
            left, right = st.columns([0.78, 0.22])
            left.caption(document.name)
            checked = right.checkbox(
                "选择",
                key=f"select-{suffix}-{document.name}",
                label_visibility="collapsed",
            )
            if checked:
                selected_names.append(document.name)

        delete_clicked = st.button(
            delete_label,
            disabled=not selected_names,
            key=f"delete-selected-{suffix}",
            use_container_width=True,
        )
        if delete_clicked:
            result = delete_documents(settings.raw_data_dir, pipeline, selected_names)
            show_delete_result(result)
            rerun_after_success(result)

        clear_clicked = st.button(
            clear_label,
            key=f"clear-{suffix}",
            type="secondary",
            use_container_width=True,
        )
        if clear_clicked:
            result = clear_raw_documents_by_suffix(settings.raw_data_dir, pipeline, suffix)
            show_delete_result(result)
            rerun_after_success(result)


def render_document_management(settings: Settings, pipeline: RagPipeline) -> None:
    """渲染侧边栏中的 Word/PDF 文档管理区域。"""
    st.header("文档管理")
    groups = group_raw_documents_by_type(settings.raw_data_dir)

    render_document_group(
        label="Word 文档",
        suffix=".docx",
        documents=groups["word"],
        settings=settings,
        pipeline=pipeline,
        delete_label="删除选中的 Word 文档",
        clear_label="清空 Word 文档",
    )
    render_document_group(
        label="PDF 文档",
        suffix=".pdf",
        documents=groups["pdf"],
        settings=settings,
        pipeline=pipeline,
        delete_label="删除选中的 PDF 文档",
        clear_label="清空 PDF 文档",
    )

    st.divider()
    confirm_clear = st.checkbox("确认清空知识库")
    if st.button("清空知识库", disabled=not confirm_clear, type="secondary", use_container_width=True):
        result = clear_knowledge_base(settings.raw_data_dir, settings.chroma_persist_dir, pipeline)
        st.success(
            f"知识库已清空：删除 {len(result.deleted_files)} 个文档，"
            f"移除 {result.deleted_chunks} 个向量分片。"
        )
        st.cache_resource.clear()
        st.rerun()


def main() -> None:
    """组装 Streamlit 页面并处理用户交互。"""
    hide_streamlit_builtin_chrome()

    settings = get_settings()
    pipeline = get_pipeline(settings)

    st.title("本地轻量 RAG 知识库问答")

    with st.sidebar:
        st.header("文档入库")
        st.session_state.setdefault("uploader_version", 0)
        uploaded_files = st.file_uploader(
            "上传 PDF 或 Word 文档",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key=f"document-uploader-{st.session_state.uploader_version}",
        )
        ingest_clicked = st.button("入库", type="primary", disabled=not uploaded_files)

        if ingest_clicked and uploaded_files:
            with st.spinner("正在解析、分片、向量化并写入本地存储..."):
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    saved_path = save_uploaded_file(uploaded_file, settings.raw_data_dir)
                    documents = load_document(saved_path)
                    ids = pipeline.ingest_documents(documents)
                    total_chunks += len(ids)
                st.success(f"入库完成：新增 {total_chunks} 个分片。")
                st.session_state.uploader_version += 1
                st.rerun()

        st.divider()
        render_document_management(settings, pipeline)

        st.divider()
        st.header("检索设置")
        top_k = st.number_input("Top K", min_value=1, max_value=12, value=settings.top_k, step=1)

    question = st.text_area(
        "请输入问题",
        height=110,
        placeholder="例如：请总结这份文档的核心内容。",
    )
    ask_clicked = st.button("提问", type="primary", disabled=not question.strip())

    if ask_clicked:
        with st.spinner("正在检索相关内容并生成回答..."):
            result = pipeline.answer(question.strip(), top_k=int(top_k))
        st.subheader("回答")
        st.write(result.answer)
        render_sources(result.sources)


if __name__ == "__main__":
    main()
