from pathlib import Path


def test_app_source_uses_utf8_chinese_ui_copy() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    expected_copy = [
        "本地轻量 RAG 知识库问答",
        "文档入库",
        "上传 PDF 或 Word 文档",
        "入库",
        "检索设置",
        "请输入问题",
        "提问",
        "回答",
        "引用来源",
        "暂无引用来源。",
    ]

    for text in expected_copy:
        assert text in source

    mojibake_markers = ["鏆", "鐭", "绛", "涓", "渚", "????"]
    for marker in mojibake_markers:
        assert marker not in source


def test_app_hides_streamlit_builtin_english_chrome() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert "#MainMenu" in source
    assert "footer" in source
    assert "stDeployButton" in source
    assert "stAppDeployButton" in source
    assert "visibility: hidden" in source
    assert "display: none" in source


def test_pipeline_is_not_cached_so_delete_methods_stay_fresh() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    pipeline_definition = source[source.index("def get_pipeline") : source.index("def render_sources")]

    assert "@st.cache_resource" not in pipeline_definition


def test_document_group_uses_checkbox_selection_and_delete_button() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert "st.checkbox" in source
    assert "selected_names.append(document.name)" in source
    assert "delete_label" in source


def test_document_management_ui_is_grouped_by_word_and_pdf() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    expected_copy = [
        "Word 文档",
        "PDF 文档",
        "清空 Word 文档",
        "清空 PDF 文档",
        "删除选中的 Word 文档",
        "删除选中的 PDF 文档",
    ]

    for text in expected_copy:
        assert text in source


def test_delete_failure_shows_friendly_locked_file_message() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert "文件可能正在被系统、浏览器、PDF 阅读器或向量库进程占用" in source
    assert "请关闭占用该文件的程序后重试" in source


def test_upload_widget_is_reset_after_successful_ingest() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert 'st.session_state.setdefault("uploader_version", 0)' in source
    assert 'key=f"document-uploader-{st.session_state.uploader_version}"' in source
    assert "st.session_state.uploader_version += 1" in source
