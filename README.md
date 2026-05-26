# 本地轻量 RAG 知识库问答系统

这是一个基于本地文件的轻量 RAG 知识库问答项目。系统支持上传 PDF 和 Word 文档，完成文档解析、文本分片、向量化存储、相似度检索，并通过 OpenAI 兼容的外部 LLM API 基于私有文档生成回答和引用来源。

运行时会优先使用 LangChain Chroma 作为向量库；如果当前环境没有安装 `langchain-chroma` / `chromadb`，会自动退回到 `data/chroma/` 下的 JSON 向量存储，方便先跑通 MVP。

## 技术栈

- 语言：Python
- Web UI：Streamlit
- RAG 编排：LangChain Document / TextSplitter 相关接口
- 文档解析：
  - PDF：`pypdf`，优先兼容 LangChain `PyPDFLoader`
  - Word：`python-docx`
- 文本分片：`RecursiveCharacterTextSplitter`
- Embedding：Ollama `nomic-embed-text:v1.5`
- 向量存储：
  - 优先：Chroma 本地持久化
  - 兜底：项目内置 JSON 向量存储
- LLM：OpenAI 兼容接口，默认配置 `https://ai-pixel.online/v1` / `gpt-5.4`
- 配置管理：`.env` + `python-dotenv`
- 测试：pytest

## 架构

系统按职责分为四层：

- UI 层：`app.py`
  - Streamlit 页面
  - 文档上传
  - Word / PDF 分组文档管理
  - 问答输入、答案展示、引用来源展示
- 应用层：`library.py`
  - raw 文档列表
  - 单个 / 批量删除
  - 按 Word / PDF 类型清空
  - 清空整个知识库
- RAG 核心层：`documents.py`、`rag.py`、`prompts.py`、`llm.py`
  - 文档解析
  - 文本分片
  - embedding
  - 向量库写入、检索、删除、清空
  - RAG prompt 构造
  - 外部 LLM 调用
- 存储层：`data/`
  - `data/raw/`：保存上传的原始 PDF / Word 文件
  - `data/chroma/`：保存 Chroma 或 JSON fallback 向量数据

## 项目结构

```text
.
├── app.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── chroma/
│       └── .gitkeep
├── src/
│   └── rag_knowledge_base/
│       ├── __init__.py
│       ├── config.py
│       ├── documents.py
│       ├── library.py
│       ├── llm.py
│       ├── prompts.py
│       └── rag.py
└── tests/
    ├── test_app_chinese_ui.py
    ├── test_config.py
    ├── test_documents.py
    ├── test_library.py
    └── test_rag_core.py
```

## 核心流程

### 文档入库

1. 用户在 Streamlit 页面上传 PDF / Word 文档。
2. 系统将原始文件保存到 `data/raw/`。
3. `documents.py` 按文件类型解析文档：
   - `.pdf` 解析为按页的 LangChain `Document`
   - `.docx` 解析为一个或多个 LangChain `Document`
4. `rag.py` 使用 `RecursiveCharacterTextSplitter` 对文档内容分片。
5. 系统调用 Ollama embedding 模型 `nomic-embed-text:v1.5` 生成向量。
6. 分片文本、metadata、向量写入本地向量库。
7. metadata 中保留 `source`、`file_name`、`chunk_index`、`chunk_id`，用于后续引用和删除。

### 问答检索

1. 用户输入问题。
2. 系统对问题生成 embedding。
3. 向量库按相似度检索 Top K 个相关分片。
4. `prompts.py` 将问题、上下文分片和来源编号拼接为 RAG prompt。
5. `llm.py` 调用外部 OpenAI 兼容 LLM API。
6. 页面展示答案，并显示引用来源、页码和分片信息。

### 文档删除

1. 页面在“文档管理”中按类型展示：
   - Word 文档
   - PDF 文档
2. 用户展开对应类型后勾选文件。
3. 点击“删除选中的 Word 文档”或“删除选中的 PDF 文档”。
4. 系统删除 `data/raw/` 中对应原始文件。
5. 系统按 `source` / `file_name` 从向量库删除对应 chunks。

### 按类型清空

1. 用户点击“清空 Word 文档”或“清空 PDF 文档”。
2. 系统删除该类型所有 raw 文件。
3. 系统同步删除这些文件对应的向量分片。
4. 其他类型文档不受影响。

### 清空知识库

1. 用户勾选“确认清空知识库”。
2. 用户点击“清空知识库”。
3. 系统清空 `data/raw/` 中所有文档。
4. 系统清空 `data/chroma/` 中的向量数据。
5. `.gitkeep` 会保留，用于维持目录结构。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

默认访问地址：

```text
http://localhost:8501
```

## 配置

复制 `.env.example` 为 `.env`，然后按需修改：

```env
RAW_DATA_DIR=data/raw
CHROMA_PERSIST_DIR=data/chroma
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text:v1.5
LLM_BASE_URL=https://ai-pixel.online/v1
LLM_MODEL=gpt-5.4
LLM_API_KEY=replace-me
CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K=4
```

## 测试

```powershell
pytest -q
```
