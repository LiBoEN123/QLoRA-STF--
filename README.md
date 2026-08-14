# QLoRA SFT 党建知识微调项目

> 7B/8B 量级（Qwen3-8B）+ QLoRA + SFT 领域大模型微调数据流水线。
> 当前阶段：**Phase 1 — 原始数据准备与统一文档抽取**。

本仓库会按阶段实现：

1. **Phase 1**：原始数据准备 + 文档抽取（统一 JSONL + manifest）。
2. Phase 2：文本清洗、去重、规范化。
3. Phase 3：SFT 数据生成（QA / messages）。
4. Phase 4：QLoRA SFT 训练。

> 当前只实现 **Phase 1**。

---

## Phase 1 — Document Extraction

### 目录约定

把原始资料按需放入 `data/raw/` 下（不强求子目录，但建议按类型组织）：

```text
data/raw/
├── pdf/
├── word/
├── markdown/
├── text/
├── html/
├── excel/
└── qa/
```

程序会**递归**扫描 `data/raw/**` 下的所有支持文件，**不依赖父目录名**判断类型——只根据文件后缀。

> `data/raw/` 是只读输入区，程序不会修改、删除、移动或重命名任何原始文件。

### 运行

```bash
# 安装依赖
python -m pip install -r requirements.txt

# 使用默认路径（data/raw → data/extracted/documents.jsonl + manifest）
python scripts/extract_documents.py

# 自定义路径
python scripts/extract_documents.py \
  --input   data/raw \
  --output  data/extracted/documents.jsonl \
  --manifest data/extracted/manifest.json \
  --log-file  logs/extract_documents.log \
  --log-level INFO
```

### 输出

| 路径 | 含义 |
| --- | --- |
| `data/extracted/documents.jsonl` | 统一格式抽取结果，每行一个 `DocumentRecord` |
| `data/extracted/manifest.json` | 抽取统计：成功/失败/跳过、warnings、每文件的 SHA256 |
| `logs/extract_documents.log` | 详细运行日志 |

### 当前支持的格式

| 后缀 | 类型 | 使用的库 |
| --- | --- | --- |
| `.pdf` | pdf | PyMuPDF (`fitz`) |
| `.docx` | docx | python-docx |
| `.md`, `.markdown` | markdown | 标准库 |
| `.txt`, `.log` | text | 标准库 |
| `.html`, `.htm` | html | BeautifulSoup4 |
| `.xlsx`, `.xlsm` | excel | openpyxl |
| `.csv` | csv | 标准库 `csv` |
| `.json` | json | 标准库 `json` |
| `.jsonl` | jsonl | 标准库 `json` |

### 当前明确不支持

- `.doc`、`.ppt`、`.pptx`
- 图片 OCR（含扫描版 PDF 的 OCR）
- 音频 / 视频
- 联网爬虫 / Playwright / Selenium
- 通过 LLM / 规则自动生成 QA
- 任何形式的 SFT messages 生成 / 模型训练

> 扫描版 PDF 会被识别并写入 `manifest.json` 的 `warnings: ["possible_scanned_pdf"]`，**不会调用 OCR**。后续会单独做 OCR Pipeline。

### 设计原则

- **统一数据模型**：`DocumentRecord` 是 Phase 1 的唯一记录结构。
- **稳定 doc_id**：使用 `SHA256(relative_path + page/sheet/index + text)` 截取前 24 位，保证重复运行 doc_id 一致。
- **相对路径**：所有 `source_path` 都使用项目内相对路径（POSIX 风格），便于跨机器迁移。
- **文件 SHA256**：每个源文件单独计算 SHA256，存到 `DocumentRecord.file_sha256` 和 `manifest`。
- **失败隔离**：单个文件失败不会中断整体任务，会在 `manifest.json` 记录 `failed`。
- **空文本不输出**：纯空白记录不会写入 `documents.jsonl`，但会在 manifest 的 `warnings` 中体现。
- **原子写入**：通过临时文件 + `os.replace` 保证 `documents.jsonl` / `manifest.json` 不会写到一半。
- **不修改原始数据**：原始文件保持只读。

### 运行测试

```bash
pip install pytest
pytest
```

### Phase 1 文档结构

```text
src/data_pipeline/
├── __init__.py
├── models.py        # DocumentRecord, compute_doc_id
├── utils.py         # SHA256, 路径处理, 文本规范化
├── registry.py      # EXTRACTORS 表
├── extractor.py     # Pipeline 编排 (run_pipeline)
└── extractors/
    ├── base.py
    ├── pdf.py
    ├── word.py
    ├── markdown.py
    ├── text.py
    ├── html.py
    ├── excel.py      # xlsx/xlsm + csv
    └── jsonl.py      # json + jsonl
```

---

## 后续阶段（暂不实现）

- Phase 2：文本清洗、MinHash/SimHash 去重、规范化。
- Phase 3：生成 SFT messages、Train/Val/Test 划分。
- Phase 4：Qwen3-8B + QLoRA SFT 训练。
