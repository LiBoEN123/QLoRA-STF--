# QLoRA SFT 党建知识微调项目

> 7B/8B 量级（Qwen3-8B）+ QLoRA + SFT 领域大模型微调数据流水线。
> 当前阶段：**Phase 2 — 文本清洗、结构恢复、质量过滤与去重**。

本仓库会按阶段实现：

1. **Phase 1**：原始数据准备 + 文档抽取（统一 JSONL + manifest）。
2. **Phase 2**：文本清洗、结构恢复、质量过滤、SimHash 去重。
3. Phase 3：SFT 数据生成（QA / messages）。
4. Phase 4：QLoRA SFT 训练。

> Phase 1 + Phase 2 已实现。Phase 2 仍然**不会**生成 SFT 训练样本，不调用任何 LLM。

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

- Phase 3：生成 SFT messages、Train/Val/Test 划分。
- Phase 4：Qwen3-8B + QLoRA SFT 训练。

---

## Phase 2 — Document Cleaning

Phase 2 在 Phase 1 的统一 JSONL 上做确定性文本清洗，**不会**调用任何 LLM / Embedding / 向量数据库，也不会构造 SFT 训练样本。

### 数据流

```text
data/raw/**
        ↓
Phase 1 Extract
        ↓
data/extracted/documents.jsonl
        ↓
Phase 2 Clean
        ↓
文本规范化 (Unicode / 空白 / 中文断行 / 标点)
        ↓
结构识别 (chapter / section / heading / subheading / list_item / paragraph / table / unknown)
        ↓
质量检测 (长度、中文比例、symbol 比例、garbled 判定)
        ↓
规则过滤 (rejected_documents.jsonl)
        ↓
重复检测 (Exact SHA256 + 64-bit SimHash)
        ↓
data/processed/documents_clean.jsonl
data/processed/rejected_documents.jsonl
data/processed/duplicates.jsonl
data/processed/quality_report.json
```

### 运行

```bash
# 默认（读 data/extracted → 写 data/processed）
python scripts/clean_documents.py

# 自定义路径
python scripts/clean_documents.py \
  --input   data/extracted/documents.jsonl \
  --output  data/processed/documents_clean.jsonl \
  --rejected data/processed/rejected_documents.jsonl \
  --duplicates data/processed/duplicates.jsonl \
  --report  data/processed/quality_report.json \
  --log-file logs/clean_documents.log \
  --log-level INFO

# 可选调参
--min-length 5
--max-length 20000
--max-symbol-ratio 0.5
--simhash-distance 3
--disable-near-dedup
```

### 输出

| 路径 | 含义 |
| --- | --- |
| `data/processed/documents_clean.jsonl` | 通过质量 + 去重的 records，附 `clean_text`、`structure_type`、`detected_section`、`quality`、`cleaning_actions` 等字段 |
| `data/processed/rejected_documents.jsonl` | 被质量规则过滤的 records（含 `reject_reasons`） |
| `data/processed/duplicates.jsonl` | exact + near duplicate，含 `canonical_doc_id` / `duplicate_doc_id` / `distance` |
| `data/processed/quality_report.json` | 聚合统计（输入/输出/保留率、文本长度分布、结构计数、清洗动作、拒因、源文件分布等） |
| `logs/clean_documents.log` | 详细运行日志 |

### 关键规则

- **确定性**：相同输入 + 相同配置 → 完全一致输出（不含时间戳）。
- **可追溯**：Phase 1 原始 `text` 字段不会被覆盖；`clean_text` 单独保存。
- **结构识别**：`第一章`/`第一节`/`一、`/`（一）`/`1.`/`（1）`/`①` 等正则在 *原始* 文本上运行，避免 NFKC 把全角括号 / 圈数字吃掉。
- **SimHash**：64-bit，自实现 FNV-1a + CJK 字符 bigram，无外部依赖；按 top-16-bit 分桶避免 O(N²)。
- **去重原则**：相同输入 → canonical record 稳定（按 `doc_id` 升序选首条）。
- **保守清洗**：宁可少清洗也不误清洗；中文标点 `， 。 ！ ？ ； ： （ ）` 在 NFKC 之后会被还原为全角。
- **Phase 2 仍不创建 SFT 训练样本**，不做 chunking，不调用任何 LLM。

### Phase 2 模块结构

```text
src/data_pipeline/
├── models.py            # Phase 1 DocumentRecord
├── utils.py             # Phase 1 utilities
├── extractor.py         # Phase 1 pipeline
├── registry.py          # Phase 1 registry
├── extractors/          # Phase 1 extractors
└── cleaning/
    ├── normalizer.py    # Unicode / 空白 / 标点 / 中文断行
    ├── structure.py     # 结构识别 + section 上下文传播
    ├── quality.py       # 质量指标 + header/footer 检测
    ├── dedup.py         # Exact SHA256 + SimHash near-dedup
    └── pipeline.py      # orchestrator (run_cleaning_pipeline)
```
