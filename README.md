# QLoRA SFT 党建知识微调项目

> 7B/8B 量级（Qwen3-8B）+ QLoRA + SFT 领域大模型微调数据流水线。
> 当前阶段：**Phase 5B — QLoRA Runtime Preflight + One-Step Smoke Training**。

本仓库按阶段实现：

1. **Phase 1 — Extraction**：原始数据准备 + 文档抽取（统一 JSONL + manifest）。
2. **Phase 2 — Cleaning**：文本清洗、结构恢复、质量过滤、SimHash 去重。
3. **Phase 3 — Semantic Unit + FAQ + Chunking**：Semantic Unit 恢复 + FAQ 配对 + 结构感知 Chunking。
4. **Phase 4 — SFT Dataset Construction**：把 FAQ SemanticUnit 转成标准 chat-style SFT messages，做数据质量审计与 Train/Val/Test 确定性划分。
5. **Phase 5A — Qwen3 Tokenizer Validation**：通过 `AutoTokenizer.apply_chat_template` 对 SFT 数据做 token 长度、prompt prefix match、coverage、thinking marker 等审计，不加载模型权重、不初始化 GPU、不训练。
6. **Phase 5B — QLoRA Runtime Preflight + One-Step Smoke Training**：在加载真实 8B 模型前复算 Phase 5A gate；在 GPU 上以 4-bit NF4 + LoRA r=16 + `assistant_only_loss=True` 跑 1 个 optimizer step，验证 forward / backward / save / reload / 推理最小链。
7. **Phase 5C — Full QLoRA SFT Training**（后续）：真正的多 epoch / packing / checkpoint / evaluation。Phase 5B 不实现。

> Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5A + Phase 5B 已实现。Phase 4 仍然**不会**调用任何 LLM、不会加载 tokenizer、不会训练模型——只把 Phase 3 已经确认的可靠 FAQ 稳定、无损、可追溯地转换成标准 Chat-style SFT 数据集。Phase 5A 仅加载 `AutoTokenizer`（CPU），不下载 8B 模型权重，不引入 QLoRA / Trainer / LoRAConfig。Phase 5B 在 GPU 上以 4-bit + LoRA 跑 1 个 optimizer step，**不**做完整 epoch 训练，不做超参搜索，不保存 merged 模型。
> Phase 3 在历史迭代中曾为了恢复少量边界 FAQ 临时引入"Phase 3.5"的额外恢复流程，当前已合并回收——仅保留唯一的"一次检测 + 一次配对"主链。

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

- Phase 5：Tokenizer Validation + QLoRA Training Preparation / Training。

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

---

## Phase 3 — Semantic Unit + FAQ Pairing + Structure-aware Chunking

Phase 3 在 Phase 2 输出上恢复源文档 record 顺序，识别 FAQ 问答关系并构建语义单元，然后做结构感知 chunking。**Phase 3 只恢复源数据中已有的 FAQ 关系并构建结构化 chunk，不会调用 LLM、不会生成新的 QA 内容、不会生成 SFT messages、不会进行模型训练。**

### 数据流

```text
data/raw/**
        ↓
Phase 1 Extract
        ↓
data/extracted/documents.jsonl
        ↓
Phase 2 Clean / Normalize / Filter / Dedup
        ↓
data/processed/documents_clean.jsonl
        ↓
Phase 3 Restore source order
        ↓
detect_question_candidate         ← 一个 detector：编号 / 段落 / 粘连题号
        ↓
pair_questions_with_answers       ← 一个 pairer：高 / 中 一次配对
        ↓
build_semantic_units              ← 一个 semantic unit 构造器
        ↓
build_chunks
        ↓
data/chunks/semantic_units.jsonl
data/chunks/chunks.jsonl
data/chunks/chunk_report.json
data/chunks/faq_audit.jsonl       ← 仅观察：未配对的候选
```

### FAQ 主链（唯一）

```text
run_chunking_pipeline
   ↓
restore_source_order                          # ordering.py
   ↓
pair_questions_with_answers                   # faq.py
   │   └── detect_question_candidate          # faq.py
   ↓
build_semantic_units                          # semantic_units.py
   ├── _build_faq_units
   └── _build_other_units
   ↓
_build_chunks                                 # pipeline.py
   ↓
build_report / build_audit_entries            # report.py / audit.py
```

历史上曾存在的"Phase 3.5 段落 FAQ 二次恢复"已合并回收。当前 detector 一次性覆盖：

* **结构化编号**（`1.` / `1、` / `①` / `一、`）→ high（带问号）/ medium（带疑问词 / 中文编号+问号）
* **粘连题号**（开头 1~3 位数字紧接中文，不接单位字）→ medium
* **段落短问句**（长度合理 + `?` + 强疑问词）→ medium

不再维护两条 pipeline 不再二次拆分。

### 运行

```bash
# 默认（读 data/processed → 写 data/chunks）
python scripts/build_chunks.py

# 自定义路径与阈值
python scripts/build_chunks.py \
  --input         data/processed/documents_clean.jsonl \
  --units-output  data/chunks/semantic_units.jsonl \
  --chunks-output data/chunks/chunks.jsonl \
  --report        data/chunks/chunk_report.json \
  --log-file      logs/build_chunks.log \
  --min-chars     100 \
  --target-chars  800 \
  --max-chars     1200 \
  --overlap-chars 100 \
  --faq-min-confidence medium
```

### 输出

| 路径 | 含义 |
| --- | --- |
| `data/chunks/semantic_units.jsonl` | 语义单元（FAQ / heading / paragraph_group / list_group / standalone），含 `unit_id`、`unit_type`、`question_text`、`answer_text`、`source_doc_ids`、`source_record_indices` |
| `data/chunks/chunks.jsonl` | 最终 chunk（faq / heading / text），含 `chunk_id`、`source_unit_ids`、`source_doc_ids`、`source_record_indices` |
| `data/chunks/chunk_report.json` | 聚合统计：FAQ 配对数、置信度分布、长度分布、split / overlap 统计、可追溯性 |
| `data/chunks/faq_audit.jsonl` | 审计旁路：未配对的候选（含已识别但无答案 / 低置信度），仅供人工核对 |
| `logs/build_chunks.log` | 详细运行日志 |

### 关键规则

- **一个 Detection Path + 一个 Pairing Path**。所有候选（结构化 / 粘连 / 段落）共用 `detect_question_candidate`，共用 `pair_questions_with_answers`。
- **精确量化后缀保护**：`100万元党费如何管理？` / `2026年党员教育工作重点是什么？` / `18岁以上人员能否申请入党？` 这类"数字+单位"的常见前缀**不会**被当作粘连题号剥离，规则以紧跟数字后的字符是否为常见单位字（万 / 元 / 岁 / 年 / % / 月 / 日 / 个 / 多 / 余）来区分。
- **Precision > Recall**：自动配对只接受 high / medium 置信度；`（1）` / `（2）` 全角括号枚举不会误识别为问题。
- **段落 FAQ 终止上一题**：检测器对段落短问句直接升级为 medium，因此上一道 FAQ 的答案范围会在此处自然终止；不再需要后处理拆分。
- **FAQ 一对一**：一道 FAQ = 一个 `SemanticUnit` = 一个 `chunk`，不做切分；短 FAQ 也不与下一道合并。
- **结构边界硬性**：chapter / section / heading / subheading 是硬边界，跨结构正文绝不合并。
- **可追溯**：`unit_id` / `chunk_id` 由 SHA256(输入内容) 决定性生成；每个 chunk 都保留 `source_unit_ids`、`source_doc_ids`、`source_record_indices`，可一路回溯到原始 Phase 1 `DocumentRecord` 与源文件。
- **确定性**：相同输入 + 相同配置 → SHA256 一致的输出文件，report 不含任何时间戳 / 当前时间字段。
- **不调用 LLM / 不生成 SFT messages / 不训练模型**：Phase 3 严格止步于「数据结构恢复 + 结构化 chunking」。
- **Audit 不参与业务控制流**：`pair_questions_with_answers` 直接返回 `unpaired` 候选，audit 只是把这份数据序列化为 JSONL，**不会**反过来驱动 SemanticUnit 重构。
- **公共 IO**：所有原子写文件逻辑（`atomic_write_json` / `atomic_write_jsonl` / `load_jsonl_records`）只维护于 `src/data_pipeline/utils.py`，Phase 2 / Phase 3 全部复用。
- **中文注释**：从 Phase 3 开始，所有新增/修改代码的注释、docstring、CLI 参数说明、日志内容均优先使用中文；类名 / 函数名 / 字段名 / JSON key / CLI 参数继续使用规范英文命名。

### Phase 3 模块结构

```text
src/data_pipeline/
├── models.py            # DocumentRecord (Phase 1)
├── utils.py             # 共享：路径、SHA、规范化、原子 IO
├── extractor.py         # Phase 1 编排
├── registry.py
├── extractors/
├── cleaning/            # Phase 2
│   ├── normalizer.py
│   ├── structure.py
│   ├── quality.py
│   ├── dedup.py
│   └── pipeline.py      # 复用 utils 的原子 IO
└── chunking/            # Phase 3
    ├── __init__.py
    ├── models.py        # ChunkingConfig / SemanticUnit / Chunk / ID 生成
    ├── ordering.py      # 源序恢复
    ├── faq.py           # 唯一 detector + 唯一 pairer
    ├── semantic_units.py# SemanticUnit 构建
    ├── splitter.py      # 超长文本安全切分
    ├── audit.py         # 旁路：序列化未配对候选
    ├── report.py        # 报告
    └── pipeline.py      # orchestrator
```

---

## Phase 4 — SFT Dataset Construction

Phase 4 直接消费 Phase 3 产出的 `semantic_units.jsonl`，只取 `unit_type == "faq"` 的单元，**不重新 detect / pair FAQ**，**不调用任何 LLM**，**不加载 tokenizer**，**不训练模型**——只把已配对的 FAQ 转成标准 Chat-style SFT 数据并做数据质量审计与 Train/Val/Test 确定性划分。

### 数据流

```text
data/raw/**
        ↓
Phase 1 Extract
        ↓
data/extracted/documents.jsonl
        ↓
Phase 2 Clean / Normalize / Filter / Dedup
        ↓
data/processed/documents_clean.jsonl
        ↓
Phase 3 Semantic Unit + FAQ + Chunking
        ↓
data/chunks/semantic_units.jsonl
        ↓
Phase 4 SFT Dataset Construction
        ├── 只取 unit_type == "faq"
        ├── 字段校验（unit_id / question / answer / confidence / traceability）
        ├── sample_id = SHA256(unit_id + q + a)[:24]
        ├── exact dedup（normalized q+a）
        ├── question conflict（同问不同答 ⇒ 整组剔除）
        ├── 确定性 split（split_group_key = SHA256(normalized_q), salt 固定）
        ├── cross-split leakage 硬校验
        ↓
data/sft/all.jsonl
data/sft/train.jsonl
data/sft/validation.jsonl
data/sft/test.jsonl
data/sft/rejected.jsonl
data/sft/dataset_report.json
```

### 运行

```bash
# 默认（读 data/chunks/semantic_units.jsonl → 写 data/sft/）
python scripts/build_sft_dataset.py

# 自定义路径与参数
python scripts/build_sft_dataset.py \
  --input             data/chunks/semantic_units.jsonl \
  --all-output        data/sft/all.jsonl \
  --train-output      data/sft/train.jsonl \
  --validation-output data/sft/validation.jsonl \
  --test-output       data/sft/test.jsonl \
  --rejected-output   data/sft/rejected.jsonl \
  --report            data/sft/dataset_report.json \
  --log-file          logs/build_sft_dataset.log \
  --min-confidence    medium \
  --train-ratio       0.80 \
  --validation-ratio  0.10 \
  --test-ratio        0.10 \
  --split-salt        qlora-sft-phase4-v1
```

### 输出

| 路径 | 含义 |
| --- | --- |
| `data/sft/all.jsonl` | 通过质量校验的 accepted samples 全量（user + assistant） |
| `data/sft/train.jsonl` | 训练集 |
| `data/sft/validation.jsonl` | 验证集 |
| `data/sft/test.jsonl` | 测试集 |
| `data/sft/rejected.jsonl` | 被剔除的样本（含 `reject_reasons` / `canonical_sample_id`） |
| `data/sft/dataset_report.json` | 数据质量报告（含 split 比例、长度分布、leakage、traceability 等） |
| `logs/build_sft_dataset.log` | 详细运行日志 |

### SFT Sample Schema

```json
{
  "sample_id": "sft_xxxxxxxxxxxxxxxxxxxxxxxx",
  "messages": [
    {"role": "user", "content": "党员应当履行哪些义务？"},
    {"role": "assistant", "content": "党员应当……"}
  ],
  "metadata": {
    "unit_id": "unit_...",
    "source_path": "...",
    "file_name": "...",
    "file_sha256": "...",
    "source_doc_ids": ["doc_..."],
    "source_record_indices": [2, 3],
    "faq_confidence": "high",
    "structure_path": ["一、基层党组织的设置"],
    "detected_section": "一、基层党组织的设置",
    "question_record_index": 2,
    "answer_record_count": 1,
    "number_style": "arabic_dot"
  }
}
```

### 关键规则

- **只消费 `unit_type == "faq"`**：heading / subheading / paragraph_group / list_group / standalone 一律不进入 SFT。
- **内容零修改**：`user.content == question_text.strip()`，`assistant.content == answer_text.strip()`；不增加"问题："/"答案："前缀；不增加 system / tool / developer 消息。
- **sample_id 稳定**：`SHA256(unit_id + question + answer)` 截前 24 位；同一输入重复运行必须一致；禁止 UUID / random / timestamp。
- **Exact dedup**：`normalized(q) + normalized(a)` 相同的样本只保留 1 条 canonical（按 sample_id 升序选首条）。
- **Question conflict**：同问不同答 ⇒ 该组所有样本**整组剔除**，不进入训练集，不静默通过。
- **确定性 split**：按 `split_group_key = SHA256(normalized_question)` 避免同问跨 split；`split_salt` 默认 `qlora-sft-phase4-v1`，必须显式持久化在 report 中。
- **Leakage 硬校验**：`cross_split_question = 0` 且 `cross_split_exact_sample = 0`；否则 pipeline 失败，不只 warning。
- **不调用 LLM / 不加载 tokenizer / 不训练模型**：Phase 4 严格止步于「FAQ → messages → 质量审计 → 划分」。
- **公共 IO**：所有原子写文件逻辑（`atomic_write_json` / `atomic_write_jsonl` / `load_jsonl_records`）只维护于 `src/data_pipeline/utils.py`，Phase 1 / 2 / 3 / 4 全部复用。
- **中文注释**：所有新增/修改代码的注释、docstring、CLI 参数说明、日志内容均优先使用中文；类名 / 函数名 / 字段名 / JSON key / CLI 参数继续使用规范英文命名。
- **类字段中文注释强制**：所有新增 dataclass / class 的每一个字段前必须存在独立中文说明注释（每个字段一行）。后续 `test_sft_dataclasses_have_chinese_field_comments` 会扫描整个 `src/data_pipeline/sft/` 下所有 dataclass，新增字段无需手改测试。

### Phase 4 模块结构

```text
src/data_pipeline/
└── sft/
    ├── __init__.py
    ├── models.py        # SFTBuildConfig / SFTSample
    └── pipeline.py      # run_sft_pipeline（唯一入口）

scripts/
└── build_sft_dataset.py  # CLI

tests/
└── test_sft_dataset.py   # 单元测试（FAQ→messages、字段校验、dedup、conflict、split、leakage、accounting、原子写、确定性、类字段中文注释约定）
```

### 后续阶段（暂不实现）

- Phase 5C：完整 epoch 训练、packing、checkpoint 调度、最终评测（Phase 5B 不实现）。

---

## Phase 5A — Qwen3 Tokenizer Validation

Phase 5A 直接消费 Phase 4 的 `data/sft/`，**只**调用 `AutoTokenizer.apply_chat_template` 做训练前 token 长度审计；不加载 `AutoModelForCausalLM`、不初始化 CUDA、不训练模型、不生成任何训练样本的 `input_ids` / `attention_mask` / `labels`。

### 数据流

```text
data/sft/
    ├── all.jsonl
    ├── train.jsonl
    ├── validation.jsonl
    └── test.jsonl
        ↓
Phase 5A Tokenizer Audit
    ├── 校验 split membership（all ⊆ train ∪ validation ∪ test，互斥）
    ├── 加载 Qwen3 tokenizer（AutoTokenizer, use_fast=True, trust_remote_code=False）
    ├── synthetic messages 渲染 + roundtrip 验证
    ├── 扫描 source messages 中是否有 <think> / </think> / reasoning_content 痕迹
    ├── 逐 sample:
    │     apply_chat_template(messages, add_generation_prompt=False, enable_thinking=False)
    │     apply_chat_template(messages[:1], add_generation_prompt=True, enable_thinking=False)
    │     校验 prompt prefix match + assistant_target_tokens > 0
    ├── 长度统计（min/mean/median/p90/p95/p99/max，full / prompt / assistant_target）
    ├── split 统计（train / validation / test 各自 count + mean/p95/max）
    ├── 候选 max_seq_length coverage（512 / 1024 / 1536 / 2048 / 3072 / 4096）
    ├── 95% / 99% / 100% coverage threshold
    └── longest 20 samples（仅定位 metadata，不复制答案）
        ↓
data/tokenizer/token_audit.jsonl
data/tokenizer/token_report.json
logs/audit_token_lengths.log
```

### 运行

```bash
# 安装依赖
pip install -r requirements.txt

# 默认（读 data/sft → 写 data/tokenizer/，加载 Qwen/Qwen3-8B tokenizer）
python scripts/audit_token_lengths.py

# 自定义路径 / 模型仓库
python scripts/audit_token_lengths.py \
    --sft-dir     data/sft \
    --model-name  Qwen/Qwen3-8B \
    --revision    main \
    --output-audit   data/tokenizer/token_audit.jsonl \
    --output-report  data/tokenizer/token_report.json \
    --log-file       logs/audit_token_lengths.log \
    --log-level      INFO
```

### 输出

| 路径 | 含义 |
| --- | --- |
| `data/tokenizer/token_audit.jsonl` | 每条 accepted SFT 样本一条轻量审计记录（不含 `input_ids` / 原文） |
| `data/tokenizer/token_report.json` | 汇总：模型/tokenizer 元数据、长度统计、split 统计、coverage、threshold、longest 20、source thinking 扫描 |
| `logs/audit_token_lengths.log` | 详细运行日志 |

### 关键规则

- **只加载 `AutoTokenizer`**（+ 可选 `AutoConfig`），**不**加载 `AutoModelForCausalLM`，**不**调用 `torch.cuda`。
- **唯一真值**：`tokenizer.apply_chat_template(...)`；**禁止**手工拼 ChatML、维护第二份 chat template、手工 `replace("<think>", "")`。
- **Non-thinking SFT**：固定 `enable_thinking=False`；不暴露 `enable_thinking=True` 作为训练路径。
- **Full conversation**：`add_generation_prompt=False`；**Prompt only**：`add_generation_prompt=True`。
- **不裁、不填**：`truncation=False`、`padding=False`，只统计 raw `len(input_ids)`。
- **Prefix property**：`full[:len(prompt)] == prompt` 必须对所有 accepted sample 成立，否则 `assistant_only_loss_feasible=False`。
- **不修改 SFT 数据**：不重写 `messages`、不动 Phase 1/2/3/4 任何上游文件。
- **不构造 `labels` / `attention_mask`**：仅判定未来能否安全做 assistant-only loss；具体 collator 留给 Phase 5B。
- **确定性**：相同 SFT 数据 + 相同 tokenizer revision + 相同 config ⇒ `token_audit.jsonl` 与 `token_report.json` byte-for-byte 一致；report 不含任何 `timestamp` / `runtime` / `hostname`。
- **不决定 max_seq_length**：只报告 95/99/100% 覆盖候选；最终训练长度由 Phase 5B 综合显存 / batch / 业务决定。
- **不启用 YaRN / 不修改 rope_scaling**：本阶段只读模型 config。
- **类字段中文注释强制**：本阶段新增的 `dataclass` 字段前必须有独立中文说明（与 Phase 4 同约定）。
- **公共 IO**：所有原子写文件逻辑继续复用 `src/data_pipeline/utils.py` 的 `atomic_write_json` / `atomic_write_jsonl` / `load_jsonl_records`。

### Phase 5A 模块结构

```text
src/data_pipeline/
└── tokenization/
    ├── __init__.py
    ├── models.py        # TokenizerAuditConfig
    └── pipeline.py      # run_tokenizer_audit（唯一入口）

scripts/
└── audit_token_lengths.py  # CLI

tests/
└── test_tokenizer_audit.py # 单元测试 + FakeTokenizer（默认不联网）
                              # 真实 Qwen3 tokenizer 集成测试需
                              # RUN_QWEN_TOKENIZER_INTEGRATION=1
```

---

## Phase 5B — QLoRA Runtime Preflight + One-Step Smoke Training

Phase 5B 是正式 QLoRA 训练前的最后一个工程验证阶段。**只跑 1 个 optimizer step**，目的是用真实 8B 模型证明：

- Phase 5A token_report 数据未被破坏；
- 当前 RTX 5090 / CUDA / PyTorch / bitsandbytes 环境可工作；
- 4-bit NF4 + LoRA + `assistant_only_loss=True` 链路可完整执行 `forward → loss → backward → optimizer step`；
- adapter 可 reload 并完成最小推理。

**不**做完整 epoch 训练、**不**做超参搜索、**不**保存 merged 模型。

### 数据流

```text
data/sft/train.jsonl
data/sft/validation.jsonl
data/tokenizer/token_report.json
data/tokenizer/token_audit.jsonl
        ↓
Phase 5B QLoRA Smoke
        ├── 复算 Phase 5A gate（不允许 max_tokens > 2048、membership 不一致、thinking marker 等）
        ├── 检查 torch / transformers / trl / peft / accelerate / datasets / bitsandbytes
        ├── 检查 CUDA availability + BF16 + Blackwell (sm_120)
        ├── 真实 BF16 matmul 计算测试
        ├── 加载 Qwen/Qwen3-8B（4-bit NF4 + double quant + bfloat16 compute）
        ├── 验证 model.is_loaded_in_4bit / Linear4bit > 0
        ├── prepare_model_for_kbit_training + LoRA r=16 alpha=32 all-linear
        ├── 选 smoke 样本：sample_id 升序前 7 条 + train P95 附近 1 条（共 8 条）
        ├── 构造 datasets.Dataset（messages-only）
        ├── 检查 TRL assistant_only_loss=True 的 labels 同时含 -100 与非 -100 token（3 条）
        ├── SFTTrainer.train(max_steps=1) — 必须 global_step == 1
        ├── 保存 LoRA adapter
        ├── 释放显存 + 重新 4-bit load + PeftModel.from_pretrained
        ├── 最小推理验证（"什么是基层党组织？"）
        ↓
data/training/smoke_config.json
data/training/smoke_report.json
outputs/phase5b-smoke/adapter/
logs/smoke_train_qlora.log
```

### 运行

```bash
# 1) 安装训练依赖（torch 单独按 CUDA wheel 安装）
pip install -r requirements-train.txt

# 2) 仅检查依赖 + GPU + BF16 + Blackwell（不下载 8B 模型）
python scripts/check_training_env.py

# 3) 运行 1-step smoke 训练（首次会下载 Qwen/Qwen3-8B 权重到 ~/.cache/huggingface）
python scripts/smoke_train_qlora.py
```

### 配置（固定）

| 维度 | 值 | 备注 |
| --- | --- | --- |
| 模型 | `Qwen/Qwen3-8B` | 与 Phase 5A 一致 |
| 量化 | `load_in_4bit=True`, `nf4`, `double quant=True` | 固定 NF4，禁止 FP4 / INT8 / GPTQ / AWQ |
| 计算 dtype | `bfloat16` | 禁止退回 fp16 |
| max_length | `2048` | 与数据真实最大 1672 匹配，禁止自动降低 |
| LoRA | `r=16, alpha=32, dropout=0.05, bias=none, all-linear` | Phase 5B 禁止超参搜索 |
| per_device_train_batch_size | `1` | |
| gradient_accumulation_steps | `1` | |
| max_steps | `1` | smoke 限定，禁止多 step |
| assistant_only_loss | `True` | TRL 内置 |
| packing | `False` | Phase 5B 禁止 |
| gradient_checkpointing | `True` | |
| seed | `42` | |

### 关键规则

- **必须先过 Phase 5A gate**：`token_report.json` 的 membership / membership missing / assistant_only_loss_feasible / prompt_prefix_mismatch / zero_assistant_target / thinking markers / max ≤ 2048 任何一项不满足，立即停止。
- **不重新生成 SFT 数据**：只读 `data/sft`、`data/tokenizer`；不修改 Phase 1~4 任何上游文件。
- **不构造第二套 collator / labels**：依赖 TRL 当前版本的 `assistant_only_loss=True`；若 TRL 无法产生正确 mask，立即停止 `TRL_ASSISTANT_MASK_UNAVAILABLE`。
- **不实现 model_factory / trainer_factory / quantization_manager**： 只有 1 个 `QLoRASmokeConfig` dataclass。
- **不保存 merged 模型**：只保存 LoRA adapter；reload 后只做最小推理验证。
- **不修改 Qwen3 chat template**：`enable_thinking=False` 由官方 tokenizer 处理；禁止手工 `replace("<think>", "")`。
- **smoke_report.json 不要求 byte-deterministic**：GPU 显存 / loss / kernel 略有差异；`smoke_config.json` 必须 deterministic。
- **类字段中文注释强制**：与 Phase 4 / 5A 同约定，新增 dataclass 每个字段必须有中文说明。

### Phase 5B 模块结构

```text
src/data_pipeline/
└── training/
    ├── __init__.py
    ├── models.py        # QLoRASmokeConfig（冻结）
    └── pipeline.py      # evaluate_phase5a_gate / select_smoke_samples / load_token_*

scripts /
├── check_training_env.py    # 环境预检（CPU 即可，依赖检查 + GPU 信息）
└── smoke_train_qlora.py     # 1-step 真实训练冒烟测试（GPU only）

tests /
└── test_training_pipeline.py
```

### 后续阶段（暂不实现）

- Phase 5C：完整 QLoRA SFT 训练（多 epoch / packing / checkpoint / 评测）。Phase 5B 已验证链路即可推进。