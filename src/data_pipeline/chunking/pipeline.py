"""Phase 3 chunking 编排器：从 clean records 到 semantic units + chunks。

完整的调用链：

::

    run_chunking_pipeline
        ↓
    restore_source_order
        ↓
    pair_questions_with_answers      # 唯一的检测 + 配对路径
        │
        └── detect_question_candidate
        ↓
    build_semantic_units             # 唯一的 SemanticUnit 构建路径
        ↓
    build_chunks                     # 唯一的 chunk 构建路径
        ↓
    write outputs (semantic_units / chunks / report / audit)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import atomic_write_json, atomic_write_jsonl, get_logger, load_jsonl_records
from .audit import AuditEntry, build_audit_entries, serialize_audit, summarize as summarize_audit
from .models import Chunk, ChunkingConfig, SemanticUnit, compute_chunk_id, serialize_chunk, serialize_unit
from .ordering import restore_source_order
from .report import build_report
from .semantic_units import build_semantic_units
from .splitter import split_long_text


@dataclass
class ChunkingResult:
    """Phase 3 一次完整运行的汇总结果。"""

    input_path: Path
    output_units: Path
    output_chunks: Path
    output_report: Path
    output_audit: Path

    input_records: int = 0
    semantic_units: int = 0
    chunks: int = 0
    unit_types: Counter = field(default_factory=Counter)
    chunks_by_type: Counter = field(default_factory=Counter)
    audit_entries: List[AuditEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunk 构造
# ---------------------------------------------------------------------------


def _make_chunk(
    units: List[SemanticUnit],
    text: str,
    *,
    chunk_index: int,
    split_from_unit: bool = False,
    chunk_type: str = "text",
) -> Chunk:
    """根据 1..n 条 SemanticUnit 构造一个 chunk。"""
    head = units[0]
    return Chunk(
        chunk_id=compute_chunk_id([u.unit_id for u in units], chunk_index, text),
        chunk_index=chunk_index,
        chunk_type=chunk_type,
        source_path=head.source_path,
        file_name=head.file_name,
        source_type=head.source_type,
        file_sha256=head.file_sha256,
        structure_path=list(head.structure_path),
        detected_section=head.detected_section,
        source_unit_ids=[u.unit_id for u in units],
        source_doc_ids=[did for u in units for did in u.source_doc_ids],
        source_record_indices=[idx for u in units for idx in u.source_record_indices],
        text=text,
        question_text=units[0].question_text if chunk_type == "faq" else None,
        answer_text=units[0].answer_text if chunk_type == "faq" else None,
        faq_confidence=units[0].faq_confidence if chunk_type == "faq" else None,
        char_count=len(text),
        split_from_unit=split_from_unit,
        metadata={"merged_from_units": len(units)} if len(units) > 1 else dict(units[0].metadata),
    )


def _build_chunks(
    units: List[SemanticUnit],
    *,
    config: ChunkingConfig,
) -> tuple[List[Chunk], int, int, int]:
    """根据 semantic units 构造 chunks（一次性完成，无中间转发层）。"""
    chunks: List[Chunk] = []
    hard_split_count = 0
    overlap_chunk_count = 0
    overlap_chars_total = 0
    chunk_index = 0

    # FAQ / 边界结构：单 chunk
    boundary_types = {"faq", "heading", "subheading", "chapter", "section"}

    i, n = 0, len(units)
    while i < n:
        u = units[i]
        if u.unit_type in boundary_types:
            chunk_type = "faq" if u.unit_type == "faq" else "heading"
            chunks.append(_make_chunk(
                [u], u.text, chunk_index=chunk_index, chunk_type=chunk_type,
            ))
            chunk_index += 1
            i += 1
            continue

        # 普通文本：贪心合并同 source + 同 structure_path
        bucket = [u]
        j = i + 1
        while j < n:
            uj = units[j]
            if uj.unit_type in boundary_types:
                break
            if uj.source_path != u.source_path or uj.structure_path != u.structure_path:
                break
            bucket.append(uj)
            j += 1

        merged_text = "\n\n".join(b.text for b in bucket if b.text)
        if len(merged_text) <= config.max_chars:
            chunks.append(_make_chunk(bucket, merged_text, chunk_index=chunk_index, chunk_type="text"))
            chunk_index += 1
            i = j
            continue

        # 超长：按 unit 单独成 chunk 并对超长 unit 切分
        for b in bucket:
            if len(b.text) <= config.max_chars:
                chunks.append(_make_chunk([b], b.text, chunk_index=chunk_index, chunk_type="text"))
                chunk_index += 1
            else:
                overlap = config.overlap_chars
                if overlap:
                    overlap_chunk_count += 1
                    overlap_chars_total += overlap
                for seg in split_long_text(b.text, max_chars=config.max_chars, overlap_chars=overlap):
                    chunks.append(_make_chunk(
                        [b], seg.text, chunk_index=chunk_index, chunk_type="text", split_from_unit=True,
                    ))
                    if seg.is_hard_split:
                        hard_split_count += 1
                    chunk_index += 1
        i = j

    return chunks, hard_split_count, overlap_chunk_count, overlap_chars_total


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_chunking_pipeline(
    input_path: Path,
    output_units: Path,
    output_chunks: Path,
    output_report: Path,
    *,
    output_audit: Optional[Path] = None,
    config: ChunkingConfig | None = None,
    logger: logging.Logger | None = None,
) -> ChunkingResult:
    """运行 Phase 3 chunking 流水线。"""
    log = logger or logging.getLogger("data_pipeline.chunking")
    cfg = config or ChunkingConfig()

    if output_audit is None:
        output_audit = output_report.parent / "faq_audit.jsonl"

    result = ChunkingResult(
        input_path=input_path,
        output_units=output_units,
        output_chunks=output_chunks,
        output_report=output_report,
        output_audit=output_audit,
    )

    log.info("Phase 3 chunking pipeline started")
    log.info("Input: %s", input_path)

    # 1) 读取 + 源序恢复
    ordered_records = restore_source_order(load_jsonl_records(input_path))
    result.input_records = len(ordered_records)
    log.info("Loaded %d records; restored source order", len(ordered_records))

    # 2) 唯一一次 FAQ 检测 + 配对 + SemanticUnit 构建
    units, unpaired = build_semantic_units(ordered_records, config=cfg)
    result.semantic_units = len(units)
    log.info("Built %d semantic units", len(units))
    for k, v in sorted(Counter(u.unit_type for u in units).items(), key=lambda x: -x[1]):
        log.info("    unit_type=%s count=%d", k, v)

    # 3) 构造 chunks
    chunks, hard_split_count, overlap_chunk_count, overlap_chars_total = _build_chunks(units, config=cfg)
    result.chunks = len(chunks)
    log.info(
        "Built %d chunks (hard_split=%d, overlap_chunks=%d)",
        len(chunks), hard_split_count, overlap_chunk_count,
    )
    for k, v in sorted(Counter(c.chunk_type for c in chunks).items(), key=lambda x: -x[1]):
        log.info("    chunk_type=%s count=%d", k, v)

    # 4) 构造 audit（直接消费 pairing 的 unpaired，不重做 pairing）
    audit_entries = build_audit_entries(
        ordered_records, unpaired,
        paired_answer_doc_ids={
            did for u in units
            if u.unit_type == "faq" and u.question_text is not None
            for did in u.source_doc_ids
        },
    )
    result.audit_entries = audit_entries
    audit_summary = summarize_audit(audit_entries)
    audit_summary["candidates_checked"] = len(audit_entries)
    log.info("Audit: %s", audit_summary)

    # 5) 写入输出
    atomic_write_jsonl(output_units, [serialize_unit(u) for u in units])
    atomic_write_jsonl(output_chunks, [serialize_chunk(c) for c in chunks])
    atomic_write_jsonl(output_audit, [serialize_audit(e) for e in audit_entries])
    atomic_write_json(
        output_report,
        build_report(
            input_records=result.input_records,
            units=units,
            chunks=chunks,
            unpaired_candidate_count=len(unpaired),
            hard_split_chunk_count=hard_split_count,
            overlap_chunk_count=overlap_chunk_count,
            overlap_chars_total=overlap_chars_total,
            audit_summary=audit_summary,
        ),
    )
    log.info("Wrote units=%d chunks=%d audit=%d → %s / %s / %s",
             len(units), len(chunks), len(audit_entries),
             output_units, output_chunks, output_audit)

    result.unit_types = Counter(u.unit_type for u in units)
    result.chunks_by_type = Counter(c.chunk_type for c in chunks)
    return result
