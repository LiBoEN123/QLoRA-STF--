"""Phase 3 编排器：从 clean records 到 semantic units + chunks。"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..utils import get_logger
from .faq import pair_questions_with_answers
from .models import Chunk, ChunkingConfig, SemanticUnit, compute_chunk_id, serialize_chunk, serialize_unit
from .ordering import restore_source_order
from .report import build_report
from .semantic_units import build_semantic_units
from .splitter import SplitSegment, split_long_text


# ---------------------------------------------------------------------------
# IO 辅助（原子写入，复用 Phase 2 模式）
# ---------------------------------------------------------------------------


def _atomic_write_jsonl(path: Path, lines: Sequence[str]) -> None:
    """以原子方式写入 JSONL（临时文件 + os.replace）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for line in lines:
                fh.write(line)
                fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """以原子方式写入 JSON，key 排序稳定。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------


@dataclass
class ChunkingResult:
    """Phase 3 运行结果聚合。"""

    input_path: Path
    output_units: Path
    output_chunks: Path
    output_report: Path

    input_records: int = 0
    semantic_units: int = 0
    chunks: int = 0
    unit_types: Counter = field(default_factory=Counter)
    chunks_by_type: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------------
# Chunk 构造
# ---------------------------------------------------------------------------


def _build_chunks_from_units(
    units: List[SemanticUnit],
    *,
    config: ChunkingConfig,
) -> tuple[List[Chunk], int, int, int]:
    """根据 semantic units 构造 chunks。"""
    return _build_chunks_impl(units, config=config)


def _build_chunks_impl(
    units: List[SemanticUnit],
    *,
    config: ChunkingConfig,
) -> tuple[List[Chunk], int, int, int]:
    """实际构造 chunks 的逻辑。"""
    chunks: List[Chunk] = []
    hard_split_count = 0
    overlap_chunk_count = 0
    overlap_chars_total = 0
    chunk_index = 0

    i = 0
    n = len(units)
    while i < n:
        u = units[i]
        # FAQ / heading → 单 chunk
        if u.unit_type == "faq":
            chunk = _make_chunk_from_unit(u, chunk_index=chunk_index, split_from_unit=False)
            chunks.append(chunk)
            chunk_index += 1
            i += 1
            continue
        if u.unit_type in ("heading", "subheading", "chapter", "section"):
            chunk = _make_chunk_from_unit(u, chunk_index=chunk_index, split_from_unit=False)
            chunk.chunk_type = "heading"
            chunks.append(chunk)
            chunk_index += 1
            i += 1
            continue

        # 普通文本：贪心合并同 source + 同 structure_path
        bucket: List[SemanticUnit] = [u]
        j = i + 1
        while j < n:
            uj = units[j]
            if uj.unit_type in ("faq", "heading", "subheading", "chapter", "section"):
                break
            if uj.source_path != u.source_path:
                break
            if uj.structure_path != u.structure_path:
                break
            bucket.append(uj)
            j += 1

        # 尝试合并 bucket
        merged_text = "\n\n".join(b.text for b in bucket if b.text)
        if len(merged_text) <= config.max_chars:
            # 单 chunk
            chunk = _chunk_from_units(bucket, merged_text, chunk_index=chunk_index)
            chunks.append(chunk)
            chunk_index += 1
            i = j
            continue

        # 超过 max_chars：按 unit 单独成 chunk，再对超长 unit 切分
        for b in bucket:
            if len(b.text) <= config.max_chars:
                chunk = _chunk_from_units([b], b.text, chunk_index=chunk_index)
                chunks.append(chunk)
                chunk_index += 1
            else:
                # 长 unit：切分
                allow_overlap = b.unit_type not in ("faq", "heading", "subheading", "chapter", "section")
                overlap = config.overlap_chars if allow_overlap else 0
                if overlap:
                    overlap_chunk_count += 1
                    overlap_chars_total += overlap
                segments = split_long_text(b.text, max_chars=config.max_chars, overlap_chars=overlap)
                for seg in segments:
                    chunk = _chunk_from_units(
                        [b],
                        seg.text,
                        chunk_index=chunk_index,
                        split_from_unit=True,
                    )
                    chunks.append(chunk)
                    if seg.is_hard_split:
                        hard_split_count += 1
                    chunk_index += 1
        i = j

    return chunks, hard_split_count, overlap_chunk_count, overlap_chars_total


def _make_chunk_from_unit(
    unit: SemanticUnit,
    *,
    chunk_index: int,
    split_from_unit: bool,
) -> Chunk:
    """从单个 SemanticUnit 生成 chunk。"""
    text = unit.text
    chunk_type = "faq" if unit.unit_type == "faq" else "text"
    return Chunk(
        chunk_id=compute_chunk_id([unit.unit_id], chunk_index, text),
        chunk_index=chunk_index,
        chunk_type=chunk_type,
        source_path=unit.source_path,
        file_name=unit.file_name,
        source_type=unit.source_type,
        file_sha256=unit.file_sha256,
        structure_path=list(unit.structure_path),
        detected_section=unit.detected_section,
        source_unit_ids=[unit.unit_id],
        source_doc_ids=list(unit.source_doc_ids),
        source_record_indices=list(unit.source_record_indices),
        text=text,
        question_text=unit.question_text,
        answer_text=unit.answer_text,
        faq_confidence=unit.faq_confidence,
        char_count=len(text),
        split_from_unit=split_from_unit,
        metadata=dict(unit.metadata),
    )


def _chunk_from_units(
    units: List[SemanticUnit],
    text: str,
    *,
    chunk_index: int,
    split_from_unit: bool = False,
) -> Chunk:
    """从多个 SemanticUnit 合并生成一个 chunk。"""
    head = units[0]
    return Chunk(
        chunk_id=compute_chunk_id([u.unit_id for u in units], chunk_index, text),
        chunk_index=chunk_index,
        chunk_type="text",
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
        char_count=len(text),
        split_from_unit=split_from_unit,
        metadata={"merged_from_units": len(units)},
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_chunking_pipeline(
    input_path: Path,
    output_units: Path,
    output_chunks: Path,
    output_report: Path,
    *,
    config: ChunkingConfig | None = None,
    logger: logging.Logger | None = None,
) -> ChunkingResult:
    """运行完整 Phase 3 chunking 流水线。"""
    log = logger or logging.getLogger("data_pipeline.chunking")
    cfg = config or ChunkingConfig()

    result = ChunkingResult(
        input_path=input_path,
        output_units=output_units,
        output_chunks=output_chunks,
        output_report=output_report,
    )

    log.info("Phase 3 chunking pipeline started")
    log.info("Input: %s", input_path)

    # 1) 读取 + 恢复源序
    raw_records = _load_records(input_path)
    result.input_records = len(raw_records)
    ordered_records = restore_source_order(raw_records)
    log.info("Loaded %d records (post-dedup); restored source order", len(ordered_records))

    # 2) FAQ 配对（仅用于统计 unpaired_candidate_count）
    paired, unpaired = pair_questions_with_answers(
        ordered_records,
        min_confidence=cfg.faq_min_confidence,
    )
    log.info(
        "FAQ candidates: paired=%d, unpaired(low/none)=%d",
        len(paired),
        len(unpaired),
    )

    # 3) 构建 semantic units
    units = build_semantic_units(ordered_records, config=cfg)
    log.info("Built %d semantic units", len(units))
    unit_type_counter: Counter = Counter(u.unit_type for u in units)
    for k, v in sorted(unit_type_counter.items(), key=lambda x: -x[1]):
        log.info("    unit_type=%s count=%d", k, v)

    # 4) 构建 chunks
    chunks, hard_split_count, overlap_chunk_count, overlap_chars_total = _build_chunks_from_units(
        units, config=cfg,
    )
    log.info("Built %d chunks (hard_split=%d, overlap_chunks=%d)",
             len(chunks), hard_split_count, overlap_chunk_count)
    chunk_type_counter: Counter = Counter(c.chunk_type for c in chunks)
    for k, v in sorted(chunk_type_counter.items(), key=lambda x: -x[1]):
        log.info("    chunk_type=%s count=%d", k, v)

    # 5) 写入输出
    unit_lines = [serialize_unit(u) for u in units]
    chunk_lines = [serialize_chunk(c) for c in chunks]
    _atomic_write_jsonl(output_units, unit_lines)
    _atomic_write_jsonl(output_chunks, chunk_lines)
    log.info("Wrote %d units → %s", len(units), output_units)
    log.info("Wrote %d chunks → %s", len(chunks), output_chunks)

    # 6) 报告
    report = build_report(
        input_records=result.input_records,
        units=units,
        chunks=chunks,
        unpaired_candidate_count=len(unpaired),
        hard_split_chunk_count=hard_split_count,
        overlap_chunk_count=overlap_chunk_count,
        overlap_chars_total=overlap_chars_total,
    )
    _atomic_write_json(output_report, report)
    log.info("Wrote report → %s", output_report)

    result.semantic_units = len(units)
    result.chunks = len(chunks)
    result.unit_types = unit_type_counter
    result.chunks_by_type = chunk_type_counter
    return result