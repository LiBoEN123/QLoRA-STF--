"""Phase 2 pipeline 编排器。

读取 Phase 1 的 ``documents.jsonl``（record 为符合 Phase 1 schema 的 dict），
输出：

* ``documents_clean.jsonl``  —— 通过清洗的 record（含清洗元数据）
* ``rejected_documents.jsonl`` —— 被质量规则过滤的 record
* ``duplicates.jsonl`` —— exact / near duplicate
* ``quality_report.json`` —— 聚合统计

pipeline 是确定性的、无状态的——相同输入 + 相同配置 → 相同输出。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..models import DocumentRecord
from . import dedup, normalizer, quality, structure
from .quality import (
    HF_MAX_LENGTH,
    HF_MIN_RATIO,
    HF_MIN_REPEATS,
    QualityConfig,
    QualityReport,
)


# ---------------------------------------------------------------------------
# Pipeline 配置
# ---------------------------------------------------------------------------


@dataclass
class CleaningConfig:
    """Phase 2 清洗管线的全部可调阈值。

    Attributes:
        min_length: 文本长度下限，低于此值将被拒。
        max_length: 文本长度上限，超过此值将被拒。
        max_symbol_ratio: 符号字符占比上限，超过将标记为疑似乱码。
        simhash_distance: SimHash 近似重复的 Hamming 距离阈值。
        enable_near_dedup: 是否启用 SimHash 近似去重。
        quality: 详细质量配置（嵌套 :class:`QualityConfig`）。
    """

    min_length: int = quality.MIN_TEXT_LENGTH
    max_length: int = quality.MAX_TEXT_LENGTH
    max_symbol_ratio: float = quality.MAX_SYMBOL_RATIO
    simhash_distance: int = 3
    enable_near_dedup: bool = True
    quality: QualityConfig = field(default_factory=QualityConfig)


# ---------------------------------------------------------------------------
# IO 辅助（原子写入）
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


# ---------------------------------------------------------------------------
# 结果 dataclass
# ---------------------------------------------------------------------------


@dataclass
class CleaningResult:
    """Phase 2 一次完整运行的汇总结果。

    Attributes:
        input_path: 输入 JSONL 路径。
        output_clean: 清洗后输出 JSONL 路径。
        output_rejected: 被拒记录输出 JSONL 路径。
        output_duplicates: 重复记录输出 JSONL 路径。
        output_report: 报告 JSON 路径。
        input_records: 输入记录数。
        output_records: 最终保留记录数。
        rejected_records: 被质量规则拒掉的记录数。
        exact_duplicates: 完全重复记录数。
        near_duplicates: 近似重复记录数。
        cleaning_actions_counts: 各清洗动作的出现次数。
        structure_counts: 结构类型计数。
        reject_reason_counts: 拒因计数。
        quality_flag_counts: 质量 flag 计数。
        header_footer_files: 被识别到页眉/页脚的源文档数。
        header_footer_texts_removed: 被剥除的页眉/页脚文本数。
        line_break_repaired: 触发换行修复的记录数。
        text_changed_count: ``clean_text`` 与原始 ``text`` 不同的记录数。
        text_length_stats: 文本长度分布统计。
        quality_stats: 质量指标汇总。
        structure_stats: 结构计数（与 ``structure_counts`` 同义）。
        source_type_counts: 源类型计数。
        top_sources: 出现记录数最多的源文件 Top N。
    """

    input_path: Path
    output_clean: Path
    output_rejected: Path
    output_duplicates: Path
    output_report: Path

    input_records: int = 0
    output_records: int = 0
    rejected_records: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0

    cleaning_actions_counts: Counter = field(default_factory=Counter)
    structure_counts: Counter = field(default_factory=Counter)
    reject_reason_counts: Counter = field(default_factory=Counter)
    quality_flag_counts: Counter = field(default_factory=Counter)
    header_footer_files: int = 0
    header_footer_texts_removed: int = 0
    line_break_repaired: int = 0
    text_changed_count: int = 0

    text_length_stats: Dict[str, float] = field(default_factory=dict)
    quality_stats: Dict[str, int] = field(default_factory=dict)
    structure_stats: Dict[str, int] = field(default_factory=dict)
    source_type_counts: Counter = field(default_factory=Counter)
    top_sources: List[Tuple[str, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IO：读取 Phase 1 records
# ---------------------------------------------------------------------------


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _serialize(record: Dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 单条 record 处理
# ---------------------------------------------------------------------------


def _process_record(
    record: Dict[str, Any],
    *,
    context_by_source: Dict[str, structure.SectionContext],
    config: CleaningConfig,
    hf_high_confidence: Dict[Tuple[str, str], bool],
) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None, Dict[str, Any] | None]:
    """对单条 record 应用清洗、结构、质量评估。

    返回 ``(clean_record 或 None, rejected_record 或 None, duplicate 或 None)``。
    三个返回值中至多有一个非 None。
    """
    doc_id = record.get("doc_id", "")
    source_path = record.get("source_path", "")
    metadata = record.get("metadata") or {}
    original_text = record.get("text", "") or ""

    # 1. 规范化。
    clean_text, actions = normalizer.normalize_full(original_text)
    line_repair_applied = "repair_line_break" in actions

    # 2. 仅移除高置信度的页眉/页脚。
    hf_key = (source_path, clean_text.strip())
    removed_hf = False
    if hf_high_confidence.get(hf_key):
        clean_text = ""
        actions.append("header_footer_removed")
        removed_hf = True

    # 3. 结构识别。在「原始」文本上运行，避免全角括号 / 圈数字被 NFKC 吃掉。
    ctx = context_by_source.setdefault(source_path, structure.SectionContext())
    info = structure.detect_structure(original_text, metadata=metadata)
    ctx.update(info)

    # 4. 质量评估。如果 record 本身就是结构标题，过短豁免。
    exempt_short = info.structure_type != "paragraph"
    quality_report = quality.evaluate_quality(
        clean_text,
        config=config.quality,
        exempt_short=exempt_short,
    )
    if removed_hf:
        quality_report.possible_header_footer = True
        quality_report.flags.append("high_confidence_header_footer")

    # 5. 决定是否 reject。
    reject_reasons: List[str] = []
    if quality_report.is_empty:
        reject_reasons.append("empty_after_cleaning")
    if quality_report.is_too_short:
        reject_reasons.append("too_short")
    if quality_report.is_too_long:
        reject_reasons.append("too_long")
    if "symbol_ratio_too_high" in quality_report.flags:
        reject_reasons.append("symbol_ratio_too_high")
    if "possible_garbled" in quality_report.flags:
        reject_reasons.append("possible_garbled")
    if "high_confidence_header_footer" in quality_report.flags:
        reject_reasons.append("high_confidence_header_footer")

    # 6. 构造增强后的 record（clean / rejected 两条路径共用）。
    detected_section = ctx.detected_section()
    augmented = dict(record)
    augmented["text"] = original_text  # 永远不要覆盖 Phase 1 的 text
    augmented["clean_text"] = clean_text
    augmented["structure_type"] = info.structure_type
    augmented["structure_level"] = info.structure_level
    augmented["detected_section"] = detected_section
    augmented["detected_heading"] = info.detected_heading
    augmented["quality"] = quality_report.to_dict()
    augmented["cleaning_actions"] = actions

    if reject_reasons:
        rejected = {
            "doc_id": doc_id,
            "source_path": source_path,
            "file_name": record.get("file_name"),
            "record_index": record.get("record_index"),
            "page_number": record.get("page_number"),
            "sheet_name": record.get("sheet_name"),
            "file_sha256": record.get("file_sha256"),
            "original_text": original_text,
            "clean_text": clean_text,
            "reject_reasons": reject_reasons,
            "structure_type": info.structure_type,
            "quality": quality_report.to_dict(),
        }
        return None, rejected, None

    return augmented, None, None


# ---------------------------------------------------------------------------
# Pipeline 驱动
# ---------------------------------------------------------------------------


def _detect_header_footer(
    records: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], bool]:
    """返回被标记为页眉/页脚的 ``(source_path, clean_text)`` 集合。"""
    grouped: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        # 使用「尚未剥离页眉页脚」时的规范化文本。
        clean, _ = normalizer.normalize_full(rec.get("text", "") or "")
        if 1 <= len(clean.strip()) <= HF_MAX_LENGTH:
            grouped[rec.get("source_path", "")].append(clean.strip())

    flagged: Dict[Tuple[str, str], bool] = {}
    for source_path, texts in grouped.items():
        candidates = quality.detect_header_footer_candidates(
            texts,
            max_length=HF_MAX_LENGTH,
            min_repeats=HF_MIN_REPEATS,
            min_ratio=HF_MIN_RATIO,
        )
        for text in candidates:
            flagged[(source_path, text)] = True
    return flagged


def _percentile(values: List[int], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return float(s[k])


def _build_report(
    result: CleaningResult,
    *,
    input_path: Path,
    config: CleaningConfig,
) -> Dict[str, Any]:
    retention_rate = (
        result.output_records / result.input_records
        if result.input_records
        else 0.0
    )
    return {
        "input_path": input_path.as_posix(),
        "input_records": result.input_records,
        "output_records": result.output_records,
        "rejected_records": result.rejected_records,
        "exact_duplicates": result.exact_duplicates,
        "near_duplicates": result.near_duplicates,
        "retention_rate": round(retention_rate, 4),
        "config": {
            "min_length": config.min_length,
            "max_length": config.max_length,
            "max_symbol_ratio": config.max_symbol_ratio,
            "simhash_distance": config.simhash_distance,
            "enable_near_dedup": config.enable_near_dedup,
        },
        "text_length": result.text_length_stats,
        "quality": dict(result.quality_stats),
        "structure": dict(result.structure_counts),
        "cleaning_actions": dict(result.cleaning_actions_counts),
        "reject_reasons": dict(result.reject_reason_counts),
        "header_footer": {
            "files_with_high_confidence_hf": result.header_footer_files,
            "high_confidence_texts_removed": result.header_footer_texts_removed,
        },
        "source_types": dict(result.source_type_counts),
        "top_sources": [
            {"source_path": p, "records": n}
            for p, n in result.top_sources
        ],
    }


def run_cleaning_pipeline(
    input_path: Path,
    output_clean: Path,
    output_rejected: Path,
    output_duplicates: Path,
    output_report: Path,
    *,
    config: CleaningConfig | None = None,
    logger: logging.Logger | None = None,
) -> CleaningResult:
    """对一个 Phase 1 JSONL 文件运行完整的清洗 pipeline。"""
    log = logger or logging.getLogger("data_pipeline.cleaning")
    cfg = config or CleaningConfig()

    result = CleaningResult(
        input_path=input_path,
        output_clean=output_clean,
        output_rejected=output_rejected,
        output_duplicates=output_duplicates,
        output_report=output_report,
    )

    started = datetime.now(timezone.utc)

    log.info("Phase 2 cleaning pipeline started")
    log.info("Input: %s", input_path)

    records = _load_records(input_path)
    result.input_records = len(records)
    log.info("Loaded %d records", len(records))

    # 页眉页脚检测遍历。
    hf_flagged = _detect_header_footer(records)
    log.info("Header/footer candidate texts: %d", len(hf_flagged))

    # 遍历 records，应用清洗 + 结构 + 质量，构造 reject 列表。
    context_by_source: Dict[str, structure.SectionContext] = {}
    clean_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []
    text_lengths: List[int] = []
    chinese_ratios: List[float] = []
    symbol_ratios: List[float] = []

    for rec in records:
        augmented, rejected, _ = _process_record(
            rec,
            context_by_source=context_by_source,
            config=cfg,
            hf_high_confidence=hf_flagged,
        )
        # 累加清洗动作 + 结构计数。
        if augmented is not None:
            for action in augmented.get("cleaning_actions", []):
                result.cleaning_actions_counts[action] += 1
            result.structure_counts[augmented["structure_type"]] += 1
            result.source_type_counts[augmented.get("source_type", "unknown")] += 1
            for f in augmented["quality"]["flags"]:
                result.quality_flag_counts[f] += 1
            if "repair_line_break" in augmented["cleaning_actions"]:
                result.line_break_repaired += 1
            if augmented["text"] != augmented["clean_text"]:
                result.text_changed_count += 1
            clean_records.append(augmented)
            text_lengths.append(augmented["quality"]["char_count"])
            chinese_ratios.append(augmented["quality"]["chinese_char_ratio"])
            symbol_ratios.append(augmented["quality"]["symbol_ratio"])
        if rejected is not None:
            for reason in rejected["reject_reasons"]:
                result.reject_reason_counts[reason] += 1
            rejected_records.append(rejected)

    log.info("After quality filtering: %d clean, %d rejected", len(clean_records), len(rejected_records))

    # 去重（exact + near）。输入按 doc_id 升序以保证完全确定性。
    dedup_input = sorted(
        (r["doc_id"], r["clean_text"]) for r in clean_records
    )
    exact_dups, near_dups = dedup.deduplicate(
        dedup_input,
        max_distance=cfg.simhash_distance,
        enable_near=cfg.enable_near_dedup,
    )
    result.exact_duplicates = len(exact_dups)
    result.near_duplicates = len(near_dups)

    dropped_ids = {d.duplicate_doc_id for d in exact_dups} | {d.duplicate_doc_id for d in near_dups}
    final_clean: List[Dict[str, Any]] = []
    for r in clean_records:
        if r["doc_id"] in dropped_ids:
            continue
        # canonical 保留在 final_clean 中。
        final_clean.append(r)

    # 原子写入所有输出。所有 JSONL 行均排序以保证确定性。
    clean_lines = sorted(_serialize(r) for r in final_clean)
    rejected_lines = sorted(_serialize(r) for r in rejected_records)
    duplicate_lines: List[str] = []
    for d in list(exact_dups) + list(near_dups):
        duplicate_lines.append(_serialize(_duplicate_to_dict(d)))
    duplicate_lines.sort()

    _atomic_write_jsonl(output_clean, clean_lines)
    _atomic_write_jsonl(output_rejected, rejected_lines)
    _atomic_write_jsonl(output_duplicates, duplicate_lines)

    # 构造 report。
    result.output_records = len(final_clean)
    result.rejected_records = len(rejected_records)
    result.text_length_stats = {
        "min": float(min(text_lengths)) if text_lengths else 0.0,
        "max": float(max(text_lengths)) if text_lengths else 0.0,
        "mean": float(statistics.fmean(text_lengths)) if text_lengths else 0.0,
        "median": float(statistics.median(text_lengths)) if text_lengths else 0.0,
        "p95": _percentile(text_lengths, 95),
    }
    result.quality_stats = {
        "possible_garbled": sum(1 for r in final_clean if r["quality"]["possible_garbled"]),
        "possible_header_footer": sum(1 for r in final_clean if r["quality"]["possible_header_footer"]),
        "too_short": sum(1 for r in final_clean if r["quality"]["is_too_short"]),
        "too_long": sum(1 for r in final_clean if r["quality"]["is_too_long"]),
        "mean_chinese_ratio": round(statistics.fmean(chinese_ratios), 4) if chinese_ratios else 0.0,
        "mean_symbol_ratio": round(statistics.fmean(symbol_ratios), 4) if symbol_ratios else 0.0,
    }
    result.structure_stats = dict(result.structure_counts)
    source_counter = Counter(r.get("source_path", "") for r in records)
    result.top_sources = source_counter.most_common(20)
    result.header_footer_files = len({k[0] for k in hf_flagged})
    result.header_footer_texts_removed = len(hf_flagged)

    duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    report = _build_report(result, input_path=input_path, config=cfg)
    _atomic_write_json(output_report, report)
    log.info("Cleaning finished in %d ms", duration_ms)
    log.info("Cleaning finished: %d clean, %d rejected, %d exact-dup, %d near-dup",
             result.output_records, result.rejected_records,
             result.exact_duplicates, result.near_duplicates)
    log.info("Outputs written to %s, %s, %s, %s",
             output_clean, output_rejected, output_duplicates, output_report)
    return result


def _duplicate_to_dict(d) -> Dict[str, Any]:
    if isinstance(d, dedup.ExactDuplicate):
        return {
            "duplicate_doc_id": d.duplicate_doc_id,
            "canonical_doc_id": d.canonical_doc_id,
            "duplicate_type": d.duplicate_type,
            "similarity": d.similarity,
            "distance": 0,
        }
    return {
        "duplicate_doc_id": d.duplicate_doc_id,
        "canonical_doc_id": d.canonical_doc_id,
        "duplicate_type": d.duplicate_type,
        "similarity": round(d.similarity, 6),
        "distance": d.distance,
    }