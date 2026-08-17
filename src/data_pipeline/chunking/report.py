"""Phase 3 质量报告生成。"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .models import Chunk, SemanticUnit


def _percentile(values: Sequence[int], pct: float) -> float:
    """计算给定百分位数（线性近邻，0..100）。"""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return float(s[k])


def _length_stats(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": round(statistics.fmean(values), 2),
        "median": float(statistics.median(values)),
        "p95": _percentile(values, 95),
    }


def build_report(
    *,
    input_records: int,
    units: List[SemanticUnit],
    chunks: List[Chunk],
    unpaired_candidate_count: int,
    hard_split_chunk_count: int,
    overlap_chunk_count: int,
    overlap_chars_total: int,
    audit_summary: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """汇总 Phase 3 运行结果为可序列化 dict。

    ``audit_summary`` 直接由 :mod:`audit` 提供的 status 计数注入，
    不会再模拟"恢复期"统计。
    """

    unit_type_counter: Counter = Counter()
    chunk_type_counter: Counter = Counter()
    faq_conf_counter: Counter = Counter()
    by_number_style: Counter = Counter()
    faq_units: List[SemanticUnit] = []
    q_lens: List[int] = []
    a_lens: List[int] = []
    q_ends_with_mark = 0
    answer_record_counts: List[int] = []
    chunk_lengths: List[int] = []
    all_source_doc_ids: set = set()
    unique_source_paths: set = set()

    for u in units:
        unit_type_counter[u.unit_type] += 1
        for did in u.source_doc_ids:
            all_source_doc_ids.add(did)
        unique_source_paths.add(u.source_path)
        if u.unit_type == "faq":
            faq_units.append(u)
            faq_conf_counter[u.faq_confidence or "none"] += 1
            by_number_style[u.metadata.get("number_style") or "none"] += 1
            if u.question_text:
                q_lens.append(len(u.question_text))
                if u.question_text.rstrip().endswith(("?", "？")):
                    q_ends_with_mark += 1
            if u.answer_text:
                a_lens.append(len(u.answer_text))
            if u.metadata.get("answer_record_count") is not None:
                answer_record_counts.append(int(u.metadata["answer_record_count"]))

    for c in chunks:
        chunk_type_counter[c.chunk_type] += 1
        chunk_lengths.append(c.char_count)
        for did in c.source_doc_ids:
            all_source_doc_ids.add(did)

    paired_count = sum(1 for u in faq_units if (u.answer_text or "").strip())

    audit_section: Dict[str, int] = {
        "candidates_checked": int((audit_summary or {}).get("candidates_checked", 0)),
        "no_answer": int((audit_summary or {}).get("no_answer", 0)),
        "low_confidence": int((audit_summary or {}).get("low_confidence", 0)),
    }

    return {
        "input_records": int(input_records),
        "semantic_units": len(units),
        "chunks": len(chunks),
        "unit_types": dict(unit_type_counter),
        "chunks_by_type": dict(chunk_type_counter),
        "faq": {
            "high_confidence": faq_conf_counter.get("high", 0),
            "medium_confidence": faq_conf_counter.get("medium", 0),
            "paired": paired_count,
            "unpaired_questions": unpaired_candidate_count,
            "by_number_style": dict(by_number_style),
        },
        "audit": audit_section,
        "faq_lengths": {
            "question": _length_stats(q_lens),
            "answer": _length_stats(a_lens),
            "answer_record_counts": _length_stats(answer_record_counts),
            "question_ends_with_mark_ratio": (
                round(q_ends_with_mark / len(q_lens), 4) if q_lens else 0.0
            ),
        },
        "length": _length_stats(chunk_lengths),
        "splitting": {
            "split_units": sum(1 for c in chunks if c.split_from_unit),
            "hard_splits": hard_split_chunk_count,
            "overlap_chunks": overlap_chunk_count,
            "overlap_chars_total": overlap_chars_total,
        },
        "traceability": {
            "unique_source_doc_ids": len(all_source_doc_ids),
            "unique_source_paths": len(unique_source_paths),
        },
    }
