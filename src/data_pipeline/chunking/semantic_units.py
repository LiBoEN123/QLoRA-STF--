"""构建 SemanticUnit：FAQ unit + 普通 record unit。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .faq import FaqCandidateRow, FaqPair, pair_questions_with_answers
from .models import ChunkingConfig, SemanticUnit, compute_unit_id


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------


def _record_source(rec: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(rec.get("source_path", "")),
        str(rec.get("file_name", "")),
        str(rec.get("source_type", "")),
        str(rec.get("file_sha256", "")),
    )


def _structure_path(record: Dict[str, Any]) -> List[str]:
    """从一条 record 派生结构路径（section → heading）。"""
    path: List[str] = []
    for key in ("detected_section", "detected_heading"):
        value = record.get(key)
        if value and str(value) not in path:
            path.append(str(value))
    return path


def _doc_ids(records: List[Dict[str, Any]]) -> List[str]:
    return [str(r.get("doc_id", "")) for r in records]


def _record_indices(records: List[Dict[str, Any]]) -> List[int]:
    return [int(r.get("record_index") or -1) for r in records]


def _make_unit(
    unit_type: str,
    text: str,
    records: List[Dict[str, Any]],
    *,
    question_text: Optional[str] = None,
    answer_text: Optional[str] = None,
    raw_question_text: Optional[str] = None,
    faq_confidence: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SemanticUnit:
    source_path, file_name, source_type, file_sha256 = _record_source(records[0])
    # 结构路径取所有 records 的最长公共前缀，缺失时退化为首条
    paths = [_structure_path(r) for r in records]
    common: List[str] = []
    for p in paths:
        if not common:
            common = list(p)
            continue
        merged: List[str] = []
        for a, b in zip(common, p):
            if a == b:
                merged.append(a)
            else:
                break
        common = merged
    detected_section = records[-1].get("detected_section") if records else None
    return SemanticUnit(
        unit_id=compute_unit_id(source_path, _doc_ids(records), unit_type, text),
        unit_type=unit_type,
        source_path=source_path,
        file_name=file_name,
        source_type=source_type,
        file_sha256=file_sha256,
        structure_path=common,
        detected_section=detected_section,
        source_doc_ids=_doc_ids(records),
        source_record_indices=_record_indices(records),
        text=text,
        question_text=question_text,
        answer_text=answer_text,
        raw_question_text=raw_question_text,
        faq_confidence=faq_confidence,
        char_count=len(text),
        metadata=metadata or {},
    )


def _format_faq_text(question: str, answer: str) -> str:
    """FAQ unit 的可读文本。"""
    q = (question or "").strip()
    a = (answer or "").strip()
    if q and a:
        return f"问题：{q}\n\n答案：{a}"
    if q:
        return f"问题：{q}"
    if a:
        return f"答案：{a}"
    return ""


# ---------------------------------------------------------------------------
# FAQ unit 构造
# ---------------------------------------------------------------------------


def _build_faq_units(
    pairs: List[FaqPair],
    ordered_records: List[Dict[str, Any]],
) -> List[Tuple[int, SemanticUnit]]:
    """把每个 FAQ pair 映射为 (record_index, SemanticUnit)。"""
    by_q_record_index: Dict[int, Dict[str, Any]] = {}
    by_record_index: Dict[int, Dict[str, Any]] = {}
    for i, r in enumerate(ordered_records):
        ri = int(r.get("record_index") or -1)
        by_q_record_index[ri] = r
        by_record_index[i] = r

    # 按 (doc_id, record_index) 反查 q record 在列表中的下标
    doc_id_to_index: Dict[str, int] = {}
    for i, r in enumerate(ordered_records):
        doc_id_to_index[str(r.get("doc_id") or "")] = i

    result: List[Tuple[int, SemanticUnit]] = []
    for p in pairs:
        q_idx = doc_id_to_index.get(p.question_doc_id)
        if q_idx is None:
            continue
        q_record = ordered_records[q_idx]

        # 收集答案 records（按出现顺序）
        ans_by_record_index: Dict[int, Dict[str, Any]] = {}
        for r in ordered_records:
            ri = int(r.get("record_index") or -1)
            if ri in set(p.answer_record_indices):
                ans_by_record_index[ri] = r
        ans_records = [
            ans_by_record_index[ri]
            for ri in p.answer_record_indices
            if ri in ans_by_record_index
        ]

        unit_records = [q_record] + ans_records
        text = _format_faq_text(p.question_text, p.answer_text)
        unit = _make_unit(
            "faq",
            text,
            unit_records,
            question_text=p.question_text,
            answer_text=p.answer_text,
            raw_question_text=p.raw_question_text,
            faq_confidence=p.confidence,
            metadata={
                "question_record_index": p.question_record_index,
                "answer_record_count": len(ans_records),
                "number_style": p.number_style,
            },
        )
        result.append((q_idx, unit))
    return result


# ---------------------------------------------------------------------------
# Non-FAQ unit：相邻同结构 records 聚合
# ---------------------------------------------------------------------------


_STRUCTURE_BOUNDARY_TYPES = {"chapter", "section", "heading", "subheading"}


def _build_other_units(
    ordered_records: List[Dict[str, Any]],
    used: set[int],
) -> List[Tuple[int, SemanticUnit]]:
    """扫描未被 FAQ 使用的 records，按相邻 structure_type 聚合成 unit。"""
    result: List[Tuple[int, SemanticUnit]] = []
    i, n = 0, len(ordered_records)
    while i < n:
        if i in used:
            i += 1
            continue
        rec = ordered_records[i]
        st = str(rec.get("structure_type") or "paragraph")

        # 边界型结构：单条成 unit
        if st in _STRUCTURE_BOUNDARY_TYPES:
            text = str(rec.get("clean_text") or rec.get("text") or "").strip()
            result.append((i, _make_unit(st, text, [rec])))
            i += 1
            continue

        # 同类型相邻 records 聚合成一个 group unit
        bucket = [rec]
        j = i + 1
        while j < n and j not in used:
            rj = ordered_records[j]
            stj = str(rj.get("structure_type") or "paragraph")
            if stj != st or stj in _STRUCTURE_BOUNDARY_TYPES:
                break
            bucket.append(rj)
            j += 1

        text = "\n\n".join(
            str(r.get("clean_text") or r.get("text") or "").strip()
            for r in bucket
        )
        unit_type = (
            "paragraph_group" if st == "paragraph"
            else "list_group" if st == "list_item"
            else "standalone"
        )
        result.append((i, _make_unit(unit_type, text, bucket)))
        i = j
    return result


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def build_semantic_units(
    ordered_records: List[Dict[str, Any]],
    *,
    config: ChunkingConfig,
) -> Tuple[List[SemanticUnit], List[FaqCandidateRow]]:
    """从已排序的 records 构建语义单元；同时返回未配对的候选行供 audit 使用。

    只有一条 pairing 路径；audit 数据由 pairing 自然产出，
    不再单独模拟一次 Phase 3 used_indices。
    """
    paired, unpaired = pair_questions_with_answers(
        ordered_records,
        min_confidence=config.faq_min_confidence,
    )

    # 标记 FAQ 使用的 record 索引
    doc_id_to_index: Dict[str, int] = {
        str(r.get("doc_id") or ""): i
        for i, r in enumerate(ordered_records)
    }
    used: set[int] = set()
    for p in paired:
        if p.question_doc_id in doc_id_to_index:
            used.add(doc_id_to_index[p.question_doc_id])
    for p in paired:
        for did in p.answer_doc_ids:
            if did in doc_id_to_index:
                used.add(doc_id_to_index[did])

    faq_units = _build_faq_units(paired, ordered_records)
    other_units = _build_other_units(ordered_records, used)

    # 按源顺序合并
    faq_units.sort(key=lambda x: x[0])
    other_units.sort(key=lambda x: x[0])
    merged = faq_units + other_units
    merged.sort(
        key=lambda x: ordered_records[x[0]].get("record_index", -1)
        if x[0] < len(ordered_records) else -1
    )
    return [u for _, u in merged], unpaired
