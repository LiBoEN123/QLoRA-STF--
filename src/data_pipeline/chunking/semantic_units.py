"""构建 SemanticUnit：FAQ unit + 普通 unit。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .faq import FaqPair, pair_questions_with_answers
from .models import ChunkingConfig, SemanticUnit, compute_unit_id


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _build_structure_path(record: Dict[str, Any]) -> List[str]:
    """从一条 record 派生结构路径。"""
    path: List[str] = []
    section = record.get("detected_section")
    heading = record.get("detected_heading")
    if section and section not in path:
        path.append(str(section))
    if heading and heading not in path:
        path.append(str(heading))
    return path


def _common_structure_path(paths: List[List[str]]) -> List[str]:
    """求一组结构路径的最长公共前缀。"""
    if not paths:
        return []
    common = list(paths[0])
    for p in paths[1:]:
        new: List[str] = []
        for a, b in zip(common, p):
            if a == b:
                new.append(a)
            else:
                break
        common = new
    return common


def _record_source(rec: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(rec.get("source_path", "")),
        str(rec.get("file_name", "")),
        str(rec.get("source_type", "")),
        str(rec.get("file_sha256", "")),
    )


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
    paths = [_build_structure_path(r) for r in records]
    structure_path = _common_structure_path(paths) if paths else []
    detected_section = (
        records[-1].get("detected_section") if records else None
    )
    return SemanticUnit(
        unit_id=compute_unit_id(source_path, _doc_ids(records), unit_type, text),
        unit_type=unit_type,
        source_path=source_path,
        file_name=file_name,
        source_type=source_type,
        file_sha256=file_sha256,
        structure_path=structure_path,
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


def _doc_ids(records: List[Dict[str, Any]]) -> List[str]:
    return [str(r.get("doc_id", "")) for r in records]


def _record_indices(records: List[Dict[str, Any]]) -> List[int]:
    return [int(r.get("record_index") or -1) for r in records]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def build_semantic_units(
    ordered_records: List[Dict[str, Any]],
    *,
    config: ChunkingConfig,
) -> List[SemanticUnit]:
    """对一组已按源序排好的 records 构建语义单元。

    流程：

    1. 调用 FAQ 配对找出 ``high/medium`` FAQ pairs。
    2. 标记被 FAQ 配对消耗的 record 索引。
    3. 顺序扫描剩余 records：
        * heading / subheading → heading unit。
        * 连续 paragraphs → paragraph_group unit。
        * 连续 list_items → list_group unit。
        * 单独 record → standalone unit。
    4. 按 source order 输出所有 unit（FAQ 与非 FAQ 交替出现，按原文位置）。
    """
    paired, _unpaired = pair_questions_with_answers(
        ordered_records,
        min_confidence=config.faq_min_confidence,
    )

    # 建立 question_doc_id → pair 的映射，方便后处理。
    pair_by_qid: Dict[str, FaqPair] = {p.question_doc_id: p for p in paired}

    used: set[int] = set()

    # 1) FAQ units
    faq_units: List[Tuple[int, SemanticUnit]] = []
    for idx, rec in enumerate(ordered_records):
        qid = str(rec.get("doc_id") or "")
        if qid not in pair_by_qid:
            continue
        if idx in used:
            continue
        pair = pair_by_qid[qid]
        q_record = rec
        # 找到答案 record 集合
        answer_records: List[Dict[str, Any]] = []
        answer_indices: List[int] = []
        for j, r in enumerate(ordered_records):
            if str(r.get("doc_id") or "") in pair.answer_doc_ids:
                answer_records.append(r)
                answer_indices.append(j)
                used.add(j)
        used.add(idx)
        all_records = [q_record] + answer_records
        text = _format_faq_text(pair.question_text, pair.answer_text)
        unit = _make_unit(
            "faq",
            text,
            all_records,
            question_text=pair.question_text,
            answer_text=pair.answer_text,
            raw_question_text=pair.raw_question_text,
            faq_confidence=pair.confidence,
            metadata={
                "question_record_index": pair.question_record_index,
                "answer_record_count": len(answer_records),
            },
        )
        faq_units.append((idx, unit))

    # 2) Non-FAQ units：扫描剩余 records，按相邻结构类型聚合
    other_units: List[Tuple[int, SemanticUnit]] = []
    i = 0
    n = len(ordered_records)
    while i < n:
        if i in used:
            i += 1
            continue
        rec = ordered_records[i]
        st = str(rec.get("structure_type") or "paragraph")

        if st in ("chapter", "section", "heading", "subheading"):
            # 边界 unit：单条
            text = str(rec.get("clean_text") or rec.get("text") or "").strip()
            unit = _make_unit(
                st,
                text,
                [rec],
            )
            other_units.append((i, unit))
            used.add(i)
            i += 1
            continue

        # 聚合相邻同结构类型 records
        bucket: List[Dict[str, Any]] = [rec]
        j = i + 1
        while j < n:
            if j in used:
                break
            rj = ordered_records[j]
            stj = str(rj.get("structure_type") or "paragraph")
            if stj != st:
                break
            if stj in ("chapter", "section", "heading", "subheading"):
                break
            bucket.append(rj)
            j += 1
        for k in range(i, j):
            used.add(k)
        text = "\n\n".join(
            str(r.get("clean_text") or r.get("text") or "").strip() for r in bucket
        )
        unit_type = "paragraph_group" if st == "paragraph" else "list_group" if st == "list_item" else "standalone"
        unit = _make_unit(unit_type, text, bucket)
        other_units.append((i, unit))
        i = j

    # 3) 按 source order 合并（先按首个 record 的 record_index 排序）
    faq_units.sort(key=lambda x: x[0])
    other_units.sort(key=lambda x: x[0])
    merged = faq_units + other_units
    merged.sort(
        key=lambda x: ordered_records[x[0]].get("record_index", -1) if x[0] < n else -1
    )
    return [u for _, u in merged]


def _format_faq_text(question: str, answer: str) -> str:
    """FAQ unit 的可读文本：'问题：xxx\\n\\n答案：xxx'。"""
    q = (question or "").strip()
    a = (answer or "").strip()
    if q and a:
        return f"问题：{q}\n\n答案：{a}"
    if q:
        return f"问题：{q}"
    if a:
        return f"答案：{a}"
    return ""