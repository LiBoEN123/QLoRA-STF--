"""Phase 3.5 候选审计输出生成。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .faq import ParagraphRecoveryResult


@dataclass(frozen=True)
class AuditEntry:
    """单条审计记录的稳定序列化形式。"""

    # 该 record 的 doc_id
    doc_id: str
    # 源文件路径
    source_path: str
    # record_index
    record_index: int

    # 候选 record 的 clean_text（审计锚点）
    text: str
    # structure_type（paragraph / list_item / heading 等）
    structure_type: str

    # 候选识别置信度（low / medium / high）
    candidate_confidence: str
    # 审计分类（recovered / no_answer / uncertain / not_question）
    classification: str
    # 判定原因（中文短语）
    reason: str

    # 前一条 record_index
    previous_record_index: Optional[int] = None
    # 前一条 record 文本（截断 120 字）
    previous_text: Optional[str] = None

    # 后一条 record_index
    next_record_index: Optional[int] = None
    # 后一条 record 文本（截断 120 字）
    next_text: Optional[str] = None
    # 后一条 record 的 structure_type
    next_record_structure_type: Optional[str] = None

    # 候选可能对应的答案 record_index 列表（用于人工核对）
    possible_answer_record_indices: List[int] = field(default_factory=list)
    # 候选答案文本预览（截断 200 字）
    possible_answer_text_preview: str = ""

    # 原始 question 文本（包含题号）
    raw_question_text: str = ""
    # 剥离题号后的 question 文本
    question_text: str = ""
    # 是否曾剥离粘连题号
    was_sticky_number_stripped: bool = False

    # 若该候选被吸收进 phase3 答案范围，对应 phase3 问题 doc_id
    absorbed_by_pair_doc_id: Optional[str] = None
    # 若该候选被恢复为 FAQ unit，对应 unit_id
    recovered_unit_id: Optional[str] = None


def build_audit_entries(
    records: List[Dict[str, Any]],
    recovery_results: List[ParagraphRecoveryResult],
) -> List[AuditEntry]:
    """把恢复判定结果转换为可审计的稳定条目。"""
    doc_id_to_source: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        did = str(rec.get("doc_id") or "")
        doc_id_to_source[did] = rec

    entries: List[AuditEntry] = []
    for r in recovery_results:
        rec = None
        for cand_rec in records:
            if int(cand_rec.get("record_index") or -1) == r.record_index:
                rec = cand_rec
                break
        if rec is None:
            rec = records[0] if records else {}
        source = doc_id_to_source.get(str(rec.get("doc_id") or ""), rec)
        entries.append(
            AuditEntry(
                doc_id=str(rec.get("doc_id") or ""),
                source_path=str(rec.get("source_path") or source.get("source_path") or ""),
                record_index=r.record_index,
                text=str(rec.get("clean_text") or rec.get("text") or "").strip(),
                structure_type=str(rec.get("structure_type") or "paragraph"),
                candidate_confidence=r.candidate.confidence,
                classification=r.classification,
                reason=r.reason,
                previous_record_index=r.previous_record_index,
                previous_text=r.previous_record_text,
                next_record_index=r.next_record_index,
                next_text=r.next_record_text,
                next_record_structure_type=r.next_record_structure_type,
                possible_answer_record_indices=list(r.answer_record_indices),
                possible_answer_text_preview=r.answer_text[:200] if r.answer_text else "",
                raw_question_text=r.raw_question_text,
                question_text=r.question_text,
                was_sticky_number_stripped=r.was_stripped,
                absorbed_by_pair_doc_id=r.absorbed_by_pair_doc_id,
                recovered_unit_id=r.recovered_unit_id,
            )
        )

    # 稳定排序：source_path, record_index, doc_id
    entries.sort(
        key=lambda e: (e.source_path, e.record_index, e.doc_id)
    )
    return entries


def serialize_audit(entry: AuditEntry) -> str:
    """把 AuditEntry 序列化为 JSONL 行。"""
    return json.dumps(
        asdict(entry),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def summarize(entries: List[AuditEntry]) -> Dict[str, int]:
    """按 classification 统计审计条目数量。"""
    out: Dict[str, int] = {}
    for e in entries:
        out[e.classification] = out.get(e.classification, 0) + 1
    return out
