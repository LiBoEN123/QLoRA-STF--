"""FAQ 候选审计：仅观测主流程，不参与业务控制流。

audit 唯一数据源是 :func:`pair_questions_with_answers` 直接返回的
:class:`FaqCandidateRow` 列表——即那些被识别为问题但因为置信度不足或
没有后续答案而未配对的 record。本模块不重新扫描 FAQ 配对逻辑。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .faq import FaqCandidateRow


@dataclass(frozen=True)
class AuditEntry:
    """单条审计记录：服务于人工核对的最小字段集。"""

    doc_id: str
    source_path: str
    record_index: int
    text: str
    structure_type: str

    confidence: str
    reason: str
    status: str  # paired | no_answer | low_confidence | unknown
    number_style: str

    next_record_index: Optional[int] = None
    next_text_preview: Optional[str] = None
    next_record_structure_type: Optional[str] = None


def build_audit_entries(
    ordered_records: List[Dict[str, Any]],
    unpaired_candidates: List[FaqCandidateRow],
    *,
    paired_answer_doc_ids: Optional[set[str]] = None,
) -> List[AuditEntry]:
    """把未配对的候选行序列化为稳定可审计条目。

    判定规则：

    * 置信度 ``>= medium`` → ``status = no_answer``（已识别为问题但找不到答案）。
    * 置信度 ``low`` / ``none`` → ``status = low_confidence``（仅观察，不配对）。
    """
    # 用 record_index 反查到 records 中的下标，便于取下一条
    ri_to_index: Dict[int, int] = {
        int(r.get("record_index") or -1): i
        for i, r in enumerate(ordered_records)
    }

    entries: List[AuditEntry] = []
    for row in unpaired_candidates:
        idx = ri_to_index.get(row.record_index)
        rec: Dict[str, Any] = ordered_records[idx] if idx is not None else {}
        text = str(rec.get("clean_text") or rec.get("text") or "").strip()
        next_rec = ordered_records[idx + 1] if idx is not None and idx + 1 < len(ordered_records) else None
        next_text = ""
        next_st: Optional[str] = None
        next_ri: Optional[int] = None
        if next_rec is not None:
            next_text = str(next_rec.get("clean_text") or next_rec.get("text") or "").strip()[:120]
            next_st = str(next_rec.get("structure_type") or "") or None
            next_ri = int(next_rec.get("record_index") or -1)

        if row.candidate.confidence in ("medium", "high"):
            status = "no_answer"
        else:
            status = "low_confidence"

        entries.append(
            AuditEntry(
                doc_id=str(rec.get("doc_id") or ""),
                source_path=str(rec.get("source_path") or ""),
                record_index=row.record_index,
                text=text,
                structure_type=str(rec.get("structure_type") or "paragraph"),
                confidence=row.candidate.confidence,
                reason=row.candidate.reason,
                status=status,
                number_style=row.candidate.number_style,
                next_record_index=next_ri,
                next_text_preview=next_text or None,
                next_record_structure_type=next_st,
            )
        )

    entries.sort(key=lambda e: (e.source_path, e.record_index, e.doc_id))
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
    """按 status 统计审计条目数量。"""
    return dict(Counter(e.status for e in entries))
