"""构建 SemanticUnit：FAQ unit + 普通 unit。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .faq import FaqPair, pair_questions_with_answers, recover_paragraph_faqs
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
# 拆分辅助：处理 paragraph_recovery pair 与 phase3 pair 答案重叠
# ---------------------------------------------------------------------------


def _split_phase3_pair_by_recovered(
    pair: FaqPair,
    records: List[Dict[str, Any]],
    absorbed_q_record_indices: set[int],
) -> Tuple[FaqPair, List[FaqPair]]:
    """把一个 phase3 pair 的答案按被恢复的 question record 切成多段。

    返回 ``(adjusted_pair, new_pairs)``：

    * ``adjusted_pair``：被截断的 phase3 pair（答案只保留到第一个吸收点之前）。
    * ``new_pairs``：每个吸收点对应的 recovered pair（顺序与吸收点顺序一致）。
    """
    if not absorbed_q_record_indices:
        return pair, []

    # 按 answer_record_indices 顺序，定位每个被吸收的 record 在 answer 中的位置
    ans_records: List[Dict[str, Any]] = []
    ans_doc_id_to_rec: Dict[str, Dict[str, Any]] = {}
    for j, r in enumerate(records):
        did = str(r.get("doc_id") or "")
        if did in pair.answer_doc_ids:
            ans_records.append(r)
            ans_doc_id_to_rec[did] = r

    # 排序后的答案 records
    ans_records_sorted = sorted(
        ans_records,
        key=lambda r: int(r.get("record_index") or -1),
    )

    # 找出每个吸收点的位置
    split_positions: List[int] = []  # 在 ans_records_sorted 中的下标
    for j, r in enumerate(ans_records_sorted):
        if int(r.get("record_index") or -1) in absorbed_q_record_indices:
            split_positions.append(j)

    if not split_positions:
        return pair, []

    # 第一段：0 到 split_positions[0]-1
    first_seg = ans_records_sorted[: split_positions[0]]
    adjusted_ans_doc_ids = [str(r.get("doc_id") or "") for r in first_seg]
    adjusted_ans_record_indices = [int(r.get("record_index") or -1) for r in first_seg]
    adjusted_ans_text = "\n\n".join(
        str(r.get("clean_text") or r.get("text") or "").strip() for r in first_seg
    )

    adjusted_pair = FaqPair(
        question_doc_id=pair.question_doc_id,
        question_record_index=pair.question_record_index,
        question_text=pair.question_text,
        raw_question_text=pair.raw_question_text,
        confidence=pair.confidence,
        answer_doc_ids=adjusted_ans_doc_ids,
        answer_record_indices=adjusted_ans_record_indices,
        answer_text=adjusted_ans_text,
        detection_source=pair.detection_source,
        recovered_unit_id=pair.recovered_unit_id,
    )

    # 中间段：每个吸收点对应的 recovered pair（已由调用方提供，这里仅占位）
    # 实际生成由 build_semantic_units 完成
    return adjusted_pair, []


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def build_semantic_units(
    ordered_records: List[Dict[str, Any]],
    *,
    config: ChunkingConfig,
    recovered_pairs: Optional[List[FaqPair]] = None,
) -> List[SemanticUnit]:
    """对一组已按源序排好的 records 构建语义单元。

    流程：

    1. 调用 Phase 3 FAQ 配对找出 ``high/medium`` FAQ pairs。
    2. （Phase 3.5）若提供 ``recovered_pairs``，则进入 paragraph recovery 拆分逻辑：
       * 对每个 recovered pair，若其 question record 落在某 phase3 pair 的
         answer 范围内，则拆分 phase3 pair 的答案为多段。
       * 同时在原位置插入新的 recovered FAQ unit。
    3. 标记被 FAQ 配对消耗的 record 索引。
    4. 顺序扫描剩余 records：
        * heading / subheading → heading unit。
        * 连续 paragraphs → paragraph_group unit。
        * 连续 list_items → list_group unit。
        * 单独 record → standalone unit。
    5. 按 source order 输出所有 unit（FAQ 与非 FAQ 交替出现，按原文位置）。
    """
    paired, _unpaired = pair_questions_with_answers(
        ordered_records,
        min_confidence=config.faq_min_confidence,
    )

    # 建立 question_doc_id → pair 的映射
    pair_by_qid: Dict[str, FaqPair] = {p.question_doc_id: p for p in paired}

    # ========== Phase 3.5：调整 phase3 pairs 的答案范围 ==========
    # 找出每个 phase3 pair 的 answer 范围内被 recovered pair 的 question 占据的记录
    recovered = list(recovered_pairs or [])
    # 按 question_record_index 排序
    recovered.sort(key=lambda p: p.question_record_index)

    # 把 recovered pair 的 question record index 按"所属 phase3 pair"分组
    ri_to_doc: Dict[int, str] = {}
    for ri, r in enumerate(ordered_records):
        ri_to_doc[int(r.get("record_index") or -1)] = str(r.get("doc_id") or "")

    absorbed_by_phase3: Dict[str, List[FaqPair]] = {}  # phase3_q_doc_id → [recovered_pair, ...]
    for rp in recovered:
        for phase3_pair in paired:
            if rp.question_record_index in phase3_pair.answer_record_indices:
                absorbed_by_phase3.setdefault(phase3_pair.question_doc_id, []).append(rp)
                break

    # 调整受影响的 phase3 pair 答案
    adjusted_phase3_pairs: Dict[str, FaqPair] = {}
    for phase3_pair in paired:
        absorbed = absorbed_by_phase3.get(phase3_pair.question_doc_id, [])
        if not absorbed:
            adjusted_phase3_pairs[phase3_pair.question_doc_id] = phase3_pair
            continue
        absorbed_q_ri = {rp.question_record_index for rp in absorbed}
        adjusted_pair, _ = _split_phase3_pair_by_recovered(
            phase3_pair, ordered_records, absorbed_q_ri,
        )
        adjusted_phase3_pairs[phase3_pair.question_doc_id] = adjusted_pair

    # ========== 构造 FAQ units（包括 phase3 + recovered） ==========
    used: set[int] = set()
    faq_units: List[Tuple[int, SemanticUnit]] = []

    # 先收集 phase3 pair 的 q/ans 索引
    for phase3_pair in paired:
        adjusted = adjusted_phase3_pairs[phase3_pair.question_doc_id]
        # 找 q 的索引
        q_idx = None
        for i, r in enumerate(ordered_records):
            if str(r.get("doc_id") or "") == phase3_pair.question_doc_id:
                q_idx = i
                break
        if q_idx is None or q_idx in used:
            continue
        q_record = ordered_records[q_idx]
        # 收集 adjusted 答案对应的 record 列表
        ans_records: List[Dict[str, Any]] = []
        for j, r in enumerate(ordered_records):
            if str(r.get("doc_id") or "") in adjusted.answer_doc_ids:
                ans_records.append(r)
                used.add(j)
        used.add(q_idx)
        all_records = [q_record] + ans_records
        text = _format_faq_text(adjusted.question_text, adjusted.answer_text)
        unit = _make_unit(
            "faq",
            text,
            all_records,
            question_text=adjusted.question_text,
            answer_text=adjusted.answer_text,
            raw_question_text=adjusted.raw_question_text,
            faq_confidence=adjusted.confidence,
            metadata={
                "question_record_index": adjusted.question_record_index,
                "answer_record_count": len(ans_records),
                "faq_detection_source": adjusted.detection_source,
            },
        )
        faq_units.append((q_idx, unit))

    # 再处理 recovered pair：跳过已被 phase3 pair 调整覆盖的 record
    # recovered pair 的 question record 不应该出现在任何 adjusted phase3 pair 的 answer 里
    for rp in recovered:
        # 找 q 的索引
        q_idx = None
        for i, r in enumerate(ordered_records):
            if int(r.get("record_index") or -1) == rp.question_record_index:
                q_idx = i
                break
        if q_idx is None or q_idx in used:
            continue
        q_record = ordered_records[q_idx]
        # 收集 answer records（跳过其他 FAQ 的 question）
        phase3_q_doc_ids = {p.question_doc_id for p in paired}
        ans_records: List[Dict[str, Any]] = []
        for j, r in enumerate(ordered_records):
            did = str(r.get("doc_id") or "")
            if did in phase3_q_doc_ids:
                continue
            if int(r.get("record_index") or -1) in rp.answer_record_indices:
                if j in used:
                    # 该 answer record 已被 phase3 占用（不应发生，但兜底）
                    continue
                ans_records.append(r)
                used.add(j)
        used.add(q_idx)
        all_records = [q_record] + ans_records
        text = _format_faq_text(rp.question_text, rp.answer_text)
        unit = _make_unit(
            "faq",
            text,
            all_records,
            question_text=rp.question_text,
            answer_text=rp.answer_text,
            raw_question_text=rp.raw_question_text,
            faq_confidence=rp.confidence,
            metadata={
                "question_record_index": rp.question_record_index,
                "answer_record_count": len(ans_records),
                "faq_detection_source": rp.detection_source,
            },
        )
        faq_units.append((q_idx, unit))

    # ========== Non-FAQ units：扫描剩余 records，按相邻结构类型聚合 ==========
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

    # ========== 按 source order 合并 ==========
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
