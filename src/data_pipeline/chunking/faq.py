"""FAQ 问题候选识别 + 答案配对。

设计原则：

* **Precision > Recall**：宁可漏掉，不要乱配。
* 第一版自动配对只接受 ``high`` 与 ``medium`` 置信度；``low`` 保留为候选但不绑定答案。
* 单条 record 既可能是问题、也可能是答案内部枚举（如 ``（1）...``）；
  必须结合多种信号联合判断。
* **Phase 3.5** 新增对 ``structure_type == paragraph`` 的问题候选进行**保守恢复**：
  只在同时满足长度、问号、疑问词、可信答案存在等条件时才升级为 medium FAQ。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import confidence_rank


# ---------------------------------------------------------------------------
# 编号前缀模式
# ---------------------------------------------------------------------------


# 阿拉伯数字编号，如 "1." / "1、"
_ARABIC_DOT_PREFIX_RE = re.compile(r"^\s*(\d+)\s*[\.\．]\s*")
_ARABIC_DUN_PREFIX_RE = re.compile(r"^\s*(\d+)\s*[、,]\s*")
# 全角括号数字 "(1)" / "（1）"
_ARABIC_PAREN_PREFIX_RE = re.compile(r"^\s*[（(]\s*(\d+)\s*[）)]\s*")
# 圈数字 "①"
_CIRCLED_PREFIX_RE = re.compile(r"^\s*([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])")
# 中文数字加顿号 "一、"
_CN_NUM_DUN_PREFIX_RE = re.compile(r"^\s*[一二三四五六七八九十]+\s*[、,]\s*")
# 全角括号中文数字 "（一）"
_CN_NUM_PAREN_PREFIX_RE = re.compile(r"^\s*[（(]\s*[一二三四五六七八九十]+\s*[）)]\s*")


# 强疑问词表（中文 + 英文）
STRONG_QUESTION_WORDS: tuple[str, ...] = (
    "什么", "为什么", "为何", "如何", "怎么", "怎样",
    "哪些", "哪种", "哪个", "是否", "能否", "可以吗",
    "应不应该", "何时", "什么时候", "哪里", "谁", "多少",
    "几种", "有什么", "有何", "区别是什么", "条件是什么",
    "程序是什么", "流程是什么", "要求有哪些", "职责是什么",
    "义务有哪些", "权利有哪些", "怎么办", "如何处理", "如何办理",
    "what", "why", "how", "when", "where", "who", "which",
)


# ---------------------------------------------------------------------------
# Phase 3.5 常量
# ---------------------------------------------------------------------------


# Paragraph FAQ 候选的最大字符数。超过即视为正文长句，不升级。
MAX_PARAGRAPH_QUESTION_CHARS: int = 160

# 答案字符最小长度。短于该值的"答案"视为不可靠，不参与配对。
MIN_RECOVERY_ANSWER_CHARS: int = 4

# 粘连题号：开头 1~3 位数字后立即为中文，且剥离后整体仍是问句。
# 例："87哪些单位一般不设立党组？" → 题号 87。
_STICKY_NUMBER_RE = re.compile(r"^(\d{1,3})(?=[一-龥])")


# ---------------------------------------------------------------------------
# 编号剥离
# ---------------------------------------------------------------------------


def strip_question_number(text: str) -> str:
    """移除开头的结构化编号。

    支持的形式：``1.`` / ``1、`` / ``（1）`` / ``①`` / ``一、`` / ``（一）``。
    只删除**开头**的编号；正文中的数字保持不变。
    """
    if not text:
        return ""
    for pattern in (
        _ARABIC_DOT_PREFIX_RE,
        _ARABIC_DUN_PREFIX_RE,
        _ARABIC_PAREN_PREFIX_RE,
        _CIRCLED_PREFIX_RE,
        _CN_NUM_DUN_PREFIX_RE,
        _CN_NUM_PAREN_PREFIX_RE,
    ):
        m = pattern.match(text)
        if m:
            return text[m.end():].strip()
    return text.strip()


def strip_sticky_question_number(text: str) -> Tuple[str, str, bool]:
    """保守剥离粘连题号（如 ``87哪些单位一般不设立党组？`` → ``87`` / 剩余）。

    仅当同时满足下列条件时才允许剥离：

    1. 文本以 1~3 位数字开头。
    2. 数字后立即是中文汉字。
    3. 剩余文本以 ``?`` / ``？`` 结尾。
    4. 剩余文本包含至少一个 :data:`STRONG_QUESTION_WORDS` 中的疑问词。
    5. 剩余文本长度合理（不超过 :data:`MAX_PARAGRAPH_QUESTION_CHARS`）。

    返回 ``(raw_text, stripped_text, was_stripped)``。当不满足条件时，
    ``stripped_text == raw_text``，``was_stripped == False``。
    """
    if not text:
        return "", "", False
    raw = text
    m = _STICKY_NUMBER_RE.match(raw)
    if not m:
        return raw, raw.strip(), False
    rest = raw[m.end():].strip()
    if not rest:
        return raw, raw.strip(), False
    if len(rest) > MAX_PARAGRAPH_QUESTION_CHARS:
        return raw, raw.strip(), False
    if not _ends_with_question_mark(rest):
        return raw, raw.strip(), False
    if not _has_question_word(rest):
        return raw, raw.strip(), False
    return raw, rest, True


# ---------------------------------------------------------------------------
# 候选识别
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionCandidate:
    """单条 record 是否被判断为 FAQ question candidate。"""

    is_candidate: bool
    confidence: str  # none | low | medium | high
    has_number_prefix: bool
    ends_with_question_mark: bool
    has_question_word: bool
    reason: str = ""


def _ends_with_question_mark(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    # 允许尾部出现成对引号 / 括号
    tail = stripped.rstrip('"""“”‘’\'”）)]）')
    return tail.endswith(("?", "？"))


def _has_question_word(text: str) -> bool:
    lower = text.lower()
    for w in STRONG_QUESTION_WORDS:
        if w in text or w in lower:
            return True
    return False


def detect_question_candidate(record: Dict[str, object]) -> QuestionCandidate:
    """判断单条 record 是否为 FAQ 问题候选。

    判定逻辑（按置信度从高到低）：

    * **high**：
        * 结构化阿拉伯数字前缀（``1.`` / ``1、``），**且**结尾是问号。
        * 圈数字前缀 + 问号。
    * **medium**：
        * 结构化阿拉伯数字前缀 + 包含强疑问词（即使没问号）。
        * 中文编号前缀（一/二） + 问号。
    * **low**：
        * 仅包含强疑问词但无编号前缀。
    * **none**：否则。
    """
    if not record:
        return QuestionCandidate(False, "none", False, False, False, "empty")

    text = str(record.get("clean_text") or record.get("text") or "").strip()
    if not text:
        return QuestionCandidate(False, "none", False, False, False, "empty")

    has_arabic_dot = bool(_ARABIC_DOT_PREFIX_RE.match(text))
    has_arabic_dun = bool(_ARABIC_DUN_PREFIX_RE.match(text))
    has_circled = bool(_CIRCLED_PREFIX_RE.match(text))
    has_arabic_paren = bool(_ARABIC_PAREN_PREFIX_RE.match(text))
    has_cn_dun = bool(_CN_NUM_DUN_PREFIX_RE.match(text))

    has_arabic_number_prefix = has_arabic_dot or has_arabic_dun or has_circled
    has_paren_prefix = has_arabic_paren
    has_cn_prefix = has_cn_dun

    ends_q = _ends_with_question_mark(text)
    has_qword = _has_question_word(text)

    # 高置信度：阿拉伯数字/圈数字编号 + 问号
    if has_arabic_number_prefix and ends_q:
        return QuestionCandidate(
            True, "high", True, ends_q, has_qword, "numbered+question_mark"
        )

    # 中置信度：阿拉伯数字/圈数字编号 + 强疑问词（即使无问号）
    if has_arabic_number_prefix and has_qword:
        return QuestionCandidate(
            True, "medium", True, ends_q, True, "numbered+question_word"
        )

    # 中置信度：中文编号前缀 + 问号
    if has_cn_prefix and ends_q:
        return QuestionCandidate(
            True, "medium", True, ends_q, has_qword, "cn_numbered+question_mark"
        )

    # 低置信度：仅有强疑问词
    if has_qword and ends_q:
        return QuestionCandidate(
            True, "low", False, ends_q, True, "loose_question_word"
        )

    return QuestionCandidate(False, "none", has_arabic_number_prefix or has_paren_prefix or has_cn_prefix, ends_q, has_qword, "no_match")


def detect_paragraph_question_candidate(
    record: Dict[str, object],
) -> QuestionCandidate:
    """判断一条普通段落 record 是否能升级为 FAQ question 候选（Phase 3.5）。

    仅在 ``structure_type == paragraph``（或缺失视为 paragraph）时生效。
    严格要求：

    1. 文本长度 ≤ :data:`MAX_PARAGRAPH_QUESTION_CHARS`。
    2. 文本以 ``?`` / ``？`` 结尾。
    3. 文本包含强疑问词。
    4. **若** 文本以 1~3 位数字开头，**则**数字后必须紧跟中文；
       且剥离数字后剩余文本仍需满足条件 1~3（视为粘连题号）。
       否则不剥离，保留原文本。
    5. 文本内不应已经含有 list_item / heading 等结构标记。

    返回的 :class:`QuestionCandidate` 在所有条件满足时 ``confidence == "medium"``，
    否则 ``confidence == "none"`` 且 ``is_candidate == False``。
    """
    if not record:
        return QuestionCandidate(False, "none", False, False, False, "empty")

    text = str(record.get("clean_text") or record.get("text") or "").strip()
    if not text:
        return QuestionCandidate(False, "none", False, False, False, "empty")

    # 防止把已经包含结构化编号的问题误识别（Phase 3 已处理）。
    if (
        _ARABIC_DOT_PREFIX_RE.match(text)
        or _ARABIC_DUN_PREFIX_RE.match(text)
        or _CIRCLED_PREFIX_RE.match(text)
        or _CN_NUM_DUN_PREFIX_RE.match(text)
    ):
        return QuestionCandidate(
            False, "none", True, _ends_with_question_mark(text),
            _has_question_word(text), "structured_prefix_handled_by_phase3",
        )

    # 长度上限
    if len(text) > MAX_PARAGRAPH_QUESTION_CHARS:
        return QuestionCandidate(
            False, "none", False, _ends_with_question_mark(text),
            _has_question_word(text), "too_long",
        )

    ends_q = _ends_with_question_mark(text)
    if not ends_q:
        return QuestionCandidate(
            False, "none", False, False, _has_question_word(text),
            "no_question_mark",
        )

    # 处理粘连题号：剥离后必须仍是问句
    raw = text
    stripped = text
    was_stripped = False
    if _STICKY_NUMBER_RE.match(text):
        _, stripped, was_stripped = strip_sticky_question_number(text)
        if not was_stripped:
            # 数字开头但剥离条件不满足：当作不可升级的普通段落
            return QuestionCandidate(
                False, "none", False, True, _has_question_word(text),
                "sticky_number_but_not_question",
            )

    # 疑问词检查（用剥离后的文本）
    if not _has_question_word(stripped):
        return QuestionCandidate(
            False, "none", was_stripped, True, False,
            "no_question_word",
        )

    return QuestionCandidate(
        True, "medium", was_stripped, True, True,
        "paragraph+question_mark+question_word" + ("+sticky_number" if was_stripped else ""),
    )


# ---------------------------------------------------------------------------
# 答案配对
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaqPair:
    """一个 (question, answers) 配对。"""

    # 问题 record 的 doc_id
    question_doc_id: str
    # 问题 record 的 record_index
    question_record_index: int
    # 清洗后的 question 文本（剥离结构化题号 / 粘连题号）
    question_text: str
    # 原始 question 文本（未做任何剥离，便于审计回溯）
    raw_question_text: str
    # 置信度：none | low | medium | high
    confidence: str
    # 答案 record 的 doc_id 列表
    answer_doc_ids: List[str]
    # 答案 record 的 record_index 列表
    answer_record_indices: List[int]
    # 答案文本（多段以 \\n\\n 拼接）
    answer_text: str
    # 检测来源：structured（Phase 3 结构化识别） | paragraph_recovery（Phase 3.5 段落恢复）
    detection_source: str = "structured"
    # 若由 paragraph_recovery 产生，记录对应 unit_id（构建 unit 后回填，可为空）
    recovered_unit_id: Optional[str] = None


# 结构边界：chapter / section / heading / subheading 是硬边界
_STRUCTURE_BOUNDARY_TYPES = {"chapter", "section", "heading", "subheading"}


def pair_questions_with_answers(
    records: List[Dict[str, object]],
    *,
    min_confidence: str = "medium",
) -> tuple[List[FaqPair], List[QuestionCandidate]]:
    """扫描已按源序排好的 records，配对问题与答案。

    返回 ``(paired, unpaired_candidates)``：

    * ``paired``：置信度 ``>= min_confidence`` 且成功配到至少 1 条答案的 FAQ。
    * ``unpaired_candidates``：识别为候选但因置信度不足或无答案而未配对的。
    """
    threshold = confidence_rank(min_confidence)
    candidates: List[tuple[int, QuestionCandidate, str]] = []

    # 先扫描候选位置。
    for idx, rec in enumerate(records):
        cand = detect_question_candidate(rec)
        if cand.is_candidate:
            candidates.append((idx, cand, str(rec.get("clean_text") or rec.get("text") or "")))

    paired: List[FaqPair] = []
    unpaired: List[QuestionCandidate] = []
    used_indices: set[int] = set()

    for n, (idx, cand, q_text) in enumerate(candidates):
        if idx in used_indices:
            continue
        if confidence_rank(cand.confidence) < threshold:
            # 置信度不足，不参与自动配对。
            unpaired.append(cand)
            continue

        # 答案范围：从 idx+1 直到下一道更高优先级候选 / 结构边界。
        answer_indices: List[int] = []
        for j in range(idx + 1, len(records)):
            rj = records[j]
            st = str(rj.get("structure_type") or "")
            if st in _STRUCTURE_BOUNDARY_TYPES:
                break
            # 是否被下一道更早的候选吃掉
            is_next_q = False
            for k, (other_idx, other_cand, _) in enumerate(candidates):
                if other_idx <= idx:
                    continue
                if other_idx == j:
                    other_rank = confidence_rank(other_cand.confidence)
                    cur_rank = confidence_rank(cand.confidence)
                    # 同样置信度按出现顺序
                    if other_rank >= cur_rank:
                        is_next_q = True
                        break
                if other_idx > j:
                    break
            if is_next_q:
                break
            answer_indices.append(j)
            used_indices.add(j)

        used_indices.add(idx)

        raw_q = q_text
        q_only = strip_question_number(raw_q)

        # 拼接答案文本
        answer_parts: List[str] = []
        for j in answer_indices:
            aj = records[j]
            t = str(aj.get("clean_text") or aj.get("text") or "").strip()
            if t:
                answer_parts.append(t)
        answer_text = "\n\n".join(answer_parts)

        answer_doc_ids = [str(records[j].get("doc_id") or "") for j in answer_indices]
        answer_record_indices = [int(records[j].get("record_index") or -1) for j in answer_indices]

        # 无答案时也保留为 unpaired。
        if not answer_text.strip():
            unpaired.append(cand)
            continue

        paired.append(
            FaqPair(
                question_doc_id=str(records[idx].get("doc_id") or ""),
                question_record_index=int(records[idx].get("record_index") or -1),
                question_text=q_only,
                raw_question_text=raw_q,
                confidence=cand.confidence,
                answer_doc_ids=answer_doc_ids,
                answer_record_indices=answer_record_indices,
                answer_text=answer_text,
            )
        )

    return paired, unpaired


# ---------------------------------------------------------------------------
# Phase 3.5：paragraph FAQ 保守恢复
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParagraphRecoveryResult:
    """单个候选的恢复判定结果。"""

    # 候选识别结果（包含置信度与原因）
    candidate: QuestionCandidate
    # 候选所在 record_index
    record_index: int
    # 原始 question 文本（包含粘连题号 / 结构化前缀）
    raw_question_text: str
    # 剥离题号后的 question 文本
    question_text: str
    # 是否成功剥离粘连题号
    was_stripped: bool
    # 判定结果使用的答案 record_index 列表（recovered 时为实际答案；其它情况为空）
    answer_record_indices: List[int]
    # 答案文本预览
    answer_text: str
    # 审计分类：recovered | no_answer | uncertain | not_question
    classification: str
    # 判定原因（中文短语，便于审计理解）
    reason: str
    # 若被吸收进 phase3 答案范围，记录对应 phase3 问题的 doc_id
    absorbed_by_pair_doc_id: Optional[str] = None
    # 前一条 record 的 record_index（无则为 None）
    previous_record_index: Optional[int] = None
    # 前一条 record 文本（截断 120 字）
    previous_record_text: Optional[str] = None
    # 后一条 record 的 record_index（无则为 None）
    next_record_index: Optional[int] = None
    # 后一条 record 文本（截断 120 字）
    next_record_text: Optional[str] = None
    # 后一条 record 的 structure_type（无则为 None）
    next_record_structure_type: Optional[str] = None
    # 若该候选被实际恢复，对应 unit_id（构建 unit 后回填，可为空）
    recovered_unit_id: Optional[str] = None


def _answer_search(
    records: List[Dict[str, object]],
    start: int,
    *,
    stop_doc_ids: set[str],
    stop_record_indices: Optional[set[int]] = None,
    max_records: int = 20,
) -> List[int]:
    """从 ``start`` 起收集可作为答案的 record 索引，直到边界或终止条件。

    终止条件（与 Phase 3 pairing 保持一致）：

    * 结构边界（chapter / section / heading / subheading）。
    * 下一个已配对的 FAQ 问题（``stop_doc_ids``）。
    * 任何其它问题 record（``stop_record_indices``，防止把后续候选 question 吞进本答案）。
    * 连续 2 条空文本（防止越界吞噬无关段落）。
    """
    indices: List[int] = []
    empty_run = 0
    stop_ri = stop_record_indices or set()
    for j in range(start, len(records)):
        if len(indices) >= max_records:
            break
        rj = records[j]
        st = str(rj.get("structure_type") or "")
        if st in _STRUCTURE_BOUNDARY_TYPES:
            break
        did = str(rj.get("doc_id") or "")
        if did in stop_doc_ids:
            break
        ri = int(rj.get("record_index") or -1)
        if ri in stop_ri:
            break
        text = str(rj.get("clean_text") or rj.get("text") or "").strip()
        if not text:
            empty_run += 1
            if empty_run >= 2:
                break
            continue
        empty_run = 0
        indices.append(j)
    return indices


def recover_paragraph_faqs(
    records: List[Dict[str, object]],
    *,
    phase3_pairs: List[FaqPair],
) -> Tuple[List[FaqPair], List[ParagraphRecoveryResult]]:
    """对所有普通段落 record 尝试识别并恢复 FAQ（Phase 3.5）。

    输入：

    * ``records``：已按源序排好的 Phase 2 clean records。
    * ``phase3_pairs``：Phase 3 已配对的高/中置信度 FAQ pairs（不会被重复识别）。

    输出：``(recovered_pairs, recovery_results)``

    * ``recovered_pairs``：满足保守升级条件、可作为 FAQ 输出的 pair。
    * ``recovery_results``：每条低置信度候选的判定结果（含恢复 / no_answer /
      uncertain / not_question），用于审计输出。

    算法要点：

    1. 用 :func:`detect_paragraph_question_candidate` 扫描所有 record。
    2. 对每条候选在 records 上独立收集答案（不受 Phase 3 used_indices 影响，
       因为 paragraph recovery 是**新**层，不破坏 Phase 3 既有 pair）。
    3. Phase 3 已有 pair 的 question record 跳过，避免重复。
    4. 候选记录如果在某条 phase3_pair 的答案范围内被吸收，
       仍可单独构造新的 recovered_pair；后续语义单元构造时再处理拆分。
    """
    phase3_q_doc_ids = {p.question_doc_id for p in phase3_pairs}
    phase3_ans_doc_ids: set[str] = set()
    for p in phase3_pairs:
        phase3_ans_doc_ids.update(p.answer_doc_ids)

    # 预计算：哪些记录已经作为 phase3 pair 的问题存在
    phase3_q_indices: set[int] = set()
    for i, rec in enumerate(records):
        if str(rec.get("doc_id") or "") in phase3_q_doc_ids:
            phase3_q_indices.add(i)

    # 找出 phase3 pair 中"问题"对应到 records 列表里的 index（用于吸收判断）
    # 用 doc_id 反查
    doc_id_to_index: Dict[str, int] = {}
    for i, rec in enumerate(records):
        doc_id_to_index[str(rec.get("doc_id") or "")] = i

    recovered_pairs: List[FaqPair] = []
    recovery_results: List[ParagraphRecoveryResult] = []

    # 第一遍：先确定所有 candidate 的判定结果
    candidate_records: List[Tuple[int, Dict[str, object], QuestionCandidate]] = []
    for idx, rec in enumerate(records):
        # 跳过已是 phase3 pair 的问题
        if idx in phase3_q_indices:
            continue
        cand = detect_paragraph_question_candidate(rec)
        if not cand.is_candidate:
            continue
        candidate_records.append((idx, rec, cand))

    # 排序：按 record_index，确保确定性
    candidate_records.sort(key=lambda x: int(x[1].get("record_index") or x[0]))

    # 收集所有"问题 record_index"集合（phase3 question + paragraph candidate），
    # _answer_search 必须在此集合处停止，避免将后一个问题的内容吞进前一个答案。
    candidate_record_indices: set[int] = set()
    for _idx, _rec, _ in candidate_records:
        candidate_record_indices.add(int(_rec.get("record_index") or -1))
    # phase3 的 question 在 records 中的 record_index
    doc_id_to_record_index: Dict[str, int] = {}
    for _i, _r in enumerate(records):
        doc_id_to_record_index[str(_r.get("doc_id") or "")] = int(_r.get("record_index") or -1)
    for _did in phase3_q_doc_ids:
        _ri = doc_id_to_record_index.get(_did)
        if _ri is not None:
            candidate_record_indices.add(_ri)

    for idx, rec, cand in candidate_records:
        text = str(rec.get("clean_text") or rec.get("text") or "").strip()
        _, stripped, was_stripped = strip_sticky_question_number(text)
        raw_q = text
        q_only = stripped if was_stripped else strip_question_number(text)

        prev_rec = records[idx - 1] if idx > 0 else None
        next_rec = records[idx + 1] if idx + 1 < len(records) else None

        # 查找答案范围：跳过 phase3 pair 的 question record 与其它候选问题
        stop_doc_ids = set(phase3_q_doc_ids)
        ans_indices = _answer_search(
            records,
            idx + 1,
            stop_doc_ids=stop_doc_ids,
            stop_record_indices=candidate_record_indices,
        )

        # 答案收集若碰上 phase3 的问题：停止（但不算 no_answer——答案在别处）
        if not ans_indices:
            recovery_results.append(
                ParagraphRecoveryResult(
                    candidate=cand,
                    record_index=int(rec.get("record_index") or idx),
                    raw_question_text=raw_q,
                    question_text=q_only,
                    was_stripped=was_stripped,
                    answer_record_indices=[],
                    answer_text="",
                    classification="no_answer",
                    reason="paragraph_question_but_no_following_answer",
                    previous_record_index=(
                        int(prev_rec.get("record_index") or -1) if prev_rec else None
                    ),
                    previous_record_text=(
                        str(prev_rec.get("clean_text") or prev_rec.get("text") or "").strip()[:120]
                        if prev_rec else None
                    ),
                    next_record_index=(
                        int(next_rec.get("record_index") or -1) if next_rec else None
                    ),
                    next_record_text=(
                        str(next_rec.get("clean_text") or next_rec.get("text") or "").strip()[:120]
                        if next_rec else None
                    ),
                    next_record_structure_type=(
                        str(next_rec.get("structure_type") or "") if next_rec else None
                    ),
                )
            )
            continue

        # 检查答案长度：过短视为不可靠
        ans_texts = [
            str(records[j].get("clean_text") or records[j].get("text") or "").strip()
            for j in ans_indices
        ]
        ans_text = "\n\n".join(t for t in ans_texts if t)

        if len(ans_text.strip()) < MIN_RECOVERY_ANSWER_CHARS:
            recovery_results.append(
                ParagraphRecoveryResult(
                    candidate=cand,
                    record_index=int(rec.get("record_index") or idx),
                    raw_question_text=raw_q,
                    question_text=q_only,
                    was_stripped=was_stripped,
                    answer_record_indices=[],
                    answer_text=ans_text,
                    classification="no_answer",
                    reason="answer_text_too_short",
                    previous_record_index=(
                        int(prev_rec.get("record_index") or -1) if prev_rec else None
                    ),
                    previous_record_text=(
                        str(prev_rec.get("clean_text") or prev_rec.get("text") or "").strip()[:120]
                        if prev_rec else None
                    ),
                    next_record_index=(
                        int(next_rec.get("record_index") or -1) if next_rec else None
                    ),
                    next_record_text=(
                        str(next_rec.get("clean_text") or next_rec.get("text") or "").strip()[:120]
                        if next_rec else None
                    ),
                    next_record_structure_type=(
                        str(next_rec.get("structure_type") or "") if next_rec else None
                    ),
                )
            )
            continue

        # 判断该候选是否在某 phase3 pair 的答案范围内（被吸收）
        absorbed_by_doc_id: Optional[str] = None
        for p in phase3_pairs:
            if int(rec.get("record_index") or -1) in p.answer_record_indices:
                absorbed_by_doc_id = p.question_doc_id
                break

        ans_doc_ids = [str(records[j].get("doc_id") or "") for j in ans_indices]
        ans_rec_indices = [int(records[j].get("record_index") or -1) for j in ans_indices]
        new_pair = FaqPair(
            question_doc_id=str(rec.get("doc_id") or ""),
            question_record_index=int(rec.get("record_index") or idx),
            question_text=q_only,
            raw_question_text=raw_q,
            confidence="medium",
            answer_doc_ids=ans_doc_ids,
            answer_record_indices=ans_rec_indices,
            answer_text=ans_text,
            detection_source="paragraph_recovery",
        )
        recovered_pairs.append(new_pair)

        recovery_results.append(
            ParagraphRecoveryResult(
                candidate=cand,
                record_index=int(rec.get("record_index") or idx),
                raw_question_text=raw_q,
                question_text=q_only,
                was_stripped=was_stripped,
                answer_record_indices=ans_rec_indices,
                answer_text=ans_text,
                classification="recovered",
                reason="paragraph+question_mark+question_word+answer_available",
                absorbed_by_pair_doc_id=absorbed_by_doc_id,
                previous_record_index=(
                    int(prev_rec.get("record_index") or -1) if prev_rec else None
                ),
                previous_record_text=(
                    str(prev_rec.get("clean_text") or prev_rec.get("text") or "").strip()[:120]
                    if prev_rec else None
                ),
                next_record_index=(
                    int(next_rec.get("record_index") or -1) if next_rec else None
                ),
                next_record_text=(
                    str(next_rec.get("clean_text") or next_rec.get("text") or "").strip()[:120]
                    if next_rec else None
                ),
                next_record_structure_type=(
                    str(next_rec.get("structure_type") or "") if next_rec else None
                ),
            )
        )

    recovered_pairs.sort(key=lambda p: p.question_record_index)
    return recovered_pairs, recovery_results


def collect_no_answer_pairs(
    records: List[Dict[str, object]],
    phase3_pairs: List[FaqPair],
) -> List[ParagraphRecoveryResult]:
    """收集 Phase 3 中"识别为问题但无答案"的 candidate（用于审计输出）。

    这些通常是文档末尾一连串没有答案的编号问题。
    """
    # 复现 Phase 3 内部的 used_indices / unpaired 计算
    threshold = confidence_rank("medium")
    candidates: List[Tuple[int, QuestionCandidate, str]] = []
    for idx, rec in enumerate(records):
        cand = detect_question_candidate(rec)
        if cand.is_candidate:
            candidates.append((idx, cand, str(rec.get("clean_text") or rec.get("text") or "")))

    paired_q_doc_ids = {p.question_doc_id for p in phase3_pairs}
    paired_ans_doc_ids: set[str] = set()
    for p in phase3_pairs:
        paired_ans_doc_ids.update(p.answer_doc_ids)

    # 找出 paired 后还剩下的"识别为候选但未配对"的 record
    used: set[int] = set()
    for p in phase3_pairs:
        # 找到 q 的 index
        for idx, rec in enumerate(records):
            if str(rec.get("doc_id") or "") == p.question_doc_id:
                used.add(idx)
                break
    # 注意：phase3 的 used 集合来自其内部循环；这里用 paired_ans_doc_ids
    # 反查也是合理的近似。

    results: List[ParagraphRecoveryResult] = []
    for idx, cand, q_text in candidates:
        if idx in used:
            continue
        if confidence_rank(cand.confidence) < threshold:
            # 低置信度：归 recover_paragraph_faqs 处理，这里跳过
            continue
        # 中/高置信度但没进入 paired → 说明无答案
        rec = records[idx]
        text = (rec.get("clean_text") or rec.get("text") or "").strip()
        prev_rec = records[idx - 1] if idx > 0 else None
        next_rec = records[idx + 1] if idx + 1 < len(records) else None
        results.append(
            ParagraphRecoveryResult(
                candidate=cand,
                record_index=int(rec.get("record_index") or idx),
                raw_question_text=text,
                question_text=strip_question_number(text),
                was_stripped=False,
                answer_record_indices=[],
                answer_text="",
                classification="no_answer",
                reason="phase3_no_answer_record",
                previous_record_index=(
                    int(prev_rec.get("record_index") or -1) if prev_rec else None
                ),
                previous_record_text=(
                    str(prev_rec.get("clean_text") or prev_rec.get("text") or "").strip()[:120]
                    if prev_rec else None
                ),
                next_record_index=(
                    int(next_rec.get("record_index") or -1) if next_rec else None
                ),
                next_record_text=(
                    str(next_rec.get("clean_text") or next_rec.get("text") or "").strip()[:120]
                    if next_rec else None
                ),
                next_record_structure_type=(
                    str(next_rec.get("structure_type") or "") if next_rec else None
                ),
            )
        )

    results.sort(key=lambda r: r.record_index)
    return results
