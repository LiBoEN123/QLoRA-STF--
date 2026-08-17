"""FAQ 问题候选识别 + 答案配对。

设计原则：

* **Precision > Recall**：宁可漏掉，不要乱配。
* 第一版自动配对只接受 ``high`` 与 ``medium`` 置信度；``low`` 保留为候选但不绑定答案。
* 单条 record 既可能是问题、也可能是答案内部枚举（如 ``（1）...``）；
  必须结合多种信号联合判断。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------------
# 答案配对
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaqPair:
    """一个 (question, answers) 配对。"""

    question_doc_id: str
    question_record_index: int
    question_text: str
    raw_question_text: str
    confidence: str
    answer_doc_ids: List[str]
    answer_record_indices: List[int]
    answer_text: str


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