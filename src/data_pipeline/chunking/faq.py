"""FAQ 问题识别与答案配对。

设计原则：

* **一个 Detection Path + 一个 Pairing Path**。
  所有候选（结构化编号 / 粘连题号 / 段落短问句）共用同一份检测与同一份配对，
  不再维护"Phase 3 + Phase 3.5"两套流程。
* **Precision > Recall**：自动配对只接受 ``high`` / ``medium`` 置信度；
  ``low`` 候选保留但不被绑定。
* 配对的答案范围自然以"下一个 high / medium 候选"或"结构边界"为终止条件，
  因此段落问句也能自动终止上一题答案，不需要二次拆分逻辑。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import confidence_rank


# ---------------------------------------------------------------------------
# 编号前缀模式
# ---------------------------------------------------------------------------


# 阿拉伯数字 + 点 / 全角点
_ARABIC_DOT_PREFIX_RE = re.compile(r"^\s*(\d+)\s*[\.\．]\s*")
# 阿拉伯数字 + 顿号 / 半角逗号
_ARABIC_DUN_PREFIX_RE = re.compile(r"^\s*(\d+)\s*[、,]\s*")
# 全角 / 半角括号阿拉伯数字
_ARABIC_PAREN_PREFIX_RE = re.compile(r"^\s*[（(]\s*(\d+)\s*[）)]\s*")
# 圈数字
_CIRCLED_PREFIX_RE = re.compile(r"^\s*([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])")
# 中文数字 + 顿号
_CN_NUM_DUN_PREFIX_RE = re.compile(r"^\s*[一二三四五六七八九十]+\s*[、,]\s*")
# 全角 / 半角括号中文数字
_CN_NUM_PAREN_PREFIX_RE = re.compile(r"^\s*[（(]\s*[一二三四五六七八九十]+\s*[）)]\s*")


# 强疑问词表（中文 + 英文），由真实 FAQ 候选语料整理得到。
STRONG_QUESTION_WORDS: tuple[str, ...] = (
    "什么", "为什么", "为何", "如何", "怎么", "怎样",
    "哪些", "哪种", "哪个", "是否", "能否", "可以吗",
    "应不应该", "何时", "什么时候", "哪里", "谁", "多少",
    "几种", "有什么", "有何", "区别是什么", "条件是什么",
    "程序是什么", "流程是什么", "要求有哪些", "职责是什么",
    "义务有哪些", "权利有哪些", "怎么办", "如何处理", "如何办理",
    "what", "why", "how", "when", "where", "who", "which",
)


# 段落 FAQ 候选的最大字符数。超过该值视为正文长句，不升级为问题。
MAX_PARAGRAPH_QUESTION_CHARS: int = 160

# 粘连题号：开头 1~3 位数字紧接中文汉字。
# 例：``87哪些单位一般不设立党组？`` → 题号 ``87``。
# 故意排除常见量词（万 / 元 / 岁 / 年 / % / 月 / 日 / 个 / 多 / 余），
# 以保留 ``100万元党费如何管理？`` / ``2026年党员教育工作重点是什么？`` / ``18岁以上...``。
_STICKY_NUMBER_RE = re.compile(r"^(\d{1,3})(?=[一-龥])")
# 数字后的"单位字"集合：出现这些字时优先视为数量 / 年份 / 金额，而不是题号。
_DIGIT_FOLLOW_UNITS = frozenset("万元岁年月日个多数余‰%")


# ---------------------------------------------------------------------------
# 候选识别
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionCandidate:
    """单条 record 的 FAQ 问题候选判定结果。

    字段设计原则：detector 是唯一的"真值源"，UI / pairing / audit 一律读此结果。
    """

    # 是否被识别为 question candidate
    is_candidate: bool
    # 置信度：none | low | medium | high
    confidence: str
    # 判定原因（中文短语，便于人工审计）
    reason: str
    # 剥离后的 question 文本（即 pairing 真正使用的 question）
    question_text: str
    # 原始 question 文本（含编号 / 粘连题号），保留以便溯源
    raw_question_text: str
    # 编号风格：arabic / circled / cn_num / sticky / paragraph / none
    number_style: str


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _ends_with_question_mark(text: str) -> bool:
    """尾部允许出现成对引号 / 括号后再以 ``?`` / ``？`` 收尾。"""
    stripped = text.rstrip()
    if not stripped:
        return False
    tail = stripped.rstrip('"""“”‘’\'”）)]）')
    return tail.endswith(("?", "？"))


def _has_question_word(text: str) -> bool:
    """快速检查是否含 :data:`STRONG_QUESTION_WORDS` 中的任意一个。"""
    for w in STRONG_QUESTION_WORDS:
        if w in text:
            return True
    return False


def _read_text(record: Dict[str, object]) -> str:
    return str(record.get("clean_text") or record.get("text") or "").strip()


def _strip_structured_prefix(text: str) -> Tuple[str, str]:
    """尝试剥离开头的结构化编号（``1.`` / ``1、`` / ``①`` / ``一、``）。

    返回 ``(style, stripped_text)``。若不匹配任何已知编号样式，
    则 ``style == "none"`` 且 ``stripped_text == 原文本``。
    """
    for style, pattern in (
        ("arabic_dot", _ARABIC_DOT_PREFIX_RE),
        ("arabic_dun", _ARABIC_DUN_PREFIX_RE),
        ("circled", _CIRCLED_PREFIX_RE),
        ("cn_num_dun", _CN_NUM_DUN_PREFIX_RE),
    ):
        m = pattern.match(text)
        if m:
            return style, text[m.end():].strip()
    return "none", text


def strip_question_number(text: str) -> str:
    """公共便利函数：剥离结构化编号（含 ``（1）`` / ``（一）``，但**不**处理粘连题号）。

    note::
        实际 detector（:func:`detect_question_candidate`）只对 ``1.`` / ``1、`` /
        ``①`` / ``一、`` 这四种样式做"问题前缀剥离"。``（1）`` / ``（一）`` 全角
        括号数字通常出现在答案内列表项中，本函数为兼容性保留它们。
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


def _try_sticky_number(text: str) -> Tuple[str, bool]:
    """尝试剥离粘连题号。

    保守判定：

    * 剩余文本必须仍以问号收尾、含强疑问词、长度合理。
    * **剥离前**必须确认数字之后的字符不是常见量词
      （万 / 元 / 岁 / 年 / % / 月 / 日 / 个 / 多 / 余），
      以避免把 ``100万元`` / ``2026年`` / ``18岁`` 这种"数字 + 单位"误判为题号。
    """
    m = _STICKY_NUMBER_RE.match(text)
    if not m:
        return text, False
    # 排除 ``100万元`` / ``2026年`` / ``18岁`` 等"数字 + 单位"组合
    next_char = text[m.end():1 + m.end()]  # 紧跟数字后的第一个字符
    if next_char and next_char in _DIGIT_FOLLOW_UNITS:
        return text, False
    rest = text[m.end():].strip()
    if (
        not rest
        or len(rest) > MAX_PARAGRAPH_QUESTION_CHARS
        or not _ends_with_question_mark(rest)
        or not _has_question_word(rest)
    ):
        return text, False
    return rest, True


# ---------------------------------------------------------------------------
# 主入口：detect_question_candidate
# ---------------------------------------------------------------------------


def detect_question_candidate(record: Dict[str, object]) -> QuestionCandidate:
    """判断一条 record 是否为 FAQ question 候选，并给出配对所用的文本。

    判定路径（优先级从高到低）：

    1. **结构化阿拉伯数字 / 圈数字编号**

       * ``1.`` / ``1、`` / ``①`` 加 **问号** → ``high``。
       * ``1.`` / ``1、`` / ``①`` 加 **强疑问词**（即使无问号）→ ``medium``。
       * ``一、`` 等中文编号加问号 → ``medium``。

    2. **粘连题号**

       开头 1~3 位数字紧接中文，剥离后仍是问句 → ``medium``。
       例如 ``87哪些单位一般不设立党组？``。

    3. **段落问句**

       无任何上述编号、长度合理、以问号收尾、含强疑问词 → ``medium``。
       例如 ``如何确定基层党组织的名称？`` / ``党员每年集中学习培训时间一般不少于多少学时？``。

    4. 否则 → ``none``。
    """
    text = _read_text(record)
    if not text:
        return QuestionCandidate(False, "none", "empty", "", "", "none")

    # 1) 结构化编号
    style, stripped = _strip_structured_prefix(text)
    if style in ("arabic_dot", "arabic_dun", "circled"):
        ends_q = _ends_with_question_mark(stripped)
        has_qword = _has_question_word(stripped)
        if ends_q:
            return QuestionCandidate(
                True, "high", "numbered+question_mark",
                stripped, text, style,
            )
        if has_qword:
            return QuestionCandidate(
                True, "medium", "numbered+question_word",
                stripped, text, style,
            )
    if style == "cn_num_dun":
        ends_q = _ends_with_question_mark(stripped)
        if ends_q:
            return QuestionCandidate(
                True, "medium", "cn_numbered+question_mark",
                stripped, text, style,
            )

    # 2) 粘连题号（开头 1~3 位数字紧接中文）
    sticky_stripped, was_sticky = _try_sticky_number(text)
    if was_sticky:
        return QuestionCandidate(
            True, "medium", "sticky_number+question_word",
            sticky_stripped, text, "sticky",
        )

    # 3) 段落问句
    if (
        len(text) <= MAX_PARAGRAPH_QUESTION_CHARS
        and _ends_with_question_mark(text)
        and _has_question_word(text)
    ):
        return QuestionCandidate(
            True, "medium", "paragraph+question_mark+question_word",
            text, text, "paragraph",
        )

    # 4) 不通过
    return QuestionCandidate(False, "none", "no_match", "", text, "none")


# ---------------------------------------------------------------------------
# FAQ Pairing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaqPair:
    """一个 (question, answers) 配对。"""

    question_doc_id: str
    question_record_index: int
    # 剥离结构化 / 粘连编号后的 question 文本，用于下游消费
    question_text: str
    # 原始 question 文本（含编号 / 粘连题号），用于审计
    raw_question_text: str
    confidence: str  # none | low | medium | high
    answer_doc_ids: List[str]
    answer_record_indices: List[int]
    answer_text: str
    # 候选检测使用的规则族，便于按 detection_source 统计
    number_style: str = "none"


# 结构边界硬性终止答案收集
_STRUCTURE_BOUNDARY_TYPES = {"chapter", "section", "heading", "subheading"}


def pair_questions_with_answers(
    records: List[Dict[str, object]],
    *,
    min_confidence: str = "medium",
) -> Tuple[List[FaqPair], List[FaqCandidateRow]]:
    """扫描已按源序排好的 records，一次性配对所有可信问题与答案。

    返回 ``(paired, unpaired_candidates)``：

    * ``paired``：置信度 ``>= min_confidence`` 且至少配到 1 条答案的 FAQ。
    * ``unpaired_candidates``：识别为候选但置信度不足或无答案，
      直接提供给 audit 序列化，**不在内部模拟第二次配对**。

    终止条件：

    * 下一道 high / medium 候选（同一 record 或后续 record）。
    * 结构边界（chapter / section / heading / subheading）。
    * records 末尾。
    """
    threshold = confidence_rank(min_confidence)

    # 第一遍：扫描所有候选（一次 detector 全跑完）
    candidate_positions: List[Tuple[int, QuestionCandidate]] = []
    for idx, rec in enumerate(records):
        cand = detect_question_candidate(rec)
        if cand.is_candidate:
            candidate_positions.append((idx, cand))

    paired: List[FaqPair] = []
    unpaired: List[FaqCandidateRow] = []
    used: set[int] = set()

    # 第二遍：按位置顺序配对
    for n, (idx, cand) in enumerate(candidate_positions):
        if idx in used:
            continue
        record_index_value = int(records[idx].get("record_index") or -1)
        if confidence_rank(cand.confidence) < threshold:
            unpaired.append(FaqCandidateRow(record_index_value, cand))
            continue

        # 答案终止点：下一个 high / medium 候选（不分等级，low 不构成边界）。
        # 这样段落问句（medium）就能自动终止上一道结构化（high）FAQ 的答案范围。
        answer_end = len(records)
        cur_rank = confidence_rank(cand.confidence)
        for m, (other_idx, other_cand) in enumerate(candidate_positions):
            if other_idx <= idx:
                continue
            # 仅当 cur_rank >= threshold 时才会进入此分支；这里直接看下一个 rank >= threshold
            if confidence_rank(other_cand.confidence) >= threshold:
                answer_end = other_idx
                break

        answer_indices: List[int] = []
        for j in range(idx + 1, min(answer_end, len(records))):
            rj = records[j]
            if str(rj.get("structure_type") or "") in _STRUCTURE_BOUNDARY_TYPES:
                break
            answer_indices.append(j)

        used.add(idx)
        for j in answer_indices:
            used.add(j)

        # 拼接答案文本
        answer_parts = [
            _read_text(records[j]) for j in answer_indices
            if _read_text(records[j])
        ]
        answer_text = "\n\n".join(answer_parts)

        record = records[idx]
        if not answer_text:
            unpaired.append(FaqCandidateRow(int(record.get("record_index") or -1), cand))
            continue

        paired.append(
            FaqPair(
                question_doc_id=str(record.get("doc_id") or ""),
                question_record_index=int(record.get("record_index") or -1),
                question_text=cand.question_text,
                raw_question_text=cand.raw_question_text,
                confidence=cand.confidence,
                answer_doc_ids=[
                    str(records[j].get("doc_id") or "") for j in answer_indices
                ],
                answer_record_indices=[
                    int(records[j].get("record_index") or -1)
                    for j in answer_indices
                ],
                answer_text=answer_text,
                number_style=cand.number_style,
            )
        )

    return paired, unpaired


# ---------------------------------------------------------------------------
# 暴露给 audit 的薄包装：带源 record_index 的候选行
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaqCandidateRow:
    """未配对候选的轻量记录：仅保留定位信息与 detector 输出，供 audit 序列化。"""

    record_index: int
    candidate: QuestionCandidate

    # 便捷读取，避免 audit 每次 ``row.candidate.confidence`` 的嵌套
    @property
    def confidence(self) -> str:
        return self.candidate.confidence

    @property
    def reason(self) -> str:
        return self.candidate.reason

    @property
    def number_style(self) -> str:
        return self.candidate.number_style
