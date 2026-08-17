"""长文本安全切分 + 段落边界识别。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


# 中英文句子结束符（保守策略：成对出现的尾部标点不计入）
_SENTENCE_END_RE = re.compile(r"(?<=[。！？；!?;])")
# 逗号、顿号（中英文）
_COMMA_RE = re.compile(r"(?<=[，,、])")


@dataclass(frozen=True)
class SplitSegment:
    """一次切分产生的片段，附带是否硬切信息。"""

    text: str
    is_hard_split: bool


def _try_split_at(text: str, max_chars: int, boundary_re: re.Pattern) -> List[str]:
    """在 ``max_chars`` 之前的最近 boundary 处切分；找不到则返回整段。"""
    if len(text) <= max_chars:
        return [text]
    # 寻找 <= max_chars 位置内最后一个 boundary
    last_pos = -1
    for m in boundary_re.finditer(text, 0, max_chars):
        last_pos = m.end()
    if last_pos <= 0:
        return [text]
    return [text[:last_pos], text[last_pos:]]


def split_long_text(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int = 0,
) -> List[SplitSegment]:
    """对超长文本按层级优先级切分：段落 > 句子 > 逗号 > 硬切。

    * 输出片段总长度均 ``<= max_chars``。
    * 仅当 ``overlap_chars > 0`` 时，相邻片段之间会保留尾部 ``overlap_chars`` 字符。
    * 真正无边界可切时返回单段并标记 ``is_hard_split=True``。
    """
    if not text or len(text) <= max_chars:
        return [SplitSegment(text, False)]

    # 先按段落切：\n\n
    paragraphs = text.split("\n\n")
    pieces: List[str] = []
    buf = ""
    for p in paragraphs:
        if not p:
            continue
        # 单段已超长 → 递归切
        if len(p) > max_chars:
            # 先 flush 缓冲
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(_split_paragraph(p, max_chars))
            continue
        # 拼接后是否会超长
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf = buf + "\n\n" + p
        else:
            pieces.append(buf)
            buf = p
    if buf:
        pieces.append(buf)

    # 应用 overlap
    if overlap_chars <= 0 or len(pieces) <= 1:
        return [SplitSegment(p, False) for p in pieces]

    overlapped: List[str] = []
    for i, p in enumerate(pieces):
        if i == 0:
            overlapped.append(p)
            continue
        prev = pieces[i - 1]
        tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        merged = tail + p
        if len(merged) > max_chars + overlap_chars:
            # 防止合并后再次超长；裁剪 tail
            allowed = max(0, max_chars - len(p))
            tail = prev[-allowed:] if allowed > 0 else ""
            merged = tail + p
        overlapped.append(merged)

    return [SplitSegment(p, False) for p in overlapped]


def _split_paragraph(text: str, max_chars: int) -> List[str]:
    """对单个超长段落按句子 → 逗号 → 硬切顺序递归切分。"""
    if len(text) <= max_chars:
        return [text]

    pieces = _try_split_at(text, max_chars, _SENTENCE_END_RE)
    if len(pieces) > 1:
        result: List[str] = []
        for p in pieces:
            if len(p) > max_chars:
                result.extend(_split_paragraph(p, max_chars))
            else:
                result.append(p)
        return result

    pieces = _try_split_at(text, max_chars, _COMMA_RE)
    if len(pieces) > 1:
        result = []
        for p in pieces:
            if len(p) > max_chars:
                result.extend(_split_paragraph(p, max_chars))
            else:
                result.append(p)
        return result

    # 硬切
    return _hard_split(text, max_chars)


def _hard_split(text: str, max_chars: int) -> List[str]:
    """按字符数硬切，可能从词中间断开。"""
    if len(text) <= max_chars:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]