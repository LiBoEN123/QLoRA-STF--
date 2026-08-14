"""Phase 2 的文本规范化工具。

本模块中的所有函数都是纯函数、确定性的、无副作用的，
不会调用 LLM、Embedding 或任何外部 NLP 库。

模块暴露的是小的可组合原语，pipeline 可以据此记录每个 record 实际执行的清洗动作。
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 必须无条件剔除的不可见字符。
_INVISIBLE_CHARS = {
    "\u200b",  # 零宽空格
    "\u200c",  # 零宽不连字
    "\u200d",  # 零宽连字
    "\ufeff",  # 零宽不换行空格 / BOM
    "\u00ad",  # 软连字符
    "\u2060",  # 词连接符
    "\u2028",  # 行分隔符
    "\u2029",  # 段分隔符
}

#: 不换行空格、细空格、表意空格等 → 普通空格。
_BLANK_NORMALIZATIONS = {
    "\u00a0": " ",  # 不换行空格
    "\u2009": " ",  # 细空格
    "\u2007": " ",  # 数字空格
    "\u2002": " ",  # 半角空格（en）
    "\u2003": " ",  # 全角空格（em）
    "\u3000": " ",  # 表意全角空格
}

#: 中文里表示句子结束的标点。
_SENTENCE_END_PUNCT = set("。！？；：…")

#: 所有不应与下一行合并的结构标记。
_HEADING_START_RE = re.compile(
    r"^\s*(?:"
    r"第[一二三四五六七八九十百千零〇两]+[章节]"  # 第X章 / 第X节
    r"|[一二三四五六七八九十]+、"  # 一、 二、
    r"|（[一二三四五六七八九十]+）"  # （一）（二）
    r"|\(\d+\)"  # (1)(2)
    r"|\d+[、.]"  # 1. 2、 3.
    r"|（\d+）"  # （1）（2）
    r"|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]"  # 圈数字
    r"|[•●○■□▪▫]"  # 项目符号
    r"|[一二三四五六七八九十]+(?=\s)"  # 单独成行的“一 二 三”（罕见标题）
    r")"
)


# ---------------------------------------------------------------------------
# Unicode 与不可见字符处理
# ---------------------------------------------------------------------------


_CJK_DETECT_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef]")


def _restore_chinese_punctuation(text: str) -> str:
    """在 CJK 上下文中把 ASCII 标点还原为全角。

    NFKC 会把 ``，`` → ``,``、``。`` → ``.`` 等，破坏中文文档的视觉风格。
    在 NFKC 之后，只要文本中存在 CJK 字符，就把对应的半角标点还原为全角。
    英文片段（URL、代码）里的真实 ASCII 标点会被 CJK 探测守卫保留。
    """
    if not _CJK_DETECT_RE.search(text):
        return text
    mapping = {
        ",": "，",
        "!": "！",
        "?": "？",
        ";": "；",
        ":": "：",
        "(": "（",
        ")": "）",
    }
    for src, dst in mapping.items():
        text = text.replace(src, dst)
    return text


def normalize_unicode(text: str) -> Tuple[str, bool]:
    """执行 NFKC 规范化，还原中文标点，剔除不可见字符。

    返回新文本与一个布尔标志，表示实际是否发生了变化（便于审计 / cleaning 元数据）。
    """
    if not text:
        return "", False
    before = text
    text = unicodedata.normalize("NFKC", text)

    # 剔除 BOM、零宽空格、零宽不换行空格、软连字符、词连接符、行/段分隔符。
    text = "".join(ch for ch in text if ch not in _INVISIBLE_CHARS)

    # 剔除其余 C0/C1 控制字符，但保留 \t \n \r。
    cleaned: List[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in "\t\n\r":
            continue
        cleaned.append(ch)
    text = "".join(cleaned)

    text = _restore_chinese_punctuation(text)
    return text, text != before


# ---------------------------------------------------------------------------
# 空白规范化
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> Tuple[str, bool]:
    """规范化空白，同时保留段落分隔。

    - 将 NBSP 等不可见空格转为普通空格。
    - 折叠行内的连续空格。
    - 去掉每行首尾空白。
    - 将单独的 ``\\r`` / ``\\r\\n`` 转为 ``\\n``。
    - 把连续 3 行及以上的空行折叠为 2 行。
    """
    if not text:
        return "", False
    before = text

    for src, dst in _BLANK_NORMALIZATIONS.items():
        text = text.replace(src, dst)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    new_lines: List[str] = []
    for line in text.split("\n"):
        # 折叠行内的连续空格，但暂不修改 tab。
        line = re.sub(r"[ \u3000]{2,}", " ", line)
        new_lines.append(line.strip())
    text = "\n".join(new_lines)

    # 把 3 行及以上的连续空行折叠到 2 行。
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    text = text.strip("\n")
    return text, text != before


# ---------------------------------------------------------------------------
# 标点规范化
# ---------------------------------------------------------------------------


_PUNCT_RUN_RE = re.compile(r"([。！？；，,])\1{2,}")


def normalize_punctuation(text: str) -> Tuple[str, bool]:
    """折叠意外重复的句末标点。

    规则：
    - ``。。。。。`` → ``。``（不做其它标点剥离）
    - ``，，，，``   → ``，``
    - ``！！！``     → ``！``
    - ``？？？``     → ``？``
    - ``;;;;``       → ``；``
    - 保留 ``……``、``——``、``?!``、``!?``、混合标点原样。
    """
    if not text:
        return "", False
    before = text

    def _collapse(match: re.Match[str]) -> str:
        return match.group(1)

    text = _PUNCT_RUN_RE.sub(_collapse, text)
    return text, text != before


# ---------------------------------------------------------------------------
# 中文断行修复
# ---------------------------------------------------------------------------


def _is_structural_line(line: str) -> bool:
    """如果该行不应与下一行合并则返回 True。"""
    if not line.strip():
        return False
    if _HEADING_START_RE.match(line):
        return True
    stripped = line.lstrip()
    # Markdown 标题。
    if stripped.startswith("#"):
        return True
    # 需要保留的引用 / 项目符号前缀。
    if stripped.startswith((">", "|", "-", "*")):
        return True
    return False


def _ends_sentence(line: str) -> bool:
    """如果该行以句末标点结尾则返回 True。"""
    stripped = line.rstrip()
    if not stripped:
        return False
    last = stripped[-1]
    return last in _SENTENCE_END_PUNCT


def repair_line_breaks(text: str) -> Tuple[str, bool]:
    """把被换行拆开的中文句子合并回单行。

    保守规则（详见规格说明）：
    - 当「当前行不以句末标点结尾」且「下一行非空 / 非结构行」时合并。
    - 保留所有空行（段落分隔）。
    - 保留所有结构标题（章、节、编号列表）。
    - 永不从空当前行向下合并——空行就是段落分隔。
    """
    if not text:
        return "", False
    lines = text.split("\n")
    if len(lines) < 2:
        return text, False

    before = text
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        # Look ahead and merge while conditions allow.
        while i + 1 < len(lines):
            nxt = lines[i + 1]
            # Don't merge across a blank line.
            if not nxt.strip():
                break
            # Never merge from an empty current line.
            if not cur.strip():
                break
            # Don't merge if the current line is itself structural.
            if _is_structural_line(cur):
                break
            # Don't merge if the next line is structural.
            if _is_structural_line(nxt):
                break
            # Don't merge if the current line ends in terminal punctuation.
            if _ends_sentence(cur):
                break
            cur = cur.rstrip() + nxt.lstrip()
            i += 1
        out.append(cur)
        i += 1

    text = "\n".join(out)
    return text, text != before


# ---------------------------------------------------------------------------
# 组合规范化
# ---------------------------------------------------------------------------


def normalize_full(text: str) -> Tuple[str, List[str]]:
    """运行完整的确定性规范化流水线。

    返回 ``(clean_text, actions)``，其中 ``actions`` 是机器可读的步骤名列表，
    可直接写入 cleaning 元数据。顺序：unicode → whitespace → line-break repair → punctuation。
    """
    actions: List[str] = []
    if not text:
        return "", actions

    text, changed = normalize_unicode(text)
    if changed:
        actions.append("unicode_nfkc")

    text, changed = normalize_whitespace(text)
    if changed:
        actions.append("collapse_spaces")

    text, changed = repair_line_breaks(text)
    if changed:
        actions.append("repair_line_break")

    text, changed = normalize_punctuation(text)
    if changed:
        actions.append("collapse_punctuation")

    return text, actions