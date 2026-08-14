"""Phase 2 清洗后的 record 质量评估。

所有启发式阈值都是模块级常量，方便调参而无需修改调用点。本模块不调用任何 LLM。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 可调阈值
# ---------------------------------------------------------------------------

MIN_TEXT_LENGTH = 5
MAX_TEXT_LENGTH = 20_000
MAX_SYMBOL_RATIO = 0.5           # 高于此值视为 possible_garbled
GARBLED_RUN_RATIO = 0.4          # 最长连续相同字符占比阈值
HF_MAX_LENGTH = 50               # 仅把短文本作为页眉页脚候选
HF_MIN_REPEATS = 5               # 同一 source 至少出现 N 次
HF_MIN_RATIO = 0.30              # 至少占同一 source 的 30%


# ---------------------------------------------------------------------------
# 字符类别辅助函数
# ---------------------------------------------------------------------------


_CJK_RE = re.compile(
    "["
    "\u4e00-\u9fff"        # CJK 统一汉字
    "\u3400-\u4dbf"        # CJK 扩展 A
    "\uf900-\ufaff"        # CJK 兼容汉字
    "\U00020000-\U0002a6df"  # CJK 扩展 B
    "\U0002a700-\U0002ebef"  # CJK 扩展 C/D/E/F
    "\u3000-\u303f"        # CJK 符号与标点（日文假名等）
    "\uff00-\uffef"        # 半角及全角形式
    "]"
)

#: 计入 ``symbol_ratio`` 的标点 / 符号。我们特意不把中文常用句末标点计入——
#: 这些是正文内容，不是噪声。
_NOISE_PUNCT_RE = re.compile(
    r"["
    r"!-/:-@\[-`{-~"  # ASCII 标点
    r"‘’“”"          # 弯引号
    r"#\$%&\*\+<=>\?\\\^_\|~"  # 其它 ASCII 符号
    r"•·●○■□▪▫★☆"
    r"]"
)


def chinese_char_count(text: str) -> int:
    """统计 ``text`` 中的 CJK + 全角字符数。"""
    return len(_CJK_RE.findall(text))


def symbol_count(text: str) -> int:
    """统计「噪声」标点 / 符号字符数。"""
    return len(_NOISE_PUNCT_RE.findall(text))


def digit_count(text: str) -> int:
    """统计 ``text`` 中的 ASCII 数字与全角数字。"""
    return sum(1 for ch in text if ch.isdigit())


# ---------------------------------------------------------------------------
# 质量评估
# ---------------------------------------------------------------------------


@dataclass
class QualityConfig:
    """质量评估的可配置阈值。"""

    min_text_length: int = MIN_TEXT_LENGTH
    max_text_length: int = MAX_TEXT_LENGTH
    max_symbol_ratio: float = MAX_SYMBOL_RATIO
    garbled_run_ratio: float = GARBLED_RUN_RATIO


@dataclass
class QualityReport:
    """单条 record 的质量指标与告警标记。"""

    char_count: int = 0
    non_whitespace_char_count: int = 0
    chinese_char_ratio: float = 0.0
    symbol_ratio: float = 0.0
    digit_ratio: float = 0.0
    line_count: int = 0
    is_empty: bool = False
    is_too_short: bool = False
    is_too_long: bool = False
    possible_garbled: bool = False
    possible_header_footer: bool = False
    longest_run_ratio: float = 0.0
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "char_count": self.char_count,
            "non_whitespace_char_count": self.non_whitespace_char_count,
            "chinese_char_ratio": round(self.chinese_char_ratio, 4),
            "symbol_ratio": round(self.symbol_ratio, 4),
            "digit_ratio": round(self.digit_ratio, 4),
            "line_count": self.line_count,
            "is_empty": self.is_empty,
            "is_too_short": self.is_too_short,
            "is_too_long": self.is_too_long,
            "possible_garbled": self.possible_garbled,
            "possible_header_footer": self.possible_header_footer,
            "longest_run_ratio": round(self.longest_run_ratio, 4),
            "flags": list(self.flags),
        }


def _longest_run(text: str) -> int:
    """``text`` 中最长连续相同字符的长度。"""
    if not text:
        return 0
    longest = 1
    cur = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 1
            prev = ch
    return longest


def evaluate_quality(
    text: str,
    *,
    config: QualityConfig | None = None,
    exempt_short: bool = False,
) -> QualityReport:
    """计算 ``text`` 的质量报告。

    ``exempt_short`` 由 pipeline 在 record 本身就是结构标题 / 列表项 / 表格时设置，
    这样 ``第一章 总则`` 这类短标题不会被仅因长度过短而拒绝。
    """
    cfg = config or QualityConfig()
    report = QualityReport()

    if not text or not text.strip():
        report.is_empty = True
        report.flags.append("empty_after_cleaning")
        return report

    report.char_count = len(text)
    report.non_whitespace_char_count = len(text.replace(" ", "").replace("\t", "").replace("\n", ""))
    report.line_count = text.count("\n") + 1
    report.chinese_char_ratio = chinese_char_count(text) / report.char_count
    report.symbol_ratio = symbol_count(text) / report.char_count
    report.digit_ratio = digit_count(text) / report.char_count
    report.longest_run_ratio = (
        _longest_run(text) / report.char_count if report.char_count else 0.0
    )

    if not exempt_short and report.char_count < cfg.min_text_length:
        report.is_too_short = True
        report.flags.append("too_short")
    if report.char_count > cfg.max_text_length:
        report.is_too_long = True
        report.flags.append("too_long")
    if report.symbol_ratio > cfg.max_symbol_ratio:
        report.possible_garbled = True
        report.flags.append("symbol_ratio_too_high")
    if (
        report.longest_run_ratio >= cfg.garbled_run_ratio
        and report.char_count >= 8
    ):
        report.possible_garbled = True
        report.flags.append("possible_garbled")

    return report


# ---------------------------------------------------------------------------
# 页眉页脚检测（按 source 文档）
# ---------------------------------------------------------------------------


@dataclass
class HeaderFooterMatch:
    text: str
    count: int
    ratio: float


def detect_header_footer_candidates(
    clean_texts: List[str],
    *,
    max_length: int = HF_MAX_LENGTH,
    min_repeats: int = HF_MIN_REPEATS,
    min_ratio: float = HF_MIN_RATIO,
) -> Dict[str, HeaderFooterMatch]:
    """识别同一文档内像页眉 / 页脚的文本。

    保守——只返回短且高频出现的精确文本。较长或中等频率的文本不会被标记，
    因为规格提醒要避免误杀 ``中国共产党`` 这类合法的常用短语。
    """
    if not clean_texts:
        return {}

    from collections import Counter

    counter: Counter[str] = Counter()
    for t in clean_texts:
        normalized = t.strip()
        if 1 <= len(normalized) <= max_length:
            counter[normalized] += 1

    total = len(clean_texts)
    matches: Dict[str, HeaderFooterMatch] = {}
    for text, count in counter.items():
        if count < min_repeats:
            continue
        ratio = count / total
        if ratio < min_ratio:
            continue
        matches[text] = HeaderFooterMatch(text=text, count=count, ratio=ratio)
    return matches