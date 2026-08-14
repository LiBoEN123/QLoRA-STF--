"""Phase 2 清洗中的结构识别。

给定一段（已经规范化过的）文本，把它分类为以下之一：

* ``chapter``     — 例如 ``第一章 总则``
* ``section``     — 例如 ``第一节 基层党组织``
* ``heading``     — 例如 ``一、基层党组织的设置``
* ``subheading``  — 例如 ``（一）基本原则``
* ``list_item``   — 例如 ``1. 坚持党的领导`` / ``（1）第一项`` / ``① ...``
* ``table``       — metadata 中已标记为表格的 record
* ``paragraph``   — 默认
* ``unknown``     — 无法分类

检测器完全是基于规则的，只依赖标准库。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千零〇两]+章[\s　]*")
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百千零〇两]+节[\s　]*")
CN_NUM_HEADING_RE = re.compile(r"^[一二三四五六七八九十]+[、,]")
CN_NUM_SUBHEADING_RE = re.compile(r"^[（(][一二三四五六七八九十]+[）)]")
NUMBERED_LIST_DOT_RE = re.compile(r"^\d+[\.\．]\s*")
NUMBERED_LIST_DUN_RE = re.compile(r"^\d+[、,]\s*")
NUMBERED_LIST_PAREN_RE = re.compile(r"^[（(]\d+[）)]")
CIRCLED_RE = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]")
BARE_HEADING_RE = re.compile(r"^第[一二三四五六七八九十百千零〇两]+[章节][^一二三四五六七八九十]")


STRUCTURE_LEVELS: dict[str, int] = {
    "chapter": 1,
    "section": 2,
    "heading": 3,
    "subheading": 4,
    "list_item": 5,
    "paragraph": 6,
    "table": 6,
    "unknown": 99,
}


@dataclass(frozen=True)
class StructureInfo:
    """单条 record 文本的结构分类结果。"""

    structure_type: str
    structure_level: int
    detected_section: Optional[str] = None
    detected_heading: Optional[str] = None


def _is_table_record(text: str, metadata: dict | None) -> bool:
    if metadata and metadata.get("kind") == "table":
        return True
    # Markdown 表格启发式：至少两行以 | 分隔的单元格。
    if text.count("|") >= 4 and "\n" in text and text.lstrip().startswith("|"):
        return True
    return False


def detect_structure(text: str, metadata: dict | None = None) -> StructureInfo:
    """对单段文本进行结构分类。

    返回的 ``detected_section`` / ``detected_heading`` 仅在当前 record 本身是
    结构标记时填充；下游代码负责把它们向下传播到后续段落。
    """
    if not text or not text.strip():
        return StructureInfo("unknown", STRUCTURE_LEVELS["unknown"])

    if _is_table_record(text, metadata):
        return StructureInfo("table", STRUCTURE_LEVELS["table"])

    stripped = text.lstrip()
    head = stripped.splitlines()[0] if stripped else ""

    if CHAPTER_RE.match(stripped):
        return StructureInfo(
            "chapter",
            STRUCTURE_LEVELS["chapter"],
            detected_section=stripped.split("\n", 1)[0].strip(),
            detected_heading=stripped.split("\n", 1)[0].strip(),
        )
    if SECTION_RE.match(stripped):
        return StructureInfo(
            "section",
            STRUCTURE_LEVELS["section"],
            detected_section=stripped.split("\n", 1)[0].strip(),
            detected_heading=stripped.split("\n", 1)[0].strip(),
        )
    if CN_NUM_HEADING_RE.match(stripped):
        return StructureInfo(
            "heading",
            STRUCTURE_LEVELS["heading"],
            detected_heading=stripped.split("\n", 1)[0].strip(),
        )
    if CN_NUM_SUBHEADING_RE.match(stripped):
        return StructureInfo(
            "subheading",
            STRUCTURE_LEVELS["subheading"],
            detected_heading=stripped.split("\n", 1)[0].strip(),
        )
    if (
        NUMBERED_LIST_PAREN_RE.match(stripped)
        or CIRCLED_RE.match(stripped)
        or NUMBERED_LIST_DOT_RE.match(stripped)
        or NUMBERED_LIST_DUN_RE.match(stripped)
    ):
        return StructureInfo("list_item", STRUCTURE_LEVELS["list_item"])

    # Markdown headings (#, ##, ###)
    if head.startswith("#"):
        level = len(head) - len(head.lstrip("#"))
        if 1 <= level <= 6:
            type_name = {1: "chapter", 2: "section", 3: "heading",
                         4: "heading", 5: "subheading", 6: "subheading"}.get(level, "heading")
            return StructureInfo(
                type_name,
                STRUCTURE_LEVELS[type_name],
                detected_heading=head.strip("# ").strip(),
            )

    return StructureInfo("paragraph", STRUCTURE_LEVELS["paragraph"])


# ---------------------------------------------------------------------------
# section 传播辅助
# ---------------------------------------------------------------------------


@dataclass
class SectionContext:
    """按 source_path 维护当前的 chapter / section / heading。"""

    current_chapter: Optional[str] = None
    current_section: Optional[str] = None
    current_heading: Optional[str] = None

    def update(self, info: StructureInfo) -> None:
        if info.structure_type == "chapter":
            self.current_chapter = info.detected_section or info.detected_heading
            self.current_section = None
            self.current_heading = info.detected_heading
        elif info.structure_type == "section":
            self.current_section = info.detected_section or info.detected_heading
            self.current_heading = info.detected_heading
        elif info.structure_type in ("heading", "subheading"):
            self.current_heading = info.detected_heading
        # list_item / paragraph / table / unknown → 不变更

    def detected_section(self) -> Optional[str]:
        if self.current_chapter and self.current_section:
            return f"{self.current_chapter} / {self.current_section}"
        if self.current_chapter:
            return self.current_chapter
        if self.current_section:
            return self.current_section
        return self.current_heading