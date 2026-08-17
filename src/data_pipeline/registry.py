"""按文件后缀注册的 extractor 表。"""

from __future__ import annotations

from pathlib import Path

from .extractors import (
    CSVExtractor,
    ExcelExtractor,
    HTMLExtractor,
    JSONExtractor,
    JSONLExtractor,
    MarkdownExtractor,
    PDFExtractor,
    TextExtractor,
    WordExtractor,
)
from .extractors.base import BaseExtractor


EXTRACTORS: dict[str, BaseExtractor] = {}


def _register(extractor: BaseExtractor) -> None:
    """将 ``extractor`` 注册到 ``EXTRACTORS`` 表，按其声明的后缀建立索引。"""
    for suffix in extractor.suffixes:
        EXTRACTORS.setdefault(suffix.lower(), extractor)


_register(PDFExtractor())
_register(WordExtractor())
_register(MarkdownExtractor())
_register(TextExtractor())
_register(HTMLExtractor())
_register(ExcelExtractor())
_register(CSVExtractor())
_register(JSONExtractor())
_register(JSONLExtractor())


def supported_suffixes() -> list[str]:
    """返回排序后的所有支持文件后缀。"""
    return sorted(EXTRACTORS.keys())


def get_extractor(path: Path) -> BaseExtractor | None:
    """返回 ``path`` 对应的 extractor；不支持则返回 None。"""
    return EXTRACTORS.get(path.suffix.lower())
