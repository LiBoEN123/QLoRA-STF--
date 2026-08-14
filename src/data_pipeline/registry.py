"""Registry of extractors keyed by file suffix."""

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
    """Return the sorted list of supported file suffixes."""
    return sorted(EXTRACTORS.keys())


def get_extractor(path: Path) -> BaseExtractor | None:
    """Return the extractor for ``path`` or ``None`` if unsupported."""
    return EXTRACTORS.get(path.suffix.lower())
