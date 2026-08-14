"""Extractor implementations for Phase 1.

Each extractor is responsible for converting a single file of a supported
type into a list of ``DocumentRecord`` instances. Extractors MUST NOT
modify the source file. They are kept deliberately simple and only do
basic structural extraction.
"""

from .base import BaseExtractor, ExtractionResult
from .pdf import PDFExtractor
from .word import WordExtractor
from .markdown import MarkdownExtractor
from .text import TextExtractor
from .html import HTMLExtractor
from .excel import ExcelExtractor, CSVExtractor
from .jsonl import JSONExtractor, JSONLExtractor

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "PDFExtractor",
    "WordExtractor",
    "MarkdownExtractor",
    "TextExtractor",
    "HTMLExtractor",
    "ExcelExtractor",
    "CSVExtractor",
    "JSONExtractor",
    "JSONLExtractor",
]
