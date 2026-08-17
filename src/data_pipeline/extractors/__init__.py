"""Phase 1 的 extractor 实现。

每个 extractor 负责将单个受支持类型的文件转换为 ``DocumentRecord`` 列表。
extractor 不应修改源文件，且保持简单，仅做基础的结构化抽取。
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
