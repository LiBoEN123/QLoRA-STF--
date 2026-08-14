"""Markdown extractor。"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import DocumentRecord, compute_doc_id
from ..utils import is_meaningful_text, normalize_text
from .base import BaseExtractor, ExtractionResult


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class MarkdownExtractor(BaseExtractor):
    source_type: str = "markdown"
    suffixes: tuple[str, ...] = (".md", ".markdown")

    def extract(
        self,
        path: Path,
        *,
        relative_path: str,
        file_sha256: str,
    ) -> ExtractionResult:
        text = self._read_text(path)
        if not is_meaningful_text(text):
            return ExtractionResult()

        record = DocumentRecord(
            doc_id=compute_doc_id(relative_path, text=text),
            source_type=self.source_type,
            source_path=relative_path,
            file_name=path.name,
            text=text,
            file_sha256=file_sha256,
            metadata={"first_heading": self._first_heading(text)},
        )
        return ExtractionResult(records=[record])

    @staticmethod
    def _read_text(path: Path) -> str:
        # Markdown 通常是 UTF-8；先剥掉 BOM，失败时回退到几种常见编码。
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _first_heading(text: str) -> str | None:
        for line in text.splitlines():
            m = _HEADING_RE.match(line.strip())
            if m:
                return m.group(2).strip()
        return None