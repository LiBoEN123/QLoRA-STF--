"""Plain text (.txt) extractor."""

from __future__ import annotations

from pathlib import Path

from ..models import DocumentRecord, compute_doc_id
from ..utils import is_meaningful_text, normalize_text
from .base import BaseExtractor, ExtractionResult


class TextExtractor(BaseExtractor):
    source_type: str = "text"
    suffixes: tuple[str, ...] = (".txt", ".log")

    def extract(
        self,
        path: Path,
        *,
        relative_path: str,
        file_sha256: str,
    ) -> ExtractionResult:
        text = self._read_text(path)
        text = normalize_text(text)
        if not is_meaningful_text(text):
            return ExtractionResult()
        record = DocumentRecord(
            doc_id=compute_doc_id(relative_path, text=text),
            source_type=self.source_type,
            source_path=relative_path,
            file_name=path.name,
            text=text,
            file_sha256=file_sha256,
            metadata={},
        )
        return ExtractionResult(records=[record])

    @staticmethod
    def _read_text(path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="replace")
