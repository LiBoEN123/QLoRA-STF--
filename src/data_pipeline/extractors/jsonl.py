"""JSON and JSONL extractors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import DocumentRecord, compute_doc_id
from ..utils import is_meaningful_text, normalize_text
from .base import BaseExtractor, ExtractionResult


_TEXT_FIELDS = ("text", "content", "body", "question", "answer", "title")
_TITLE_FIELDS = ("title",)


def _gather_text_fields(obj: dict[str, Any]) -> dict[str, str]:
    """Extract a permissive subset of text-like fields from a JSON object."""
    bits: dict[str, str] = {}
    for field in _TEXT_FIELDS:
        if field in obj and isinstance(obj[field], str):
            value = obj[field].strip()
            if value:
                bits[field] = value
    return bits


def _summarize_text(bits: dict[str, str]) -> str:
    """Combine extracted text fields into a single readable string."""
    if not bits:
        return ""
    if "question" in bits and "answer" in bits:
        return (
            f"question: {bits['question']}\n"
            f"answer: {bits['answer']}"
        )
    if "title" in bits and "content" in bits:
        return f"{bits['title']}\n\n{bits['content']}"
    # Fallback: join whatever we found in stable order.
    parts = [bits[k] for k in _TEXT_FIELDS if k in bits]
    return "\n\n".join(parts)


class JSONExtractor(BaseExtractor):
    source_type: str = "json"
    suffixes: tuple[str, ...] = (".json",)

    def extract(
        self,
        path: Path,
        *,
        relative_path: str,
        file_sha256: str,
    ) -> ExtractionResult:
        raw = self._read_text(path)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON: {exc}") from exc

        records: list[DocumentRecord] = []
        items = self._as_iterable(data)
        for index, entry in enumerate(items):
            if not isinstance(entry, dict):
                continue
            bits = _gather_text_fields(entry)
            text = _summarize_text(bits)
            text = normalize_text(text)
            if not is_meaningful_text(text):
                continue
            title = None
            for key in _TITLE_FIELDS:
                if isinstance(entry.get(key), str) and entry[key].strip():
                    title = entry[key].strip()
                    break
            record = DocumentRecord(
                doc_id=compute_doc_id(
                    relative_path,
                    record_index=index,
                    text=text,
                ),
                source_type=self.source_type,
                source_path=relative_path,
                file_name=path.name,
                title=title,
                record_index=index,
                text=text,
                file_sha256=file_sha256,
                metadata=self._safe_metadata(entry),
            )
            records.append(record)
        return ExtractionResult(records=records)

    @staticmethod
    def _as_iterable(data: Any) -> list[Any]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    @staticmethod
    def _safe_metadata(entry: dict[str, Any]) -> dict[str, Any]:
        """Keep top-level scalars as metadata, but never overwrite text."""
        meta: dict[str, Any] = {}
        for key, value in entry.items():
            if key in _TEXT_FIELDS:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                meta[key] = value
        return meta

    @staticmethod
    def _read_text(path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="replace")


class JSONLExtractor(BaseExtractor):
    source_type: str = "jsonl"
    suffixes: tuple[str, ...] = (".jsonl",)

    def extract(
        self,
        path: Path,
        *,
        relative_path: str,
        file_sha256: str,
    ) -> ExtractionResult:
        raw = self._read_text(path)
        records: list[DocumentRecord] = []
        warnings: list[str] = []
        for index, line in enumerate(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"invalid_jsonl_line:{index + 1}")
                continue
            if not isinstance(entry, dict):
                continue
            bits = _gather_text_fields(entry)
            text = _summarize_text(bits)
            text = normalize_text(text)
            if not is_meaningful_text(text):
                continue
            title = None
            for key in _TITLE_FIELDS:
                if isinstance(entry.get(key), str) and entry[key].strip():
                    title = entry[key].strip()
                    break
            record = DocumentRecord(
                doc_id=compute_doc_id(
                    relative_path,
                    record_index=index,
                    text=text,
                ),
                source_type=self.source_type,
                source_path=relative_path,
                file_name=path.name,
                title=title,
                record_index=index,
                text=text,
                file_sha256=file_sha256,
                metadata=JSONExtractor._safe_metadata(entry),
            )
            records.append(record)
        return ExtractionResult(records=records, warnings=warnings)

    @staticmethod
    def _read_text(path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="replace")
