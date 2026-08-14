"""Unified DocumentRecord data model used by all extractors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_SOURCE_TYPES: tuple[str, ...] = (
    "pdf",
    "docx",
    "markdown",
    "text",
    "html",
    "excel",
    "csv",
    "json",
    "jsonl",
)


@dataclass
class DocumentRecord:
    """A normalized record extracted from a source document.

    One physical file may produce multiple ``DocumentRecord`` instances
    (for example one per PDF page, one per Word section, one per Excel sheet,
    or one per JSON/JSONL line).
    """

    doc_id: str
    source_type: str
    source_path: str
    file_name: str
    title: str | None = None
    section: str | None = None
    page_number: int | None = None
    sheet_name: str | None = None
    record_index: int | None = None
    text: str = ""
    file_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)


def compute_doc_id(
    source_path: str,
    *,
    page_number: int | None = None,
    sheet_name: str | None = None,
    record_index: int | None = None,
    section: str | None = None,
    text: str = "",
    prefix: str = "doc",
) -> str:
    """Compute a stable, deterministic ID for a document record.

    The same input always yields the same ``doc_id`` so that re-running
    extraction does not produce duplicate records.
    """
    parts: list[str] = [source_path]
    if page_number is not None:
        parts.append(f"page={page_number}")
    if sheet_name is not None:
        parts.append(f"sheet={sheet_name}")
    if record_index is not None:
        parts.append(f"rec={record_index}")
    if section is not None:
        parts.append(f"section={section}")
    parts.append(text)
    payload = "\n".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"{prefix}_{digest[:24]}"


def serialize_record(record: DocumentRecord) -> str:
    """Serialize a record to a single JSONl line (UTF-8 safe)."""
    return json.dumps(
        record.to_jsonable(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
