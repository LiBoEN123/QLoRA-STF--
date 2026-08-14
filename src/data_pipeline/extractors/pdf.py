"""PDF extractor using PyMuPDF (fitz)."""

from __future__ import annotations

from pathlib import Path

from ..models import DocumentRecord, compute_doc_id
from ..utils import is_meaningful_text, normalize_text
from .base import BaseExtractor, ExtractionResult


class PDFExtractor(BaseExtractor):
    source_type: str = "pdf"
    suffixes: tuple[str, ...] = (".pdf",)

    def extract(
        self,
        path: Path,
        *,
        relative_path: str,
        file_sha256: str,
    ) -> ExtractionResult:
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "PyMuPDF (fitz) is required for PDF extraction. "
                "Install it via `pip install PyMuPDF`."
            ) from exc

        result = ExtractionResult()
        scanned_pages: list[int] = []

        with fitz.open(path) as doc:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                # ``get_text("text")`` returns the textual layer only.
                raw_text = page.get_text("text") or ""
                text = normalize_text(raw_text)
                page_number = page_index + 1  # 1-based for humans

                if not is_meaningful_text(text):
                    scanned_pages.append(page_number)
                    # Still emit a placeholder record so that downstream
                    # tooling can see the page exists, but mark it as
                    # possibly scanned via metadata.
                    record = DocumentRecord(
                        doc_id=compute_doc_id(
                            relative_path,
                            page_number=page_number,
                            text="",
                        ),
                        source_type=self.source_type,
                        source_path=relative_path,
                        file_name=path.name,
                        page_number=page_number,
                        text="",
                        file_sha256=file_sha256,
                        metadata={"possible_scanned_pdf": True},
                    )
                    result.records.append(record)
                    continue

                record = DocumentRecord(
                    doc_id=compute_doc_id(
                        relative_path,
                        page_number=page_number,
                        text=text,
                    ),
                    source_type=self.source_type,
                    source_path=relative_path,
                    file_name=path.name,
                    page_number=page_number,
                    text=text,
                    file_sha256=file_sha256,
                    metadata={},
                )
                result.records.append(record)

        if scanned_pages:
            result.warnings.append("possible_scanned_pdf")
            # Attach the affected pages per-record for downstream visibility.
            for record in result.records:
                if (
                    record.page_number is not None
                    and record.page_number in scanned_pages
                    and record.metadata.get("possible_scanned_pdf") is not True
                ):
                    record.metadata["possible_scanned_pdf"] = (
                        record.page_number in scanned_pages
                    )

        return result
