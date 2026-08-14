"""使用 BeautifulSoup4 的 HTML extractor。"""

from __future__ import annotations

from pathlib import Path

from ..models import DocumentRecord, compute_doc_id
from ..utils import is_meaningful_text, normalize_text
from .base import BaseExtractor, ExtractionResult


class HTMLExtractor(BaseExtractor):
    source_type: str = "html"
    suffixes: tuple[str, ...] = (".html", ".htm")

    def extract(
        self,
        path: Path,
        *,
        relative_path: str,
        file_sha256: str,
    ) -> ExtractionResult:
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "beautifulsoup4 is required for HTML extraction. "
                "Install it via `pip install beautifulsoup4`."
            ) from exc

        raw_html = self._read_text(path)
        soup = BeautifulSoup(raw_html, "html.parser")

        # 去掉下游文本永远用不到的噪声标签。
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = None
        if soup.title and soup.title.string:
            title = normalize_text(soup.title.string) or None

        body = soup.body or soup
        text = normalize_text(body.get_text(separator="\n"))
        if not is_meaningful_text(text):
            return ExtractionResult()

        record = DocumentRecord(
            doc_id=compute_doc_id(relative_path, text=text, section=title),
            source_type=self.source_type,
            source_path=relative_path,
            file_name=path.name,
            title=title,
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