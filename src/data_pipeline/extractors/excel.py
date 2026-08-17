"""Excel 与 CSV extractor。"""

from __future__ import annotations

import csv
from pathlib import Path

from ..models import DocumentRecord, compute_doc_id
from ..utils import is_meaningful_text, normalize_text
from .base import BaseExtractor, ExtractionResult


def _format_sheet_as_table(rows: list[list[str]]) -> str:
    """把二维单元格字符串列表渲染为 Markdown 风格的表格。

    完全为空的行 / 列会被跳过，以减少噪声且不改变保留单元格的含义。
    """
    if not rows:
        return ""
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    # 丢掉整列都为空白的列。
    keep_cols: list[int] = []
    for c in range(width):
        if any(rows[r][c].strip() for r in range(len(rows))):
            keep_cols.append(c)
    if not keep_cols:
        return ""
    rows = [[r[c] for c in keep_cols] for r in rows]

    widths = [max(len(rows[r][c].strip()) for r in range(len(rows))) for c in range(len(keep_cols))]
    lines: list[str] = []
    for r_idx, row in enumerate(rows):
        cells = [row[c].strip().ljust(widths[c]) for c in range(len(keep_cols))]
        lines.append("| " + " | ".join(cells) + " |")
        if r_idx == 0:
            lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(lines)


class ExcelExtractor(BaseExtractor):
    """基于 openpyxl 的 Excel (.xlsx / .xlsm) 文件抽取器。

    每个 sheet 渲染成 Markdown 表格字符串并作为一条 ``DocumentRecord`` 输出。
    """

    source_type: str = "excel"
    suffixes: tuple[str, ...] = (".xlsx", ".xlsm")

    def extract(
        self,
        path: Path,
        *,
        relative_path: str,
        file_sha256: str,
    ) -> ExtractionResult:
        try:
            from openpyxl import load_workbook  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "openpyxl is required for Excel extraction. "
                "Install it via `pip install openpyxl`."
            ) from exc

        result = ExtractionResult()
        # ``data_only=True`` 让我们拿到公式的计算结果而非公式本身。
        workbook = load_workbook(filename=str(path), data_only=True, read_only=True)
        try:
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                rows: list[list[str]] = []
                for row in sheet.iter_rows(values_only=True):
                    rows.append(["" if v is None else str(v) for v in row])
                rendered = _format_sheet_as_table(rows)
                rendered = normalize_text(rendered)
                if not is_meaningful_text(rendered):
                    continue
                record = DocumentRecord(
                    doc_id=compute_doc_id(
                        relative_path,
                        sheet_name=sheet_name,
                        text=rendered,
                    ),
                    source_type=self.source_type,
                    source_path=relative_path,
                    file_name=path.name,
                    sheet_name=sheet_name,
                    text=rendered,
                    file_sha256=file_sha256,
                    metadata={"rows": len(rows)},
                )
                result.records.append(record)
        finally:
            workbook.close()

        return result


class CSVExtractor(BaseExtractor):
    """CSV (.csv) 文件抽取器。

    自动嗅探分隔符，将整个 CSV 渲染为 Markdown 表格字符串并作为一条
    ``DocumentRecord`` 输出。
    """

    source_type: str = "csv"
    suffixes: tuple[str, ...] = (".csv",)

    def extract(
        self,
        path: Path,
        *,
        relative_path: str,
        file_sha256: str,
    ) -> ExtractionResult:
        # 嗅探编码；检测失败时回退到 UTF-8。
        text = self._read_text(path)
        rows = self._parse_csv(text)
        rendered = _format_sheet_as_table(rows)
        rendered = normalize_text(rendered)
        if not is_meaningful_text(rendered):
            return ExtractionResult()
        record = DocumentRecord(
            doc_id=compute_doc_id(
                relative_path,
                text=rendered,
                sheet_name=path.stem,
            ),
            source_type=self.source_type,
            source_path=relative_path,
            file_name=path.name,
            sheet_name=path.stem,
            text=rendered,
            file_sha256=file_sha256,
            metadata={"rows": len(rows)},
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

    @staticmethod
    def _parse_csv(text: str) -> list[list[str]]:
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(text.splitlines(), dialect)
        return [row for row in reader]
