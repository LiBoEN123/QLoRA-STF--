"""所有 extractor 共用的 DocumentRecord 数据模型。"""

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
    """从源文档抽取出的标准化 record。

    一个物理文件可能产出多条 ``DocumentRecord``（例如每个 PDF 页一条、
    每个 Word 节一条、每个 Excel sheet 一条、或者每行 JSONL 一条）。
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
        """返回可 JSON 序列化的 dict 表示。"""
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
    """为一条 record 计算稳定、确定的 ID。

    相同输入始终得到相同 ``doc_id``，因此重复运行抽取不会产生重复记录。
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
    """把 record 序列化为一行 JSONL（UTF-8 安全）。"""
    return json.dumps(
        record.to_jsonable(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
