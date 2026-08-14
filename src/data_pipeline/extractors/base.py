"""文档 extractor 的基类与结果类型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DocumentRecord


@dataclass
class ExtractionResult:
    """单个文件抽取的结果。"""

    records: list["DocumentRecord"] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BaseExtractor(ABC):
    """所有 extractor 的抽象基类。"""

    #: 存入 ``DocumentRecord.source_type`` 的类型标签。
    source_type: str = ""

    #: 该 extractor 处理的小写文件后缀（如 ``.pdf``）。
    suffixes: tuple[str, ...] = ()

    @abstractmethod
    def extract(
        self,
        path: Path,
        *,
        relative_path: str,
        file_sha256: str,
    ) -> ExtractionResult:
        """从 ``path`` 抽取结构化 records。

        Parameters
        ----------
        path:
            源文件的绝对路径。
        relative_path:
            项目内相对路径，用于稳定生成 ``doc_id`` 与填充 ``source_path``。
        file_sha256:
            文件内容的预计算 SHA256 摘要。
        """

    @classmethod
    def handles(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.suffixes
