"""Base class and result type for document extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DocumentRecord


@dataclass
class ExtractionResult:
    """The outcome of extracting a single file."""

    records: list["DocumentRecord"] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""

    #: Source type label stored in ``DocumentRecord.source_type``.
    source_type: str = ""

    #: Lower-case file suffixes this extractor handles (e.g. ``.pdf``).
    suffixes: tuple[str, ...] = ()

    @abstractmethod
    def extract(
        self,
        path: Path,
        *,
        relative_path: str,
        file_sha256: str,
    ) -> ExtractionResult:
        """Extract structured records from ``path``.

        Parameters
        ----------
        path:
            Absolute path to the source file.
        relative_path:
            Project-relative path used for stable ``doc_id`` generation
            and for the ``source_path`` field.
        file_sha256:
            Pre-computed SHA256 digest of the file contents.
        """

    @classmethod
    def handles(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.suffixes
