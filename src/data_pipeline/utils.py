"""Shared utilities for the document extraction pipeline."""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from pathlib import Path
from typing import Iterable

LOGGER_NAME = "data_pipeline"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger for the data pipeline."""
    return logging.getLogger(name or LOGGER_NAME)


def to_relative_path(path: Path, root: Path) -> str:
    """Return a POSIX-style path relative to ``root`` if possible.

    Falls back to the absolute path if ``path`` is not under ``root``.
    The return value uses forward slashes so output is portable across
    Windows / Linux / WSL.
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def calculate_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA256 hex digest of a file's contents."""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def normalize_text(text: str) -> str:
    """Apply minimal, lossless text normalization.

    - Unicode ``NFKC`` normalization.
    - Translate stray carriage returns to newlines.
    - Strip trailing whitespace on each line.
    - Collapse runs of more than two blank lines.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(cleaned_lines)
    # Collapse 3+ blank lines down to 2.
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text.strip()


def is_meaningful_text(text: str, min_chars: int = 1) -> bool:
    """Return True if a piece of text is non-empty after stripping."""
    if not text:
        return False
    return len(text.strip()) >= min_chars


def iter_supported_files(
    root: Path,
    suffixes: Iterable[str],
) -> list[Path]:
    """Recursively find files under ``root`` whose suffix is supported."""
    suffixes_lc = {s.lower() for s in suffixes}
    if not root.exists():
        return []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in suffixes_lc:
            files.append(path)
    return files
