"""文档抽取流水线的共享工具函数。"""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from pathlib import Path
from typing import Iterable

LOGGER_NAME = "data_pipeline"


def get_logger(name: str | None = None) -> logging.Logger:
    """返回一个配置好的数据流水线 logger。"""
    return logging.getLogger(name or LOGGER_NAME)


def to_relative_path(path: Path, root: Path) -> str:
    """尽可能返回 ``path`` 相对于 ``root`` 的 POSIX 风格路径。

    如果 ``path`` 不在 ``root`` 下，则回退为绝对路径。
    返回值统一使用正斜杠，保证 Windows / Linux / WSL 间可移植。
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def calculate_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """计算文件内容的 SHA256 十六进制摘要。"""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def normalize_text(text: str) -> str:
    """对文本做最小化、无损的规范化。

    - Unicode ``NFKC`` 规范化。
    - 将孤立的回车转为换行。
    - 去掉每行行尾空白。
    - 折叠 3 行以上的连续空行为 2 行。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(cleaned_lines)
    # 把 3 行及以上的连续空行折叠为 2 行。
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text.strip()


def is_meaningful_text(text: str, min_chars: int = 1) -> bool:
    """如果文本 strip 后非空则返回 True。"""
    if not text:
        return False
    return len(text.strip()) >= min_chars


def iter_supported_files(
    root: Path,
    suffixes: Iterable[str],
) -> list[Path]:
    """递归查找 ``root`` 下后缀受支持的所有文件。"""
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
