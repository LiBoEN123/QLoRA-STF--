"""Qwen3-8B QLoRA SFT 项目的数据流水线包。

Phase 1：从原始文件中抽取文档并转换为统一的 JSONL 格式。
"""

from .models import DocumentRecord
from .registry import EXTRACTORS, get_extractor

__all__ = ["DocumentRecord", "EXTRACTORS", "get_extractor"]
