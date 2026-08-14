"""Data pipeline package for Qwen3-8B QLoRA SFT project.

Phase 1: Document extraction from raw files into a unified JSONL format.
"""

from .models import DocumentRecord
from .registry import EXTRACTORS, get_extractor

__all__ = ["DocumentRecord", "EXTRACTORS", "get_extractor"]
