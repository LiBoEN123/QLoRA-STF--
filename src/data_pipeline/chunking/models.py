"""Phase 3 数据模型与配置。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking 阶段所有可调阈值，集中放置避免散落 magic numbers。"""

    min_chars: int = 100
    target_chars: int = 800
    max_chars: int = 1200
    overlap_chars: int = 100
    faq_min_confidence: str = "medium"

    def __post_init__(self) -> None:
        # 强制约束：min <= target <= max；overlap < max
        if not (self.min_chars <= self.target_chars <= self.max_chars):
            raise ValueError(
                f"非法配置：min_chars({self.min_chars}) <= "
                f"target_chars({self.target_chars}) <= max_chars({self.max_chars}) 必须同时成立"
            )
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars 必须为非负数")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars 必须小于 max_chars")

        # FAQ 置信度阈值
        valid_levels = {"none", "low", "medium", "high"}
        if self.faq_min_confidence not in valid_levels:
            raise ValueError(
                f"faq_min_confidence 必须是 {valid_levels} 之一，得到 {self.faq_min_confidence!r}"
            )


def confidence_rank(level: str) -> int:
    """将置信度字符串映射为可比较整数。"""
    return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(level, 0)


# ---------------------------------------------------------------------------
# 记录结构
# ---------------------------------------------------------------------------


@dataclass
class SemanticUnit:
    """一个语义单元，由 1..n 条 Phase 2 clean record 组成。"""

    unit_id: str
    unit_type: str
    source_path: str
    file_name: str
    source_type: str
    file_sha256: str

    structure_path: List[str] = field(default_factory=list)
    detected_section: Optional[str] = None

    source_doc_ids: List[str] = field(default_factory=list)
    source_record_indices: List[int] = field(default_factory=list)

    text: str = ""

    question_text: Optional[str] = None
    answer_text: Optional[str] = None
    raw_question_text: Optional[str] = None
    faq_confidence: Optional[str] = None

    char_count: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        """返回可 JSON 序列化的 dict 表示。"""
        return asdict(self)


@dataclass
class Chunk:
    """最终供下游消费的 chunk。"""

    chunk_id: str
    chunk_index: int
    chunk_type: str  # faq | text | heading

    source_path: str
    file_name: str
    source_type: str
    file_sha256: str

    structure_path: List[str] = field(default_factory=list)
    detected_section: Optional[str] = None

    source_unit_ids: List[str] = field(default_factory=list)
    source_doc_ids: List[str] = field(default_factory=list)
    source_record_indices: List[int] = field(default_factory=list)

    text: str = ""

    question_text: Optional[str] = None
    answer_text: Optional[str] = None
    faq_confidence: Optional[str] = None

    char_count: int = 0
    split_from_unit: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        """返回可 JSON 序列化的 dict 表示。"""
        return asdict(self)


# ---------------------------------------------------------------------------
# 稳定 ID 生成
# ---------------------------------------------------------------------------


def _sorted_hash(parts: List[str]) -> str:
    """对一组字符串排序后计算 SHA256，保证同输入同输出。"""
    payload = "\u241F".join(sorted(parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_unit_id(
    source_path: str,
    source_doc_ids: List[str],
    unit_type: str,
    text: str,
) -> str:
    """为语义单元生成确定性的 unit_id。"""
    payload = "\u241E".join(
        [
            source_path,
            _sorted_hash(source_doc_ids),
            unit_type,
            text,
        ]
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"unit_{digest[:24]}"


def compute_chunk_id(
    source_unit_ids: List[str],
    chunk_index: int,
    text: str,
) -> str:
    """为 chunk 生成确定性的 chunk_id。"""
    payload = "\u241E".join(
        [
            _sorted_hash(source_unit_ids),
            f"chunk_index={chunk_index}",
            text,
        ]
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"chunk_{digest[:24]}"


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------


def serialize_unit(unit: SemanticUnit) -> str:
    """把 SemanticUnit 序列化为 JSONL 行。"""
    return json.dumps(unit.to_jsonable(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def serialize_chunk(chunk: Chunk) -> str:
    """把 Chunk 序列化为 JSONL 行。"""
    return json.dumps(chunk.to_jsonable(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_key(record: Dict[str, Any]) -> Tuple:
    """在缺失字段时给出稳定的 fallback。"""
    return (
        record.get("source_path", ""),
        record.get("page_number") if record.get("page_number") is not None else -1,
        record.get("sheet_name") or "",
        record.get("record_index") if record.get("record_index") is not None else -1,
        record.get("doc_id", ""),
    )