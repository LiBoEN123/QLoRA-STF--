"""Phase 3 语义单元 + 结构感知 chunking 子包。"""

from .audit import AuditEntry, build_audit_entries, serialize_audit, summarize as summarize_audit
from .faq import (
    MAX_PARAGRAPH_QUESTION_CHARS,
    STRONG_QUESTION_WORDS,
    FaqCandidateRow,
    FaqPair,
    QuestionCandidate,
    detect_question_candidate,
    pair_questions_with_answers,
    strip_question_number,
)
from .models import (
    Chunk,
    ChunkingConfig,
    SemanticUnit,
    compute_chunk_id,
    compute_unit_id,
)
from .ordering import restore_source_order
from .pipeline import ChunkingResult, run_chunking_pipeline
from .semantic_units import build_semantic_units
from .splitter import split_long_text


__all__ = [
    # models
    "Chunk",
    "ChunkingConfig",
    "SemanticUnit",
    "compute_unit_id",
    "compute_chunk_id",
    # ordering
    "restore_source_order",
    # faq
    "QuestionCandidate",
    "FaqPair",
    "FaqCandidateRow",
    "STRONG_QUESTION_WORDS",
    "MAX_PARAGRAPH_QUESTION_CHARS",
    "strip_question_number",
    "detect_question_candidate",
    "pair_questions_with_answers",
    # audit
    "AuditEntry",
    "build_audit_entries",
    "serialize_audit",
    "summarize_audit",
    # semantic units
    "build_semantic_units",
    # splitter
    "split_long_text",
    # pipeline
    "ChunkingResult",
    "run_chunking_pipeline",
]
