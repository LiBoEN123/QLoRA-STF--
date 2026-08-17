"""Phase 3 + Phase 3.5 语义单元 + 结构感知 chunking 子包。"""

from .audit import AuditEntry, build_audit_entries, serialize_audit, summarize as summarize_audit
from .faq import (
    MAX_PARAGRAPH_QUESTION_CHARS,
    FaqPair,
    ParagraphRecoveryResult,
    QuestionCandidate,
    STRONG_QUESTION_WORDS,
    collect_no_answer_pairs,
    detect_paragraph_question_candidate,
    detect_question_candidate,
    pair_questions_with_answers,
    recover_paragraph_faqs,
    strip_question_number,
    strip_sticky_question_number,
)
from .models import (
    Chunk,
    ChunkingConfig,
    SemanticUnit,
    compute_chunk_id,
    compute_unit_id,
)
from .ordering import restore_source_order
from .semantic_units import build_semantic_units
from .splitter import split_long_text
from .pipeline import (
    ChunkingResult,
    run_chunking_pipeline,
)

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
    "ParagraphRecoveryResult",
    "STRONG_QUESTION_WORDS",
    "MAX_PARAGRAPH_QUESTION_CHARS",
    "strip_question_number",
    "strip_sticky_question_number",
    "detect_question_candidate",
    "detect_paragraph_question_candidate",
    "pair_questions_with_answers",
    "recover_paragraph_faqs",
    "collect_no_answer_pairs",
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
