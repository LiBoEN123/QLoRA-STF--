"""Phase 3 语义单元 + 结构感知 chunking 子包。"""

from .models import (
    Chunk,
    ChunkingConfig,
    SemanticUnit,
    compute_chunk_id,
    compute_unit_id,
)
from .ordering import restore_source_order
from .faq import (
    QuestionCandidate,
    detect_question_candidate,
    pair_questions_with_answers,
    strip_question_number,
)
from .semantic_units import build_semantic_units
from .splitter import split_long_text
from .pipeline import (
    ChunkingResult,
    run_chunking_pipeline,
)

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "SemanticUnit",
    "compute_unit_id",
    "compute_chunk_id",
    "ChunkingResult",
    "restore_source_order",
    "QuestionCandidate",
    "strip_question_number",
    "detect_question_candidate",
    "pair_questions_with_answers",
    "build_semantic_units",
    "split_long_text",
    "run_chunking_pipeline",
]