"""一次性审计脚本：列出所有 low-confidence 候选及其上下文。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline.chunking.faq import (
    STRONG_QUESTION_WORDS,
    detect_question_candidate,
    pair_questions_with_answers,
)
from data_pipeline.chunking.ordering import restore_source_order
from data_pipeline.chunking.models import ChunkingConfig

INPUT = PROJECT_ROOT / "data" / "processed" / "documents_clean.jsonl"


def main() -> int:
    records = []
    with INPUT.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    ordered = restore_source_order(records)
    cfg = ChunkingConfig(faq_min_confidence="high")
    paired_high, unpaired_high = pair_questions_with_answers(ordered, min_confidence="high")
    paired_med, unpaired_med = pair_questions_with_answers(ordered, min_confidence="medium")
    paired_low, unpaired_low = pair_questions_with_answers(ordered, min_confidence="low")

    # 找出所有 low-confidence 候选
    low_candidates = []
    for idx, rec in enumerate(ordered):
        cand = detect_question_candidate(rec)
        if cand.confidence == "low":
            low_candidates.append((idx, rec, cand))

    print(f"Total records: {len(ordered)}")
    print(f"Low-confidence candidates: {len(low_candidates)}")
    print(f"Paired at high: {len(paired_high)}, at medium: {len(paired_med)}, at low: {len(paired_low)}")
    print()
    print("=" * 100)
    for idx, rec, cand in low_candidates:
        prev_rec = ordered[idx - 1] if idx > 0 else None
        next_rec = ordered[idx + 1] if idx + 1 < len(ordered) else None
        text = (rec.get("clean_text") or rec.get("text") or "").strip()
        print(f"\n--- candidate idx={idx} record_index={rec.get('record_index')} ---")
        print(f"doc_id: {rec.get('doc_id')}")
        print(f"source_path: {rec.get('source_path')}")
        print(f"structure_type: {rec.get('structure_type')}")
        print(f"candidate: conf={cand.confidence} reason={cand.reason}")
        print(f"text: {text[:200]}")
        print(f"text_len: {len(text)}")
        if prev_rec:
            prev_text = (prev_rec.get("clean_text") or prev_rec.get("text") or "").strip()
            print(f"prev structure_type: {prev_rec.get('structure_type')}")
            print(f"prev text: {prev_text[:120]}")
        else:
            print("prev: <none>")
        if next_rec:
            next_text = (next_rec.get("clean_text") or next_rec.get("text") or "").strip()
            print(f"next structure_type: {next_rec.get('structure_type')}")
            print(f"next text: {next_text[:200]}")
        else:
            print("next: <none>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
