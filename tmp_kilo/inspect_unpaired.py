"""详细审计脚本：拆解 16 unpaired 的真实来源。"""

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
    cfg = ChunkingConfig(faq_min_confidence="medium")
    paired_med, unpaired_med = pair_questions_with_answers(ordered, min_confidence="medium")
    paired_low, unpaired_low = pair_questions_with_answers(ordered, min_confidence="low")
    paired_none, unpaired_none = pair_questions_with_answers(ordered, min_confidence="none")

    print(f"Total records: {len(ordered)}")
    print(f"Paired at medium: {len(paired_med)}, unpaired: {len(unpaired_med)}")
    print(f"Paired at low: {len(paired_low)}, unpaired: {len(unpaired_low)}")
    print(f"Paired at none: {len(paired_none)}, unpaired: {len(unpaired_none)}")
    print()
    # 统计 confidence 分布
    all_cands = []
    for idx, rec in enumerate(ordered):
        cand = detect_question_candidate(rec)
        if cand.is_candidate:
            all_cands.append((idx, rec, cand))
    from collections import Counter
    cnt = Counter(c.confidence for _, _, c in all_cands)
    print(f"Candidate confidence distribution: {dict(cnt)}")
    print()

    # 看 paired_med + unpaired_med 是否对应
    # unpaired_med 包含: low confidence 候选 + medium+confidence 无答案
    # 区分两者
    low_only = [c for c in unpaired_med if c.confidence == "low"]
    med_no_ans = [c for c in unpaired_med if c.confidence in ("medium", "high")]
    print(f"unpaired_med breakdown: low_confidence={len(low_only)}, med_or_high_no_answer={len(med_no_ans)}")
    print()

    # 看 med_no_ans 具体是什么
    if med_no_ans:
        print("=== medium/high candidates with no answer (genuine no-answer) ===")
        # 找到对应的源 record
        cand_indices = []
        for idx, rec, cand in all_cands:
            if cand.confidence in ("medium", "high"):
                cand_indices.append((idx, rec, cand))
        # 用 used_indices 思路：paired 中使用的 indices
        # 用 doc_id 集合更简单
        paired_doc_ids = {p.question_doc_id for p in paired_med}
        for idx, rec, cand in cand_indices:
            if rec.get("doc_id") not in paired_doc_ids:
                text = (rec.get("clean_text") or rec.get("text") or "").strip()
                next_rec = ordered[idx + 1] if idx + 1 < len(ordered) else None
                next_text = (next_rec.get("clean_text") or next_rec.get("text") or "").strip() if next_rec else None
                next_st = next_rec.get("structure_type") if next_rec else None
                print(f"  idx={idx} rec_idx={rec.get('record_index')} conf={cand.confidence} reason={cand.reason}")
                print(f"    text: {text[:80]}")
                print(f"    next_structure_type: {next_st}")
                print(f"    next_text: {next_text[:80] if next_text else None}")
                print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
