"""Phase 5D-R — Runtime Chat Template Compatibility Probe。

使用 validation.jsonl 验证真实 transformers 5.5.0 + Qwen3-8B tokenizer 行为：

- ``prompt_ids`` = ``apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)``
- ``full_ids`` = ``apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)``
- 必须 ``prefix_equal = True``。

输出：APPLY_CHAT_TEMPLATE_RUNTIME_PROBE section。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _apply_chat_template_ids(tokenizer, messages, *, add_generation_prompt):
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt,
    )
    if hasattr(out, "ids") and not hasattr(out, "keys"):
        return list(out.ids)
    if hasattr(out, "keys") and "input_ids" in out.keys():
        ids = out["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    if isinstance(out, list):
        return list(out)
    raise TypeError(f"unexpected: {type(out).__name__}")


def main():
    from transformers import AutoTokenizer

    val_path = ROOT / "data/sft/validation.jsonl"
    records = []
    with val_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    # 取前 3 条 sample_id 排序的
    records.sort(key=lambda r: r["sample_id"])
    sample = records[:3]

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        use_fast=True,
        local_files_only=True,
    )
    print("APPLY_CHAT_TEMPLATE_RUNTIME_PROBE:")
    print(f"  tokenizer class: {type(tokenizer).__name__}")
    print(f"  validation samples: {len(sample)}")
    print()

    all_pass = True
    for i, r in enumerate(sample, 1):
        msgs = r["messages"]
        prompt_obj = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=True, add_generation_prompt=True,
        )
        full_obj = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
        )
        prompt_ids = _apply_chat_template_ids(
            tokenizer, msgs[:-1], add_generation_prompt=True,
        )
        full_ids = _apply_chat_template_ids(
            tokenizer, msgs, add_generation_prompt=False,
        )
        prompt_type = type(prompt_obj).__name__
        full_type = type(full_obj).__name__
        prefix_equal = (
            len(prompt_ids) <= len(full_ids)
            and list(full_ids[: len(prompt_ids)]) == list(prompt_ids)
        )
        masked = len(prompt_ids)
        supervised = len(full_ids) - len(prompt_ids)
        print(f"  [{i}] sample_id={r['sample_id']}")
        print(f"      prompt type: {prompt_type}")
        print(f"      full type:   {full_type}")
        print(f"      prompt tokens: {len(prompt_ids)}")
        print(f"      full tokens:   {len(full_ids)}")
        print(f"      prefix_equal:  {prefix_equal}")
        print(f"      masked:        {masked}")
        print(f"      supervised:    {supervised}")
        if not prefix_equal:
            all_pass = False
            print(f"      [FAIL] prefix mismatch!")
        print()

    print(f"PASS: {all_pass}")
    if not all_pass:
        print("CHAT_TEMPLATE_PREFIX_MISMATCH")
        sys.exit(1)


if __name__ == "__main__":
    main()