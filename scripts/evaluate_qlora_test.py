"""Phase 5D — Held-out Test Evaluation runtime。

不做任何训练；只：

1. 冻结 evaluation protocol；
2. test.jsonl 指纹 + leakage audit；
3. 加载 BASE（4-bit / NF4 / BF16）→ assistant-only teacher-forced loss
   + 79 条 greedy generation；
4. 释放 BASE；
5. 加载 BASE + best_adapter → 同 79 条 loss + generation；
6. 规范化 + char F1 + ROUGE-L + paired wins/ties；
7. 通过 ``PHASE5D_CODE = PASS`` gate。

输出（runtime artifact，不提交 Git）：

```
data/training/phase5d/
├── eval_manifest.json
├── leakage_audit.json
├── base_test_metrics.json
├── finetuned_test_metrics.json
├── paired_metrics.json
├── test_predictions.jsonl
├── qualitative_review.md
└── phase5d_report.json
```

退出码：

- 0：PHASE5D_CODE = PASS, READY_FOR_PHASE5E = YES
- 10：参数 / 文件 / adapter / tokenizer 异常
- 20：leakage / sample_id / count 异常
- 30：BASE 评测异常（OOM / NaN / loss 缺失）
- 40：FINETUNED 评测异常
- 50：gate 未通过（test loss / generation metrics / empty / regression）
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Force offline mode for HuggingFace — model + tokenizer must already be cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data_pipeline.training import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_REVISION,
    Phase5DEvalConfig,
    aggregate_metrics,
    char_f1,
    eval_gate,
    exact_match,
    fingerprint_test,
    leakage_audit,
    normalize_for_eval,
    paired_metrics,
    rouge_l_f1,
    write_eval_manifest,
)
from data_pipeline.training.phase5c_b import (  # noqa: E402
    compute_adapter_signature,
    verify_best_adapter_matches_checkpoint,
)
from data_pipeline.utils import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    get_logger,
    load_jsonl_records,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_TOKENIZER_SIGNATURE = (
    "4ac4fed6d76f2cddf5ee756990b7bd4dc40655f96e4c6be460dc3299e73651c0"
)
DEFAULT_BEST_ADAPTER_DIR = Path("outputs/phase5c-b/best_adapter")
DEFAULT_BEST_CHECKPOINT_DIR = Path("outputs/phase5c-b/checkpoints/checkpoint-288")
DEFAULT_OUTPUT_DIR = Path("data/training/phase5d")
EXPECTED_LORA_LAYERS = 252
EXPECTED_TEST_COUNT = 79
MAX_SEQ_LENGTH = 2048


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_head(cwd: Path) -> Tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True
        ).strip()
    except Exception:
        commit = "UNKNOWN"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=str(cwd), text=True
            ).strip()
        )
    except Exception:
        dirty = False
    return commit, dirty


def _tokenizer_signature(token_audit_path) -> str:
    """按 Phase 5C-A 算法重算 tokenizer signature。"""
    p = Path(token_audit_path)
    if not p.exists():
        return ""
    rows = load_jsonl_records(p)
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["sample_id"]):
        h.update(r["sample_id"].encode("utf-8"))
        h.update(b"|")
        h.update(str(r.get("split", "")).encode("utf-8"))
        h.update(b"|")
        h.update(str(r.get("total_tokens", 0)).encode("utf-8"))
        h.update(b"|")
        h.update(str(r.get("prompt_tokens", 0)).encode("utf-8"))
        h.update(b"|")
        h.update(str(r.get("assistant_target_tokens", 0)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _reference_assistant(messages: List[Dict[str, Any]]) -> str:
    """从最后一条 assistant 取 reference（如果不存在则抛错）。"""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return str(m.get("content", ""))
    raise ValueError("messages 中找不到 assistant reference")


def _prompt_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """reference assistant 之前的 messages。"""
    return [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in messages
        if m.get("role") != "assistant"
    ] or messages[:-1]


def _apply_chat_template_ids(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> List[int]:
    """调用 ``apply_chat_template(tokenize=True)`` 并强制返回 ``List[int]``。

    实测（transformers 5.5.0）：``apply_chat_template(..., tokenize=True,
    return_dict=True)`` 返回 ``BatchEncoding`` dict-like 对象，包含
    ``input_ids`` / ``attention_mask``。token 列表位于 ``obj['input_ids']``。
    返回单样本时 ``input_ids`` 已经是 ``List[int]``，没有 batch 维度。

    容错：

    - 如果 ``obj['input_ids']`` 是 nested list（带 batch 维度），取 ``[0]``。
    - 如果 ``obj`` 本身就是 list（旧版本 transformers 兼容），直接 list()。
    - 如果 ``obj`` 有 ``.ids`` 属性（旧版本 fast tokenizer 兼容），取 ``.ids``。
    """
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(out, "ids") and not hasattr(out, "keys"):
        # tokenizers.Encoding-like
        return list(out.ids)
    if hasattr(out, "keys") and "input_ids" in out.keys():
        ids = out["input_ids"]
        # batch dim?
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    if isinstance(out, list):
        return list(out)
    raise TypeError(
        f"apply_chat_template returned unexpected type: {type(out).__name__}; "
        f"value={repr(out)[:200]}"
    )


def _per_sample_assistant_mask(
    tokenizer: Any,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """校验 assistant-only mask 是否真正生效。

    Qwen3 的 chat template **不包含** ``{% generation %}`` 关键字，因此
    ``return_assistant_tokens_mask=True`` 会返回全 0。这里使用与
    Phase 5C-B 一致的 prompt-prefix 切片法：

    1. ``prompt_ids`` = tokenize(messages[:-1] + add_generation_prompt=True)；
    2. ``full_ids`` = tokenize(messages + add_generation_prompt=False)；
    3. 验证 ``full_ids[:len(prompt_ids)] == prompt_ids``；
    4. ``prompt_len = len(prompt_ids)``；
    5. ``masked_tokens = supervised_tokens = full_ids 长度 - prompt 长度``。
    """
    msgs = record["messages"]
    prompt_ids = _apply_chat_template_ids(
        tokenizer, msgs[:-1], add_generation_prompt=True,
    )
    full_ids = _apply_chat_template_ids(
        tokenizer, msgs, add_generation_prompt=False,
    )
    seq_len = len(full_ids)
    if not (len(prompt_ids) <= len(full_ids)
            and list(full_ids[:len(prompt_ids)]) == list(prompt_ids)):
        # 退化情况：prompt 与 full 前缀不一致。仍记录但保守报 masked=0
        prompt_len = 0
        masked = 0
    else:
        prompt_len = len(prompt_ids)
        masked = seq_len - prompt_len
    return {
        "sample_id": record.get("sample_id", ""),
        "sequence_length": seq_len,
        "masked_tokens": int(masked),
        "supervised_tokens": int(masked),
        "prompt_len": int(prompt_len),
    }


def _assistant_only_loss(
    model: Any,
    tokenizer: Any,
    record: Dict[str, Any],
    device: str,
    torch: Any,
) -> Tuple[float, int]:
    """对单条样本计算 assistant-only teacher-forced loss。

    使用 prompt-prefix 切片法（Qwen3 chat template 不支持 ``{% generation %}``）：

    - ``prompt_ids`` = tokenize(msgs[:-1] + add_generation_prompt=True)
    - ``full_ids`` = tokenize(msgs + add_generation_prompt=False)
    - labels[i] = full_ids[i] if i >= len(prompt_ids) else -100
    - 监督 token = full_ids 长度 - prompt 长度
    """
    msgs = record["messages"]
    prompt_ids = _apply_chat_template_ids(
        tokenizer, msgs[:-1], add_generation_prompt=True,
    )
    full_ids = _apply_chat_template_ids(
        tokenizer, msgs, add_generation_prompt=False,
    )
    if not (len(prompt_ids) <= len(full_ids)
            and list(full_ids[:len(prompt_ids)]) == list(prompt_ids)):
        # prompt 与 full 前缀不一致 ⇒ 无法安全切片；返回 NaN
        return float("nan"), 0
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + list(full_ids[prompt_len:])
    sup = len(full_ids) - prompt_len
    if sup == 0:
        return float("nan"), 0

    input_ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device)
    attn_t = torch.ones_like(input_ids_t)
    with torch.inference_mode():
        out = model(
            input_ids=input_ids_t,
            attention_mask=attn_t,
            labels=labels_t,
        )
    loss = out.loss
    if loss is None or not torch.isfinite(loss).item():
        return float("nan"), sup
    return float(loss.item()), sup


def _greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt_messages: List[Dict[str, str]],
    max_new_tokens: int,
    torch: Any,
) -> Tuple[str, int, bool]:
    """对 prompt_messages 跑 greedy generation。

    Returns (generated_text, generated_token_count, hit_cap)
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_ids = out[0][inputs["input_ids"].shape[-1]:]
    n_new = int(new_ids.shape[-1])
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    hit_cap = (n_new >= max_new_tokens)
    return text, n_new, hit_cap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5D held-out test evaluation")
    p.add_argument("--best-adapter-dir", default=str(DEFAULT_BEST_ADAPTER_DIR), type=Path)
    p.add_argument("--best-checkpoint-dir", default=str(DEFAULT_BEST_CHECKPOINT_DIR), type=Path)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=Path)
    p.add_argument("--token-audit", default="data/tokenizer/token_audit.jsonl", type=Path)
    p.add_argument("--token-report", default="data/tokenizer/token_report.json", type=Path)
    p.add_argument("--test-jsonl", default="data/sft/test.jsonl", type=Path)
    p.add_argument("--train-jsonl", default="data/sft/train.jsonl", type=Path)
    p.add_argument("--validation-jsonl", default="data/sft/validation.jsonl", type=Path)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--base-only", action="store_true", help="只跑 BASE；调试用")
    p.add_argument("--finetuned-only", action="store_true", help="只跑 FINETUNED；调试用")
    p.add_argument("--resume", action="store_true", help="如果 test_predictions.jsonl 已有则跳过已完成 sample")
    return p.parse_args()


def _run(args: argparse.Namespace, log: Any) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: git baseline
    # ------------------------------------------------------------------
    git_commit, git_dirty = _git_head(ROOT)
    log.info("[repo] HEAD=%s dirty=%s", git_commit, git_dirty)

    # ------------------------------------------------------------------
    # Step 2: tokenizer signature
    # ------------------------------------------------------------------
    actual_tok_sig = _tokenizer_signature(args.token_audit)
    if actual_tok_sig != EXPECTED_TOKENIZER_SIGNATURE:
        log.error(
            "[tokenizer] signature mismatch: got=%s expected=%s",
            actual_tok_sig[:16] if actual_tok_sig else "<missing>",
            EXPECTED_TOKENIZER_SIGNATURE[:16],
        )
        atomic_write_json(output_dir / "phase5d_report.json", {
            "status": "BLOCKED",
            "code": "TOKENIZER_DRIFT",
            "tokenizer_signature_actual": actual_tok_sig,
            "tokenizer_signature_expected": EXPECTED_TOKENIZER_SIGNATURE,
        })
        return 10

    # ------------------------------------------------------------------
    # Step 3: best adapter gate (signature + LoRA layers)
    # ------------------------------------------------------------------
    best_st = args.best_adapter_dir / "adapter_model.safetensors"
    if not best_st.exists():
        log.error("[adapter] missing: %s", best_st)
        atomic_write_json(output_dir / "phase5d_report.json", {
            "status": "BLOCKED",
            "code": "BEST_ADAPTER_INVALID",
            "reason": f"missing {best_st}",
        })
        return 10

    # config JSON checks
    cfg_json = json.loads((args.best_adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg_json.get("r") != 16:
        log.error("[adapter] r != 16")
        return 10
    if cfg_json.get("lora_alpha") != 32:
        log.error("[adapter] lora_alpha != 32")
        return 10
    if abs(float(cfg_json.get("lora_dropout", 0)) - 0.05) > 1e-6:
        log.error("[adapter] dropout != 0.05")
        return 10
    if cfg_json.get("base_model_name_or_path") != "Qwen/Qwen3-8B":
        log.error("[adapter] base_model_name_or_path != Qwen/Qwen3-8B")
        return 10

    log.info("[adapter] computing tensor signature...")
    import torch  # local import; unsloth env may already have torch
    best_sig = compute_adapter_signature(best_st, torch=torch)
    log.info("[adapter] n_tensors=%d sig=%s", best_sig.n_tensors, best_sig.signature[:16])

    # Verify LoRA layers by reading adapter tensor names; 504 = 252 × 2
    if best_sig.n_tensors != EXPECTED_LORA_LAYERS * 2:
        log.warning(
            "[adapter] n_tensors=%d != %d; will verify after load",
            best_sig.n_tensors, EXPECTED_LORA_LAYERS * 2,
        )

    # ------------------------------------------------------------------
    # Step 4: test fingerprint + leakage audit
    # ------------------------------------------------------------------
    test_fp = fingerprint_test(args.test_jsonl)
    if test_fp.record_count != EXPECTED_TEST_COUNT:
        log.error("[test] count=%d != %d", test_fp.record_count, EXPECTED_TEST_COUNT)
        return 20

    train_records = load_jsonl_records(args.train_jsonl)
    val_records = load_jsonl_records(args.validation_jsonl)
    test_records = load_jsonl_records(args.test_jsonl)
    leak = leakage_audit(test_records, train_records, val_records)
    leak_dict = leak.to_dict()
    atomic_write_json(output_dir / "leakage_audit.json", leak_dict)
    if not leak.is_clean:
        log.error("[leakage] %s", leak_dict)
        return 20

    # ------------------------------------------------------------------
    # Step 5: compute generation_max_new_tokens from references
    # ------------------------------------------------------------------
    # We need tokenizer to compute token lengths; load tokenizer once here.
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_NAME,
        revision=DEFAULT_REVISION,
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    log.info("[tokenizer] loaded: %s", type(tokenizer).__name__)

    ref_token_lengths: List[int] = []
    for r in test_records:
        ref = _reference_assistant(r["messages"])
        toks = tokenizer(ref, add_special_tokens=False)["input_ids"]
        ref_token_lengths.append(len(toks))
    max_ref_tokens = max(ref_token_lengths) if ref_token_lengths else 0
    generation_max_new_tokens = min(
        2048,
        max(256, max_ref_tokens + 64),
    )
    log.info(
        "[generation] max_reference_assistant_tokens=%d, generation_max_new_tokens=%d",
        max_ref_tokens, generation_max_new_tokens,
    )

    # ------------------------------------------------------------------
    # Step 6: write frozen eval_manifest
    # ------------------------------------------------------------------
    manifest_path = output_dir / "eval_manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not existing.get("protocol_frozen"):
                log.error("[manifest] existing manifest not protocol_frozen=True")
                return 10
            # Compare generation_max_new_tokens + tokenizer sig + test sample_id sig
            if existing.get("generation_max_new_tokens") != generation_max_new_tokens:
                log.error("[manifest] EVAL_PROTOCOL_MISMATCH: generation_max_new_tokens drift")
                return 10
            if existing.get("tokenizer_signature") != EXPECTED_TOKENIZER_SIGNATURE:
                log.error("[manifest] EVAL_PROTOCOL_MISMATCH: tokenizer_signature drift")
                return 10
            if existing.get("test_sample_id_signature") != test_fp.sample_id_signature:
                log.error("[manifest] EVAL_PROTOCOL_MISMATCH: test_sample_id_signature drift")
                return 10
            log.info("[manifest] existing frozen manifest is consistent; reuse")
        except Exception as e:
            log.error("[manifest] failed to parse existing manifest: %s", e)
            return 10
    else:
        manifest_payload = {
            "git": {
                "branch": "main",
                "commit": git_commit,
                "dirty": git_dirty,
            },
            "backend": "unsloth",
            "model_name": DEFAULT_MODEL_NAME,
            "model_revision": DEFAULT_REVISION,
            "quantization": {
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "double_quant": True,
                "max_length": MAX_SEQ_LENGTH,
            },
            "best_adapter_path": str(args.best_adapter_dir),
            "best_adapter_tensor_signature": best_sig.to_dict(),
            "best_adapter_signature_sha256": best_sig.signature,
            "tokenizer_signature": EXPECTED_TOKENIZER_SIGNATURE,
            "test_jsonl_path": str(args.test_jsonl),
            "test_jsonl_sha256": test_fp.file_sha256,
            "test_count": test_fp.record_count,
            "test_sample_id_signature": test_fp.sample_id_signature,
            "test_record_count": test_fp.record_count,
            "max_reference_assistant_tokens": int(max_ref_tokens),
            "generation_max_new_tokens": int(generation_max_new_tokens),
            "assistant_only_loss": True,
            "enable_thinking": False,
            "do_sample": False,
            "normalization": {
                "method": "NFKC + strip + collapse whitespace; lexical via unicodedata.category P*/S*",
                "deterministic": True,
            },
            "metrics": {
                "char_f1": "normalize_for_eval + Counter overlap",
                "rouge_l_f1": "normalize_for_eval + LCS / (|p|+|r|)",
                "exact_match": "normalize_for_eval == normalize_for_eval",
            },
            "seed": 42,
            "expected_lora_layers": EXPECTED_LORA_LAYERS,
        }
        write_eval_manifest(manifest_path, payload=manifest_payload, logger=log)

    # ------------------------------------------------------------------
    # Step 7: assistant mask sanity (3 samples)
    # ------------------------------------------------------------------
    log.info("[mask] sanity check on 3 test samples (first / middle / last by sample_id)")
    sorted_test = sorted(test_records, key=lambda r: r["sample_id"])
    mask_samples_to_check = [sorted_test[0], sorted_test[len(sorted_test) // 2], sorted_test[-1]]
    mask_results: List[Dict[str, Any]] = []
    for r in mask_samples_to_check:
        info = _per_sample_assistant_mask(tokenizer, r)
        mask_results.append(info)
        log.info(
            "[mask] sample_id=%s seq_len=%d masked=%d supervised=%d",
            info["sample_id"], info["sequence_length"], info["masked_tokens"], info["supervised_tokens"],
        )
        if info["sequence_length"] > MAX_SEQ_LENGTH:
            log.error("[mask] TEST_ASSISTANT_MASK_FAIL: sequence_length > %d", MAX_SEQ_LENGTH)
            return 10
        if info["masked_tokens"] <= 0 or info["supervised_tokens"] <= 0:
            log.error("[mask] TEST_ASSISTANT_MASK_FAIL: masked/supervised <= 0")
            return 10

    atomic_write_json(output_dir / "test_assistant_mask_check.json", {
        "checked_samples": mask_results,
        "pass": True,
    })

    # ------------------------------------------------------------------
    # Step 8: load unsloth backend + helper functions from train script
    # ------------------------------------------------------------------
    # Reuse existing backend & helpers so we don't duplicate logic
    sys.path.insert(0, str(SCRIPTS))
    from train_qlora_sft import (  # type: ignore
        _ensure_torch, _check_gpu, _bytes_to_gib,
        _flatten_log_history,
    )
    from data_pipeline.training.backends import resolve_backend  # type: ignore

    torch = _ensure_torch()
    gpu_info = _check_gpu(torch, log)
    log.info("[gpu] %s total_gib=%.3f", gpu_info.get("device_name"), _bytes_to_gib(gpu_info.get("total_memory_bytes", 0)))

    config = Phase5DEvalConfig(
        model_name=DEFAULT_MODEL_NAME,
        revision=DEFAULT_REVISION,
        max_new_tokens=generation_max_new_tokens,
    )

    backend = resolve_backend("unsloth")

    # ------------------------------------------------------------------
    # Step 9: Build per-sample result state from existing predictions (resume)
    # ------------------------------------------------------------------
    pred_path = output_dir / "test_predictions.jsonl"
    existing_preds: Dict[str, Dict[str, Any]] = {}
    if args.resume and pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                existing_preds[rec["sample_id"]] = rec
            except Exception:
                pass
        log.info("[resume] loaded %d existing predictions", len(existing_preds))

    # ------------------------------------------------------------------
    # Step 10: BASE eval (loss + generation)
    # ------------------------------------------------------------------
    base_test_loss = float("nan")
    base_test_ppl = float("nan")
    base_per_sample: List[Dict[str, Any]] = []
    if not args.finetuned_only:
        log.info("[base] loading BASE Qwen3-8B 4-bit...")
        base_model, base_tokenizer = backend.load_model_and_tokenizer(
            model_name=DEFAULT_MODEL_NAME,
            revision=DEFAULT_REVISION,
            torch=torch,
            config=config,
        )
        base_model.eval()

        # Sanity: 4-bit loaded
        is_4bit = backend.is_loaded_in_4bit(base_model)
        linear4bit = backend.count_linear4bit(base_model)
        log.info("[base] is_4bit=%s linear4bit=%d", is_4bit, linear4bit)
        if not is_4bit or linear4bit <= 0:
            log.error("[base] BASE_NOT_4BIT")
            return 30

        # Loss
        log.info("[base] computing teacher-forced assistant-only loss on %d samples...", len(test_records))
        t0 = time.time()
        total_loss = 0.0
        total_sup = 0
        loss_n = 0
        for r in test_records:
            loss, sup = _assistant_only_loss(base_model, base_tokenizer, r, base_model.device, torch)
            if math.isfinite(loss) and sup > 0:
                total_loss += loss * sup
                total_sup += sup
                loss_n += 1
        base_runtime = time.time() - t0
        if total_sup > 0:
            base_test_loss = total_loss / total_sup
            base_test_ppl = float(math.exp(base_test_loss)) if math.isfinite(base_test_loss) else float("nan")
        log.info(
            "[base] loss PASS  loss=%.6f ppl=%.6f sup_tokens=%d runtime=%.1fs",
            base_test_loss, base_test_ppl, total_sup, base_runtime,
        )
        if not math.isfinite(base_test_loss):
            log.error("[base] BASE_TEST_LOSS_NON_FINITE")
            return 30

        # Generation
        log.info("[base] generating 79/79...")
        for idx, r in enumerate(test_records):
            sid = r["sample_id"]
            if sid in existing_preds and existing_preds[sid].get("base_generation") is not None:
                base_per_sample.append(existing_preds[sid])
                continue
            prompt_msgs = _prompt_messages(r["messages"])
            ref = _reference_assistant(r["messages"])
            text, n_new, hit_cap = _greedy_generate(
                base_model, base_tokenizer, prompt_msgs,
                max_new_tokens=generation_max_new_tokens, torch=torch,
            )
            base_em = exact_match(text, ref)
            base_cf1 = char_f1(text, ref)["f1"]
            base_rl = rouge_l_f1(text, ref)["f1"]
            entry = {
                "sample_id": sid,
                "prompt_messages": prompt_msgs,
                "reference": ref,
                "base_generation": text,
                "base_generated_tokens": int(n_new),
                "base_empty": (n_new == 0),
                "base_hit_max_tokens": bool(hit_cap),
                "base_normalized_exact_match": int(base_em),
                "base_char_f1": round(base_cf1, 6),
                "base_rouge_l_f1": round(base_rl, 6),
            }
            base_per_sample.append(entry)
            if (idx + 1) % 10 == 0 or idx == 0 or (idx + 1) == len(test_records):
                log.info("[base] generation %d/%d", idx + 1, len(test_records))

        # Save progress
        _dump_predictions(pred_path, base_per_sample, finetuned_field=None)
        _save_base_metrics(output_dir, base_per_sample, base_test_loss, base_test_ppl, base_runtime)

        # Release base
        del base_model, base_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log.info("[base] released")

    # ------------------------------------------------------------------
    # Step 11: FINETUNED eval (loss + generation)
    # ------------------------------------------------------------------
    ft_test_loss = float("nan")
    ft_test_ppl = float("nan")
    ft_per_sample: List[Dict[str, Any]] = []
    if not args.base_only:
        log.info("[finetuned] loading BASE + best_adapter...")
        ft_model, ft_tokenizer = backend.load_model_and_tokenizer(
            model_name=DEFAULT_MODEL_NAME,
            revision=DEFAULT_REVISION,
            torch=torch,
            config=config,
        )
        # Load adapter
        try:
            if hasattr(ft_model, "load_adapter"):
                ft_model.load_adapter(str(args.best_adapter_dir))
                peft_model = ft_model
            else:
                from peft import PeftModel  # type: ignore
                inner = getattr(ft_model, "model", ft_model)
                peft_model = PeftModel.from_pretrained(inner, str(args.best_adapter_dir))
        except Exception as e:
            log.error("[finetuned] ADAPTER_RELOAD_FAIL: %s", e)
            return 40

        peft_model.eval()

        lora_layers = backend.count_lora_layers(peft_model)
        log.info("[finetuned] LoRA layers = %d", lora_layers)
        if lora_layers != EXPECTED_LORA_LAYERS:
            log.error(
                "[finetuned] LORA_LAYERS_DRIFT: %d != %d",
                lora_layers, EXPECTED_LORA_LAYERS,
            )
            return 40

        # Loss
        log.info("[finetuned] computing teacher-forced assistant-only loss on %d samples...", len(test_records))
        t0 = time.time()
        total_loss = 0.0
        total_sup = 0
        for r in test_records:
            loss, sup = _assistant_only_loss(peft_model, ft_tokenizer, r, peft_model.device, torch)
            if math.isfinite(loss) and sup > 0:
                total_loss += loss * sup
                total_sup += sup
        ft_runtime = time.time() - t0
        if total_sup > 0:
            ft_test_loss = total_loss / total_sup
            ft_test_ppl = float(math.exp(ft_test_loss)) if math.isfinite(ft_test_loss) else float("nan")
        log.info(
            "[finetuned] loss PASS  loss=%.6f ppl=%.6f sup_tokens=%d runtime=%.1fs",
            ft_test_loss, ft_test_ppl, total_sup, ft_runtime,
        )
        if not math.isfinite(ft_test_loss):
            log.error("[finetuned] FINETUNED_TEST_LOSS_NON_FINITE")
            return 40

        # Generation
        log.info("[finetuned] generating 79/79...")
        for idx, r in enumerate(test_records):
            sid = r["sample_id"]
            prompt_msgs = _prompt_messages(r["messages"])
            ref = _reference_assistant(r["messages"])
            # If base result already in existing_preds but finetuned missing, fill only finetuned
            entry = None
            for existing in base_per_sample:
                if existing.get("sample_id") == sid:
                    entry = dict(existing)
                    break
            if entry is None:
                entry = {"sample_id": sid, "prompt_messages": prompt_msgs, "reference": ref}
            if entry.get("finetuned_generation") is not None:
                ft_per_sample.append(entry)
                continue
            text, n_new, hit_cap = _greedy_generate(
                peft_model, ft_tokenizer, prompt_msgs,
                max_new_tokens=generation_max_new_tokens, torch=torch,
            )
            entry["finetuned_generation"] = text
            entry["finetuned_generated_tokens"] = int(n_new)
            entry["finetuned_empty"] = (n_new == 0)
            entry["finetuned_hit_max_tokens"] = bool(hit_cap)
            entry["finetuned_normalized_exact_match"] = int(exact_match(text, ref))
            entry["finetuned_char_f1"] = round(char_f1(text, ref)["f1"], 6)
            entry["finetuned_rouge_l_f1"] = round(rouge_l_f1(text, ref)["f1"], 6)
            ft_per_sample.append(entry)
            if (idx + 1) % 10 == 0 or idx == 0 or (idx + 1) == len(test_records):
                log.info("[finetuned] generation %d/%d", idx + 1, len(test_records))

        _dump_predictions(pred_path, ft_per_sample, finetuned_field="finetuned_generation")
        _save_finetuned_metrics(output_dir, ft_per_sample, ft_test_loss, ft_test_ppl, ft_runtime)

        del peft_model, ft_model, ft_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log.info("[finetuned] released")

    # ------------------------------------------------------------------
    # Step 12: paired metrics + gate
    # ------------------------------------------------------------------
    return _finalize(
        args, output_dir, git_commit, git_dirty, best_sig, test_fp,
        base_per_sample, base_test_loss, base_test_ppl,
        ft_per_sample, ft_test_loss, ft_test_ppl,
        leak, log,
    )


def _dump_predictions(path: Path, entries: List[Dict[str, Any]], finetuned_field: Optional[str]) -> None:
    """Dump jsonl preserving all known fields. Used for resume support."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        by_id[e["sample_id"]] = e
    lines = [json.dumps(by_id[k], ensure_ascii=False) for k in sorted(by_id.keys())]
    atomic_write_jsonl(path, lines)


def _save_base_metrics(
    output_dir: Path,
    per_sample: List[Dict[str, Any]],
    loss: float,
    ppl: float,
    runtime: float,
) -> None:
    metrics_for_agg = []
    for e in per_sample:
        metrics_for_agg.append({
            "exact_match": e.get("base_normalized_exact_match", 0),
            "char_f1": e.get("base_char_f1", 0.0),
            "rouge_l_f1": e.get("base_rouge_l_f1", 0.0),
            "generated_tokens": e.get("base_generated_tokens", 0),
            "hit_generation_cap": e.get("base_hit_max_tokens", False),
        })
    agg = aggregate_metrics(metrics_for_agg)
    atomic_write_json(output_dir / "base_test_metrics.json", {
        "test_loss": float(loss),
        "test_perplexity": float(ppl),
        "runtime_seconds": round(runtime, 3),
        **agg,
        "empty_count": sum(1 for m in metrics_for_agg if m["generated_tokens"] == 0),
        "cap_hit_count": sum(1 for m in metrics_for_agg if m["hit_generation_cap"]),
    })


def _save_finetuned_metrics(
    output_dir: Path,
    per_sample: List[Dict[str, Any]],
    loss: float,
    ppl: float,
    runtime: float,
) -> None:
    metrics_for_agg = []
    for e in per_sample:
        metrics_for_agg.append({
            "exact_match": e.get("finetuned_normalized_exact_match", 0),
            "char_f1": e.get("finetuned_char_f1", 0.0),
            "rouge_l_f1": e.get("finetuned_rouge_l_f1", 0.0),
            "generated_tokens": e.get("finetuned_generated_tokens", 0),
            "hit_generation_cap": e.get("finetuned_hit_max_tokens", False),
        })
    agg = aggregate_metrics(metrics_for_agg)
    atomic_write_json(output_dir / "finetuned_test_metrics.json", {
        "test_loss": float(loss),
        "test_perplexity": float(ppl),
        "runtime_seconds": round(runtime, 3),
        **agg,
        "empty_count": sum(1 for m in metrics_for_agg if m["generated_tokens"] == 0),
        "cap_hit_count": sum(1 for m in metrics_for_agg if m["hit_generation_cap"]),
    })


def _finalize(
    args: argparse.Namespace,
    output_dir: Path,
    git_commit: str,
    git_dirty: bool,
    best_sig: Any,
    test_fp: Any,
    base_per_sample: List[Dict[str, Any]],
    base_loss: float,
    base_ppl: float,
    ft_per_sample: List[Dict[str, Any]],
    ft_loss: float,
    ft_ppl: float,
    leak: Any,
    log: Any,
) -> int:
    # Build full per-sample list by sample_id
    by_id: Dict[str, Dict[str, Any]] = {}
    for e in base_per_sample:
        by_id[e["sample_id"]] = dict(e)
    for e in ft_per_sample:
        sid = e["sample_id"]
        if sid not in by_id:
            by_id[sid] = dict(e)
        else:
            for k, v in e.items():
                if k not in by_id[sid] or by_id[sid].get(k) is None:
                    by_id[sid][k] = v

    pred_path = output_dir / "test_predictions.jsonl"
    lines = [json.dumps(by_id[k], ensure_ascii=False) for k in sorted(by_id.keys())]
    atomic_write_jsonl(pred_path, lines)

    # Base aggregate
    base_metrics_for_agg = [
        {
            "exact_match": by_id[k].get("base_normalized_exact_match", 0),
            "char_f1": by_id[k].get("base_char_f1", 0.0),
            "rouge_l_f1": by_id[k].get("base_rouge_l_f1", 0.0),
            "generated_tokens": by_id[k].get("base_generated_tokens", 0),
            "hit_generation_cap": by_id[k].get("base_hit_max_tokens", False),
        }
        for k in sorted(by_id.keys())
    ]
    base_agg = aggregate_metrics(base_metrics_for_agg)
    ft_metrics_for_agg = [
        {
            "exact_match": by_id[k].get("finetuned_normalized_exact_match", 0),
            "char_f1": by_id[k].get("finetuned_char_f1", 0.0),
            "rouge_l_f1": by_id[k].get("finetuned_rouge_l_f1", 0.0),
            "generated_tokens": by_id[k].get("finetuned_generated_tokens", 0),
            "hit_generation_cap": by_id[k].get("finetuned_hit_max_tokens", False),
        }
        for k in sorted(by_id.keys())
    ]
    ft_agg = aggregate_metrics(ft_metrics_for_agg)

    atomic_write_json(output_dir / "base_test_metrics.json", {
        "test_loss": float(base_loss),
        "test_perplexity": float(base_ppl),
        **base_agg,
        "empty_count": sum(1 for m in base_metrics_for_agg if m["generated_tokens"] == 0),
        "cap_hit_count": sum(1 for m in base_metrics_for_agg if m["hit_generation_cap"]),
    })
    atomic_write_json(output_dir / "finetuned_test_metrics.json", {
        "test_loss": float(ft_loss),
        "test_perplexity": float(ft_ppl),
        **ft_agg,
        "empty_count": sum(1 for m in ft_metrics_for_agg if m["generated_tokens"] == 0),
        "cap_hit_count": sum(1 for m in ft_metrics_for_agg if m["hit_generation_cap"]),
    })

    # Paired metrics
    base_cf1_scores = [m["char_f1"] for m in base_metrics_for_agg]
    ft_cf1_scores = [m["char_f1"] for m in ft_metrics_for_agg]
    base_rl_scores = [m["rouge_l_f1"] for m in base_metrics_for_agg]
    ft_rl_scores = [m["rouge_l_f1"] for m in ft_metrics_for_agg]
    paired_cf1 = paired_metrics(ft_cf1_scores, base_cf1_scores)
    paired_rl = paired_metrics(ft_rl_scores, base_rl_scores)

    atomic_write_json(output_dir / "paired_metrics.json", {
        "char_f1": paired_cf1,
        "rouge_l_f1": paired_rl,
        "tolerance": 1e-12,
        "n": len(by_id),
    })

    # Qualitative review
    _write_qualitative_review(output_dir / "qualitative_review.md", by_id, paired_cf1)

    # Eval gate
    gate = eval_gate(
        base_test_loss=base_loss,
        finetuned_test_loss=ft_loss,
        base_metrics=base_agg,
        finetuned_metrics=ft_agg,
        paired_char_f1=paired_cf1,
        paired_rouge_l=paired_rl,
        finetuned_empty_count=sum(1 for m in ft_metrics_for_agg if m["generated_tokens"] == 0),
        leakage_audit_result=leak,
        n_test=len(by_id),
        base_eval_completed=len(base_per_sample) == 79,
        finetuned_eval_completed=len(ft_per_sample) == 79,
    )

    test_loss_improvement = (base_loss - ft_loss) if (math.isfinite(base_loss) and math.isfinite(ft_loss)) else 0.0
    test_loss_improvement_pct = (
        test_loss_improvement / base_loss * 100.0
        if math.isfinite(base_loss) and base_loss > 0 and math.isfinite(ft_loss)
        else 0.0
    )

    report = {
        "status": "PASS" if gate.code == "PASS" else "BLOCKED",
        "code": gate.code,
        "ready_for_phase5e": bool(gate.ready_for_phase5e),
        "git": {"branch": "main", "commit": git_commit, "dirty": git_dirty},
        "test_count": int(len(by_id)),
        "best_adapter_signature": best_sig.to_dict(),
        "tokenizer_signature": EXPECTED_TOKENIZER_SIGNATURE,
        "leakage_audit": leak.to_dict(),
        "base_test_loss": float(base_loss),
        "base_test_perplexity": float(base_ppl),
        "finetuned_test_loss": float(ft_loss),
        "finetuned_test_perplexity": float(ft_ppl),
        "test_loss_improvement": float(test_loss_improvement),
        "test_loss_improvement_pct": float(test_loss_improvement_pct),
        "base_aggregate": base_agg,
        "finetuned_aggregate": ft_agg,
        "paired_char_f1": paired_cf1,
        "paired_rouge_l_f1": paired_rl,
        "gate": gate.to_dict(),
        "reasons": list(gate.reasons),
        "generation_max_new_tokens": int(json.loads((output_dir / "eval_manifest.json").read_text(encoding="utf-8"))["generation_max_new_tokens"]),
    }
    atomic_write_json(output_dir / "phase5d_report.json", report)

    log.info("=" * 60)
    log.info("[result] PHASE5D_CODE = %s", gate.code)
    log.info("[result] READY_FOR_PHASE5E = %s", gate.ready_for_phase5e)
    log.info("[result] base_loss=%.6f ft_loss=%.6f improvement=%.6f (%.3f%%)",
             base_loss, ft_loss, test_loss_improvement, test_loss_improvement_pct)
    log.info("[result] base char_f1=%.4f ft char_f1=%.4f", base_agg["mean_char_f1"], ft_agg["mean_char_f1"])
    log.info("[result] base rouge_l=%.4f ft rouge_l=%.4f", base_agg["mean_rouge_l_f1"], ft_agg["mean_rouge_l_f1"])
    log.info("[result] paired cf1: ft_wins=%d base_wins=%d ties=%d",
             paired_cf1["finetuned_wins"], paired_cf1["base_wins"], paired_cf1["ties"])
    log.info("[result] paired rl: ft_wins=%d base_wins=%d ties=%d",
             paired_rl["finetuned_wins"], paired_rl["base_wins"], paired_rl["ties"])
    log.info("=" * 60)

    return 0 if gate.code == "PASS" else 50


def _write_qualitative_review(path: Path, by_id: Dict[str, Dict[str, Any]], paired_cf1: Dict[str, Any]) -> None:
    items = []
    for sid, e in by_id.items():
        delta = (e.get("finetuned_char_f1", 0.0) or 0.0) - (e.get("base_char_f1", 0.0) or 0.0)
        items.append({
            "sample_id": sid,
            "delta": delta,
            "data": e,
        })
    items.sort(key=lambda x: x["delta"], reverse=True)
    top5 = items[:5]
    bottom5 = items[-5:][::-1]
    median = items[len(items) // 2]

    def _fmt_entry(label: str, entry: Dict[str, Any]) -> str:
        d = entry["data"]
        lines = []
        lines.append(f"### {label}  (sample_id=`{entry['sample_id']}`, delta_char_f1={entry['delta']:+.4f})")
        lines.append("")
        lines.append("**Prompt:**")
        for m in d.get("prompt_messages", []):
            lines.append(f"- {m['role']}: {m['content'][:200]}")
        lines.append("")
        lines.append(f"**Reference:**")
        ref_text = d.get("reference", "") or ""
        lines.append(f"```\n{ref_text[:500]}\n```")
        lines.append("")
        lines.append("**BASE:**")
        base_text = d.get("base_generation", "") or ""
        lines.append(f"```\n{base_text[:500]}\n```")
        lines.append("")
        lines.append("**FINETUNED:**")
        ft_text = d.get("finetuned_generation", "") or ""
        lines.append(f"```\n{ft_text[:500]}\n```")
        lines.append("")
        lines.append(
            "| metric | base | finetuned |\n|---|---|---|\n"
            f"| char_f1 | {d.get('base_char_f1', 0):.4f} | {d.get('finetuned_char_f1', 0):.4f} |\n"
            f"| rouge_l_f1 | {d.get('base_rouge_l_f1', 0):.4f} | {d.get('finetuned_rouge_l_f1', 0):.4f} |\n"
            f"| exact_match | {d.get('base_normalized_exact_match', 0)} | {d.get('finetuned_normalized_exact_match', 0)} |\n"
            f"| generated_tokens | {d.get('base_generated_tokens', 0)} | {d.get('finetuned_generated_tokens', 0)} |\n"
            f"| hit_max_tokens | {d.get('base_hit_max_tokens', False)} | {d.get('finetuned_hit_max_tokens', False)} |"
        )
        lines.append("")
        return "\n".join(lines)

    md = []
    md.append("# Phase 5D — Qualitative Review (frozen)")
    md.append("")
    md.append(
        "Deterministic selection by ``finetuned_char_f1 - base_char_f1``."
        "Not used for any parameter tuning — human inspection only."
    )
    md.append("")
    md.append("## Top 5 Improvements (finetuned > base)")
    md.append("")
    for i, e in enumerate(top5, 1):
        md.append(_fmt_entry(f"#{i} improvement", e))
        md.append("")
    md.append("## Top 5 Regressions (base > finetuned)")
    md.append("")
    for i, e in enumerate(bottom5, 1):
        md.append(_fmt_entry(f"#{i} regression", e))
        md.append("")
    md.append("## Median sample (mid)")
    md.append("")
    md.append(_fmt_entry("median", median))

    path.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    log = get_logger("evaluate_qlora_test")
    log.setLevel(getattr(__import__("logging"), args.log_level.upper(), __import__("logging").INFO))
    # Add file handler
    log_path = args.output_dir / "evaluate_qlora_test.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import logging
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    try:
        return _run(args, log)
    except Exception as e:
        import traceback
        log.error("UNCAUGHT EXCEPTION: %s", e)
        log.error(traceback.format_exc())
        atomic_write_json(args.output_dir / "phase5d_report.json", {
            "status": "BLOCKED",
            "code": "UNCAUGHT_EXCEPTION",
            "reason": str(e),
            "traceback": traceback.format_exc(),
        })
        return 50


if __name__ == "__main__":
    sys.exit(main())
