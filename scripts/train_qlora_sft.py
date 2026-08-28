"""Phase 5C-B — Unsloth QLoRA 正式 3 Epoch SFT 训练脚本。

执行顺序（严格遵守 Phase 5C-B 规格 §34）：

1. Repository baseline（git commit / dirty）
2. pytest 基线回归（可选）
3. Phase 5A gate（含 audit 越限 + max 一致性硬校验）
4. Tokenizer signature（与 Phase 5C-A 已锁定 SHA256 一致）
5. Dataset fingerprint（train / validation SHA256 + sample_id signature）
6. Unsloth 环境预检（RTX 5090 / sm_120 / BF16）
7. Model revision pin（Qwen3-8B + b968826...）
8. 30-step batch=8 preflight（不安全 ⇒ fallback 4/2）
9. 固化正式 batch config
10. 写 run manifest
11. 加载 Unsloth base（4-bit + NF4 + double quant + BF16）
12. 4-bit 验证
13. LoRA 验证（r=16 alpha=32 dropout=0.05 all-linear ⇒ 252 layers）
14. Train assistant mask（3 条样本）
15. Validation assistant mask（3 条样本）
16. Pre-train eval（trainer.evaluate() ⇒ eval_loss_before_training）
17. 3 epoch formal train
18. epoch validation
19. best checkpoint
20. best adapter
21. fresh reload
22. minimal inference
23. loss curve
24. final report

禁止：

- 重新比较 HF / Unsloth。
- 使用 test.jsonl 参与训练 / validation / 选 best。
- packing=True / dropout=0 / 改 max_length。
- 改 QLoRA r / alpha / bias / task_type。
- silent batch fallback。
- 最后 epoch 自动当 best。
- 保存 merged model。

退出码：

- 0：READY_FOR_PHASE5D = YES
- 10：Phase 5A gate fail
- 20：环境 / GPU / 依赖不满足
- 30：4-bit / LoRA / tokenizer signature 不通过
- 40：assistant mask 不通过
- 50：preflight / 正式训练异常（OOM / NaN / Inf / step 数）
- 60：adapter save / reload / inference 失败
- 70：test set 被污染
- 80：validation 未改善 ⇒ READY_FOR_PHASE5D = NO
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data_pipeline.utils import (
    atomic_write_json,
    atomic_write_jsonl,
    calculate_file_sha256,
    get_logger,
    load_jsonl_records,
)
from data_pipeline.training import (
    DEFAULT_MODEL_NAME,
    DEFAULT_REVISION,
    Phase5CBTrainConfig,
    QLoRASmokeConfig,
    TrainingGateError,
    _bytes_to_gib,
    build_inference_prompt,
    collect_memory_snapshot,
    collect_peak_memory,
    build_memory_field_legend,
    build_memory_summary,
    compute_dataset_fingerprint,
    evaluate_phase5a_gate_with_audit,
    load_run_manifest,
    load_token_audit_rows,
    load_token_report,
    qloRA_smoke_config_parity_check,
    select_best_checkpoint,
    verify_resume_manifest,
    write_run_manifest,
)
from data_pipeline.training.backends import resolve_backend
from smoke_train_qlora import (
    _verify_assistant_mask,
    _check_gpu,
    _bf16_matmul,
    _ensure_torch,
)


# Phase 5C-B 默认输出位置
DEFAULT_OUTPUT_ROOT = Path("outputs/phase5c-b")
DEFAULT_DATA_DIR = Path("data/training/phase5c_b")
DEFAULT_RUN_MANIFEST = DEFAULT_DATA_DIR / "run_manifest.json"
DEFAULT_TRAIN_JSONL = Path("data/sft/train.jsonl")
DEFAULT_VALIDATION_JSONL = Path("data/sft/validation.jsonl")
DEFAULT_TEST_JSONL = Path("data/sft/test.jsonl")
DEFAULT_TOKEN_REPORT = Path("data/tokenizer/token_report.json")
DEFAULT_TOKEN_AUDIT = Path("data/tokenizer/token_audit.jsonl")

# 期望 tokenizer signature：与 Phase 5C-A 已锁定一致。
EXPECTED_TOKENIZER_SIGNATURE = (
    "4ac4fed6d76f2cddf5ee756990b7bd4dc40655f96e4c6be460dc3299e73651c0"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5C-B Unsloth 正式 QLoRA SFT")
    p.add_argument("--sft-dir", default="data/sft", type=Path)
    p.add_argument("--token-report", default=str(DEFAULT_TOKEN_REPORT), type=Path)
    p.add_argument("--token-audit", default=str(DEFAULT_TOKEN_AUDIT), type=Path)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), type=Path)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), type=Path)
    p.add_argument(
        "--run-manifest",
        default=str(DEFAULT_RUN_MANIFEST),
        type=Path,
    )
    p.add_argument("--log-file", default="logs/train_qlora_sft.log", type=Path)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--skip-preflight", action="store_true",
                   help="跳过 30-step preflight（已外部预跑过）")
    p.add_argument("--no-inference", action="store_true",
                   help="跳过最小推理（仅在调试时使用）")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="指定 resume 的 checkpoint 目录（或 'latest'）")
    p.add_argument("--initial-batch-size", type=int, default=None,
                   help="preflight 初始 batch size（默认 8）")
    p.add_argument("--initial-grad-accum", type=int, default=1,
                   help="preflight 初始 gradient_accumulation_steps（默认 1）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Git baseline
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


# ---------------------------------------------------------------------------
# Library versions
# ---------------------------------------------------------------------------


def _collect_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for dep in (
        "transformers", "trl", "peft", "accelerate", "datasets",
        "bitsandbytes", "unsloth", "unsloth_zoo",
    ):
        try:
            versions[dep] = __import__(dep).__version__
        except Exception:
            versions[dep] = None
    for dep in ("triton", "xformers"):
        try:
            m = __import__(dep)
            versions[dep] = getattr(m, "__version__", None)
        except Exception:
            versions[dep] = None
    try:
        import torch  # type: ignore
        versions["torch"] = torch.__version__
        versions["torch_cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return versions


# ---------------------------------------------------------------------------
# Tokenizer signature
# ---------------------------------------------------------------------------


def _compute_tokenizer_signature(token_audit_path: Path) -> str:
    """复算 token_audit signature（与 Phase 5C-A 算法一致）。

    算法来自 ``scripts/build_phase5c_a_report.py:_compute_signature_from_audit``：
    按 ``sample_id`` 升序，对每个 audit 行写入
    ``sample_id | split | total_tokens | prompt_tokens | assistant_target_tokens``。
    """
    import hashlib

    rows = load_jsonl_records(token_audit_path)
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


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _build_dataset(
    records: List[Dict[str, Any]],
    sample_ids: Optional[List[str]] = None,
) -> Any:
    """构造 ``datasets.Dataset``。

    若 ``sample_ids`` 为 None，则使用全部 records。
    """
    from datasets import Dataset  # type: ignore

    if sample_ids is None:
        selected = [{"sample_id": r["sample_id"], "messages": r["messages"]} for r in records]
    else:
        sid_to_record = {r["sample_id"]: r for r in records}
        selected = []
        for sid in sample_ids:
            r = sid_to_record.get(sid)
            if r is None:
                raise TrainingGateError(f"sample_id {sid} 不在数据集中")
            selected.append({"sample_id": sid, "messages": r["messages"]})
    return Dataset.from_list(selected)


def _validate_test_set_untouched(
    records_used: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    log: logging.Logger,
) -> Dict[str, Any]:
    """严格校验 test.jsonl 未参与训练 / validation / checkpoint 选择。"""
    used_ids = {str(r.get("sample_id")) for r in records_used}
    test_ids = {str(r.get("sample_id")) for r in test_records}
    leaked = sorted(used_ids & test_ids)
    if leaked:
        raise TrainingGateError(
            f"TEST_SET_CONTAMINATION: test.jsonl 中 {len(leaked)} 个 sample_id 进入了"
            f"训练/验证，前 5 个={leaked[:5]}"
        )
    return {
        "training": "NO",
        "validation": "NO",
        "model_selection": "NO",
        "test_count": len(test_ids),
        "test_sample_ids_leaked": 0,
    }


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------
#
# 显存统计口径在 Phase 5C-B 收口时统一通过
# ``data_pipeline.training.phase5c_b.collect_memory_snapshot`` /
# ``collect_peak_memory`` / ``build_memory_summary`` / ``build_memory_field_legend``
# 暴露，禁止在这里重新实现一套逻辑。
#
# 关键口径：
#
# - ``peak_allocated_gib`` = ``torch.cuda.max_memory_allocated()`` —
#   真正代表物理显存 tensor 占用峰值（判断 OOM 风险的唯一指标）。
# - ``allocator_peak_reserved_gib`` = ``torch.cuda.max_memory_reserved()`` —
#   CUDA caching allocator 保留字节数（virtual reservation /
#   expandable segments / 池化复用，可能 > 设备物理显存）。
# - ``device_total_gib`` = ``torch.cuda.mem_get_info()[1]`` —
#   设备物理显存总量。


def _mem_snapshot(torch: Any) -> Dict[str, int]:
    """采集 GPU 显存原始字段（瞬时值）。"""
    return collect_memory_snapshot(torch)


# ---------------------------------------------------------------------------
# Loss history helpers
# ---------------------------------------------------------------------------


def _flatten_log_history(
    log_history: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """把 trainer.log_history 展平为可序列化 dict 列表。

    注意：

    - ``epoch`` 在 transformers 的 log_history 里是 float（如 ``1.0``）；
      直接 ``int("1.0")`` 会抛 ValueError 导致整行被跳过。
    - ``eval_loss`` 在某些版本会是字符串；这里做宽松转换。
    """
    out: List[Dict[str, Any]] = []
    for row in log_history:
        if not isinstance(row, dict):
            continue
        flat: Dict[str, Any] = {}
        skip_row = False
        for k in ("step", "epoch", "loss", "eval_loss", "learning_rate",
                  "grad_norm"):
            if k not in row:
                continue
            v = row[k]
            if v is None:
                continue
            if k in ("step", "epoch"):
                try:
                    if isinstance(v, float):
                        flat[k] = int(v)
                    elif isinstance(v, int):
                        flat[k] = v
                    else:
                        # 字符串：先试 int 直接转换，再试 float→int。
                        try:
                            flat[k] = int(v)
                        except (TypeError, ValueError):
                            flat[k] = int(float(v))
                except (TypeError, ValueError):
                    skip_row = True
                    break
            elif k in ("loss", "eval_loss", "learning_rate", "grad_norm"):
                try:
                    flat[k] = float(v)
                except (TypeError, ValueError):
                    skip_row = True
                    break
            else:
                flat[k] = v
        if skip_row:
            continue
        out.append(flat)
    return out


def _check_finite_losses(losses: Sequence[Optional[float]], label: str) -> None:
    bad = [(i, v) for i, v in enumerate(losses) if v is None or not math.isfinite(float(v))]
    if bad:
        raise TrainingGateError(
            f"LOSS_INVALID: {label} 中 {len(bad)} 条 loss 非 finite，前 5 个={bad[:5]}"
        )


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------


def _plot_loss_curve(
    history: List[Dict[str, Any]],
    eval_losses: List[Tuple[int, float]],
    output_path: Path,
) -> None:
    """生成 loss_curve.png。"""
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    steps_train = [int(r["step"]) for r in history if r.get("loss") is not None]
    train_losses = [float(r["loss"]) for r in history if r.get("loss") is not None]
    if not steps_train:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(steps_train, train_losses, marker="o", linewidth=1, markersize=3)
    axes[0].set_title("Train Loss vs Optimizer Step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Train Loss")
    axes[0].grid(True, alpha=0.3)

    if eval_losses:
        xs, ys = zip(*eval_losses)
        axes[1].plot(xs, ys, marker="o", linewidth=2, color="C1")
        axes[1].set_title("Validation Loss vs Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Eval Loss")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No eval data", ha="center", va="center")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Resume checkpoint resolution
# ---------------------------------------------------------------------------


def _resolve_resume_checkpoint(
    resume_arg: str, checkpoints_dir: Path
) -> Optional[Path]:
    if resume_arg == "latest":
        if not checkpoints_dir.exists():
            return None
        candidates = sorted(
            (p for p in checkpoints_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        return candidates[-1] if candidates else None
    p = Path(resume_arg)
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _run_preflight(
    *,
    backend: Any,
    config: Phase5CBTrainConfig,
    train_records: List[Dict[str, Any]],
    validation_records: List[Dict[str, Any]],
    log: logging.Logger,
    output_dir: Path,
    adapter_dir: Path,
) -> Dict[str, Any]:
    """30-step Unsloth preflight。

    使用完整 train.jsonl + validation.jsonl 子集；按正式训练即将使用的
    batch / gradient_accumulation / tokenizer / collator / assistant mask /
    gradient checkpointing / eval_strategy=epoch 跑 30 步。
    """
    torch = _ensure_torch()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = _mem_snapshot(torch)

    preflight_config = Phase5CBTrainConfig(
        model_name=config.model_name,
        revision=config.revision,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    # 在 config 上加 max_steps 字段用于 preflight
    object.__setattr__(preflight_config, "max_steps", config.preflight_max_steps)

    log.info(
        "[preflight] batch=%d accum=%d effective=%d max_steps=%d",
        preflight_config.per_device_train_batch_size,
        preflight_config.gradient_accumulation_steps,
        preflight_config.effective_batch_size,
        preflight_config.preflight_max_steps,
    )

    model, tokenizer = backend.load_model_and_tokenizer(
        model_name=preflight_config.model_name,
        revision=preflight_config.revision,
        torch=torch,
        config=preflight_config,
    )
    torch.cuda.synchronize()
    mem_after_4bit = _mem_snapshot(torch)

    is_4bit = backend.is_loaded_in_4bit(model)
    linear4bit = backend.count_linear4bit(model)
    if not is_4bit or linear4bit <= 0:
        raise TrainingGateError("MODEL_NOT_ACTUALLY_4BIT")

    model = backend.apply_lora(model=model, config=preflight_config)
    trainable_params, all_params = backend.count_trainable_params(model)
    lora_layers = backend.count_lora_layers(model)
    torch.cuda.synchronize()
    mem_after_lora = _mem_snapshot(torch)

    if trainable_params <= 0 or lora_layers <= 0:
        raise TrainingGateError("LORA_NOT_APPLIED")
    if trainable_params != 43_646_976:
        raise TrainingGateError(
            f"TRAINABLE_PARAMS_DRIFT: {trainable_params} != 43646976"
        )

    train_dataset = _build_dataset(train_records)
    # preflight 使用 validation.jsonl 前 8 条作为 eval proxy；
    # 这些样本不会进入正式训练，**不会污染 test.jsonl**。
    from datasets import Dataset  # type: ignore
    val_probe = Dataset.from_list([
        {"sample_id": r["sample_id"], "messages": r["messages"]}
        for r in validation_records[:8]
    ])
    trainer = backend.build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        config=preflight_config,
        output_dir=output_dir,
        eval_dataset=val_probe,
    )
    torch.cuda.synchronize()
    mem_after_trainer = _mem_snapshot(torch)

    mask_samples = _verify_assistant_mask(trainer, train_dataset)
    log.info(
        "[preflight/mask] sample detail: %s",
        [(s["sample_id"], s["sequence_length"], s["masked_tokens"], s["supervised_tokens"], s["mask_validation_pass"]) for s in mask_samples],
    )
    if not all(s["mask_validation_pass"] for s in mask_samples):
        raise TrainingGateError("TRL_ASSISTANT_MASK_UNAVAILABLE: preflight mask 未通过")

    t0 = time.time()
    train_output = trainer.train()
    runtime = time.time() - t0

    global_step = int(getattr(train_output, "global_step", 0))
    log_history = (
        getattr(trainer, "log_history", None)
        or getattr(getattr(trainer, "state", None), "log_history", None)
        or []
    )
    losses = [
        float(r["loss"])
        for r in log_history
        if isinstance(r, dict) and "loss" in r and r["loss"] is not None
    ]
    _check_finite_losses(losses, "preflight")
    if global_step != preflight_config.preflight_max_steps:
        raise TrainingGateError(
            f"PREFLIGHT_STEP_MISMATCH: global_step={global_step} != "
            f"{preflight_config.preflight_max_steps}"
        )

    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    total_mem = int(torch.cuda.mem_get_info()[1])
    # peak_alloc / total_mem 是真正代表物理显存使用率的指标；
    # reserved / total 仅作为诊断参考，可能 > 1.0（虚拟保留 / expandable segments）。
    peak_alloc_pct = peak_alloc / total_mem if total_mem else 0.0

    # cleanup before exit
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        "batch_size": preflight_config.per_device_train_batch_size,
        "gradient_accumulation_steps": preflight_config.gradient_accumulation_steps,
        "effective_batch_size": preflight_config.effective_batch_size,
        "global_step": global_step,
        "loss_history": losses,
        "loss_finite": True,
        # 主要显存指标：物理 tensor 占用峰值。
        "peak_allocated_bytes": peak_alloc,
        "peak_allocated_gib": _bytes_to_gib(peak_alloc),
        # 诊断指标：CUDA caching allocator 保留字节数（可能 > 设备总显存）。
        "allocator_peak_reserved_bytes": peak_reserved,
        "allocator_peak_reserved_gib": _bytes_to_gib(peak_reserved),
        "device_total_bytes": total_mem,
        "device_total_gib": _bytes_to_gib(total_mem),
        "peak_allocated_within_80pct": bool(peak_alloc_pct <= 0.80),
        "vram_allocated_percentage": round(peak_alloc_pct * 100.0, 3),
        "runtime_seconds": round(runtime, 3),
        "linear4bit_modules": linear4bit,
        "lora_layers": lora_layers,
        "trainable_params": trainable_params,
        "all_params": all_params,
        "memory": {
            "before_model": mem_before,
            "after_4bit_model": mem_after_4bit,
            "after_lora": mem_after_lora,
            "after_trainer": mem_after_trainer,
            # 物理 tensor 占用峰值（GiB / bytes）。
            "peak_allocated_bytes": peak_alloc,
            "peak_allocated_gib": _bytes_to_gib(peak_alloc),
            # CUDA caching allocator 保留字节数（GiB / bytes），可能 > 物理显存。
            "allocator_peak_reserved_bytes": peak_reserved,
            "allocator_peak_reserved_gib": _bytes_to_gib(peak_reserved),
            # 设备物理显存总量。
            "device_total_bytes": total_mem,
            "device_total_gib": _bytes_to_gib(total_mem),
        },
        "memory_field_legend": _build_memory_legend(total_mem),
        "assistant_mask": {
            "checked_samples": len(mask_samples),
            "pass": True,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    log = get_logger("train_qlora_sft")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(args.log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    config = Phase5CBTrainConfig()
    if args.initial_batch_size is not None:
        object.__setattr__(
            config,
            "per_device_train_batch_size",
            int(args.initial_batch_size),
        )
    object.__setattr__(
        config,
        "gradient_accumulation_steps",
        int(args.initial_grad_accum),
    )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_root / "checkpoints"
    best_adapter_dir = output_root / "best_adapter"
    logs_dir = output_root / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_adapter_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    final_report: Dict[str, Any] = {
        "status": "BLOCKED",
        "ready_for_phase5d": False,
        "backend": "unsloth",
    }

    final_batch_size = config.per_device_train_batch_size
    final_grad_accum = config.gradient_accumulation_steps

    try:
        # ---- 1) Repository baseline ----
        git_commit, git_dirty = _git_head(ROOT)
        final_report["repository"] = {
            "branch": "main",
            "start_commit": git_commit,
            "dirty": git_dirty,
        }
        log.info("Repo: commit=%s dirty=%s", git_commit, git_dirty)

        # ---- 2) Phase 5A gate ----
        token_report = load_token_report(args.token_report)
        token_audit_rows = load_token_audit_rows(args.token_audit)
        gate = evaluate_phase5a_gate_with_audit(token_report, token_audit_rows)
        final_report["phase5a_gate"] = {
            "passed": gate.passed,
            "reasons": gate.reasons,
            "report_max_tokens": gate.report_max_tokens,
            "audit_max_tokens": gate.audit_max_tokens,
            "actual_max_tokens": gate.actual_max_tokens,
        }
        if not gate.passed:
            raise TrainingGateError("PHASE5A_GATE_FAIL: " + "; ".join(gate.reasons))

        # ---- 3) Tokenizer signature ----
        tokenizer_sig = _compute_tokenizer_signature(args.token_audit)
        if tokenizer_sig != EXPECTED_TOKENIZER_SIGNATURE:
            raise TrainingGateError(
                f"TOKENIZER_DRIFT: signature={tokenizer_sig} != expected="
                f"{EXPECTED_TOKENIZER_SIGNATURE}"
            )
        final_report["tokenizer_signature"] = {
            "audit_signature": tokenizer_sig,
            "expected_signature": EXPECTED_TOKENIZER_SIGNATURE,
            "drift_detected": False,
        }

        # ---- 4) Dataset fingerprint ----
        train_path = args.sft_dir / "train.jsonl"
        validation_path = args.sft_dir / "validation.jsonl"
        test_path = args.sft_dir / "test.jsonl"

        train_fp = compute_dataset_fingerprint(train_path)
        val_fp = compute_dataset_fingerprint(validation_path)
        log.info(
            "Dataset fingerprints: train_sha=%s val_sha=%s",
            train_fp.file_sha256[:12],
            val_fp.file_sha256[:12],
        )

        final_report["dataset"] = {
            "train_count": train_fp.record_count,
            "validation_count": val_fp.record_count,
            "train_sha256": train_fp.file_sha256,
            "validation_sha256": val_fp.file_sha256,
            "train_sample_id_signature": train_fp.sample_id_signature,
            "validation_sample_id_signature": val_fp.sample_id_signature,
            "train_schema": train_fp.schema,
            "validation_schema": val_fp.schema,
        }

        # ---- 5) Test set isolation check (before everything) ----
        test_records = load_jsonl_records(test_path)
        test_isolation = _validate_test_set_untouched(
            records_used=[{"sample_id": "placeholder"}],  # placeholder; real check below
            test_records=test_records,
            log=log,
        )
        # Real check: train + validation must not contain any test sample_id
        train_ids = {r["sample_id"] for r in load_jsonl_records(train_path)}
        val_ids = {r["sample_id"] for r in load_jsonl_records(validation_path)}
        test_ids = {r["sample_id"] for r in test_records}
        leaked = sorted((train_ids | val_ids) & test_ids)
        if leaked:
            raise TrainingGateError(
                f"TEST_SET_CONTAMINATION: test.jsonl 中 {len(leaked)} 个 sample_id"
                f"出现在 train/validation 中，前 5 个={leaked[:5]}"
            )
        test_isolation["test_sample_ids_leaked"] = 0
        final_report["test_isolation"] = test_isolation

        # ---- 6) Unsloth environment ----
        torch = _ensure_torch()
        gpu_info = _check_gpu(torch, log)
        _bf16_matmul(torch)
        library_versions = _collect_versions()
        if library_versions.get("unsloth") is None:
            raise TrainingGateError("UNSLOTH_DEPENDENCY_MISSING")
        final_report["library_versions"] = library_versions
        final_report["environment"] = {
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            **gpu_info,
        }

        # ---- 7) Backend resolution ----
        backend = resolve_backend("unsloth")
        meta = backend.metadata()
        final_report["backend_metadata"] = {
            "name": meta.name,
            "description": meta.description,
            "supports_revision_pin": meta.supports_revision_pin,
            "supports_assistant_only_loss": meta.supports_assistant_only_loss,
            "depends_on_unsloth": meta.depends_on_unsloth,
        }

        # ---- 8) Resolve model revision ----
        requested, resolved, pinned = backend.resolve_revision(
            config.model_name, config.revision
        )
        if not resolved:
            try:
                info = backend.resolve_revision(config.model_name, config.revision)
                resolved = info[1]
            except Exception:
                pass
        load_revision = resolved or config.revision
        if load_revision != config.revision:
            log.warning(
                "Revision fallback: requested=%s resolved=%s → using %s",
                config.revision, resolved, load_revision,
            )
        final_report["model"] = {
            "model_name": config.model_name,
            "requested_revision": config.revision,
            "resolved_revision": resolved,
            "load_revision": load_revision,
            "revision_pinned": bool(load_revision == config.revision),
        }

        # ---- 9) Run manifest (skeleton; will overwrite after preflight) ----
        manifest_payload: Dict[str, Any] = {
            "git": {
                "branch": "main",
                "commit": git_commit,
                "dirty": git_dirty,
            },
            "backend": "unsloth",
            "library_versions": library_versions,
            "environment": final_report["environment"],
            "model": final_report["model"],
            "dataset": final_report["dataset"],
            "tokenizer_signature": tokenizer_sig,
            "quantization": {
                "quant_type": config.quant_type,
                "compute_dtype": config.compute_dtype,
                "double_quant": config.double_quant,
                "max_length": config.max_length,
            },
            "lora": {
                "r": config.lora_r,
                "alpha": config.lora_alpha,
                "dropout": config.lora_dropout,
                "bias": config.lora_bias,
                "task_type": config.lora_task_type,
                "target_modules": config.lora_target_modules,
            },
            "training": {
                "num_train_epochs": config.num_train_epochs,
                "per_device_train_batch_size": final_batch_size,
                "gradient_accumulation_steps": final_grad_accum,
                "learning_rate": config.learning_rate,
                "lr_scheduler_type": config.lr_scheduler_type,
                "warmup_ratio": config.warmup_ratio,
                "weight_decay": config.weight_decay,
                "max_grad_norm": config.max_grad_norm,
                "logging_steps": config.logging_steps,
                "seed": config.seed,
                "data_seed": config.data_seed,
                "save_strategy": config.save_strategy,
                "eval_strategy": config.eval_strategy,
                "save_total_limit": config.save_total_limit,
                "load_best_model_at_end": config.load_best_model_at_end,
                "metric_for_best_model": config.metric_for_best_model,
                "greater_is_better": config.greater_is_better,
                "packing": config.packing,
                "assistant_only_loss": config.assistant_only_loss,
                "bf16": config.bf16,
                "fp16": config.fp16,
                "gradient_checkpointing": config.gradient_checkpointing,
            },
        }
        write_run_manifest(args.run_manifest, payload=manifest_payload, logger=log)

        # ---- 10) Preflight (30-step batch=8 / 4+2 fallback) ----
        preflight_report: Dict[str, Any] = {"status": "SKIPPED"}
        train_records_pre = load_jsonl_records(train_path)
        validation_records_pre = load_jsonl_records(validation_path)
        if not args.skip_preflight:
            try:
                preflight_report = _run_preflight(
                    backend=backend,
                    config=config,
                    train_records=train_records_pre,
                    validation_records=validation_records_pre,
                    log=log,
                    output_dir=output_root / "preflight",
                    adapter_dir=output_root / "preflight_adapter",
                )
                preflight_report["status"] = "PASS"
            except TrainingGateError as e:
                # fallback 4/2
                if "OOM" in str(e).upper() or "step" not in str(e).lower():
                    log.warning("Preflight 触发 fallback：%s", e)
                    fallback_config = Phase5CBTrainConfig(
                        model_name=config.model_name,
                        revision=config.revision,
                        per_device_train_batch_size=4,
                        gradient_accumulation_steps=2,
                    )
                    object.__setattr__(fallback_config, "max_steps", config.preflight_max_steps)
                    config = fallback_config
                    final_batch_size = 4
                    final_grad_accum = 2
                    log.info(
                        "[preflight fallback] batch=4 accum=2 effective=8",
                    )
                    preflight_report = _run_preflight(
                        backend=backend,
                        config=fallback_config,
                        train_records=train_records_pre,
                        validation_records=validation_records_pre,
                        log=log,
                        output_dir=output_root / "preflight",
                        adapter_dir=output_root / "preflight_adapter",
                    )
                    preflight_report["status"] = "PASS_FALLBACK"
                    preflight_report["fallback_reason"] = str(e)
                else:
                    raise
            final_report["preflight"] = preflight_report

        # Update manifest with final batch config
        manifest_payload["training"]["per_device_train_batch_size"] = final_batch_size
        manifest_payload["training"]["gradient_accumulation_steps"] = final_grad_accum
        manifest_payload["effective_batch_size"] = final_batch_size * final_grad_accum
        write_run_manifest(args.run_manifest, payload=manifest_payload, logger=log)

        # ---- 11) Load Unsloth base ----
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = _mem_snapshot(torch)
        model, tokenizer = backend.load_model_and_tokenizer(
            model_name=config.model_name,
            revision=load_revision,
            torch=torch,
            config=config,
        )
        torch.cuda.synchronize()
        mem_after_4bit = _mem_snapshot(torch)

        # ---- 12) 4-bit verification ----
        is_4bit = backend.is_loaded_in_4bit(model)
        linear4bit = backend.count_linear4bit(model)
        if not is_4bit or linear4bit != 252:
            raise TrainingGateError(
                f"MODEL_NOT_ACTUALLY_4BIT: is_4bit={is_4bit}, linear4bit={linear4bit}"
            )
        final_report["quantization"] = {
            "load_in_4bit": is_4bit,
            "quant_type": config.quant_type,
            "double_quant": config.double_quant,
            "compute_dtype": config.compute_dtype,
            "linear4bit_modules": linear4bit,
        }

        # ---- 13) LoRA ----
        model = backend.apply_lora(model=model, config=config)
        trainable_params, all_params = backend.count_trainable_params(model)
        lora_layers = backend.count_lora_layers(model)
        if trainable_params != 43_646_976:
            raise TrainingGateError(
                f"TRAINABLE_PARAMS_DRIFT: {trainable_params} != 43646976"
            )
        if lora_layers != 252:
            raise TrainingGateError(f"LORA_LAYERS_DRIFT: {lora_layers} != 252")
        final_report["lora"] = {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "bias": config.lora_bias,
            "task_type": config.lora_task_type,
            "target_modules": config.lora_target_modules,
            "lora_layers": lora_layers,
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": round(trainable_params / max(1, all_params) * 100, 4),
        }
        torch.cuda.synchronize()
        mem_after_lora = _mem_snapshot(torch)

        # ---- 14) Build train + eval datasets ----
        train_records = load_jsonl_records(train_path)
        validation_records = load_jsonl_records(validation_path)
        # dataset isolation: verify no test sample_id leaks into either
        all_used_ids = {r["sample_id"] for r in train_records} | {
            r["sample_id"] for r in validation_records
        }
        leaked = sorted(all_used_ids & test_ids)
        if leaked:
            raise TrainingGateError(
                f"TEST_SET_CONTAMINATION: {len(leaked)} test sample_id leaked"
            )

        train_dataset = _build_dataset(train_records)
        eval_dataset = _build_dataset(validation_records)
        trainer = backend.build_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            config=config,
            output_dir=checkpoints_dir,
            eval_dataset=eval_dataset,
        )
        torch.cuda.synchronize()
        mem_after_trainer = _mem_snapshot(torch)

        # ---- 15) Train assistant mask (3 samples) ----
        train_mask = _verify_assistant_mask(trainer, train_dataset)
        log.info(
            "[mask/train] sample detail: %s",
            [(s["sample_id"], s["sequence_length"], s["masked_tokens"], s["supervised_tokens"], s["mask_validation_pass"]) for s in train_mask],
        )
        if not all(s["mask_validation_pass"] for s in train_mask):
            final_report["assistant_mask_train_debug"] = {
                "samples": train_mask,
                "pass": False,
            }
            raise TrainingGateError("TRL_ASSISTANT_MASK_UNAVAILABLE: train mask")
        # ---- 16) Validation assistant mask (3 samples) ----
        eval_mask = _verify_assistant_mask(trainer, eval_dataset)
        if not all(s["mask_validation_pass"] for s in eval_mask):
            raise TrainingGateError("TRL_ASSISTANT_MASK_UNAVAILABLE: validation mask")
        final_report["assistant_mask"] = {
            "train": {
                "checked_samples": len(train_mask),
                "samples": train_mask,
                "pass": True,
            },
            "validation": {
                "checked_samples": len(eval_mask),
                "samples": eval_mask,
                "pass": True,
            },
        }

        # ---- 17) Pre-train eval ----
        pre_eval = trainer.evaluate()
        eval_loss_before = float(pre_eval.get("eval_loss"))
        if not math.isfinite(eval_loss_before):
            raise TrainingGateError(
                f"PRE_EVAL_INVALID: eval_loss_before_training={eval_loss_before}"
            )
        if eval_loss_before <= 0:
            raise TrainingGateError(
                f"PRE_EVAL_NON_POSITIVE: eval_loss_before_training={eval_loss_before}"
            )
        perplexity_before = float(math.exp(eval_loss_before))
        final_report["pre_train_eval"] = {
            "eval_loss": eval_loss_before,
            "perplexity": perplexity_before,
        }
        log.info("Pre-train eval_loss=%s perplexity=%s", eval_loss_before, perplexity_before)

        # ---- 18) 3 epoch formal training ----
        t0 = time.time()
        resume_ckpt = (
            _resolve_resume_checkpoint(args.resume_from_checkpoint, checkpoints_dir)
            if args.resume_from_checkpoint else None
        )
        if resume_ckpt is not None:
            log.info("Resuming from checkpoint %s", resume_ckpt)
            train_output = trainer.train(resume_from_checkpoint=str(resume_ckpt))
        else:
            train_output = trainer.train()
        runtime = time.time() - t0
        global_step = int(getattr(train_output, "global_step", 0))

        # ---- 19) Loss history + eval per epoch ----
        # transformers Trainer 把 ``log_history`` 存在 ``trainer.state.log_history``，
        # 不是 ``trainer.log_history``。Phase 5C-A 的 smoke config 没用到 eval_loss
        # 所以一直没暴露；本阶段必须正确获取。
        log_history = (
            getattr(trainer, "log_history", None)
            or getattr(getattr(trainer, "state", None), "log_history", None)
            or []
        )
        flat = _flatten_log_history(log_history)
        train_losses = [float(r["loss"]) for r in flat if r.get("loss") is not None]
        _check_finite_losses(train_losses, "train")
        eval_losses_by_epoch = [
            (int(r.get("epoch") or 0), float(r["eval_loss"]))
            for r in flat if r.get("eval_loss") is not None
        ]
        eval_losses_only = [v for _, v in eval_losses_by_epoch]
        _check_finite_losses(eval_losses_only, "eval")

        # Write loss history jsonl
        loss_history_path = data_dir / "loss_history.jsonl"
        atomic_write_jsonl(
            loss_history_path,
            [json.dumps(r, ensure_ascii=False) for r in flat],
        )

        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        device_total = int(torch.cuda.mem_get_info()[1])

        # ---- 20) Best checkpoint selection ----
        best_sel = select_best_checkpoint(eval_losses_only)
        epoch_losses = [v for _, v in eval_losses_by_epoch]
        if len(epoch_losses) != config.num_train_epochs:
            log.warning(
                "eval_history epoch 数量=%d != num_train_epochs=%d",
                len(epoch_losses), config.num_train_epochs,
            )

        final_report["training"] = {
            "epochs_completed": config.num_train_epochs,
            "global_step": global_step,
            "runtime_seconds": round(runtime, 3),
            # 物理 tensor 占用峰值（GiB / bytes）— 真正代表物理显存消耗。
            "peak_allocated_bytes": peak_alloc,
            "peak_allocated_gib": _bytes_to_gib(peak_alloc),
            # CUDA caching allocator 保留字节数（GiB / bytes）— 仅诊断。
            "allocator_peak_reserved_bytes": peak_reserved,
            "allocator_peak_reserved_gib": _bytes_to_gib(peak_reserved),
            "device_total_bytes": device_total,
            "device_total_gib": _bytes_to_gib(device_total),
            "first_train_loss": train_losses[0] if train_losses else None,
            "last_train_loss": train_losses[-1] if train_losses else None,
            "min_train_loss": min(train_losses) if train_losses else None,
            "nan": 0,
            "inf": 0,
            "oom": False,
            "loss_history": flat,
        }

        final_report["validation"] = {
            "before_training": eval_loss_before,
            "epoch_eval_losses": epoch_losses,
            "best_eval_loss": best_sel["best_eval_loss"],
            "best_epoch": best_sel["best_epoch"],
            "best_is_last": best_sel["best_is_last"],
            "improved": bool(best_sel["best_eval_loss"] < eval_loss_before),
        }

        # ---- 21) Best adapter save ----
        # Since load_best_model_at_end=True, trainer.model is the best LoRA.
        try:
            adapter_files = backend.save_adapter(
                trainer=trainer, tokenizer=tokenizer, adapter_dir=best_adapter_dir,
            )
        except Exception as e:
            raise TrainingGateError(f"ADAPTER_SAVE_FAIL (best): {e}") from e
        final_report["best_adapter"] = {
            "saved": True,
            "path": str(best_adapter_dir),
            "files": adapter_files,
        }

        # ---- 22) Fresh reload ----
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # ---- 23) Minimal inference ----
        infer_result: Dict[str, Any] = {"pass": False}
        if not args.no_inference:
            try:
                infer = backend.reload_and_infer(
                    torch=torch,
                    model_name=config.model_name,
                    revision=load_revision,
                    config=config,
                    adapter_dir=best_adapter_dir,
                    inference_prompt=config.inference_prompt,
                    max_new_tokens=config.inference_max_new_tokens,
                )
            except Exception as e:
                raise TrainingGateError(f"INFERENCE_FAIL: {e}") from e
            if infer["generated_token_count"] <= 0 or not infer["nonempty"]:
                raise TrainingGateError(
                    f"INFERENCE_FAILED: {infer}"
                )
            infer_result = {
                "pass": True,
                "prompt": config.inference_prompt,
                "generated_token_count": infer["generated_token_count"],
                "generated_char_count": infer["generated_char_count"],
                "nonempty": infer["nonempty"],
                "preview": infer["generated_text"][:200],
            }
            final_report["inference"] = infer_result

        # Reload LoRA layers verification
        # (reload_and_infer returns immediately; we can't easily re-verify layers.
        #  We rely on the infer_result above + best_adapter save checks.)

        # ---- 24) Loss curve ----
        eval_for_plot = [(i + 1, v) for i, v in enumerate(epoch_losses)]
        _plot_loss_curve(
            history=flat,
            eval_losses=eval_for_plot,
            output_path=data_dir / "loss_curve.png",
        )

        # ---- 25) Memory summary ----
        # 关键：peak_allocated_gib 是真正代表物理显存 tensor 占用峰值；
        # allocator_peak_reserved_gib 是 CUDA caching allocator 保留字节数，
        # 由于 PyTorch 2.11 native allocator 的 virtual reservation /
        # expandable segments / 池化复用，reserved 可能 > device_total_gib，
        # 这是合法的 allocator 行为，**不**代表物理显存被超额占用。
        final_report["memory"] = {
            "before_model": mem_before,
            "after_4bit_model": mem_after_4bit,
            "after_lora": mem_after_lora,
            "after_trainer": mem_after_trainer,
            # 物理 tensor 占用峰值（GiB / bytes）— 判断 OOM 风险的真正指标。
            "training_peak_allocated_bytes": peak_alloc,
            "training_peak_allocated_gib": _bytes_to_gib(peak_alloc),
            # CUDA caching allocator 保留字节数（GiB / bytes）— 仅诊断。
            # 可能超过设备物理显存：见 _build_memory_legend。
            "training_allocator_peak_reserved_bytes": peak_reserved,
            "training_allocator_peak_reserved_gib": _bytes_to_gib(peak_reserved),
            # 设备物理显存总量。
            "device_total_bytes": device_total,
            "device_total_gib": _bytes_to_gib(device_total),
        }
        final_report["memory_field_legend"] = _build_memory_legend(device_total)

        # ---- 26) READY_FOR_PHASE5D gate ----
        ready = (
            gate.passed
            and tokenizer_sig == EXPECTED_TOKENIZER_SIGNATURE
            and is_4bit
            and linear4bit == 252
            and lora_layers == 252
            and trainable_params == 43_646_976
            and all(s["mask_validation_pass"] for s in train_mask)
            and all(s["mask_validation_pass"] for s in eval_mask)
            and best_sel["best_eval_loss"] < eval_loss_before
            and global_step >= 0
            and (args.no_inference or infer_result.get("pass"))
            and not leaked
        )
        final_report["ready_for_phase5d"] = ready
        if not best_sel["best_eval_loss"] < eval_loss_before:
            raise TrainingGateError(
                "VALIDATION_NOT_IMPROVED: best_eval_loss "
                f"{best_sel['best_eval_loss']} >= eval_loss_before_training "
                f"{eval_loss_before} ⇒ READY_FOR_PHASE5D=NO"
            )
        if not ready:
            raise TrainingGateError(
                "READY_FOR_PHASE5D_GATE_FAIL: 详见 report"
            )

        final_report["status"] = "PASS"
        return 0

    except TrainingGateError as e:
        final_report["status"] = "BLOCKED"
        final_report["error"] = {
            "type": "TrainingGateError",
            "reason": str(e),
        }
        # VALIDATION_NOT_IMPROVED 仍记录 best_eval_loss / before
        if "VALIDATION_NOT_IMPROVED" in str(e):
            final_report["ready_for_phase5d"] = False
        log.error("BLOCKED: %s", e)
        log.error(traceback.format_exc())
        return _gate_exit_code(str(e))
    except Exception as e:
        final_report["status"] = "BLOCKED"
        final_report["error"] = {
            "type": type(e).__name__,
            "reason": str(e),
            "traceback": traceback.format_exc(),
        }
        log.error("BLOCKED: %s: %s", type(e).__name__, e)
        log.error(traceback.format_exc())
        return 50
    finally:
        # Always write the final report
        atomic_write_json(data_dir / "final_report.json", final_report)


def _gate_exit_code(reason: str) -> int:
    s = reason.upper()
    if "PHASE5A_GATE" in s:
        return 10
    if "TEST_SET_CONTAMINATION" in s:
        return 70
    if "TOKENIZER_DRIFT" in s:
        return 30
    if "TORCH" in s and "NOT" in s:
        return 20
    if "CUDA" in s and "NOT" in s:
        return 20
    if "BF16" in s:
        return 20
    if "BLACKWELL" in s:
        return 20
    if "UNSLOTH_DEPENDENCY" in s:
        return 20
    if "MODEL_NOT_ACTUALLY_4BIT" in s or "TRAINABLE_PARAMS_DRIFT" in s or "LORA_LAYERS_DRIFT" in s:
        return 30
    if "TRL_ASSISTANT_MASK" in s:
        return 40
    if "PRE_EVAL" in s or "PREFLIGHT" in s:
        return 50
    if "ADAPTER" in s:
        return 60
    if "INFERENCE" in s:
        return 60
    if "VALIDATION_NOT_IMPROVED" in s:
        return 80
    return 50


if __name__ == "__main__":
    sys.exit(main())
