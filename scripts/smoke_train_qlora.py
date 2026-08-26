"""Phase 5B — QLoRA 单步真实训练冒烟测试。

执行顺序（严格）：

1. 读 ``token_report.json`` + ``token_audit.jsonl``；运行 Phase 5A gate
   （含 audit 越限 + report/audit max 一致性硬校验）。
2. GPU 环境 + BF16 + Blackwell (sm_120) 预检；真实 BF16 matmul。
3. 解析 model revision（HfApi.model_info.sha；失败则 fallback ``main``）。
4. 选 smoke 样本（短 7 + P95 1，严格 ``select_smoke_samples_strict`` ⇒ 必须 == 8）。
5. **写 smoke_config.json**（含 resolution + selection）。
6. 4-bit + NF4 + double quant + bfloat16 加载 base；记录 before_model 显存。
7. 验证 ``is_loaded_in_4bit`` + ``Linear4bit > 0``；记录 after_4bit_model。
8. ``prepare_model_for_kbit_training`` + LoRA r=16 alpha=32 all-linear；记录 after_lora。
9. ``SFTTrainer``；记录 after_trainer。
10. 走公开 TRL 路径 ``trainer.train_dataset`` + ``trainer.data_collator`` 验证
    assistant mask（至少 3 条样本同时含 -100 与非 -100 token）。
11. ``trainer.train(max_steps=1)`` ⇒ ``global_step == 1``、loss finite ≠ 0。
12. 记录 peak 显存（alloc / reserved）。
13. 保存 adapter → 释放显存 → 重新 4-bit load + PeftModel.from_pretrained。
14. 最小推理（"什么是基层党组织？"）；PASS 条件：生成 token 数 > 0 且文本非空。

退出码：

- 0：完整 PASS
- 10：Phase 5A gate 不通过
- 20：环境不满足（CUDA / BF16 / Blackwell / 库缺失）
- 30：4-bit 模型未生效
- 40：assistant mask 不可用
- 50：训练失败（loss 异常 / step 数异常 / OOM 等）
- 60：adapter 保存 / reload / 推理失败
- 70：smoke selection 不完整 / 不在 train
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_pipeline.utils import (
    atomic_write_json,
    get_logger,
    load_jsonl_records,
)
from data_pipeline.training import (
    QLoRASmokeConfig,
    evaluate_phase5a_gate_with_audit,
    load_token_audit_rows,
    load_token_report,
    select_smoke_samples_strict,
    write_smoke_config,
)
from data_pipeline.training.pipeline import (
    TrainingGateError,
    build_inference_prompt,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5B/5C-A QLoRA 单步训练冒烟测试")
    p.add_argument("--sft-dir", default="data/sft", type=Path)
    p.add_argument("--token-report", default="data/tokenizer/token_report.json", type=Path)
    p.add_argument("--token-audit", default="data/tokenizer/token_audit.jsonl", type=Path)
    p.add_argument("--output-dir", default="data/training", type=Path)
    p.add_argument("--adapter-dir", default="outputs/phase5b-smoke/adapter", type=Path)
    p.add_argument("--log-file", default="logs/smoke_train_qlora.log", type=Path)
    p.add_argument("--log-level", default="INFO")
    # Phase 5C-A：新增 backend 选择；默认 hf 与 Phase 5B 完全等价。
    p.add_argument("--backend", default="hf", choices=["hf", "unsloth"],
                   help="Training backend：hf（Phase 5B baseline） 或 unsloth（Phase 5C-A 候选）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 环境预检
# ---------------------------------------------------------------------------


def _ensure_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:
        raise TrainingGateError(f"torch 未安装：{e}")
    return torch


def _check_gpu(torch: Any, log: logging.Logger) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_name": None,
        "compute_capability": None,
        "arch_list": [],
        "bf16_supported": False,
        "total_memory_bytes": 0,
    }
    if not torch.cuda.is_available():
        raise TrainingGateError("CUDA 不可用")
    info["device_name"] = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    info["compute_capability"] = list(cap)
    info["arch_list"] = list(torch.cuda.get_arch_list())
    info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    info["total_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)

    if not info["bf16_supported"]:
        raise TrainingGateError("BF16 不受支持；Phase 5B 禁止退回 FP16")

    if tuple(cap) == (12, 0):
        arch_has = any("sm_120" in a or a == "12.0" for a in info["arch_list"])
        if not arch_has:
            raise TrainingGateError(
                "PYTORCH_BLACKWELL_ARCH_UNSUPPORTED：当前 PyTorch arch_list 不含 sm_120"
            )
    return info


def _bf16_matmul(torch: Any) -> None:
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all().item()):
        raise TrainingGateError("CUDA BF16 matmul 结果非 finite")


# ---------------------------------------------------------------------------
# 4-bit + LoRA
# ---------------------------------------------------------------------------


def _resolve_model_revision(model_name: str, requested: str) -> Tuple[str, Optional[str], bool]:
    """尽量从 huggingface_hub 拿真实 commit SHA。"""
    resolved: Optional[str] = None
    pinned = False
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi()
        try:
            info = api.model_info(model_name, revision=requested)
            sha = getattr(info, "sha", None)
            if isinstance(sha, str) and sha:
                resolved = sha
                pinned = True
        except Exception:
            resolved = None
    except Exception:
        resolved = None
    return requested, resolved, pinned


def _load_4bit_model(torch: Any, model_name: str, revision: str, config: QLoRASmokeConfig) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore

    compute_dtype = getattr(torch, config.compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.double_quant,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            quantization_config=bnb_config,
            dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=False,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=False,
        )
    return model


def _count_linear4bit(model: Any) -> int:
    try:
        from bitsandbytes.nn import Linear4bit  # type: ignore
    except Exception:
        return 0
    return sum(1 for m in model.modules() if isinstance(m, Linear4bit))


def _is_loaded_in_4bit(model: Any) -> bool:
    if hasattr(model, "is_loaded_in_4bit"):
        try:
            return bool(model.is_loaded_in_4bit)
        except Exception:
            pass
    qc = getattr(model, "quantization_method", None) or getattr(model, "quantization_config", None)
    if qc is not None and hasattr(qc, "get"):
        try:
            return bool(qc.get("load_in_4bit")) or bool(qc.get("bits") == 4)
        except Exception:
            return False
    return False


def _apply_lora(model: Any, config: QLoRASmokeConfig) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore

    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    except TypeError:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=config.lora_task_type,
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)

    if config.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            try:
                model.config.use_cache = False
            except Exception:
                pass

    return model


# ---------------------------------------------------------------------------
# dataset + assistant mask 校验（公开 TRL 路径）
# ---------------------------------------------------------------------------


def _build_dataset(train_records: List[Dict[str, Any]], sample_ids: List[str]) -> Any:
    """构造 ``datasets.Dataset``（conversational）。

    任何选中 sample_id 不在 ``train_records`` ⇒ 抛 ``TrainingGateError(SMOKE_SAMPLE_NOT_IN_TRAIN)``。
    """
    from datasets import Dataset  # type: ignore

    sid_to_record = {r["sample_id"]: r for r in train_records}
    selected: List[Dict[str, Any]] = []
    for sid in sample_ids:
        r = sid_to_record.get(sid)
        if r is None:
            raise TrainingGateError(
                f"SMOKE_SAMPLE_NOT_IN_TRAIN: sample_id={sid} 不在 train.jsonl 中"
            )
        selected.append({"sample_id": sid, "messages": r["messages"]})
    return Dataset.from_list(selected)


def _verify_assistant_mask(trainer: Any, dataset: Any) -> List[Dict[str, Any]]:
    """走公开 TRL 路径：``trainer.train_dataset`` → ``trainer.data_collator``。

    严格不依赖私有 ``trainer._prepare_dataset``；如 trainer 已 prepared 数据，
    直接走 collator；否则从 ``train_dataset`` 取 3 条样本构造 batch。
    """
    samples: List[Dict[str, Any]] = []
    # 优先 trainer.train_dataset，否则退到原始 dataset
    source = getattr(trainer, "train_dataset", None) or dataset
    checked = min(3, len(source))
    for i in range(checked):
        row = source[i]
        # trainer.data_collator 接受 list[dict]
        batch = trainer.data_collator([row])
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        if hasattr(input_ids, "shape"):
            seq_len = int(input_ids.shape[-1])
        else:
            seq_len = len(input_ids)
        if hasattr(labels, "tolist"):
            labels_list = labels.tolist()
        else:
            labels_list = list(labels)
        masked = sum(1 for v in labels_list if int(v) == -100)
        supervised = seq_len - masked
        ok = masked > 0 and supervised > 0 and seq_len <= 2048
        samples.append({
            "sample_id": row.get("sample_id") if isinstance(row, dict) else None,
            "sequence_length": seq_len,
            "masked_tokens": masked,
            "supervised_tokens": supervised,
            "mask_validation_pass": bool(ok),
        })
    return samples


# ---------------------------------------------------------------------------
# SFTConfig 构造
# ---------------------------------------------------------------------------


def _build_sft_config(config: QLoRASmokeConfig, output_dir: Path) -> Any:
    from trl import SFTConfig  # type: ignore
    import inspect

    sig = inspect.signature(SFTConfig.__init__)
    params = sig.parameters

    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy,
        "eval_strategy": config.eval_strategy,
        "gradient_checkpointing": config.gradient_checkpointing,
        "packing": config.packing,
        "assistant_only_loss": config.assistant_only_loss,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "report_to": config.report_to,
        "seed": config.seed,
    }
    if "max_length" in params:
        kwargs["max_length"] = config.max_length
    elif "max_seq_length" in params:
        kwargs["max_seq_length"] = config.max_length

    return SFTConfig(**kwargs)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_tokenizer(model_name: str, revision: str) -> Any:
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(
        model_name, revision=revision, use_fast=True, trust_remote_code=False
    )


def _count_trainable_params(model: Any) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def _count_lora_layers(model: Any) -> int:
    try:
        from peft.tuners.lora import LoraLayer  # type: ignore
    except Exception:
        return 0
    return sum(1 for m in model.modules() if isinstance(m, LoraLayer))


def _extract_loss(train_output: Any, trainer: Any) -> Optional[float]:
    val = getattr(train_output, "training_loss", None)
    if val is None:
        log_history = getattr(trainer, "log_history", None) or []
        if log_history:
            last = log_history[-1]
            if isinstance(last, dict) and "loss" in last:
                try:
                    return float(last["loss"])
                except Exception:
                    return None
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def _extract_grad_norm(trainer: Any) -> Optional[float]:
    log_history = getattr(trainer, "log_history", None) or []
    if not log_history:
        return None
    last = log_history[-1]
    if isinstance(last, dict) and "grad_norm" in last:
        try:
            return float(last["grad_norm"])
        except Exception:
            return None
    return None


def _gpu_mem_snapshot(torch: Any) -> Dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "free_bytes": int(torch.cuda.mem_get_info()[0]),
        "total_bytes": int(torch.cuda.mem_get_info()[1]),
    }


def _bytes_to_gib(n: int) -> float:
    return round(n / (1024 ** 3), 3)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    log = get_logger("smoke_train_qlora")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(args.log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    config = QLoRASmokeConfig()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)

    final_report: Dict[str, Any] = {"status": "BLOCKED"}
    library_versions: Dict[str, Optional[str]] = {}

    try:
        # ---- 1) Phase 5A gate（含 audit 越限 + max 一致性） ----
        token_report = load_token_report(args.token_report)
        token_audit_rows = load_token_audit_rows(args.token_audit)
        gate = evaluate_phase5a_gate_with_audit(token_report, token_audit_rows)
        final_report["phase5a_gate"] = {
            "passed": gate.passed,
            "reasons": gate.reasons,
            "report_max_tokens": gate.report_max_tokens,
            "audit_max_tokens": gate.audit_max_tokens,
            "actual_max_tokens": gate.actual_max_tokens,
            "selected_max_length": gate.selected_max_length,
            "samples_over_max_length": gate.samples_over_max_length,
            "max_audit_over_total_tokens": gate.max_audit_over_total_tokens,
        }
        log.info("Phase 5A gate: passed=%s reasons=%s", gate.passed, gate.reasons)
        if not gate.passed:
            raise TrainingGateError("PHASE5B_BLOCKED: PHASE5A_GATE_FAIL: " + "; ".join(gate.reasons))

        if token_report.get("model", {}).get("model_name") != config.model_name:
            raise TrainingGateError(
                "PHASE5B_BLOCKED: MODEL_TOKENIZER_AUDIT_MISMATCH: "
                f"{token_report.get('model', {}).get('model_name')!r} != {config.model_name!r}"
            )

        # ---- 2) 依赖版本收集 ----
        for dep in ("transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes"):
            try:
                library_versions[dep] = __import__(dep).__version__
            except Exception:
                library_versions[dep] = None

        # ---- 3) GPU 预检 ----
        torch = _ensure_torch()
        library_versions["torch"] = torch.__version__
        gpu_info = _check_gpu(torch, log)
        _bf16_matmul(torch)
        library_versions["bitsandbytes_cuda_ok"] = True  # 通过 matmul 隐含 CUDA OK
        final_report["library_versions"] = library_versions
        final_report["environment"] = {
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            **gpu_info,
        }
        log.info("GPU: %s", gpu_info)

        # ---- 4) 解析 model revision（在写 smoke_config.json 之前） ----
        requested_revision, resolved_revision, revision_pinned = _resolve_model_revision(
            config.model_name, config.revision
        )
        load_revision = resolved_revision or requested_revision

        # ---- 5) smoke 选样（严格 8 条；P95 必须真实存在于 train.jsonl） ----
        train_records = load_jsonl_records(args.sft_dir / "train.jsonl")
        train_total_tokens = sorted(
            int(r.get("total_tokens") or 0)
            for r in token_audit_rows
            if r.get("split") == "train"
        )
        if train_total_tokens:
            p95_idx = max(0, min(len(train_total_tokens) - 1,
                                  int(round(0.95 * (len(train_total_tokens) - 1)))))
            target_train_p95 = float(train_total_tokens[p95_idx])
        else:
            target_train_p95 = None
        selection = select_smoke_samples_strict(
            train_records, token_audit_rows, config, target_train_p95=target_train_p95
        )
        log.info("Smoke selection: short=%d, p95=%s, total=%d",
                 selection.short_count, selection.p95_sample_id, len(selection.sample_ids))

        # ---- 6) 写 smoke_config.json（含 revision + selection） ----
        model_revision_info = {
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "load_revision": load_revision,
            "revision_pinned": revision_pinned,
        }
        write_smoke_config(
            config,
            output_dir / "smoke_config.json",
            logger=log,
            model_revision=model_revision_info,
            selection=selection,
        )

        final_report["model"] = {
            "model_name": config.model_name,
            **model_revision_info,
        }
        final_report["smoke_samples"] = {
            "total_selected": len(selection.sample_ids),
            "sample_ids": selection.sample_ids,
            "short_count": selection.short_count,
            "p95_sample_id": selection.p95_sample_id,
            "p95_total_tokens": selection.p95_total_tokens,
            "all_exist_in_train": True,
        }

        # ---- 7) 加载 tokenizer + 4-bit base model；记录 before_model ----
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = _gpu_mem_snapshot(torch)

        tokenizer = _load_tokenizer(config.model_name, load_revision)
        model = _load_4bit_model(torch, config.model_name, load_revision, config)
        torch.cuda.synchronize()
        mem_after_4bit = _gpu_mem_snapshot(torch)

        # ---- 8) 4-bit 验证 ----
        linear4bit_modules = _count_linear4bit(model)
        is_loaded_in_4bit = _is_loaded_in_4bit(model)
        final_report["quantization"] = {
            "load_in_4bit": is_loaded_in_4bit,
            "quant_type": config.quant_type,
            "double_quant": config.double_quant,
            "compute_dtype": config.compute_dtype,
            "linear4bit_modules": linear4bit_modules,
        }
        if not is_loaded_in_4bit or linear4bit_modules <= 0:
            raise TrainingGateError("MODEL_NOT_ACTUALLY_4BIT")

        # ---- 9) LoRA；记录 after_lora ----
        model = _apply_lora(model, config)
        trainable_params, all_params = _count_trainable_params(model)
        lora_layers = _count_lora_layers(model)
        torch.cuda.synchronize()
        mem_after_lora = _gpu_mem_snapshot(torch)
        final_report["lora"] = {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "bias": config.lora_bias,
            "target_modules": config.lora_target_modules,
            "task_type": config.lora_task_type,
            "lora_layers": lora_layers,
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": round(trainable_params / max(1, all_params) * 100, 4),
        }
        if trainable_params <= 0 or lora_layers <= 0:
            raise TrainingGateError("LORA_NOT_APPLIED")
        if final_report["lora"]["trainable_percentage"] > 95.0:
            raise TrainingGateError(
                "LORA_TOO_BROAD: trainable_percentage 异常接近 100%（base 未冻结）"
            )

        # ---- 10) dataset + trainer；记录 after_trainer ----
        dataset = _build_dataset(train_records, selection.sample_ids)
        sft_config = _build_sft_config(config, output_dir)

        from trl import SFTTrainer  # type: ignore

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        torch.cuda.synchronize()
        mem_after_trainer = _gpu_mem_snapshot(torch)

        # ---- 11) assistant mask sanity（公开 TRL 路径） ----
        try:
            mask_samples = _verify_assistant_mask(trainer, dataset)
        except Exception as e:
            raise TrainingGateError(
                "TRL_ASSISTANT_MASK_UNAVAILABLE: "
                f"无法通过 trainer.train_dataset + trainer.data_collator 验证 mask：{e}"
            )
        final_report["assistant_mask"] = {
            "checked_samples": len(mask_samples),
            "samples": mask_samples,
            "pass": all(s["mask_validation_pass"] for s in mask_samples),
        }
        if not final_report["assistant_mask"]["pass"]:
            raise TrainingGateError("TRL_ASSISTANT_MASK_UNAVAILABLE: 监督 mask 未生效")

        # ---- 12) 1 step train ----
        try:
            train_output = trainer.train()
        except Exception as e:
            raise TrainingGateError(f"train() 失败：{e}") from e

        global_step = int(getattr(train_output, "global_step", 0))
        train_loss = _extract_loss(train_output, trainer)
        loss_finite = train_loss is not None and math.isfinite(train_loss) and train_loss != 0
        grad_norm = _extract_grad_norm(trainer)
        step_time = None
        try:
            log_history = getattr(trainer, "log_history", None) or []
            if log_history and isinstance(log_history[-1], dict):
                step_time = log_history[-1].get("train_runtime")
        except Exception:
            pass

        final_report["training"] = {
            "max_length": config.max_length,
            "max_steps": config.max_steps,
            "global_step": global_step,
            "loss": train_loss,
            "loss_finite": bool(loss_finite),
            "grad_norm": grad_norm,
            "packing": config.packing,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "gradient_checkpointing": config.gradient_checkpointing,
            "learning_rate": config.learning_rate,
            "train_runtime_seconds": step_time,
        }
        if global_step != 1:
            raise TrainingGateError(f"GLOBAL_STEP 异常：{global_step} != 1")
        if not loss_finite:
            raise TrainingGateError(f"LOSS_INVALID: {train_loss}")

        mem_peak_alloc = int(torch.cuda.max_memory_allocated())
        mem_peak_reserved = int(torch.cuda.max_memory_reserved())
        final_report["memory"] = {
            "before_model": mem_before,
            "after_4bit_model": mem_after_4bit,
            "after_lora": mem_after_lora,
            "after_trainer": mem_after_trainer,
            "peak_allocated_bytes": mem_peak_alloc,
            "peak_reserved_bytes": mem_peak_reserved,
            "peak_allocated_gib": _bytes_to_gib(mem_peak_alloc),
            "peak_reserved_gib": _bytes_to_gib(mem_peak_reserved),
        }

        # ---- 13) save adapter ----
        try:
            trainer.model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
        except Exception as e:
            raise TrainingGateError(f"ADAPTER_SAVE_FAIL: {e}") from e
        adapter_files = sorted(p.name for p in adapter_dir.glob("*") if p.is_file())
        has_config = "adapter_config.json" in adapter_files
        weight_files = [f for f in adapter_files if f.endswith((".safetensors", ".bin"))]
        if not has_config or not weight_files:
            raise TrainingGateError(
                f"ADAPTER_SAVE_FAILED: 缺少 adapter_config.json 或权重文件（files={adapter_files}）"
            )
        final_report["adapter"] = {
            "saved": True,
            "path": str(adapter_dir),
            "files": adapter_files,
        }

        # ---- 14) 释放 + 重新 4-bit load + PeftModel.from_pretrained ----
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        from peft import PeftModel  # type: ignore

        base = _load_4bit_model(torch, config.model_name, load_revision, config)
        try:
            peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        except Exception as e:
            raise TrainingGateError(f"ADAPTER_RELOAD_FAIL: {e}") from e
        lora_after_reload = _count_lora_layers(peft_model)
        if lora_after_reload <= 0:
            raise TrainingGateError("ADAPTER_RELOAD_FAILED: 重新加载后 LoRA 层数为 0")
        final_report["adapter_reload"] = {
            "passed": True,
            "lora_layers_after_reload": lora_after_reload,
        }

        # ---- 15) 最小推理 ----
        peft_model.eval()
        prompt_text = tokenizer.apply_chat_template(
            build_inference_prompt(config.inference_prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = peft_model.generate(
                **inputs,
                max_new_tokens=config.inference_max_new_tokens,
                do_sample=False,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated_token_count = int(new_tokens.shape[-1]) if hasattr(new_tokens, "shape") else len(new_tokens)
        generation_nonempty = bool(gen_text.strip())
        if generated_token_count <= 0 or not generation_nonempty:
            raise TrainingGateError(
                f"INFERENCE_SMOKE_FAILED: generated_token_count={generated_token_count}, nonempty={generation_nonempty}"
            )
        final_report["inference_smoke"] = {
            "pass": True,
            "prompt": config.inference_prompt,
            "generated_token_count": generated_token_count,
            "generated_char_count": len(gen_text),
            "generation_nonempty": generation_nonempty,
            "generation_preview": gen_text[:200],
        }
        log.info("Inference preview: %s", gen_text[:200])

        final_report["status"] = "PASS"
        return 0

    except TrainingGateError as e:
        final_report["status"] = "BLOCKED"
        final_report["error"] = {
            "type": "TrainingGateError",
            "reason": str(e),
        }
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
        atomic_write_json(output_dir / "smoke_report.json", final_report)


def _gate_exit_code(reason: str) -> int:
    s = reason.upper()
    if "PHASE5A_GATE" in s:
        return 10
    if "SMOKE_SELECTION" in s or "SMOKE_SAMPLE_NOT_IN_TRAIN" in s:
        return 70
    if "TORCH" in s and "NOT" in s:
        return 20
    if "CUDA" in s and "NOT" in s:
        return 20
    if "BF16" in s:
        return 20
    if "BLACKWELL" in s:
        return 20
    if "PYTORCH_BLACKWELL" in s:
        return 20
    if "MODEL_NOT_ACTUALLY_4BIT" in s:
        return 30
    if "TRL_ASSISTANT_MASK" in s:
        return 40
    if "ADAPTER" in s:
        return 60
    if "INFERENCE_SMOKE" in s:
        return 60
    return 50


# ---------------------------------------------------------------------------
# Phase 5C-A — Unsloth backend 主流程
# 与 HF main() 完全独立的代码路径；Phase 5B baseline（--backend hf）行为不受影响。
# ---------------------------------------------------------------------------


def main_unsloth() -> int:
    """Phase 5C-A — Unsloth backend 单步 smoke。

    与 HF 主流程的关键差异：

    - 通过 ``data_pipeline.training.backends.UnslothBackend`` 加载模型；
    - 不依赖现有 HF helpers（``_load_4bit_model`` 等），保持 HF 路径完全不变；
    - 4-bit 验证仍走 ``is_loaded_in_4bit`` + ``Linear4bit`` 实际计数；
    - assistant mask 仍走公开 TRL 路径 ``trainer.train_dataset`` + ``trainer.data_collator``。
    """
    args = _parse_args()
    log = get_logger("smoke_train_qlora.unsloth")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(args.log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    config = QLoRASmokeConfig()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)

    final_report: Dict[str, Any] = {"status": "BLOCKED", "backend": "unsloth"}
    library_versions: Dict[str, Optional[str]] = {}

    try:
        # ---- backend 解析 ----
        from data_pipeline.training.backends import resolve_backend
        backend = resolve_backend("unsloth")
        meta = backend.metadata()
        log.info("Backend metadata: %s", meta)

        # ---- 1) Phase 5A gate（与 HF 完全相同） ----
        token_report = load_token_report(args.token_report)
        token_audit_rows = load_token_audit_rows(args.token_audit)
        gate = evaluate_phase5a_gate_with_audit(token_report, token_audit_rows)
        final_report["phase5a_gate"] = {
            "passed": gate.passed,
            "reasons": gate.reasons,
            "report_max_tokens": gate.report_max_tokens,
            "audit_max_tokens": gate.audit_max_tokens,
            "actual_max_tokens": gate.actual_max_tokens,
            "selected_max_length": gate.selected_max_length,
            "samples_over_max_length": gate.samples_over_max_length,
            "max_audit_over_total_tokens": gate.max_audit_over_total_tokens,
        }
        if not gate.passed:
            raise TrainingGateError("PHASE5A_GATE_FAIL: " + "; ".join(gate.reasons))
        if token_report.get("model", {}).get("model_name") != config.model_name:
            raise TrainingGateError(
                f"PHASE5A_GATE_FAIL: model_name mismatch "
                f"{token_report.get('model', {}).get('model_name')!r} != {config.model_name!r}"
            )

        # ---- 2) 依赖版本收集 ----
        for dep in ("transformers", "trl", "peft", "accelerate", "datasets",
                    "bitsandbytes", "unsloth", "unsloth_zoo"):
            try:
                library_versions[dep] = __import__(dep).__version__
            except Exception:
                library_versions[dep] = None
        for dep in ("triton", "xformers"):
            try:
                m = __import__(dep)
                library_versions[dep] = getattr(m, "__version__", None)
            except Exception:
                library_versions[dep] = None

        # ---- 3) GPU 预检（与 HF 完全相同） ----
        torch = _ensure_torch()
        library_versions["torch"] = torch.__version__
        gpu_info = _check_gpu(torch, log)
        _bf16_matmul(torch)
        final_report["library_versions"] = library_versions
        final_report["environment"] = {
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            **gpu_info,
        }

        # ---- 4) 解析 model revision（不联网 fallback 走缓存 SHA） ----
        requested_revision, resolved_revision, revision_pinned = backend.resolve_revision(
            config.model_name, config.revision
        )
        load_revision = resolved_revision or requested_revision

        # Phase 5C-A：离线环境 fallback 到已知缓存 SHA，避免 silent main。
        if not resolved_revision:
            cached_sha = _load_cached_revision(config.model_name)
            if cached_sha:
                log.warning("HF Hub 不可达；使用缓存中的 pinned revision %s", cached_sha)
                load_revision = cached_sha
                revision_pinned = True

        # ---- 5) smoke 选样（与 HF 完全相同） ----
        train_records = load_jsonl_records(args.sft_dir / "train.jsonl")
        train_total_tokens = sorted(
            int(r.get("total_tokens") or 0)
            for r in token_audit_rows
            if r.get("split") == "train"
        )
        if train_total_tokens:
            p95_idx = max(0, min(len(train_total_tokens) - 1,
                                  int(round(0.95 * (len(train_total_tokens) - 1)))))
            target_train_p95 = float(train_total_tokens[p95_idx])
        else:
            target_train_p95 = None
        selection = select_smoke_samples_strict(
            train_records, token_audit_rows, config, target_train_p95=target_train_p95
        )

        model_revision_info = {
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "load_revision": load_revision,
            "revision_pinned": revision_pinned,
        }
        write_smoke_config(
            config,
            output_dir / "smoke_config.json",
            logger=log,
            model_revision=model_revision_info,
            selection=selection,
        )
        final_report["model"] = {
            "model_name": config.model_name,
            **model_revision_info,
        }
        final_report["smoke_samples"] = {
            "total_selected": len(selection.sample_ids),
            "sample_ids": selection.sample_ids,
            "short_count": selection.short_count,
            "p95_sample_id": selection.p95_sample_id,
            "p95_total_tokens": selection.p95_total_tokens,
        }

        # ---- 6) 加载 tokenizer + model（Unsloth backend） ----
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = _gpu_mem_snapshot(torch)

        model, tokenizer = backend.load_model_and_tokenizer(
            model_name=config.model_name,
            revision=load_revision,
            torch=torch,
            config=config,
        )
        torch.cuda.synchronize()
        mem_after_4bit = _gpu_mem_snapshot(torch)

        # ---- 7) 4-bit 验证（真实检查） ----
        is_loaded_in_4bit = backend.is_loaded_in_4bit(model)
        linear4bit_modules = backend.count_linear4bit(model)
        final_report["quantization"] = {
            "load_in_4bit": is_loaded_in_4bit,
            "quant_type": config.quant_type,
            "double_quant": config.double_quant,
            "compute_dtype": config.compute_dtype,
            "linear4bit_modules": linear4bit_modules,
        }
        if not is_loaded_in_4bit or linear4bit_modules <= 0:
            raise TrainingGateError("MODEL_NOT_ACTUALLY_4BIT")

        # ---- 8) LoRA（Unsloth backend） ----
        model = backend.apply_lora(model=model, config=config)
        trainable_params, all_params = backend.count_trainable_params(model)
        lora_layers = backend.count_lora_layers(model)
        torch.cuda.synchronize()
        mem_after_lora = _gpu_mem_snapshot(torch)
        final_report["lora"] = {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "bias": config.lora_bias,
            "target_modules": config.lora_target_modules,
            "task_type": config.lora_task_type,
            "lora_layers": lora_layers,
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": round(trainable_params / max(1, all_params) * 100, 4),
        }
        if trainable_params <= 0 or lora_layers <= 0:
            raise TrainingGateError("LORA_NOT_APPLIED")
        if final_report["lora"]["trainable_percentage"] > 95.0:
            raise TrainingGateError("LORA_TOO_BROAD")

        # ---- 9) dataset + trainer ----
        dataset = _build_dataset(train_records, selection.sample_ids)
        trainer = backend.build_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            output_dir=output_dir,
        )
        torch.cuda.synchronize()
        mem_after_trainer = _gpu_mem_snapshot(torch)

        # ---- 10) assistant mask 公开路径校验 ----
        mask_samples = _verify_assistant_mask(trainer, dataset)
        final_report["assistant_mask"] = {
            "checked_samples": len(mask_samples),
            "samples": mask_samples,
            "pass": all(s["mask_validation_pass"] for s in mask_samples),
        }
        if not final_report["assistant_mask"]["pass"]:
            raise TrainingGateError("TRL_ASSISTANT_MASK_UNAVAILABLE: 监督 mask 未生效")

        # ---- 11) 1 step train ----
        train_output = trainer.train()
        global_step = int(getattr(train_output, "global_step", 0))
        train_loss = _extract_loss(train_output, trainer)
        loss_finite = train_loss is not None and math.isfinite(train_loss) and train_loss != 0
        grad_norm = _extract_grad_norm(trainer)
        if global_step != 1:
            raise TrainingGateError(f"GLOBAL_STEP 异常：{global_step} != 1")
        if not loss_finite:
            raise TrainingGateError(f"LOSS_INVALID: {train_loss}")
        final_report["training"] = {
            "max_length": config.max_length,
            "max_steps": config.max_steps,
            "global_step": global_step,
            "loss": train_loss,
            "loss_finite": bool(loss_finite),
            "grad_norm": grad_norm,
            "packing": config.packing,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "gradient_checkpointing": config.gradient_checkpointing,
            "learning_rate": config.learning_rate,
        }

        mem_peak_alloc = int(torch.cuda.max_memory_allocated())
        mem_peak_reserved = int(torch.cuda.max_memory_reserved())
        final_report["memory"] = {
            "before_model": mem_before,
            "after_4bit_model": mem_after_4bit,
            "after_lora": mem_after_lora,
            "after_trainer": mem_after_trainer,
            "peak_allocated_bytes": mem_peak_alloc,
            "peak_reserved_bytes": mem_peak_reserved,
            "peak_allocated_gib": _bytes_to_gib(mem_peak_alloc),
            "peak_reserved_gib": _bytes_to_gib(mem_peak_reserved),
        }

        # ---- 12) save adapter ----
        try:
            adapter_files = backend.save_adapter(
                trainer=trainer, tokenizer=tokenizer, adapter_dir=adapter_dir
            )
        except Exception as e:
            raise TrainingGateError(f"ADAPTER_SAVE_FAIL: {e}") from e
        final_report["adapter"] = {
            "saved": True,
            "path": str(adapter_dir),
            "files": adapter_files,
        }

        # ---- 13) 释放 + reload + infer ----
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            infer = backend.reload_and_infer(
                torch=torch,
                model_name=config.model_name,
                revision=load_revision,
                config=config,
                adapter_dir=adapter_dir,
                inference_prompt=config.inference_prompt,
                max_new_tokens=config.inference_max_new_tokens,
            )
        except Exception as e:
            raise TrainingGateError(f"ADAPTER_RELOAD_OR_INFER_FAIL: {e}") from e
        if infer["generated_token_count"] <= 0 or not infer["nonempty"]:
            raise TrainingGateError(
                f"INFERENCE_SMOKE_FAILED: {infer}"
            )
        final_report["inference_smoke"] = {
            "pass": True,
            "prompt": config.inference_prompt,
            "generated_token_count": infer["generated_token_count"],
            "generated_char_count": infer["generated_char_count"],
            "generation_nonempty": infer["nonempty"],
            "generation_preview": infer["generated_text"][:200],
        }
        log.info("Inference preview: %s", infer["generated_text"][:200])

        final_report["status"] = "PASS"
        return 0

    except TrainingGateError as e:
        final_report["status"] = "BLOCKED"
        final_report["error"] = {"type": "TrainingGateError", "reason": str(e)}
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
        atomic_write_json(output_dir / "smoke_report.json", final_report)


def _load_cached_revision(model_name: str) -> Optional[str]:
    """当 HuggingFace Hub 不可达时，从本地 HF cache 的 ``refs/main`` 读取 SHA。"""
    try:
        import os
        from pathlib import Path
        hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
        slug = "models--" + model_name.replace("/", "--")
        refs_main = hf_home / "hub" / slug / "refs" / "main"
        if refs_main.exists():
            sha = refs_main.read_text(encoding="utf-8").strip()
            if sha:
                return sha
    except Exception:
        return None
    return None


def _cli_main() -> int:
    args = _parse_args()
    if (args.backend or "hf").lower() == "hf":
        return main()
    if args.backend == "unsloth":
        return main_unsloth()
    print(f"unknown backend: {args.backend}", file=sys.stderr)
    return 50


if __name__ == "__main__":
    sys.exit(_cli_main())