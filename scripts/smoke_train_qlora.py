"""Phase 5B — QLoRA 单步真实训练冒烟测试。

执行流程：

1. 读 ``token_report.json`` + ``token_audit.jsonl``；运行 Phase 5A gate。
2. GPU 环境 + BF16 + Blackwell 预检。
3. 4-bit + NF4 + double quant 加载 ``Qwen/Qwen3-8B``；验证 Linear4bit > 0。
4. ``prepare_model_for_kbit_training`` + LoRA r=16 alpha=32 all-linear。
5. 选 smoke 样本（短 7 + P95 1），构造 ``datasets.Dataset.from_list``。
6. ``SFTTrainer(assistant_only_loss=True, packing=False, max_steps=1, bf16=True)``；
   训练前对 3 条样本做 label mask sanity check。
7. 1 个 optimizer step；要求 ``global_step == 1``、``loss finite``。
8. 保存 adapter → 释放显存 → 重新 4-bit load + PeftModel.from_pretrained → 最小推理。

只输出关键运行信息到 ``smoke_report.json``；长回答不写文件。

退出码：

- 0：完整 PASS
- 10：Phase 5A gate 不通过
- 20：环境不满足（CUDA / BF16 / Blackwell / 库缺失）
- 30：4-bit 模型未生效
- 40：assistant mask 不可用
- 50：训练失败（loss 异常 / step 数异常 / OOM 等）
- 60：adapter 保存 / reload / 推理失败
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
    select_smoke_samples,
    write_smoke_config,
)
from data_pipeline.training.pipeline import (
    PHASE5A_MAX_TOKEN_LIMIT,
    TrainingGateError,
    build_inference_prompt,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5B QLoRA 单步训练冒烟测试")
    p.add_argument("--sft-dir", default="data/sft", type=Path)
    p.add_argument("--token-report", default="data/tokenizer/token_report.json", type=Path)
    p.add_argument("--token-audit", default="data/tokenizer/token_audit.jsonl", type=Path)
    p.add_argument("--output-dir", default="data/training", type=Path)
    p.add_argument("--adapter-dir", default="outputs/phase5b-smoke/adapter", type=Path)
    p.add_argument("--log-file", default="logs/smoke_train_qlora.log", type=Path)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step：环境预检（与 check_training_env.py 共享语义，但脚本内强制顺序）
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
# Step：4-bit + LoRA 加载
# ---------------------------------------------------------------------------


def _resolve_model_revision(model_name: str, requested: str) -> Tuple[str, Optional[str], bool]:
    """尽量从 HF cache / huggingface_hub 拿真实 commit SHA。"""
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
    """加载 4-bit + NF4 + double quant + bfloat16 compute 的 base model。"""
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
        # 旧版 transformers 不接受 dtype= → 用 torch_dtype
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
    """统计 ``bnb.nn.Linear4bit`` 数量。"""
    try:
        from bitsandbytes.nn import Linear4bit  # type: ignore
    except Exception:
        return 0
    return sum(1 for m in model.modules() if isinstance(m, Linear4bit))


def _is_loaded_in_4bit(model: Any) -> bool:
    """尽量从 transformers / accelerate 取 is_loaded_in_4bit。"""
    if hasattr(model, "is_loaded_in_4bit"):
        try:
            return bool(model.is_loaded_in_4bit)
        except Exception:
            pass
    # 兜底：存在 quant_config 且方法返回 True
    qc = getattr(model, "quantization_method", None) or getattr(model, "quantization_config", None)
    if qc is not None and hasattr(qc, "get"):
        try:
            return bool(qc.get("load_in_4bit")) or bool(qc.get("bits") == 4)
        except Exception:
            return False
    return False


def _apply_lora(model: Any, tokenizer: Any, config: QLoRASmokeConfig) -> Any:
    """应用 LoRA（all-linear）并返回 PeftModel。"""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore

    # 先 prepare
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

    # gradient_checkpointing 开启 + use_cache=False（PEFT/Transformers 兼容方式）
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
# Step：dataset 准备 + mask 校验
# ---------------------------------------------------------------------------


def _build_dataset(train_records: List[Dict[str, Any]], sample_ids: List[str]) -> Any:
    """构造 ``datasets.Dataset``（conversational）。"""
    from datasets import Dataset  # type: ignore

    selected: List[Dict[str, Any]] = []
    sid_to_record = {r["sample_id"]: r for r in train_records}
    for sid in sample_ids:
        r = sid_to_record.get(sid)
        if r is None:
            continue
        # 仅保留 messages + sample_id；metadata 留给定位
        selected.append({"sample_id": sid, "messages": r["messages"]})
    return Dataset.from_list(selected)


def _verify_assistant_mask(trainer: Any, tokenizer: Any, dataset: Any, log: logging.Logger) -> List[Dict[str, Any]]:
    """对 trainer 处理后的 3 条样本检查 labels 同时含 -100 与非 -100 token。"""
    import torch  # type: ignore

    samples = []
    checked = min(3, len(dataset))
    for i in range(checked):
        row = dataset[i]
        features = trainer._prepare_dataset([row], tokenizer) if hasattr(trainer, "_prepare_dataset") else None
        if features is None:
            # TRL 路径不固定；尝试 dataset.map(trainer._prepare_dataset) 或 trainer.data_collator
            batch = trainer.data_collator([row])
        else:
            batch = trainer.data_collator(features)
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        seq_len = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else len(input_ids)
        if hasattr(labels, "tolist"):
            labels_list = labels.tolist()
        else:
            labels_list = list(labels)
        masked = sum(1 for v in labels_list if int(v) == -100)
        supervised = seq_len - masked
        ok = masked > 0 and supervised > 0
        samples.append({
            "sample_id": row.get("sample_id"),
            "sequence_length": seq_len,
            "masked_tokens": masked,
            "supervised_tokens": supervised,
            "mask_validation_pass": bool(ok),
        })
    return samples


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _build_sft_config(config: QLoRASmokeConfig, output_dir: Path) -> Any:
    """构造 ``trl.SFTConfig``（基于当前安装版本）。"""
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

    # max_length vs max_seq_length
    if "max_length" in params:
        kwargs["max_length"] = config.max_length
    elif "max_seq_length" in params:
        kwargs["max_seq_length"] = config.max_length

    return SFTConfig(**kwargs)


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
    try:
        # ---- 1) Phase 5A gate ----
        token_report = load_token_report(args.token_report)
        token_audit_rows = load_token_audit_rows(args.token_audit)
        gate = evaluate_phase5a_gate_with_audit(token_report, token_audit_rows)
        final_report["phase5a_gate"] = {
            "passed": gate.passed,
            "reasons": gate.reasons,
            "actual_max_tokens": gate.actual_max_tokens,
            "selected_max_length": gate.selected_max_length,
            "samples_over_max_length": gate.samples_over_max_length,
        }
        log.info("Phase 5A gate: %s", gate)
        if not gate.passed:
            raise TrainingGateError("PHASE5B_BLOCKED: PHASE5A_GATE_FAIL: " + "; ".join(gate.reasons))

        # model 一致性
        if token_report.get("model", {}).get("model_name") != config.model_name:
            raise TrainingGateError(
                "PHASE5B_BLOCKED: MODEL_TOKENIZER_AUDIT_MISMATCH: "
                f"{token_report.get('model', {}).get('model_name')!r} != {config.model_name!r}"
            )

        # ---- 2) GPU 预检 ----
        torch = _ensure_torch()
        gpu_info = _check_gpu(torch, log)
        _bf16_matmul(torch)
        final_report["environment"] = {
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            **gpu_info,
        }
        log.info("GPU: %s", gpu_info)

        # ---- 3) 写 smoke_config.json ----
        write_smoke_config(config, output_dir / "smoke_config.json", logger=log)

        # ---- 4) 加载 4-bit base model ----
        requested_revision, resolved_revision, revision_pinned = _resolve_model_revision(
            config.model_name, config.revision
        )
        # 尝试使用 resolved；拿不到则用 requested
        load_revision = resolved_revision or requested_revision
        tokenizer = _load_tokenizer(config.model_name, load_revision)
        model = _load_4bit_model(torch, config.model_name, load_revision, config)

        final_report["model"] = {
            "model_name": config.model_name,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "revision_pinned": revision_pinned,
        }
        log.info("Loaded 4-bit base: %s revision=%s pinned=%s",
                 config.model_name, load_revision, revision_pinned)

        # ---- 5) 4-bit 验证 ----
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

        # ---- 6) LoRA ----
        model = _apply_lora(model, tokenizer, config)
        trainable_params, all_params = _count_trainable_params(model)
        lora_layers = _count_lora_layers(model)
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

        # ---- 7) smoke 选样 ----
        train_records = load_jsonl_records(args.sft_dir / "train.jsonl")
        # 找 train P95
        train_total_tokens = sorted(
            int(r.get("total_tokens") or 0)
            for r in token_audit_rows
            if r.get("split") == "train"
        )
        p95_idx = max(0, min(len(train_total_tokens) - 1,
                              int(round(0.95 * (len(train_total_tokens) - 1))))) if train_total_tokens else 0
        target_train_p95 = float(train_total_tokens[p95_idx]) if train_total_tokens else None
        selection = select_smoke_samples(
            train_records, token_audit_rows, config, target_train_p95=target_train_p95
        )
        final_report["smoke_samples"] = {
            "total_selected": len(selection.sample_ids),
            "sample_ids": selection.sample_ids,
            "short_count": selection.short_count,
            "p95_sample_id": selection.p95_sample_id,
            "p95_total_tokens": selection.p95_total_tokens,
        }
        log.info("Smoke selection: %s", selection)

        # ---- 8) dataset + trainer ----
        dataset = _build_dataset(train_records, selection.sample_ids)
        sft_config = _build_sft_config(config, output_dir)

        # 显存阶段 A
        torch.cuda.reset_peak_memory_stats()
        mem_a_alloc = int(torch.cuda.memory_allocated())
        mem_a_reserved = int(torch.cuda.memory_reserved())

        from trl import SFTTrainer  # type: ignore
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        mem_b_alloc = int(torch.cuda.memory_allocated())
        mem_b_reserved = int(torch.cuda.memory_reserved())

        # ---- 9) mask sanity ----
        try:
            mask_samples = _verify_assistant_mask(trainer, tokenizer, dataset, log)
        except Exception as e:
            raise TrainingGateError(
                "TRL_ASSISTANT_MASK_UNAVAILABLE: "
                f"无法检查 TRL labels 同时含 -100 与非 -100 token：{e}"
            )
        final_report["assistant_mask"] = {
            "checked_samples": len(mask_samples),
            "samples": mask_samples,
            "pass": all(s["mask_validation_pass"] for s in mask_samples),
        }
        if not final_report["assistant_mask"]["pass"]:
            raise TrainingGateError("TRL_ASSISTANT_MASK_UNAVAILABLE: 监督 mask 未生效")

        # ---- 10) 1 step train ----
        try:
            train_output = trainer.train()
        except Exception as e:
            raise TrainingGateError(f"train() 失败：{e}") from e

        global_step = int(getattr(train_output, "global_step", 0))
        # loss
        train_loss = _extract_loss(train_output, trainer)
        loss_finite = train_loss is not None and math.isfinite(train_loss) and train_loss != 0
        grad_norm = _extract_grad_norm(trainer)

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
        }

        if global_step != 1:
            raise TrainingGateError(f"GLOBAL_STEP 异常：{global_step} != 1")
        if not loss_finite:
            raise TrainingGateError(f"LOSS_INVALID: {train_loss}")

        mem_peak_alloc = int(torch.cuda.max_memory_allocated())
        mem_peak_reserved = int(torch.cuda.max_memory_reserved())
        final_report["memory"] = {
            "before_trainer_allocated": mem_a_alloc,
            "before_trainer_reserved": mem_a_reserved,
            "after_trainer_allocated": mem_b_alloc,
            "after_trainer_reserved": mem_b_reserved,
            "peak_allocated": mem_peak_alloc,
            "peak_reserved": mem_peak_reserved,
        }

        # ---- 11) save adapter ----
        try:
            trainer.model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
        except Exception as e:
            raise TrainingGateError(f"ADAPTER_SAVE_FAIL: {e}") from e
        adapter_files = sorted(p.name for p in adapter_dir.glob("*") if p.is_file())
        final_report["adapter"] = {"saved": True, "path": str(adapter_dir), "files": adapter_files}

        # ---- 12) reload + inference ----
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
        final_report["adapter_reload"] = {
            "passed": lora_after_reload > 0,
            "lora_layers_after_reload": lora_after_reload,
        }

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
        final_report["inference_smoke"] = {
            "pass": True,
            "prompt": config.inference_prompt,
            "generated_token_count": int(new_tokens.shape[-1]),
            "generated_char_count": len(gen_text),
            "generation_nonempty": bool(gen_text.strip()),
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
    if "TORCH" in s and "NOT" in s:
        return 20
    if "CUDA" in s and "NOT" in s:
        return 20
    if "BF16" in s:
        return 20
    if "BLACKWELL" in s:
        return 20
    if "MODEL_NOT_ACTUALLY_4BIT" in s:
        return 30
    if "TRL_ASSISTANT_MASK" in s:
        return 40
    if "ADAPTER" in s:
        return 60
    return 50


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
    # SFTTrainer.returncode 风格 TrainOutput：training_loss 字段
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


if __name__ == "__main__":
    sys.exit(main())