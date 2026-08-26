"""Phase 5C-A — HF vs Unsloth A/B benchmark + batch probe。

该脚本只做 benchmark / probe；不替换 ``smoke_train_qlora.py`` 的任何逻辑。

- 必读 ``data/tokenizer/token_report.json`` + ``token_audit.jsonl``；
- 必跑 ``Phase 5A gate``；
- 必使用 ``select_smoke_samples_strict`` 选 8 条 smoke 样本；
- 与 ``smoke_train_qlora.py`` 共享 ``QLoRASmokeConfig`` / gate / 选样；
- 通过 ``data_pipeline.training.backends.resolve_backend`` 切 backend；
- 严格不修改 max_length / packing / assistant_only_loss / NF4 / double_quant / LoRA；
- A/B benchmark：每个 backend 跑 ``max_steps`` 步（默认 20），记录：
    - model_load_seconds
    - trainer_init_seconds
    - training_runtime_seconds
    - mean / median step time
    - samples/sec
    - total tokens / sec
    - peak allocated / reserved VRAM
- Batch probe：依次 batch=1, 2, 4（每档 max_steps=3）；
  仅当 batch=4 peak_reserved < 24 GiB 时才允许 probe batch=8。

报告输出：

- ``data/training/phase5c_a/backend_benchmark.json``
- ``data/training/phase5c_a/batch_probe.json``
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from data_pipeline.utils import (
    atomic_write_json,
    get_logger,
    load_jsonl_records,
)
from data_pipeline.training import (
    QLoRASmokeConfig,
    TrainingGateError,
    build_inference_prompt,
    evaluate_phase5a_gate_with_audit,
    load_token_audit_rows,
    load_token_report,
    select_smoke_samples_strict,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5C-A — HF vs Unsloth A/B benchmark")
    p.add_argument("--backend", required=True, choices=["hf", "unsloth"])
    p.add_argument("--sft-dir", default="data/sft", type=Path)
    p.add_argument("--token-report", default="data/tokenizer/token_report.json", type=Path)
    p.add_argument("--token-audit", default="data/tokenizer/token_audit.jsonl", type=Path)
    p.add_argument("--output-dir", default="data/training/phase5c_a", type=Path)
    p.add_argument("--adapter-dir", type=Path, default=None,
                   help="默认 outputs/phase5c_a/<backend>/adapter")
    p.add_argument("--log-file", default="logs/benchmark_qlora_backends.log", type=Path)
    p.add_argument("--log-level", default="INFO")
    # benchmark
    p.add_argument("--benchmark-steps", type=int, default=20)
    p.add_argument("--skip-benchmark", action="store_true",
                   help="仅做 1-step smoke；用于只想验证 backend 接入的场景")
    # batch probe
    p.add_argument("--batch-probe-steps", type=int, default=3)
    p.add_argument("--probe-batches", default="1,2,4",
                   help="逗号分隔的 batch size 列表（probe 顺序）；默认 '1,2,4'；"
                        "如允许自动追加 8，会按需扩展。")
    p.add_argument("--skip-batch-probe", action="store_true",
                   help="不做 batch probe；仅跑 A/B benchmark。")
    p.add_argument("--include-batch-8", action="store_true",
                   help="强制包含 batch=8 probe（无视显存判断；仅诊断使用）。")
    p.add_argument("--enable-batch-8", action="store_true",
                   help="batch=4 peak_reserved < 24 GiB 时自动追加 batch=8 probe。"
                        "默认 False — 不开启。")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 共享工具
# ---------------------------------------------------------------------------


def _ensure_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:
        raise TrainingGateError(f"torch 未安装：{e}")
    return torch


def _check_gpu(torch: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
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
        raise TrainingGateError("BF16 不受支持")
    return info


def _bf16_matmul(torch: Any) -> None:
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all().item()):
        raise TrainingGateError("CUDA BF16 matmul 结果非 finite")


def _bytes_to_gib(n: int) -> float:
    return round(n / (1024 ** 3), 3)


def _gpu_mem_snapshot(torch: Any) -> Dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "free_bytes": int(torch.cuda.mem_get_info()[0]),
        "total_bytes": int(torch.cuda.mem_get_info()[1]),
    }


def _load_cached_revision(model_name: str) -> Optional[str]:
    try:
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


def _build_dataset(train_records: List[Dict[str, Any]], sample_ids: List[str]) -> Any:
    from datasets import Dataset  # type: ignore

    sid_to_record = {r["sample_id"]: r for r in train_records}
    selected: List[Dict[str, Any]] = []
    for sid in sample_ids:
        r = sid_to_record.get(sid)
        if r is None:
            raise TrainingGateError(f"SMOKE_SAMPLE_NOT_IN_TRAIN: {sid}")
        selected.append({"sample_id": sid, "messages": r["messages"]})
    return Dataset.from_list(selected)


def _verify_assistant_mask(trainer: Any, dataset: Any) -> List[Dict[str, Any]]:
    """走公开 TRL 路径 ``trainer.train_dataset`` + ``trainer.data_collator`` 校验 mask。"""
    samples: List[Dict[str, Any]] = []
    source = getattr(trainer, "train_dataset", None) or dataset
    checked = min(3, len(source))
    for i in range(checked):
        row = source[i]
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


def _build_dataset_for_batch(
    train_records: List[Dict[str, Any]],
    longest_sample_ids: List[str],
    batch_size: int,
    *,
    require_unique: bool = True,
) -> Any:
    """构造 batch probe dataset：包含最长样本 + 多个真实不同的 train sample。

    - 先放 ``len(longest_sample_ids)`` 条最长样本；
    - 然后按 sample_id 升序补充到 ``batch_size`` 条；
    - 不允许重复同一条 sample_id（除非 ``require_unique=False``）。
    """
    from datasets import Dataset  # type: ignore

    sid_to_record = {r["sample_id"]: r for r in train_records if r.get("sample_id")}
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    # 1) 最长样本优先
    for sid in longest_sample_ids:
        r = sid_to_record.get(sid)
        if r is None:
            raise TrainingGateError(f"BATCH_PROBE_SAMPLE_MISSING: {sid}")
        selected.append({"sample_id": sid, "messages": r["messages"]})
        seen.add(sid)
    # 2) 按 sample_id 升序补足到 batch_size
    if len(selected) < batch_size:
        for sid in sorted(sid_to_record.keys()):
            if len(selected) >= batch_size:
                break
            if sid in seen:
                continue
            selected.append({"sample_id": sid, "messages": sid_to_record[sid]["messages"]})
            seen.add(sid)
    if require_unique and len(selected) != len(set(s["sample_id"] for s in selected)):
        raise TrainingGateError("BATCH_PROBE_NOT_UNIQUE")
    if len(selected) < batch_size:
        raise TrainingGateError(
            f"BATCH_PROBE_INSUFFICIENT_SAMPLES: needed={batch_size}, got={len(selected)}"
        )
    return Dataset.from_list(selected[:batch_size])


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


def _extract_loss_history(trainer: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in getattr(trainer, "log_history", None) or []:
        if not isinstance(entry, dict):
            continue
        if "loss" in entry:
            out.append({
                "step": entry.get("step"),
                "loss": entry.get("loss"),
                "grad_norm": entry.get("grad_norm"),
                "learning_rate": entry.get("learning_rate"),
                "epoch": entry.get("epoch"),
            })
    return out


def _compute_step_times(log_history: List[Dict[str, Any]]) -> List[float]:
    """从 log_history 中提取相邻 step 的 wall-clock 时间差。"""
    times: List[Dict[str, Any]] = []
    for entry in log_history or []:
        if not isinstance(entry, dict):
            continue
        if "train_runtime" in entry or "step" in entry:
            times.append(entry)
    if not times:
        return []
    # 如果 log_history 不含 wall_time，则退化为单值（train_runtime）
    if "train_runtime" in times[-1] and len(times) == 1:
        return [float(times[-1]["train_runtime"])]
    # 启发式：用相邻 step 的 wall_time 差
    wall_times = [t.get("wall_time") for t in times if "wall_time" in t]
    if len(wall_times) >= 2:
        deltas: List[float] = []
        for a, b in zip(wall_times[:-1], wall_times[1:]):
            try:
                deltas.append(float(b) - float(a))
            except Exception:
                continue
        return deltas
    return []


# ---------------------------------------------------------------------------
# A/B benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    backend: str
    status: str = "BLOCKED"
    model_load_seconds: float = 0.0
    lora_apply_seconds: float = 0.0
    trainer_init_seconds: float = 0.0
    training_runtime_seconds: float = 0.0
    mean_step_seconds: float = 0.0
    median_step_seconds: float = 0.0
    total_tokens: int = 0
    supervised_tokens: int = 0
    supervised_tokens_per_sec: float = 0.0
    samples: int = 0
    samples_per_sec: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    peak_allocated_gib: float = 0.0
    peak_reserved_gib: float = 0.0
    loss_history: List[Dict[str, Any]] = field(default_factory=list)
    loss_finite: bool = False
    error: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _run_single_benchmark(
    backend_name: str,
    *,
    config: QLoRASmokeConfig,
    train_records: List[Dict[str, Any]],
    sample_ids: List[str],
    load_revision: str,
    torch: Any,
    max_steps: int,
    adapter_dir: Path,
) -> BenchmarkResult:
    """运行单个 backend 的 A/B benchmark（max_steps 步 + save）。"""
    res = BenchmarkResult(backend=backend_name)
    try:
        from data_pipeline.training.backends import resolve_backend
        backend = resolve_backend(backend_name)
        res.extra["backend_metadata"] = {
            "name": backend.metadata().name,
            "description": backend.metadata().description,
            "depends_on_unsloth": backend.metadata().depends_on_unsloth,
        }

        # QLoRASmokeConfig 是 frozen 且 __post_init__ 强制 max_steps=1；
        # 这里通过 shim 派生一个允许 max_steps=N 的配置副本，用于 benchmark。
        class _CfgShim:
            def __init__(self, base: QLoRASmokeConfig, max_steps_: int):
                for f in base.__dataclass_fields__:
                    setattr(self, f, getattr(base, f))
                self.max_steps = max_steps_

        cfg_obj = _CfgShim(config, max_steps)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # ---- load ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model, tokenizer = backend.load_model_and_tokenizer(
            model_name=config.model_name,
            revision=load_revision,
            torch=torch,
            config=cfg_obj,
        )
        torch.cuda.synchronize()
        res.model_load_seconds = round(time.perf_counter() - t0, 3)

        # 4-bit 校验
        is_4bit = backend.is_loaded_in_4bit(model)
        l4bit = backend.count_linear4bit(model)
        res.extra["quantization"] = {
            "load_in_4bit": is_4bit,
            "linear4bit_modules": l4bit,
            "quant_type": config.quant_type,
            "double_quant": config.double_quant,
            "compute_dtype": config.compute_dtype,
        }
        if not is_4bit or l4bit <= 0:
            raise TrainingGateError("MODEL_NOT_ACTUALLY_4BIT")

        # ---- LoRA ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model = backend.apply_lora(model=model, config=cfg_obj)
        torch.cuda.synchronize()
        res.lora_apply_seconds = round(time.perf_counter() - t0, 3)

        trainable, total = backend.count_trainable_params(model)
        lora_layers = backend.count_lora_layers(model)
        res.extra["lora"] = {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "bias": config.lora_bias,
            "target_modules": config.lora_target_modules,
            "task_type": config.lora_task_type,
            "lora_layers": lora_layers,
            "trainable_params": trainable,
            "all_params": total,
            "trainable_percentage": round(trainable / max(1, total) * 100, 4),
        }
        if lora_layers <= 0:
            raise TrainingGateError("LORA_NOT_APPLIED")

        # ---- dataset + trainer ----
        dataset = _build_dataset(train_records, sample_ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        trainer = backend.build_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=cfg_obj,
            output_dir=adapter_dir.parent,
        )
        torch.cuda.synchronize()
        res.trainer_init_seconds = round(time.perf_counter() - t0, 3)

        # ---- assistant mask ----
        mask_samples = _verify_assistant_mask(trainer, dataset)
        mask_pass = all(s["mask_validation_pass"] for s in mask_samples)
        res.extra["assistant_mask"] = {
            "checked_samples": len(mask_samples),
            "pass": mask_pass,
        }
        if not mask_pass:
            raise TrainingGateError("TRL_ASSISTANT_MASK_UNAVAILABLE")

        # ---- train ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_output = trainer.train()
        torch.cuda.synchronize()
        res.training_runtime_seconds = round(time.perf_counter() - t0, 3)
        res.loss_history = _extract_loss_history(trainer)
        step_times = _compute_step_times(getattr(trainer, "log_history", []) or [])
        if step_times:
            res.mean_step_seconds = round(statistics.mean(step_times), 4)
            res.median_step_seconds = round(statistics.median(step_times), 4)
        else:
            # 退化为总时间 / 步数
            if max_steps > 0:
                res.mean_step_seconds = round(res.training_runtime_seconds / max_steps, 4)
                res.median_step_seconds = res.mean_step_seconds

        # tokens
        res.samples = len(dataset)
        res.samples_per_sec = round(res.samples / max(0.001, res.training_runtime_seconds), 4)

        # total / supervised tokens：优先用 trainer.train_dataset（已被 Unsloth /
        # TRL 处理过，含 input_ids / labels）；否则回退到原 dataset + collator。
        supervised_tokens = 0
        total_tokens = 0
        try:
            td = getattr(trainer, "train_dataset", None) or dataset
            for i in range(min(len(td), res.samples)):
                row = td[i]
                if isinstance(row, dict) and "input_ids" in row:
                    ids = row["input_ids"]
                    labels = row.get("labels")
                    if hasattr(ids, "shape"):
                        seq = int(ids.shape[-1])
                    else:
                        seq = len(ids)
                    if labels is not None:
                        if hasattr(labels, "tolist"):
                            labels_list = labels.tolist()
                        else:
                            labels_list = list(labels)
                        masked = sum(1 for v in labels_list if int(v) == -100)
                    else:
                        masked = 0
                else:
                    batch = trainer.data_collator([row])
                    id_keys = [k for k in ("input_ids", "ids") if k in batch]
                    lbl_keys = [k for k in ("labels",) if k in batch]
                    if not id_keys or not lbl_keys:
                        continue
                    ids = batch[id_keys[0]][0]
                    labels = batch[lbl_keys[0]][0]
                    seq = int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)
                    masked = int(sum(
                        1 for v in (labels.tolist() if hasattr(labels, "tolist") else list(labels))
                        if int(v) == -100
                    ))
                total_tokens += seq
                supervised_tokens += max(0, seq - masked)
        except Exception as e:
            res.extra["token_count_error"] = f"{type(e).__name__}: {e}"
        res.total_tokens = int(total_tokens)
        res.supervised_tokens = int(supervised_tokens)
        res.supervised_tokens_per_sec = round(
            res.supervised_tokens / max(0.001, res.training_runtime_seconds), 4
        )

        train_loss = _extract_loss(train_output, trainer)
        res.loss_finite = train_loss is not None and math.isfinite(train_loss) and train_loss != 0
        res.extra["final_loss"] = train_loss
        res.extra["global_step"] = int(getattr(train_output, "global_step", 0))
        if not res.loss_finite:
            raise TrainingGateError(f"LOSS_INVALID: {train_loss}")
        if res.extra["global_step"] != max_steps:
            raise TrainingGateError(
                f"GLOBAL_STEP 异常：{res.extra['global_step']} != {max_steps}"
            )

        res.peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
        res.peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
        res.peak_allocated_gib = _bytes_to_gib(res.peak_allocated_bytes)
        res.peak_reserved_gib = _bytes_to_gib(res.peak_reserved_bytes)

        # ---- save adapter ----
        try:
            adapter_files = backend.save_adapter(
                trainer=trainer, tokenizer=tokenizer, adapter_dir=adapter_dir
            )
            res.extra["adapter_files"] = adapter_files
        except Exception as e:
            raise TrainingGateError(f"ADAPTER_SAVE_FAIL: {e}") from e

        # ---- reload + infer ----
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
                config=cfg_obj,
                adapter_dir=adapter_dir,
                inference_prompt=config.inference_prompt,
                max_new_tokens=config.inference_max_new_tokens,
            )
        except Exception as e:
            raise TrainingGateError(f"ADAPTER_RELOAD_OR_INFER_FAIL: {e}") from e
        if infer["generated_token_count"] <= 0 or not infer["nonempty"]:
            raise TrainingGateError(f"INFERENCE_SMOKE_FAILED: {infer}")
        res.extra["inference_smoke"] = {
            "generated_token_count": infer["generated_token_count"],
            "generated_char_count": infer["generated_char_count"],
            "nonempty": infer["nonempty"],
            "preview": infer["generated_text"][:200],
        }

        res.status = "PASS"
        return res

    except TrainingGateError as e:
        res.status = "BLOCKED"
        res.error = {"type": "TrainingGateError", "reason": str(e)}
        return res
    except Exception as e:
        res.status = "BLOCKED"
        res.error = {
            "type": type(e).__name__,
            "reason": str(e),
            "traceback": traceback.format_exc(),
        }
        return res


# ---------------------------------------------------------------------------
# Batch probe
# ---------------------------------------------------------------------------


@dataclass
class BatchProbeResult:
    backend: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    recommended_batch_size: int = 0
    recommended_gradient_accumulation_steps: int = 1
    status: str = "BLOCKED"
    reason: str = ""


def _run_batch_probe(
    backend_name: str,
    *,
    config: QLoRASmokeConfig,
    train_records: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
    load_revision: str,
    torch: Any,
    gpu_total_bytes: int,
    probe_batches: List[int],
    batch_probe_steps: int,
    base_adapter_dir: Path,
    include_batch_8: bool,
    enable_batch_8: bool,
) -> BatchProbeResult:
    """Batch probe：测试 batch=1,2,4,（可选 8）。

    - 每档 max_steps = ``batch_probe_steps``；
    - probe batch 必须包含 train 中最长的真实样本；
    - batch=8 仅在 batch=4 peak_reserved < 24 GiB 且 ``enable_batch_8`` 才允许；
    - ``include_batch_8`` 强制包含（无视显存判断，仅诊断）。
    """
    res = BatchProbeResult(backend=backend_name)

    # 选最长 train 样本的 sample_id
    train_rows = [r for r in audit_rows if r.get("split") == "train"]
    if not train_rows:
        res.reason = "no train audit rows"
        return res
    longest = sorted(train_rows, key=lambda r: int(r.get("total_tokens") or 0), reverse=True)
    longest_sample_ids = [r["sample_id"] for r in longest[:max(probe_batches)]]

    safety_limit_bytes = int(0.80 * gpu_total_bytes)

    actual_probe = list(probe_batches)
    batch4_result: Optional[Dict[str, Any]] = None

    for batch_size in probe_batches:
        # QLoRASmokeConfig 是 frozen + __post_init__ 校验 max_steps=1；
        # 这里通过 shim 派生一个允许 max_steps=N + batch_size=B 的副本。
        class _CfgShim:
            def __init__(self, base: QLoRASmokeConfig, batch_size: int, max_steps_: int):
                for f in base.__dataclass_fields__:
                    setattr(self, f, getattr(base, f))
                self.per_device_train_batch_size = batch_size
                self.max_steps = max_steps_

        cfg_obj = _CfgShim(config, batch_size, batch_probe_steps)

        # dataset 包含最长样本 + 不同 sample
        dataset = _build_dataset_for_batch(
            train_records, longest_sample_ids, batch_size, require_unique=True
        )

        try:
            from data_pipeline.training.backends import resolve_backend
            backend = resolve_backend(backend_name)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model, tokenizer = backend.load_model_and_tokenizer(
                model_name=config.model_name,
                revision=load_revision,
                torch=torch,
                config=cfg_obj,
            )
            is_4bit = backend.is_loaded_in_4bit(model)
            if not is_4bit:
                raise TrainingGateError("MODEL_NOT_ACTUALLY_4BIT")
            model = backend.apply_lora(model=model, config=cfg_obj)

            probe_adapter_dir = base_adapter_dir.parent / f"probe_b{batch_size}"
            probe_adapter_dir.mkdir(parents=True, exist_ok=True)

            trainer = backend.build_trainer(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                config=cfg_obj,
                output_dir=probe_adapter_dir.parent,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            train_output = trainer.train()
            torch.cuda.synchronize()
            runtime = round(time.perf_counter() - t0, 3)
            loss = _extract_loss(train_output, trainer)
            loss_finite = loss is not None and math.isfinite(loss) and loss != 0
            peak_alloc = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())

            try:
                backend.save_adapter(
                    trainer=trainer, tokenizer=tokenizer, adapter_dir=probe_adapter_dir
                )
                save_ok = True
            except Exception:
                save_ok = False

            del trainer
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            entry = {
                "batch_size": batch_size,
                "max_steps": batch_probe_steps,
                "runtime_seconds": runtime,
                "mean_step_seconds": round(runtime / batch_probe_steps, 4),
                "loss": loss,
                "loss_finite": bool(loss_finite),
                "global_step": int(getattr(train_output, "global_step", 0)),
                "peak_allocated_bytes": peak_alloc,
                "peak_reserved_bytes": peak_reserved,
                "peak_allocated_gib": _bytes_to_gib(peak_alloc),
                "peak_reserved_gib": _bytes_to_gib(peak_reserved),
                "peak_reserved_within_80pct": peak_reserved <= safety_limit_bytes,
                "adapter_save": save_ok,
                "oom": False,
            }
            res.results.append(entry)
            if batch_size == 4:
                batch4_result = entry
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                res.results.append({
                    "batch_size": batch_size,
                    "max_steps": batch_probe_steps,
                    "oom": True,
                    "error": str(e)[:300],
                })
                torch.cuda.empty_cache()
                continue
            raise

    # batch=8 判断
    add_batch_8 = False
    if include_batch_8:
        add_batch_8 = True
        res.reason = "include_batch_8 强制开启"
    elif enable_batch_8 and batch4_result is not None:
        if batch4_result["peak_reserved_bytes"] < 24 * (1024 ** 3):
            add_batch_8 = True
            res.reason = (
                f"batch=4 peak_reserved={batch4_result['peak_reserved_gib']} GiB < 24 GiB；"
                "允许追加 batch=8 probe。"
            )
        else:
            res.reason = (
                f"batch=4 peak_reserved={batch4_result['peak_reserved_gib']} GiB >= 24 GiB；"
                "不追加 batch=8 probe。"
            )
    else:
        res.reason = res.reason or "默认不包含 batch=8 probe（需 --enable-batch-8 或 --include-batch-8）。"

    if add_batch_8 and 8 not in actual_probe:
        actual_probe.append(8)
        res.results.extend(_run_one_probe_batch(
            backend_name, 8, config, train_records, longest_sample_ids,
            load_revision, torch, safety_limit_bytes,
            base_adapter_dir.parent,
        ))

    # 推荐 batch：peak_reserved <= 80% total 的最大 batch
    safe_results = [r for r in res.results if r.get("peak_reserved_within_80pct") and not r.get("oom")]
    if safe_results:
        rec = max(safe_results, key=lambda r: r["batch_size"])
        res.recommended_batch_size = int(rec["batch_size"])
        res.recommended_gradient_accumulation_steps = 1  # Phase 5C-A 不变
    else:
        # 至少回退到 batch=1（如果成功过）
        ok1 = [r for r in res.results if r["batch_size"] == 1 and not r.get("oom") and r.get("loss_finite")]
        if ok1:
            res.recommended_batch_size = 1
        res.recommended_gradient_accumulation_steps = 1

    res.status = "PASS" if any(not r.get("oom") for r in res.results) else "BLOCKED"
    return res


def _run_one_probe_batch(
    backend_name: str,
    batch_size: int,
    config: QLoRASmokeConfig,
    train_records: List[Dict[str, Any]],
    longest_sample_ids: List[str],
    load_revision: str,
    torch: Any,
    safety_limit_bytes: int,
    base_output_dir: Path,
) -> List[Dict[str, Any]]:
    """单独跑一档 batch probe。"""
    out: List[Dict[str, Any]] = []
    try:
        class _CfgShim:
            def __init__(self, base: QLoRASmokeConfig, batch_size: int, max_steps_: int):
                for f in base.__dataclass_fields__:
                    setattr(self, f, getattr(base, f))
                self.per_device_train_batch_size = batch_size
                self.max_steps = max_steps_

        cfg_obj = _CfgShim(config, batch_size, 3)
        from data_pipeline.training.backends import resolve_backend
        backend = resolve_backend(backend_name)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer = backend.load_model_and_tokenizer(
            model_name=config.model_name,
            revision=load_revision,
            torch=torch,
            config=cfg_obj,
        )
        is_4bit = backend.is_loaded_in_4bit(model)
        if not is_4bit:
            out.append({"batch_size": batch_size, "max_steps": 3,
                        "error": "MODEL_NOT_ACTUALLY_4BIT"})
            return out
        model = backend.apply_lora(model=model, config=cfg_obj)
        dataset = _build_dataset_for_batch(
            train_records, longest_sample_ids, batch_size, require_unique=True
        )
        probe_adapter_dir = base_output_dir / f"probe_b{batch_size}"
        probe_adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer = backend.build_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=cfg_obj,
            output_dir=base_output_dir,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_output = trainer.train()
        torch.cuda.synchronize()
        runtime = round(time.perf_counter() - t0, 3)
        loss = _extract_loss(train_output, trainer)
        loss_finite = loss is not None and math.isfinite(loss) and loss != 0
        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        try:
            backend.save_adapter(
                trainer=trainer, tokenizer=tokenizer, adapter_dir=probe_adapter_dir
            )
            save_ok = True
        except Exception:
            save_ok = False
        out.append({
            "batch_size": batch_size,
            "max_steps": 3,
            "runtime_seconds": runtime,
            "mean_step_seconds": round(runtime / 3, 4),
            "loss": loss,
            "loss_finite": bool(loss_finite),
            "global_step": int(getattr(train_output, "global_step", 0)),
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_gib": _bytes_to_gib(peak_alloc),
            "peak_reserved_gib": _bytes_to_gib(peak_reserved),
            "peak_reserved_within_80pct": peak_reserved <= safety_limit_bytes,
            "adapter_save": save_ok,
            "oom": False,
        })
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "OOM" in str(e):
            out.append({"batch_size": batch_size, "max_steps": 3,
                        "oom": True, "error": str(e)[:300]})
            torch.cuda.empty_cache()
        else:
            out.append({"batch_size": batch_size, "max_steps": 3,
                        "error": f"{type(e).__name__}: {str(e)[:200]}"})
    except Exception as e:
        out.append({"batch_size": batch_size, "max_steps": 3,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    log = get_logger("benchmark_qlora_backends")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(args.log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.adapter_dir is None:
        adapter_dir = Path(f"outputs/phase5c_a/{args.backend}/adapter")
    else:
        adapter_dir = args.adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)

    final: Dict[str, Any] = {"status": "BLOCKED", "backend": args.backend}

    try:
        config = QLoRASmokeConfig()
        torch = _ensure_torch()
        gpu_info = _check_gpu(torch)
        _bf16_matmul(torch)
        final["environment"] = {
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            **gpu_info,
        }
        log.info("GPU: %s", gpu_info)

        # gate
        token_report = load_token_report(args.token_report)
        token_audit_rows = load_token_audit_rows(args.token_audit)
        gate = evaluate_phase5a_gate_with_audit(token_report, token_audit_rows)
        if not gate.passed:
            raise TrainingGateError("PHASE5A_GATE_FAIL: " + "; ".join(gate.reasons))
        final["phase5a_gate"] = {"passed": gate.passed, "reasons": gate.reasons}

        # revision
        from data_pipeline.training.backends import resolve_backend
        backend = resolve_backend(args.backend)
        requested, resolved, pinned = backend.resolve_revision(config.model_name, config.revision)
        load_revision = resolved or requested
        if not resolved:
            cached = _load_cached_revision(config.model_name)
            if cached:
                log.warning("HF Hub 不可达；使用缓存 SHA %s", cached)
                load_revision = cached
                pinned = True
        final["model"] = {
            "model_name": config.model_name,
            "requested_revision": requested,
            "resolved_revision": resolved,
            "load_revision": load_revision,
            "revision_pinned": pinned,
        }

        # 选样
        train_records = load_jsonl_records(args.sft_dir / "train.jsonl")
        totals = sorted(
            int(r.get("total_tokens") or 0)
            for r in token_audit_rows
            if r.get("split") == "train"
        )
        if totals:
            p95_idx = max(0, min(len(totals) - 1, int(round(0.95 * (len(totals) - 1)))))
            target_p95 = float(totals[p95_idx])
        else:
            target_p95 = None
        selection = select_smoke_samples_strict(
            train_records, token_audit_rows, config, target_train_p95=target_p95
        )
        final["smoke_samples"] = {
            "total_selected": len(selection.sample_ids),
            "sample_ids": selection.sample_ids,
            "short_count": selection.short_count,
            "p95_sample_id": selection.p95_sample_id,
            "p95_total_tokens": selection.p95_total_tokens,
        }

        # ---- A/B benchmark ----
        benchmark_report: Optional[BenchmarkResult] = None
        if not args.skip_benchmark:
            log.info("=== A/B benchmark (%s): %d steps ===", args.backend, args.benchmark_steps)
            benchmark_report = _run_single_benchmark(
                args.backend,
                config=config,
                train_records=train_records,
                sample_ids=selection.sample_ids,
                load_revision=load_revision,
                torch=torch,
                max_steps=args.benchmark_steps,
                adapter_dir=adapter_dir,
            )
            log.info("Benchmark result: %s", benchmark_report.status)
            final["benchmark"] = {
                "backend": benchmark_report.backend,
                "status": benchmark_report.status,
                "model_load_seconds": benchmark_report.model_load_seconds,
                "lora_apply_seconds": benchmark_report.lora_apply_seconds,
                "trainer_init_seconds": benchmark_report.trainer_init_seconds,
                "training_runtime_seconds": benchmark_report.training_runtime_seconds,
                "mean_step_seconds": benchmark_report.mean_step_seconds,
                "median_step_seconds": benchmark_report.median_step_seconds,
                "total_tokens": benchmark_report.total_tokens,
                "supervised_tokens": benchmark_report.supervised_tokens,
                "samples": benchmark_report.samples,
                "samples_per_sec": benchmark_report.samples_per_sec,
                "supervised_tokens_per_sec": benchmark_report.supervised_tokens_per_sec,
                "peak_allocated_gib": benchmark_report.peak_allocated_gib,
                "peak_reserved_gib": benchmark_report.peak_reserved_gib,
                "loss_finite": benchmark_report.loss_finite,
                "loss_history": benchmark_report.loss_history,
                "error": benchmark_report.error,
                "extra": benchmark_report.extra,
            }
            atomic_write_json(output_dir / f"backend_benchmark_{args.backend}.json",
                              final["benchmark"])

        # ---- Batch probe ----
        batch_probe_report: Optional[BatchProbeResult] = None
        if not args.skip_batch_probe:
            probe_batches = [int(x) for x in args.probe_batches.split(",") if x.strip()]
            log.info("=== Batch probe (%s): %s ===", args.backend, probe_batches)
            batch_probe_report = _run_batch_probe(
                args.backend,
                config=config,
                train_records=train_records,
                audit_rows=token_audit_rows,
                load_revision=load_revision,
                torch=torch,
                gpu_total_bytes=int(gpu_info["total_memory_bytes"]),
                probe_batches=probe_batches,
                batch_probe_steps=args.batch_probe_steps,
                base_adapter_dir=adapter_dir,
                include_batch_8=args.include_batch_8,
                enable_batch_8=args.enable_batch_8,
            )
            log.info("Batch probe result: %s", batch_probe_report.status)
            final["batch_probe"] = {
                "backend": batch_probe_report.backend,
                "status": batch_probe_report.status,
                "reason": batch_probe_report.reason,
                "results": batch_probe_report.results,
                "recommended_batch_size": batch_probe_report.recommended_batch_size,
                "recommended_gradient_accumulation_steps": batch_probe_report.recommended_gradient_accumulation_steps,
            }
            atomic_write_json(output_dir / f"batch_probe_{args.backend}.json",
                              final["batch_probe"])

        final["status"] = "PASS"
        atomic_write_json(output_dir / f"phase5c_a_run_{args.backend}.json", final)
        return 0

    except TrainingGateError as e:
        final["status"] = "BLOCKED"
        final["error"] = {"type": "TrainingGateError", "reason": str(e)}
        log.error("BLOCKED: %s", e)
        log.error(traceback.format_exc())
        atomic_write_json(output_dir / f"phase5c_a_run_{args.backend}.json", final)
        return 30
    except Exception as e:
        final["status"] = "BLOCKED"
        final["error"] = {"type": type(e).__name__, "reason": str(e), "traceback": traceback.format_exc()}
        log.error("BLOCKED: %s: %s", type(e).__name__, e)
        log.error(traceback.format_exc())
        atomic_write_json(output_dir / f"phase5c_a_run_{args.backend}.json", final)
        return 50
    finally:
        pass


if __name__ == "__main__":
    sys.exit(main())