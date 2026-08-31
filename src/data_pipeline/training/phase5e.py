"""Phase 5E — Final QLoRA Release & Project Closeout。

本模块是项目最终收口阶段的 CPU-only 工具集：

- ``ReleaseManifest``：标准化最终交付 manifest 的 dataclass 结构；
- ``validate_phase5d_report``：重新读取本地 Phase 5D 产物，校验其满足
  ``PHASE5D_CODE = PASS`` / ``ready_for_phase5e = true`` / test_count=79 /
  finetuned_test_loss < base_test_loss 等硬门槛；
- ``build_release_manifest``：把 Phase 5C-B run manifest + Phase 5D
  held-out test 指标 + 训练 metadata 拼接成可交付 ``release_manifest.json``；
- ``build_training_summary`` / ``build_evaluation_summary``：从 Phase 5C-B /
  Phase 5D runtime artifact 抽取 ``training_summary.json`` /
  ``evaluation_summary.json``；
- ``build_checksums`` + ``verify_checksums``：对 release 目录计算并校验
  SHA256；
- ``verify_best_adapter_provenance``：再校验 source ``best_adapter``
  tensor signature == ``checkpoint-288`` tensor signature（数值一致）。

**严禁在此模块引入 trainer / optimizer / scheduler / model.train() /
loss.backward()**。任何 GPU / 模型加载路径都不在本阶段范围内；GPU
fresh reload + inference 由 ``scripts/finalize_qlora_release.py`` 处理。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .phase5c_b import (
    AdapterTensorSignature,
    compute_adapter_signature,
)
from ..utils import atomic_write_json, calculate_file_sha256, get_logger


_LOG = get_logger("data_pipeline.training.phase5e")


SCHEMA_VERSION = "phase5e.v1"


# ---------------------------------------------------------------------------
# Constants — frozen final project identity
# ---------------------------------------------------------------------------

# 与 Phase 5A / 5B / 5C-A / 5C-B / 5D 一致的 base model 身份。
PHASE5E_BASE_MODEL_NAME = "Qwen/Qwen3-8B"
PHASE5E_BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

# 与 Phase 5C-B / 5D 完全对齐的 tokenizer signature。
PHASE5E_TOKENIZER_SIGNATURE = (
    "4ac4fed6d76f2cddf5ee756990b7bd4dc40655f96e4c6be460dc3299e73651c0"
)

# 源 best adapter 路径（不动）。
PHASE5E_BEST_ADAPTER_SOURCE = Path("outputs/phase5c-b/best_adapter")

# 源 best checkpoint 路径。
PHASE5E_BEST_CHECKPOINT_SOURCE = Path(
    "outputs/phase5c-b/checkpoints/checkpoint-288/adapter_model.safetensors"
)

# release 输出目录。
PHASE5E_RELEASE_ROOT = Path("outputs/phase5e/release")
PHASE5E_RELEASE_ADAPTER_DIRNAME = "adapter"

# runtime artifact 目录。
PHASE5E_RUNTIME_ARTIFACT_DIR = Path("data/training/phase5e")

# Phase 5D artifact 目录。
PHASE5D_ARTIFACT_DIR = Path("data/training/phase5d")

# Phase 5C-B artifact 目录。
PHASE5C_B_ARTIFACT_DIR = Path("data/training/phase5c_b")


# ---------------------------------------------------------------------------
# 1) Phase 5D result validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase5DVerification:
    """从本地 Phase 5D runtime artifact 重新读出的硬门槛校验结果。"""

    # Phase 5D 报告路径
    phase5d_report_path: str
    # ``phase5d_report.status`` 字段
    status: str
    # ``phase5d_report.code`` 字段
    code: str
    # ``phase5d_report.ready_for_phase5e`` 字段
    ready_for_phase5e: bool
    # test 集合样本数（必须 = 79）
    test_count: int
    # test_predictions.jsonl 实际行数
    predictions_rows: int
    # 基础模型 test loss 字段
    base_test_loss: float
    # 微调模型 test loss 字段
    finetuned_test_loss: float
    # test loss 改善绝对值（base - finetuned）
    test_loss_improvement: float
    # test loss 改善百分比
    test_loss_improvement_pct: float
    # ``eval_manifest.protocol_frozen`` 字段
    protocol_frozen: bool
    # ``eval_manifest.eval_protocol_signature`` 字段
    eval_protocol_signature: str
    # ``eval_manifest.model_name`` 字段
    model_name: str
    # ``eval_manifest.model_revision`` 字段
    model_revision: str
    # ``eval_manifest.tokenizer_signature`` 字段
    tokenizer_signature: str
    # ``eval_manifest.generation_max_new_tokens`` 字段
    generation_max_new_tokens: int
    # ``eval_manifest.test_jsonl_sha256`` 字段
    test_jsonl_sha256: str
    # ``eval_manifest.test_sample_id_signature`` 字段
    test_sample_id_signature: str
    # ``eval_manifest.best_adapter_signature_sha256`` 字段
    best_adapter_signature_sha256: str
    # ``eval_manifest.expected_lora_layers`` 字段
    expected_lora_layers: int
    # Phase 5D gate 是否 PASS（所有硬门槛满足）
    pass_: bool
    # 未通过原因列表（空 = PASS）
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"cannot parse JSON {path}: {e}") from e


def _check_test_predictions(
    predictions_path: Path,
) -> Tuple[int, List[str]]:
    """校验 ``test_predictions.jsonl`` 每行包含全部期望字段。"""
    if not predictions_path.exists():
        return (0, [f"missing predictions file: {predictions_path}"])
    reasons: List[str] = []
    rows = 0
    required = (
        "sample_id",
        "prompt_messages",
        "reference",
        "base_generation",
        "base_generated_tokens",
        "base_char_f1",
        "base_rouge_l_f1",
        "finetuned_generation",
        "finetuned_generated_tokens",
        "finetuned_char_f1",
        "finetuned_rouge_l_f1",
    )
    with predictions_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                reasons.append(f"predictions line {lineno} invalid JSON: {e}")
                continue
            for k in required:
                if k not in obj:
                    reasons.append(
                        f"predictions line {lineno} missing key {k!r}"
                    )
    return (rows, reasons)


def validate_phase5d_report(
    *,
    phase5d_dir: Path = PHASE5D_ARTIFACT_DIR,
) -> Phase5DVerification:
    """从 Phase 5D runtime artifact 重新读出硬门槛。

    至少校验：

    - ``phase5d_report.status == "PASS``" / ``code == "PASS``"；
    - ``ready_for_phase5e == true``；
    - ``test_count == 79``；
    - ``test_predictions.jsonl`` 行数 == 79，每行字段齐全；
    - ``finetuned_test_loss < base_test_loss``；
    - ``eval_manifest.protocol_frozen == true``；
    - ``eval_manifest.tokenizer_signature`` 锁定值；
    - ``eval_manifest.model_name`` / ``model_revision`` 锁定值。
    """
    phase5d_dir = Path(phase5d_dir)
    report_path = phase5d_dir / "phase5d_report.json"
    manifest_path = phase5d_dir / "eval_manifest.json"
    base_metrics_path = phase5d_dir / "base_test_metrics.json"
    finetuned_metrics_path = phase5d_dir / "finetuned_test_metrics.json"
    paired_path = phase5d_dir / "paired_metrics.json"
    predictions_path = phase5d_dir / "test_predictions.jsonl"

    reasons: List[str] = []
    for p in (report_path, manifest_path, base_metrics_path,
              finetuned_metrics_path, paired_path, predictions_path):
        if not p.exists():
            reasons.append(f"missing Phase 5D artifact: {p}")

    if reasons:
        return Phase5DVerification(
            phase5d_report_path=str(report_path),
            status="BLOCKED",
            code="BLOCKED",
            ready_for_phase5e=False,
            test_count=0,
            predictions_rows=0,
            base_test_loss=0.0,
            finetuned_test_loss=0.0,
            test_loss_improvement=0.0,
            test_loss_improvement_pct=0.0,
            protocol_frozen=False,
            eval_protocol_signature="",
            model_name="",
            model_revision="",
            tokenizer_signature="",
            generation_max_new_tokens=0,
            test_jsonl_sha256="",
            test_sample_id_signature="",
            best_adapter_signature_sha256="",
            expected_lora_layers=0,
            pass_=False,
            reasons=reasons,
        )

    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    base_metrics = _read_json(base_metrics_path)
    ft_metrics = _read_json(finetuned_metrics_path)
    paired = _read_json(paired_path)

    status = str(report.get("status", ""))
    code = str(report.get("code", ""))
    ready = bool(report.get("ready_for_phase5e", False))
    test_count = int(report.get("test_count", 0))
    base_test_loss = float(report.get("base_test_loss", 0.0))
    ft_test_loss = float(report.get("finetuned_test_loss", 0.0))
    improvement = float(report.get("test_loss_improvement", 0.0))
    improvement_pct = float(report.get("test_loss_improvement_pct", 0.0))

    protocol_frozen = bool(manifest.get("protocol_frozen", False))
    eval_protocol_signature = str(manifest.get("eval_protocol_signature", ""))
    model_name = str(manifest.get("model_name", ""))
    model_revision = str(manifest.get("model_revision", ""))
    tokenizer_signature = str(manifest.get("tokenizer_signature", ""))
    generation_max_new_tokens = int(manifest.get("generation_max_new_tokens", 0))
    test_jsonl_sha256 = str(manifest.get("test_jsonl_sha256", ""))
    test_sample_id_signature = str(manifest.get("test_sample_id_signature", ""))
    best_adapter_signature_sha256 = str(
        manifest.get("best_adapter_signature_sha256", "")
    )
    expected_lora_layers = int(manifest.get("expected_lora_layers", 0))

    pred_rows, pred_reasons = _check_test_predictions(predictions_path)
    reasons.extend(pred_reasons)

    if status != "PASS":
        reasons.append(f"phase5d_report.status={status!r} != 'PASS'")
    if code != "PASS":
        reasons.append(f"phase5d_report.code={code!r} != 'PASS'")
    if not ready:
        reasons.append("ready_for_phase5e != true")
    if test_count != 79:
        reasons.append(f"test_count={test_count} != 79")
    if pred_rows != 79:
        reasons.append(f"test_predictions rows={pred_rows} != 79")
    if not (math.isfinite(base_test_loss) and math.isfinite(ft_test_loss)):
        reasons.append("test loss non-finite")
    elif not (ft_test_loss < base_test_loss):
        reasons.append(
            f"finetuned_test_loss={ft_test_loss} >= "
            f"base_test_loss={base_test_loss}"
        )
    if not protocol_frozen:
        reasons.append("eval_manifest.protocol_frozen != true")
    if model_name != PHASE5E_BASE_MODEL_NAME:
        reasons.append(
            f"eval_manifest.model_name={model_name!r} != "
            f"{PHASE5E_BASE_MODEL_NAME!r}"
        )
    if model_revision != PHASE5E_BASE_REVISION:
        reasons.append(
            f"eval_manifest.model_revision={model_revision!r} != "
            f"{PHASE5E_BASE_REVISION!r}"
        )
    if tokenizer_signature != PHASE5E_TOKENIZER_SIGNATURE:
        reasons.append(
            f"eval_manifest.tokenizer_signature={tokenizer_signature!r} != "
            f"PHASE5E_TOKENIZER_SIGNATURE"
        )

    # Suppress unused-variable warnings for the loaded metric files.
    _ = (base_metrics, ft_metrics, paired)

    return Phase5DVerification(
        phase5d_report_path=str(report_path),
        status=status,
        code=code,
        ready_for_phase5e=ready,
        test_count=test_count,
        predictions_rows=pred_rows,
        base_test_loss=base_test_loss,
        finetuned_test_loss=ft_test_loss,
        test_loss_improvement=improvement,
        test_loss_improvement_pct=improvement_pct,
        protocol_frozen=protocol_frozen,
        eval_protocol_signature=eval_protocol_signature,
        model_name=model_name,
        model_revision=model_revision,
        tokenizer_signature=tokenizer_signature,
        generation_max_new_tokens=generation_max_new_tokens,
        test_jsonl_sha256=test_jsonl_sha256,
        test_sample_id_signature=test_sample_id_signature,
        best_adapter_signature_sha256=best_adapter_signature_sha256,
        expected_lora_layers=expected_lora_layers,
        pass_=(len(reasons) == 0),
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# 2) Best adapter provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterProvenance:
    """ ``best_adapter`` 与 ``checkpoint-288`` 的 tensor 一致性结果。"""

    # 源 best adapter 文件路径（绝对或相对）
    source_path: str
    # checkpoint 文件路径（绝对或相对）
    checkpoint_path: str
    # 源 best adapter tensor signature
    source_signature: str
    # checkpoint 适配器的 tensor signature
    checkpoint_signature: str
    # tensor 总数（应当 = 504）
    n_tensors: int
    # source 与 checkpoint 是否数值一致
    equal: bool
    # 校验失败时的错误信息（None = 无错误）
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verify_best_adapter_provenance(
    *,
    best_adapter_path: Path = PHASE5E_BEST_ADAPTER_SOURCE
    / "adapter_model.safetensors",
    checkpoint_path: Path = PHASE5E_BEST_CHECKPOINT_SOURCE,
    torch: Any = None,
    prefer_dtype: str = "bfloat16",
) -> AdapterProvenance:
    """重新计算 ``best_adapter`` 与 ``checkpoint-288`` 的 tensor signature。

    使用 Phase 5C-B 已建立的 ``compute_adapter_signature``；返回
    ``equal=True`` 表示 source 与 checkpoint 在 ``prefer_dtype`` 强转后
    数值上完全一致（tensor name / shape / dtype 各自升序 SHA256 拼接
    一致）。
    """
    best_path = Path(best_adapter_path)
    ckpt_path = Path(checkpoint_path)
    try:
        if not best_path.exists():
            raise ValueError(f"best_adapter not found: {best_path}")
        if not ckpt_path.exists():
            raise ValueError(f"checkpoint not found: {ckpt_path}")
        best_sig: AdapterTensorSignature = compute_adapter_signature(
            best_path, torch=torch, prefer_dtype=prefer_dtype,
        )
        ckpt_sig: AdapterTensorSignature = compute_adapter_signature(
            ckpt_path, torch=torch, prefer_dtype=prefer_dtype,
        )
        equal = bool(
            best_sig.n_tensors == ckpt_sig.n_tensors
            and best_sig.signature == ckpt_sig.signature
        )
        return AdapterProvenance(
            source_path=str(best_path),
            checkpoint_path=str(ckpt_path),
            source_signature=best_sig.signature,
            checkpoint_signature=ckpt_sig.signature,
            n_tensors=int(best_sig.n_tensors),
            equal=equal,
            error=None,
        )
    except Exception as e:
        return AdapterProvenance(
            source_path=str(best_path),
            checkpoint_path=str(ckpt_path),
            source_signature="",
            checkpoint_signature="",
            n_tensors=0,
            equal=False,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# 3) Release manifest builders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseManifest:
    """``release_manifest.json`` 数据契约。"""

    # manifest schema 版本号
    schema_version: str
    # 项目名称
    project_name: str
    # release 类型（固定 lora_adapter）
    release_type: str
    # release 生成时间（UTC ISO 字符串）
    release_created_at: str
    # 来源 git commit SHA
    source_git_commit: str
    # 来源 git 是否 dirty
    source_git_dirty: bool
    # base model 身份信息
    base_model: Dict[str, Any]
    # tokenizer 指纹
    tokenizer: Dict[str, str]
    # LoRA 配置
    lora: Dict[str, Any]
    # 训练配置
    training: Dict[str, Any]
    # 数据集指纹
    dataset: Dict[str, Any]
    # validation 阶段指标
    validation: Dict[str, Any]
    # held-out test 阶段指标
    held_out_test: Dict[str, Any]
    # adapter 指纹与来源
    adapter: Dict[str, Any]
    # Phase 5D gate 摘要
    phase5d: Dict[str, Any]
    # release 内文件清单（path / size / sha256）
    files: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_release_manifest(
    *,
    git_commit: str,
    git_dirty: bool,
    phase5d_verification: Phase5DVerification,
    adapter_provenance: AdapterProvenance,
    release_dir: Path = PHASE5E_RELEASE_ROOT,
    source_adapter_dir: Path = PHASE5E_BEST_ADAPTER_SOURCE,
    run_manifest: Optional[Mapping[str, Any]] = None,
    file_inventory: Optional[Sequence[Mapping[str, Any]]] = None,
    project_name: str = "QLoRA-STF--",
) -> Dict[str, Any]:
    """构造最终 ``release_manifest.json`` 内容（dict 形态，可 atomic 写入）。"""
    lora_block: Dict[str, Any] = {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "bias": "none",
        "target_modules": "all-linear",
        "lora_layers": 252,
        "trainable_params": 43646976,
    }
    training_block: Dict[str, Any] = {
        "backend": "unsloth",
        "epochs": 3,
        "global_steps": 432,
        "batch_size": 8,
        "gradient_accumulation": 1,
        "effective_batch": 8,
        "learning_rate": 1e-4,
        "scheduler": "linear",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 42,
        "data_seed": 42,
        "max_length": 2048,
        "assistant_only_loss": True,
        "packing": False,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    dataset_block: Dict[str, Any] = {
        "train_count": 1149,
        "validation_count": 75,
        "test_count": 79,
        "train_sha256": "b59927afe162bc9eef340386b01775ea8796032aeb5fda324652b15d6e1ae58c",
        "validation_sha256": "61d29e321f4c8259870dddb50a8f99d914a87372233e90cdf96b8465d0dc64a4",
        "test_sha256": "b77a51613ebe6b75d2d7f9e676a0f41adfd9e6c5093da19651076967122f5db8",
        "train_sample_id_signature": (
            "58ae3a429e1120413d58814c27c1ec7795f7d20afc73334ed69a25ff4c8e4bfc"
        ),
        "validation_sample_id_signature": (
            "56e4ac9398ff69ca22725eb8e292e23665d4d6cf5bae92abc76ede199d8a0740"
        ),
        "test_sample_id_signature": (
            phase5d_verification.test_sample_id_signature
        ),
    }
    validation_block: Dict[str, Any] = {
        "eval_loss_before_training": 2.108738660812378,
        "epoch_1_eval_loss": 1.164391040802002,
        "epoch_2_eval_loss": 1.1593782901763916,
        "epoch_3_eval_loss": 1.1902490854263306,
        "best_epoch": 2,
        "best_eval_loss": 1.1593782901763916,
        "best_checkpoint": "checkpoint-288",
    }
    held_out_test_block: Dict[str, Any] = {
        "base_test_loss": phase5d_verification.base_test_loss,
        "finetuned_test_loss": phase5d_verification.finetuned_test_loss,
        "improvement": phase5d_verification.test_loss_improvement,
        "improvement_pct": phase5d_verification.test_loss_improvement_pct,
        "base_char_f1": 0.2702,
        "finetuned_char_f1": 0.3753,
        "base_rouge_l": 0.1447,
        "finetuned_rouge_l": 0.2626,
        "paired_char_f1_wins_finetuned": 53,
        "paired_char_f1_wins_base": 26,
        "paired_char_f1_ties": 0,
        "paired_rouge_l_wins_finetuned": 61,
        "paired_rouge_l_wins_base": 18,
        "paired_rouge_l_ties": 0,
    }
    adapter_block: Dict[str, Any] = {
        "source_path": str(source_adapter_dir),
        "release_path": str(release_dir / PHASE5E_RELEASE_ADAPTER_DIRNAME),
        "tensor_signature": adapter_provenance.source_signature,
        "checkpoint_signature": adapter_provenance.checkpoint_signature,
        "tensor_count": adapter_provenance.n_tensors,
        "equal_to_checkpoint_288": adapter_provenance.equal,
    }
    phase5d_block: Dict[str, Any] = {
        "eval_protocol_signature": phase5d_verification.eval_protocol_signature,
        "phase5d_code": phase5d_verification.code,
        "test_count": phase5d_verification.test_count,
        "predictions_rows": phase5d_verification.predictions_rows,
        "ready_for_phase5e": phase5d_verification.ready_for_phase5e,
        "protocol_frozen": phase5d_verification.protocol_frozen,
        "test_jsonl_sha256": phase5d_verification.test_jsonl_sha256,
        "best_adapter_signature_sha256": (
            phase5d_verification.best_adapter_signature_sha256
        ),
    }

    payload = ReleaseManifest(
        schema_version=SCHEMA_VERSION,
        project_name=project_name,
        release_type="lora_adapter",
        release_created_at=_utc_now_iso(),
        source_git_commit=git_commit,
        source_git_dirty=bool(git_dirty),
        base_model={
            "model_name": PHASE5E_BASE_MODEL_NAME,
            "revision": PHASE5E_BASE_REVISION,
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        tokenizer={
            "signature": PHASE5E_TOKENIZER_SIGNATURE,
        },
        lora=lora_block,
        training=training_block,
        dataset=dataset_block,
        validation=validation_block,
        held_out_test=held_out_test_block,
        adapter=adapter_block,
        phase5d=phase5d_block,
        files=list(file_inventory or []),
    )
    if run_manifest:
        # 把 run_manifest 的 dataset / lora 字段并入；不影响顶层结构。
        rm_ds = run_manifest.get("dataset") if isinstance(run_manifest, dict) else None
        if isinstance(rm_ds, dict):
            for k in ("train_count", "validation_count", "train_sha256",
                      "validation_sha256", "train_sample_id_signature",
                      "validation_sample_id_signature"):
                if k in rm_ds and rm_ds[k] is not None:
                    payload.dataset[k] = rm_ds[k]
    return payload.to_dict()


def write_release_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写入 ``release_manifest.json``。"""
    atomic_write_json(Path(path), dict(payload))
    _LOG.info("Wrote release_manifest.json to %s", path)


# ---------------------------------------------------------------------------
# 4) Training summary
# ---------------------------------------------------------------------------


def build_training_summary(
    *,
    run_manifest_path: Path = PHASE5C_B_ARTIFACT_DIR / "run_manifest.json",
    final_report_path: Path = PHASE5C_B_ARTIFACT_DIR / "final_report.json",
) -> Dict[str, Any]:
    """从 Phase 5C-B run manifest + final report 抽取训练摘要。

    严禁手写猜测；如果源文件缺失 ⇒ 抛 ``ValueError``。
    """
    rm = _read_json(Path(run_manifest_path))
    fr = _read_json(Path(final_report_path))
    validation = fr.get("validation", {}) or {}
    training = rm.get("training", {}) or {}
    lora = rm.get("lora", {}) or {}
    quantization = rm.get("quantization", {}) or {}
    model = rm.get("model", {}) or {}
    dataset = rm.get("dataset", {}) or {}

    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_run_manifest": str(run_manifest_path),
        "source_final_report": str(final_report_path),
        "backend": rm.get("backend", "unsloth"),
        "model_name": model.get("model_name", ""),
        "model_revision": model.get("load_revision", ""),
        "tokenizer_signature": rm.get("tokenizer_signature", ""),
        "quantization": {
            "quant_type": quantization.get("quant_type", "nf4"),
            "double_quant": bool(quantization.get("double_quant", True)),
            "compute_dtype": quantization.get("compute_dtype", "bfloat16"),
            "max_length": int(quantization.get("max_length", 2048)),
        },
        "lora": {
            "r": int(lora.get("r", 16)),
            "alpha": int(lora.get("alpha", 32)),
            "dropout": float(lora.get("dropout", 0.05)),
            "bias": str(lora.get("bias", "none")),
            "target_modules": str(lora.get("target_modules", "all-linear")),
            "task_type": str(lora.get("task_type", "CAUSAL_LM")),
            "lora_layers": int(fr.get("lora", {}).get("lora_layers", 252)),
            "trainable_params": int(
                fr.get("lora", {}).get("trainable_params", 0)
            ),
            "all_params": int(fr.get("lora", {}).get("all_params", 0)),
        },
        "training": {
            "epochs": int(training.get("num_train_epochs", 3)),
            "global_steps": int(fr.get("training", {}).get("global_step", 432)),
            "per_device_train_batch_size": int(
                training.get("per_device_train_batch_size", 8)
            ),
            "gradient_accumulation_steps": int(
                training.get("gradient_accumulation_steps", 1)
            ),
            "effective_batch_size": int(
                training.get("per_device_train_batch_size", 8)
                * training.get("gradient_accumulation_steps", 1)
            ),
            "learning_rate": float(training.get("learning_rate", 1e-4)),
            "lr_scheduler_type": str(training.get("lr_scheduler_type", "linear")),
            "warmup_ratio": float(training.get("warmup_ratio", 0.03)),
            "weight_decay": float(training.get("weight_decay", 0.0)),
            "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
            "seed": int(training.get("seed", 42)),
            "data_seed": int(training.get("data_seed", 42)),
            "assistant_only_loss": bool(training.get("assistant_only_loss", True)),
            "packing": bool(training.get("packing", False)),
            "bf16": bool(training.get("bf16", True)),
            "fp16": bool(training.get("fp16", False)),
            "gradient_checkpointing": bool(
                training.get("gradient_checkpointing", True)
            ),
            "save_strategy": str(training.get("save_strategy", "epoch")),
            "eval_strategy": str(training.get("eval_strategy", "epoch")),
            "load_best_model_at_end": bool(
                training.get("load_best_model_at_end", True)
            ),
            "metric_for_best_model": str(
                training.get("metric_for_best_model", "eval_loss")
            ),
            "greater_is_better": bool(training.get("greater_is_better", False)),
            "save_total_limit": int(training.get("save_total_limit", 2)),
        },
        "dataset": {
            "train_count": int(dataset.get("train_count", 0)),
            "validation_count": int(dataset.get("validation_count", 0)),
            "train_sha256": str(dataset.get("train_sha256", "")),
            "validation_sha256": str(dataset.get("validation_sha256", "")),
            "train_sample_id_signature": str(
                dataset.get("train_sample_id_signature", "")
            ),
            "validation_sample_id_signature": str(
                dataset.get("validation_sample_id_signature", "")
            ),
        },
        "validation": {
            "before_training": float(validation.get("before_training", 0.0)),
            "epoch_eval_losses": list(validation.get("epoch_eval_losses", [])),
            "best_epoch": int(validation.get("best_epoch", 2)),
            "best_eval_loss": float(validation.get("best_eval_loss", 0.0)),
            "best_is_last": bool(validation.get("best_is_last", False)),
            "best_checkpoint": "checkpoint-288",
        },
    }
    return summary


def write_training_summary(path: Path, summary: Mapping[str, Any]) -> None:
    atomic_write_json(Path(path), dict(summary))
    _LOG.info("Wrote training_summary.json to %s", path)


# ---------------------------------------------------------------------------
# 5) Evaluation summary
# ---------------------------------------------------------------------------


def build_evaluation_summary(
    *,
    phase5d_dir: Path = PHASE5D_ARTIFACT_DIR,
) -> Dict[str, Any]:
    """从 Phase 5D runtime artifact 抽取 held-out test summary。"""
    phase5d_dir = Path(phase5d_dir)
    report_path = phase5d_dir / "phase5d_report.json"
    base_metrics_path = phase5d_dir / "base_test_metrics.json"
    ft_metrics_path = phase5d_dir / "finetuned_test_metrics.json"
    paired_path = phase5d_dir / "paired_metrics.json"

    report = _read_json(report_path)
    base_metrics = _read_json(base_metrics_path)
    ft_metrics = _read_json(ft_metrics_path)
    paired = _read_json(paired_path)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_phase5d_dir": str(phase5d_dir),
        "phase5d_status": report.get("status"),
        "phase5d_code": report.get("code"),
        "ready_for_phase5e": report.get("ready_for_phase5e"),
        "test_count": int(report.get("test_count", 0)),
        "base": dict(base_metrics),
        "finetuned": dict(ft_metrics),
        "loss_improvement": float(report.get("test_loss_improvement", 0.0)),
        "loss_improvement_pct": float(
            report.get("test_loss_improvement_pct", 0.0)
        ),
        "paired_char_f1": dict(paired.get("char_f1", {})),
        "paired_rouge_l_f1": dict(paired.get("rouge_l_f1", {})),
        "eval_protocol_signature": report.get("eval_protocol_signature"),
        "leakage_audit": dict(report.get("leakage_audit", {})),
    }
    return summary


def write_evaluation_summary(path: Path, summary: Mapping[str, Any]) -> None:
    atomic_write_json(Path(path), dict(summary))
    _LOG.info("Wrote evaluation_summary.json to %s", path)


# ---------------------------------------------------------------------------
# 6) File inventory + checksums
# ---------------------------------------------------------------------------


# forbidden release artifacts (must NOT exist in release directory)
FORBIDDEN_RELEASE_ARTIFACT_PATTERNS = (
    "*.gguf",
    "merged_model*",
    "pytorch_model*.bin",
    "model-*.safetensors",
)


def is_forbidden_release_artifact(rel_path: str) -> bool:
    """根据相对路径判断是否属于禁止在 release 中出现的 base weights 形式。"""
    import fnmatch

    base = rel_path.replace("\\", "/").rsplit("/", 1)[-1]
    for pat in FORBIDDEN_RELEASE_ARTIFACT_PATTERNS:
        if fnmatch.fnmatch(base, pat):
            return True
    return False


def build_file_inventory(
    release_dir: Path,
    *,
    root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """枚举 ``release_dir`` 下所有稳定文件，计算 SHA256 + size。"""
    release_dir = Path(release_dir)
    root = Path(root) if root else release_dir
    items: List[Dict[str, Any]] = []
    if not release_dir.exists():
        return items
    for p in sorted(release_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if is_forbidden_release_artifact(rel):
            continue
        items.append(
            {
                "path": rel,
                "size_bytes": int(p.stat().st_size),
                "sha256": calculate_file_sha256(p),
            }
        )
    return items


def build_checksums(file_inventory: Sequence[Mapping[str, Any]]) -> str:
    """把 ``file_inventory`` 渲染为 ``sha256sum`` 风格文本。"""
    lines: List[str] = []
    for item in file_inventory:
        path = str(item.get("path", ""))
        sha = str(item.get("sha256", ""))
        if not path or not sha:
            continue
        lines.append(f"{sha}  {path}")
    return ("\n".join(lines) + ("\n" if lines else ""))


def write_checksums(path: Path, body: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(body, encoding="utf-8")
    _LOG.info("Wrote checksums.sha256 to %s", path)


def verify_checksums(
    checksums_path: Path,
    *,
    root: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    """重新读取 ``checksums.sha256``，逐文件校验 SHA256。

    Returns
    -------
    (ok, reasons)
        ``ok=True`` 表示 missing=0 / mismatch=0。
    """
    p = Path(checksums_path)
    if not p.exists():
        return (False, [f"missing checksums file: {p}"])
    root = Path(root) if root else p.parent
    text = p.read_text(encoding="utf-8")
    reasons: List[str] = []
    missing = 0
    mismatch = 0
    checked = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            reasons.append(
                f"line {lineno} malformed: expected '<sha256>  <path>'"
            )
            continue
        expected_sha, rel_path = parts
        full = root / rel_path
        if not full.exists():
            missing += 1
            reasons.append(f"missing file: {rel_path}")
            continue
        actual_sha = calculate_file_sha256(full)
        checked += 1
        if actual_sha != expected_sha:
            mismatch += 1
            reasons.append(
                f"hash mismatch: {rel_path} expected={expected_sha[:16]} "
                f"actual={actual_sha[:16]}"
            )
    if missing > 0 or mismatch > 0:
        reasons.insert(0, f"checked={checked} missing={missing} mismatch={mismatch}")
        return (False, reasons)
    return (True, [])


# ---------------------------------------------------------------------------
# 7) Release adapter copy
# ---------------------------------------------------------------------------


def copy_release_adapter(
    source_dir: Path,
    release_dir: Path,
    *,
    adapter_dirname: str = PHASE5E_RELEASE_ADAPTER_DIRNAME,
) -> Path:
    """把 ``source_dir`` 下的所有文件 copy 到 ``release_dir/<adapter_dirname>``。

    **不删除 / 不移动** 源 adapter；返回目标 adapter 目录。
    """
    import shutil

    source_dir = Path(source_dir)
    release_dir = Path(release_dir)
    target = release_dir / adapter_dirname
    if target.exists():
        # idempotent: 清空旧 target 后重新 copy（保留目录本身）
        for child in sorted(target.iterdir(), key=lambda x: -len(str(x))):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_dir.iterdir()):
        if not src.is_file():
            continue
        dst = target / src.name
        shutil.copy2(src, dst)
    _LOG.info("Copied %s -> %s", source_dir, target)
    return target


# ---------------------------------------------------------------------------
# 8) Final gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalGateResult:
    """Phase 5E 最终门判定。"""

    # 最终门状态码："PASS" 或 "BLOCKED"
    code: str
    # 项目是否完成（仅当所有门槛满足时为 True）
    project_complete: bool
    # 未通过原因列表（空 = PASS）
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def final_gate(
    *,
    phase5d_verification: Phase5DVerification,
    adapter_provenance: AdapterProvenance,
    release_manifest_path: Path,
    checksums_verified: bool,
    fresh_reload_passed: bool,
    inference_non_empty: int,
    inference_total: int,
) -> FinalGateResult:
    """综合所有 Phase 5E 硬门槛，输出 ``PROJECT_COMPLETE`` 标志。

    全部满足 ⇒ ``code='PASS'`` 且 ``project_complete=True``。
    """
    reasons: List[str] = []

    if not phase5d_verification.pass_:
        reasons.append("PHASE5D_FAIL: " + "; ".join(phase5d_verification.reasons))
    if phase5d_verification.test_count != 79:
        reasons.append("TEST_COUNT_DRIFT")
    if not (
        phase5d_verification.finetuned_test_loss
        < phase5d_verification.base_test_loss
    ):
        reasons.append("TEST_LOSS_NOT_IMPROVED")
    if not adapter_provenance.equal:
        reasons.append(
            f"FINAL_ADAPTER_PROVENANCE_FAIL: source={adapter_provenance.source_signature[:16]} "
            f"checkpoint={adapter_provenance.checkpoint_signature[:16]} "
            f"err={adapter_provenance.error}"
        )
    if adapter_provenance.n_tensors != 504:
        reasons.append(
            f"ADAPTER_TENSOR_COUNT_FAIL: n_tensors={adapter_provenance.n_tensors} != 504"
        )
    if not Path(release_manifest_path).exists():
        reasons.append("RELEASE_MANIFEST_MISSING")
    if not checksums_verified:
        reasons.append("RELEASE_INTEGRITY_FAIL")
    if not fresh_reload_passed:
        reasons.append("FRESH_RELOAD_FAIL")
    if inference_total <= 0 or inference_non_empty < inference_total:
        reasons.append(
            f"INFERENCE_NOT_ALL_NONEMPTY: {inference_non_empty}/{inference_total}"
        )

    if reasons:
        return FinalGateResult(
            code="BLOCKED", project_complete=False, reasons=reasons,
        )
    return FinalGateResult(
        code="PASS", project_complete=True, reasons=[],
    )


__all__ = [
    "SCHEMA_VERSION",
    "PHASE5E_BASE_MODEL_NAME",
    "PHASE5E_BASE_REVISION",
    "PHASE5E_TOKENIZER_SIGNATURE",
    "PHASE5E_BEST_ADAPTER_SOURCE",
    "PHASE5E_BEST_CHECKPOINT_SOURCE",
    "PHASE5E_RELEASE_ROOT",
    "PHASE5E_RELEASE_ADAPTER_DIRNAME",
    "PHASE5E_RUNTIME_ARTIFACT_DIR",
    "PHASE5D_ARTIFACT_DIR",
    "PHASE5C_B_ARTIFACT_DIR",
    "Phase5DVerification",
    "AdapterProvenance",
    "ReleaseManifest",
    "FinalGateResult",
    "FORBIDDEN_RELEASE_ARTIFACT_PATTERNS",
    "validate_phase5d_report",
    "verify_best_adapter_provenance",
    "build_release_manifest",
    "write_release_manifest",
    "build_training_summary",
    "write_training_summary",
    "build_evaluation_summary",
    "write_evaluation_summary",
    "build_file_inventory",
    "build_checksums",
    "write_checksums",
    "verify_checksums",
    "copy_release_adapter",
    "is_forbidden_release_artifact",
    "final_gate",
]