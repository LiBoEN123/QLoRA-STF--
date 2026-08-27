"""Phase 5C-B — 正式 3 Epoch QLoRA SFT 配置 + 数据指纹 + 训练辅助。

本模块只承载 Phase 5C-B 新增的语义，**禁止**改写 Phase 5B / 5C-A 已验证的
训练超参（max_length / NF4 / double quant / BF16 / LoRA r16a32d0.05 /
assistant_only_loss / packing 等）。Phase 5B ``QLoRASmokeConfig`` 不变。

新增字段：

- ``num_train_epochs = 3``
- ``per_device_train_batch_size = 8``（默认 8；显式回退 4/2 时记录）
- ``gradient_accumulation_steps = 1``
- ``learning_rate = 1e-4``
- ``lr_scheduler_type = "linear"``
- ``warmup_ratio = 0.03``
- ``weight_decay = 0.0``
- ``max_grad_norm = 1.0``
- ``logging_steps = 5``
- ``save_strategy = "epoch"``
- ``eval_strategy = "epoch"``
- ``save_total_limit = 2``
- ``load_best_model_at_end = True``
- ``metric_for_best_model = "eval_loss"``
- ``greater_is_better = False``
- ``seed = 42``
- ``data_seed = 42``

辅助函数：

- ``compute_dataset_fingerprint(jsonl_path)``：SHA256(file content)。
- ``compute_sample_id_signature(jsonl_path)``：基于 ``sample_id`` 升序拼接
  的 SHA256；data 改变 ⇒ signature 改变。
- ``verify_resume_manifest(...)``：检查 resume checkpoint / adapter 与当前
  manifest 是否兼容；任何关键字段不一致 ⇒ ``RESUME_CONFIG_MISMATCH``。
- ``select_best_checkpoint(... )``：根据 eval_loss 选 best checkpoint。
- ``write_run_manifest(...)`` / ``load_run_manifest(...)``：读写
  ``run_manifest.json``。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ..utils import atomic_write_json, get_logger, load_jsonl_records


_LOG = get_logger("data_pipeline.training.phase5c_b")


# Phase 5C-B 正式训练配置默认值。禁止修改已验证 QLoRA 字段。
DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
# 必须 pin 的 revision；与 Phase 5C-A 已加载 SHA 一致。
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_QUANT_TYPE = "nf4"
DEFAULT_COMPUTE_DTYPE = "bfloat16"
DEFAULT_DOUBLE_QUANT = True

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_BIAS = "none"
DEFAULT_LORA_TASK_TYPE = "CAUSAL_LM"
DEFAULT_LORA_TARGET_MODULES = "all-linear"

# 训练参数：固化。
DEFAULT_NUM_TRAIN_EPOCHS = 3
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_LR_SCHEDULER_TYPE = "linear"
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STRATEGY = "epoch"
DEFAULT_EVAL_STRATEGY = "epoch"
DEFAULT_SAVE_TOTAL_LIMIT = 2
DEFAULT_LOAD_BEST_MODEL_AT_END = True
DEFAULT_METRIC_FOR_BEST_MODEL = "eval_loss"
DEFAULT_GREATER_IS_BETTER = False
DEFAULT_GRADIENT_CHECKPOINTING = True
DEFAULT_PACKING = False
DEFAULT_ASSISTANT_ONLY_LOSS = True
DEFAULT_BF16 = True
DEFAULT_FP16 = False
DEFAULT_REPORT_TO = "none"
DEFAULT_SEED = 42
DEFAULT_DATA_SEED = 42

# Pre-flight：30-step batch=8 workload。
DEFAULT_PREFLIGHT_MAX_STEPS = 30

# 推理：固定 prompt。
DEFAULT_INFERENCE_PROMPT = "什么是基层党组织？"
DEFAULT_INFERENCE_MAX_NEW_TOKENS = 64


@dataclass(frozen=True)
class Phase5CBTrainConfig:
    """Phase 5C-B 正式 3 Epoch QLoRA SFT 配置。

    字段全部冻结；``__post_init__`` 校验硬不变量。
    """

    # 基础模型仓库名；与 Phase 5A / 5B / 5C-A 一致。
    model_name: str = DEFAULT_MODEL_NAME
    # 必须 pin 的 commit SHA；禁止退回 main / dynamic。
    revision: str = DEFAULT_REVISION

    # 训练最大 token 长度；与数据集真实最大 token (1672) 匹配，留足 buffer。
    max_length: int = DEFAULT_MAX_LENGTH
    # 4-bit 量化类型；Phase 5C-B 固定 NF4。
    quant_type: str = DEFAULT_QUANT_TYPE
    # 量化张量的计算 dtype；本阶段固定 bfloat16，不退化为 fp16。
    compute_dtype: str = DEFAULT_COMPUTE_DTYPE
    # 是否启用 bitsandbytes double quantization；Phase 5C-B 固定 True。
    double_quant: bool = DEFAULT_DOUBLE_QUANT

    # LoRA 低秩矩阵的 rank；与 Phase 5B baseline 一致。
    lora_r: int = DEFAULT_LORA_R
    # LoRA 缩放参数 alpha；与 Phase 5B baseline 一致。
    lora_alpha: int = DEFAULT_LORA_ALPHA
    # LoRA 层训练时的 dropout；禁止为 0（Unsloth fast path 也不允许）。
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    # LoRA bias 处理方式；固定 none。
    lora_bias: str = DEFAULT_LORA_BIAS
    # LoRA 任务类型；固定 CAUSAL_LM。
    lora_task_type: str = DEFAULT_LORA_TASK_TYPE
    # LoRA 目标模块；与 Phase 5B 一致使用 all-linear。
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES

    # 总训练 epoch 数；Phase 5C-B 固定 3。
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS
    # 每设备训练 batch size；默认 8（可显式回退到 4）。
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE
    # 梯度累积步数；与 batch size 共同决定 effective batch。
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    # 学习率；正式训练固定 1e-4。
    learning_rate: float = DEFAULT_LEARNING_RATE
    # 学习率调度器类型；Phase 5C-B 固定 linear。
    lr_scheduler_type: str = DEFAULT_LR_SCHEDULER_TYPE
    # warmup 比例；与 Phase 5B smoke 的 warmup_steps=0 不同，正式训练用 ratio。
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    # 权重衰减；标准 AdamW 默认 0.0。
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    # 梯度裁剪阈值；Phase 5C-B 固定 1.0。
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    # 日志步数；5 步一次。
    logging_steps: int = DEFAULT_LOGGING_STEPS

    # checkpoint 保存策略；Phase 5C-B 固定 epoch。
    save_strategy: str = DEFAULT_SAVE_STRATEGY
    # evaluation 策略；Phase 5C-B 固定 epoch。
    eval_strategy: str = DEFAULT_EVAL_STRATEGY
    # 最多保留 checkpoint 数；Phase 5C-B 固定 2。
    save_total_limit: int = DEFAULT_SAVE_TOTAL_LIMIT
    # 训练结束时是否加载 best checkpoint；Phase 5C-B 固定 True。
    load_best_model_at_end: bool = DEFAULT_LOAD_BEST_MODEL_AT_END
    # best checkpoint 选择指标；固定 eval_loss。
    metric_for_best_model: str = DEFAULT_METRIC_FOR_BEST_MODEL
    # best 是否越大越好；eval_loss 越小越好 ⇒ False。
    greater_is_better: bool = DEFAULT_GREATER_IS_BETTER

    # 是否启用梯度检查点；Phase 5C-B 固定 True。
    gradient_checkpointing: bool = DEFAULT_GRADIENT_CHECKPOINTING
    # 是否开启 packing；Phase 5C-B 禁止。
    packing: bool = DEFAULT_PACKING
    # TRL SFTTrainer 的 assistant_only_loss；本阶段必须 True。
    assistant_only_loss: bool = DEFAULT_ASSISTANT_ONLY_LOSS
    # 是否启用 bf16 训练；本阶段必须 True。
    bf16: bool = DEFAULT_BF16
    # 是否启用 fp16 训练；本阶段必须 False。
    fp16: bool = DEFAULT_FP16
    # Trainer 上报目标；本阶段固定 none。
    report_to: str = DEFAULT_REPORT_TO
    # 全局随机种子；与 Phase 5B baseline 一致。
    seed: int = DEFAULT_SEED
    # 数据采样随机种子；与 seed 区分。
    data_seed: int = DEFAULT_DATA_SEED

    # 30-step preflight 的 max_steps。
    preflight_max_steps: int = DEFAULT_PREFLIGHT_MAX_STEPS

    # 正式推理使用的固定 prompt；与 Phase 5B 一致。
    inference_prompt: str = DEFAULT_INFERENCE_PROMPT
    # 推理允许的最大生成 token 数。
    inference_max_new_tokens: int = DEFAULT_INFERENCE_MAX_NEW_TOKENS

    def __post_init__(self) -> None:
        """冻结配置基本不变量校验。"""
        if self.max_length != DEFAULT_MAX_LENGTH:
            raise ValueError(f"max_length 必须固定为 {DEFAULT_MAX_LENGTH}，得到 {self.max_length}")
        if self.quant_type != "nf4":
            raise ValueError(f"quant_type 必须固定为 'nf4'，得到 {self.quant_type!r}")
        if self.compute_dtype != "bfloat16":
            raise ValueError(f"compute_dtype 必须固定为 'bfloat16'，得到 {self.compute_dtype!r}")
        if not self.double_quant:
            raise ValueError("double_quant 必须为 True（Phase 5C-B 固定开启）")
        if self.bf16 is not True:
            raise ValueError("bf16 必须为 True（Phase 5C-B 禁止退回 fp16）")
        if self.fp16 is not False:
            raise ValueError("fp16 必须为 False（Phase 5C-B 禁止启用 fp16）")
        if self.packing:
            raise ValueError("packing 必须为 False（Phase 5C-B 禁止开启）")
        if not self.assistant_only_loss:
            raise ValueError("assistant_only_loss 必须为 True（Phase 5C-B 禁止退回）")
        if self.lora_r != DEFAULT_LORA_R:
            raise ValueError(f"lora_r 必须固定为 {DEFAULT_LORA_R}，得到 {self.lora_r}")
        if self.lora_alpha != DEFAULT_LORA_ALPHA:
            raise ValueError(f"lora_alpha 必须固定为 {DEFAULT_LORA_ALPHA}，得到 {self.lora_alpha}")
        if self.lora_dropout != DEFAULT_LORA_DROPOUT:
            raise ValueError(f"lora_dropout 必须固定为 {DEFAULT_LORA_DROPOUT}，得到 {self.lora_dropout}")
        if self.lora_target_modules != "all-linear":
            raise ValueError(
                f"lora_target_modules 必须为 'all-linear'，得到 {self.lora_target_modules!r}"
            )
        if self.lora_bias != "none":
            raise ValueError(f"lora_bias 必须为 'none'，得到 {self.lora_bias!r}")
        if self.num_train_epochs != 3:
            raise ValueError(f"num_train_epochs 必须为 3，得到 {self.num_train_epochs}")
        if not self.load_best_model_at_end:
            raise ValueError("load_best_model_at_end 必须为 True")
        if self.metric_for_best_model != "eval_loss":
            raise ValueError(
                f"metric_for_best_model 必须为 'eval_loss'，得到 {self.metric_for_best_model!r}"
            )
        if self.greater_is_better is not False:
            raise ValueError("greater_is_better 必须为 False")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size 必须 > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps 必须 > 0")
        if self.save_strategy != "epoch":
            raise ValueError(
                f"save_strategy 必须为 'epoch'，得到 {self.save_strategy!r}"
            )
        if self.eval_strategy != "epoch":
            raise ValueError(
                f"eval_strategy 必须为 'epoch'，得到 {self.eval_strategy!r}"
            )
        if self.inference_max_new_tokens <= 0:
            raise ValueError("inference_max_new_tokens 必须 > 0")
        if self.preflight_max_steps <= 0:
            raise ValueError("preflight_max_steps 必须 > 0")

    # ------------------------------------------------------------------
    # 派生 / 序列化
    # ------------------------------------------------------------------

    @property
    def effective_batch_size(self) -> int:
        """``per_device_train_batch_size * gradient_accumulation_steps``。"""
        return int(self.per_device_train_batch_size) * int(self.gradient_accumulation_steps)

    def to_dict(self) -> Dict[str, Any]:
        """转 dict 供 manifest 序列化；保持字段稳定顺序。"""
        return asdict(self)


@dataclass(frozen=True)
class DatasetFingerprint:
    """数据集指纹：用于 run manifest + resume 校验。"""

    # 整个 JSONL 文件的 SHA256
    file_sha256: str

    # 记录数
    record_count: int

    # sample_id 升序拼接的 SHA256（顺序确定性）
    sample_id_signature: str

    # 字段名列表（用于校验 schema 不变）
    schema: List[str] = field(default_factory=list)


def compute_dataset_fingerprint(
    jsonl_path: Path,
    *,
    expected_sample_id_field: str = "sample_id",
) -> DatasetFingerprint:
    """计算一个 JSONL 文件的指纹。

    包含：

    - file_sha256：整个文件内容的 SHA256。
    - record_count：行数（JSONL 有效记录）。
    - sample_id_signature：所有 ``sample_id`` 升序拼接的 SHA256。
    - schema：第一条记录的字段名列表（保证字段集合稳定）。

    Raises
    ------
    ValueError
        JSONL 缺失 / 解析失败 / 缺少 ``sample_id``。
    """
    if not jsonl_path.exists():
        raise ValueError(f"dataset 文件不存在：{jsonl_path}")
    try:
        records = load_jsonl_records(jsonl_path)
    except Exception as e:
        raise ValueError(f"无法解析 JSONL {jsonl_path}: {e}") from e

    file_sha = hashlib.sha256()
    with jsonl_path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            file_sha.update(chunk)

    sample_ids: List[str] = []
    schema: List[str] = []
    for r in records:
        sid = r.get(expected_sample_id_field)
        if sid is None:
            raise ValueError(
                f"record 缺少 {expected_sample_id_field!r} 字段：{jsonl_path}"
            )
        sample_ids.append(str(sid))
        if not schema:
            schema = sorted(r.keys())

    sample_ids_sorted = sorted(sample_ids)
    sig = hashlib.sha256()
    for sid in sample_ids_sorted:
        sig.update(sid.encode("utf-8"))
        sig.update(b"\n")

    return DatasetFingerprint(
        file_sha256=file_sha.hexdigest(),
        record_count=len(records),
        sample_id_signature=sig.hexdigest(),
        schema=schema,
    )


def compute_sample_id_signature(
    sample_ids: Sequence[str],
) -> str:
    """基于 ``sample_id`` 升序拼接的 SHA256。

    与 :func:`compute_dataset_fingerprint` 的 ``sample_id_signature``
    算法一致：输入不同 ⇒ signature 不同。
    """
    sig = hashlib.sha256()
    for sid in sorted(str(s) for s in sample_ids):
        sig.update(sid.encode("utf-8"))
        sig.update(b"\n")
    return sig.hexdigest()


def select_best_checkpoint(
    epoch_eval_losses: Sequence[float],
) -> Dict[str, Any]:
    """根据 eval_loss 选 best epoch / checkpoint。

    Parameters
    ----------
    epoch_eval_losses:
        每个 epoch 末的 eval_loss；长度 = num_train_epochs。

    Returns
    -------
    dict
        包含：

        - ``best_epoch``：1-indexed；eval_loss 最低的 epoch。
        - ``best_eval_loss``：最低 eval_loss 值。
        - ``best_is_last``：True iff best epoch == 最后一 epoch。
        - ``all_eval_losses``：每 epoch eval_loss 列表。
    """
    if not epoch_eval_losses:
        raise ValueError("epoch_eval_losses 不能为空")

    losses = [float(x) for x in epoch_eval_losses]
    best_idx = min(range(len(losses)), key=lambda i: losses[i])
    return {
        "best_epoch": int(best_idx + 1),
        "best_eval_loss": float(losses[best_idx]),
        "best_is_last": bool(best_idx == len(losses) - 1),
        "all_eval_losses": losses,
    }


def verify_resume_manifest(
    *,
    manifest: Mapping[str, Any],
    config: Phase5CBTrainConfig,
    train_fingerprint: DatasetFingerprint,
    validation_fingerprint: DatasetFingerprint,
    backend_name: str,
) -> List[str]:
    """校验当前 manifest 是否兼容 resume。

    Returns
    -------
    list of str
        不兼容原因列表（空 = 全部匹配）。
    """
    reasons: List[str] = []

    # 1) backend
    mb = manifest.get("backend")
    if mb != backend_name:
        reasons.append(f"backend mismatch: manifest={mb!r} current={backend_name!r}")

    # 2) revision
    rev = manifest.get("model", {}).get("resolved_revision") or manifest.get("model", {}).get(
        "load_revision"
    )
    if rev != config.revision:
        reasons.append(f"revision mismatch: manifest={rev!r} current={config.revision!r}")

    # 3) token-level 训练字段
    mf_quant = manifest.get("quantization", {})
    if mf_quant.get("quant_type") != config.quant_type:
        reasons.append(
            f"quant_type mismatch: manifest={mf_quant.get('quant_type')!r} "
            f"current={config.quant_type!r}"
        )
    if mf_quant.get("compute_dtype") != config.compute_dtype:
        reasons.append(
            f"compute_dtype mismatch: manifest={mf_quant.get('compute_dtype')!r} "
            f"current={config.compute_dtype!r}"
        )
    if bool(mf_quant.get("double_quant")) != bool(config.double_quant):
        reasons.append(
            f"double_quant mismatch: manifest={mf_quant.get('double_quant')!r} "
            f"current={config.double_quant!r}"
        )
    if int(mf_quant.get("max_length") or 0) != int(config.max_length):
        reasons.append(
            f"max_length mismatch: manifest={mf_quant.get('max_length')!r} "
            f"current={config.max_length!r}"
        )

    mf_lora = manifest.get("lora", {})
    if int(mf_lora.get("r") or 0) != int(config.lora_r):
        reasons.append(f"lora_r mismatch: manifest={mf_lora.get('r')!r} current={config.lora_r!r}")
    if int(mf_lora.get("alpha") or 0) != int(config.lora_alpha):
        reasons.append(
            f"lora_alpha mismatch: manifest={mf_lora.get('alpha')!r} current={config.lora_alpha!r}"
        )
    if float(mf_lora.get("dropout") or 0.0) != float(config.lora_dropout):
        reasons.append(
            f"lora_dropout mismatch: manifest={mf_lora.get('dropout')!r} "
            f"current={config.lora_dropout!r}"
        )

    # 4) training
    mf_train = manifest.get("training", {})
    if int(mf_train.get("num_train_epochs") or 0) != int(config.num_train_epochs):
        reasons.append(
            f"num_train_epochs mismatch: manifest={mf_train.get('num_train_epochs')!r} "
            f"current={config.num_train_epochs!r}"
        )
    if int(mf_train.get("per_device_train_batch_size") or 0) != int(
        config.per_device_train_batch_size
    ):
        reasons.append(
            "per_device_train_batch_size mismatch: "
            f"manifest={mf_train.get('per_device_train_batch_size')!r} "
            f"current={config.per_device_train_batch_size!r}"
        )
    if int(mf_train.get("gradient_accumulation_steps") or 0) != int(
        config.gradient_accumulation_steps
    ):
        reasons.append(
            "gradient_accumulation_steps mismatch: "
            f"manifest={mf_train.get('gradient_accumulation_steps')!r} "
            f"current={config.gradient_accumulation_steps!r}"
        )
    if bool(mf_train.get("packing")) != bool(config.packing):
        reasons.append(
            f"packing mismatch: manifest={mf_train.get('packing')!r} "
            f"current={config.packing!r}"
        )
    if bool(mf_train.get("assistant_only_loss")) != bool(config.assistant_only_loss):
        reasons.append(
            f"assistant_only_loss mismatch: "
            f"manifest={mf_train.get('assistant_only_loss')!r} "
            f"current={config.assistant_only_loss!r}"
        )

    # 5) dataset fingerprint
    mf_dataset = manifest.get("dataset", {})
    if mf_dataset.get("train_sha256") != train_fingerprint.file_sha256:
        reasons.append(
            f"train SHA256 mismatch: manifest={mf_dataset.get('train_sha256')!r} "
            f"current={train_fingerprint.file_sha256!r}"
        )
    if mf_dataset.get("validation_sha256") != validation_fingerprint.file_sha256:
        reasons.append(
            "validation SHA256 mismatch: "
            f"manifest={mf_dataset.get('validation_sha256')!r} "
            f"current={validation_fingerprint.file_sha256!r}"
        )
    if mf_dataset.get("train_sample_id_signature") != train_fingerprint.sample_id_signature:
        reasons.append("train sample_id signature mismatch")
    if (
        mf_dataset.get("validation_sample_id_signature")
        != validation_fingerprint.sample_id_signature
    ):
        reasons.append("validation sample_id signature mismatch")

    return reasons


# ---------------------------------------------------------------------------
# Run Manifest
# ---------------------------------------------------------------------------


def write_run_manifest(
    path: Path,
    *,
    payload: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> None:
    """原子写入 ``run_manifest.json``。"""
    log = logger or _LOG
    atomic_write_json(Path(path), dict(payload))
    log.info("Wrote run_manifest.json to %s", path)


def load_run_manifest(path: Path) -> Dict[str, Any]:
    """读取 ``run_manifest.json``；缺失或非法 JSON 抛 ``ValueError``。"""
    if not path.exists():
        raise ValueError(f"run_manifest.json 不存在：{path}")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"无法解析 run_manifest.json {path}: {e}") from e


# ---------------------------------------------------------------------------
# Backward / Smoke compatibility helpers
# ---------------------------------------------------------------------------


def qloRA_smoke_config_parity_check(smoke: "QLoRASmokeConfigLike") -> List[str]:  # noqa: F821
    """验证 ``QLoRASmokeConfig`` 与 Phase 5C-B 固定超参一致（保留 Phase 5B 语义）。

    不可在 Phase 5C-B 引入的差异：

    - max_length=2048 / NF4 / BF16 / double quant / LoRA r16 a32 d0.05；
    - assistant_only_loss=True / packing=False。

    Returns
    -------
    list of str
        差异原因列表（空 = 完全一致）。
    """
    reasons: List[str] = []
    if smoke.max_length != DEFAULT_MAX_LENGTH:
        reasons.append(f"smoke.max_length={smoke.max_length} != {DEFAULT_MAX_LENGTH}")
    if smoke.quant_type != DEFAULT_QUANT_TYPE:
        reasons.append(f"smoke.quant_type={smoke.quant_type!r} != 'nf4'")
    if smoke.compute_dtype != DEFAULT_COMPUTE_DTYPE:
        reasons.append(f"smoke.compute_dtype={smoke.compute_dtype!r} != 'bfloat16'")
    if not smoke.double_quant:
        reasons.append("smoke.double_quant != True")
    if smoke.lora_r != DEFAULT_LORA_R:
        reasons.append(f"smoke.lora_r={smoke.lora_r} != {DEFAULT_LORA_R}")
    if smoke.lora_alpha != DEFAULT_LORA_ALPHA:
        reasons.append(f"smoke.lora_alpha={smoke.lora_alpha} != {DEFAULT_LORA_ALPHA}")
    if smoke.lora_dropout != DEFAULT_LORA_DROPOUT:
        reasons.append(f"smoke.lora_dropout={smoke.lora_dropout} != {DEFAULT_LORA_DROPOUT}")
    if smoke.assistant_only_loss is not True:
        reasons.append("smoke.assistant_only_loss != True")
    if smoke.packing is not False:
        reasons.append("smoke.packing != False")
    if smoke.bf16 is not True:
        reasons.append("smoke.bf16 != True")
    if smoke.fp16 is not False:
        reasons.append("smoke.fp16 != False")
    return reasons


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_REVISION",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_NUM_TRAIN_EPOCHS",
    "DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE",
    "DEFAULT_GRADIENT_ACCUMULATION_STEPS",
    "Phase5CBTrainConfig",
    "DatasetFingerprint",
    "compute_dataset_fingerprint",
    "compute_sample_id_signature",
    "select_best_checkpoint",
    "verify_resume_manifest",
    "write_run_manifest",
    "load_run_manifest",
    "qloRA_smoke_config_parity_check",
]
