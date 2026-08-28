"""Phase 5B/5C — 训练相关模块。

Phase 5B：``QLoRASmokeConfig`` + ``evaluate_phase5a_gate`` + 选样 + 最小推理。
Phase 5C-A：backend 抽象（``HfBackend`` / ``UnslothBackend``）。
Phase 5C-B：正式 3 epoch QLoRA SFT 配置 + 数据指纹 + 训练辅助。
"""

from .models import QLoRASmokeConfig
from .phase5c_b import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_REVISION,
    AdapterTensorSignature,
    DatasetFingerprint,
    Phase5CBTrainConfig,
    _bytes_to_gib,
    build_memory_field_legend,
    build_memory_summary,
    collect_memory_snapshot,
    collect_peak_memory,
    compute_adapter_signature,
    compute_dataset_fingerprint,
    compute_sample_id_signature,
    load_run_manifest,
    qloRA_smoke_config_parity_check,
    select_best_checkpoint,
    verify_best_adapter_matches_checkpoint,
    verify_resume_manifest,
    write_run_manifest,
)
from .pipeline import (
    Phase5AGateResult,
    SmokeSampleSelection,
    TrainingGateError,
    build_inference_prompt,
    evaluate_phase5a_gate,
    evaluate_phase5a_gate_with_audit,
    load_token_audit_rows,
    load_token_report,
    select_smoke_samples,
    select_smoke_samples_strict,
    write_smoke_config,
)

__all__ = [
    # Phase 5B
    "QLoRASmokeConfig",
    "Phase5AGateResult",
    "SmokeSampleSelection",
    "TrainingGateError",
    "build_inference_prompt",
    "evaluate_phase5a_gate",
    "evaluate_phase5a_gate_with_audit",
    "load_token_audit_rows",
    "load_token_report",
    "select_smoke_samples",
    "select_smoke_samples_strict",
    "write_smoke_config",
    # Phase 5C-B
    "Phase5CBTrainConfig",
    "DatasetFingerprint",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_REVISION",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_NUM_TRAIN_EPOCHS",
    "DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE",
    "DEFAULT_GRADIENT_ACCUMULATION_STEPS",
    "compute_dataset_fingerprint",
    "compute_sample_id_signature",
    "select_best_checkpoint",
    "verify_resume_manifest",
    "write_run_manifest",
    "load_run_manifest",
    "qloRA_smoke_config_parity_check",
    # Phase 5C-B 收口
    "AdapterTensorSignature",
    "_bytes_to_gib",
    "collect_memory_snapshot",
    "collect_peak_memory",
    "build_memory_summary",
    "build_memory_field_legend",
    "compute_adapter_signature",
    "verify_best_adapter_matches_checkpoint",
]
