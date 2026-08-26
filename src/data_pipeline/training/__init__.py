"""Phase 5B — QLoRA 单步训练冒烟测试相关代码。

当前阶段仅实现：

- ``QLoRASmokeConfig``：训练/采样/超参静态配置。
- ``evaluate_phase5a_gate``：基于已写出的 ``token_report.json`` 复算硬门槛。
- ``select_smoke_samples``：从 ``train.jsonl`` 中确定性选 8 条 smoke 样本。
- ``build_inference_prompt``：最小推理 prompt 构造。

不引入：trainer factory / model factory / quantization manager / lora manager
等抽象。Phase 5C 之前不扩展。
"""

from .models import QLoRASmokeConfig
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
]