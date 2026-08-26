"""Phase 5B — QLoRA 训练相关的小工具：

- ``evaluate_phase5a_gate``：基于 ``token_report.json`` 复算硬门槛。
- ``select_smoke_samples``：从 ``train.jsonl`` 中确定性选 smoke 样本。
- ``build_inference_prompt``：最小推理 prompt 构造。

本模块不引入 trainer / model / quantization 抽象；仅做配置与样本挑选。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ..utils import get_logger, load_jsonl_records
from .models import QLoRASmokeConfig


# Phase 5A 训练门槛 token 上限；任何样本超长 ⇒ 立即阻断，禁止改 max_length 跳过。
PHASE5A_MAX_TOKEN_LIMIT = 2048


# 顶层异常：Phase 5B 任意硬性条件不满足时抛出。
class TrainingGateError(RuntimeError):
    """Phase 5B 任意硬性条件不满足时抛出。"""


@dataclass(frozen=True)
class Phase5AGateResult:
    """``evaluate_phase5a_gate`` / ``evaluate_phase5a_gate_with_audit`` 的返回结果。"""

    # 总体是否通过
    passed: bool

    # 失败原因列表（人类可读）；全部为空表示通过
    reasons: List[str] = field(default_factory=list)

    # 最终认定的实际最大 token 数（默认来自 report，audit 一致时也来自 audit）
    actual_max_tokens: int = 0

    # 报告本身记录的 max（length_stats.total_tokens.max）
    report_max_tokens: int = 0

    # token_audit.jsonl 中观察到的最大 total_tokens
    audit_max_tokens: int = 0

    # 触发 100% 覆盖所需的 max_length（与 token_report.coverage_thresholds 一致）
    selected_max_length: int = 0

    # 超过 2048 的样本数；必须为 0
    samples_over_max_length: int = 0

    # token_audit 中越限样本中最大的 total_tokens（reason 附带）
    max_audit_over_total_tokens: int = 0


@dataclass(frozen=True)
class SmokeSampleSelection:
    """``select_smoke_samples`` 的返回结果。"""

    # 选中的 sample_id 列表（去重、确定性顺序）
    sample_ids: List[str]

    # 短样本数量（按 sample_id 升序）
    short_count: int

    # 1 条 P95 附近样本的 sample_id（可能为 None，若 train 样本不足）
    p95_sample_id: str | None

    # P95 样本的真实 total_tokens
    p95_total_tokens: int = 0


def evaluate_phase5a_gate(token_report: Mapping[str, Any]) -> Phase5AGateResult:
    """复算 Phase 5A gate；任何不满足返回 ``Phase5AGateResult(passed=False, reasons=...)``。

    硬门槛：

    1. ``dataset_membership.set_equal_all_vs_union == true``
    2. ``missing_all_sample_ids`` 与 ``missing_split_sample_ids`` 必须为空
    3. ``assistant_only_loss_feasible == true``
    4. ``prompt_prefix_mismatch_count == 0``
    5. ``zero_assistant_target_count == 0``
    6. ``roundtrip_check.ok == true``
    7. ``tokenizer.chat_template_present == true``
    8. ``source_thinking_markers.think_open / think_close / reasoning_content`` 全为 0
    9. ``length_stats.total_tokens.max <= 2048``
    10. ``model.model_name == "Qwen/Qwen3-8B"``

    本函数仅消费 ``token_report``；audit 行越限/不一致检查在
    ``evaluate_phase5a_gate_with_audit`` 中处理。
    """
    reasons: List[str] = []

    membership = token_report.get("dataset_membership") or {}
    if membership.get("set_equal_all_vs_union") is not True:
        reasons.append("dataset_membership.set_equal_all_vs_union != true")
    if membership.get("missing_all_sample_ids"):
        reasons.append(f"missing_all_sample_ids 非空（{len(membership['missing_all_sample_ids'])} 条）")
    if membership.get("missing_split_sample_ids"):
        reasons.append(
            f"missing_split_sample_ids 非空（{len(membership['missing_split_sample_ids'])} 条）"
        )

    if token_report.get("assistant_only_loss_feasible") is not True:
        reasons.append("assistant_only_loss_feasible != true")
    if (token_report.get("prompt_prefix_mismatch_count") or 0) != 0:
        reasons.append(f"prompt_prefix_mismatch_count={token_report.get('prompt_prefix_mismatch_count')}")
    if (token_report.get("zero_assistant_target_count") or 0) != 0:
        reasons.append(f"zero_assistant_target_count={token_report.get('zero_assistant_target_count')}")

    roundtrip = token_report.get("roundtrip_check") or {}
    if roundtrip.get("ok") is not True:
        reasons.append("roundtrip_check.ok != true")

    tokenizer_info = token_report.get("tokenizer") or {}
    if tokenizer_info.get("chat_template_present") is not True:
        reasons.append("tokenizer.chat_template_present != true")

    thinking = token_report.get("source_thinking_markers") or {}
    for k in ("think_open", "think_close", "reasoning_content"):
        if (thinking.get(k) or 0) != 0:
            reasons.append(f"source_thinking_markers.{k}={thinking.get(k)}")

    length_stats = token_report.get("length_stats") or {}
    total_tokens = length_stats.get("total_tokens") or {}
    report_max = int(total_tokens.get("max") or 0)
    if report_max > PHASE5A_MAX_TOKEN_LIMIT:
        reasons.append(
            f"length_stats.total_tokens.max={report_max} > {PHASE5A_MAX_TOKEN_LIMIT}（数据与 SFT 集合已失同步）"
        )

    model = token_report.get("model") or {}
    if model.get("model_name") != QLoRASmokeConfig.model_name:
        reasons.append(
            f"model.model_name={model.get('model_name')!r} != 'Qwen/Qwen3-8B'"
        )

    coverage_thresholds = token_report.get("coverage_thresholds") or {}
    selected_max_length = int(coverage_thresholds.get("smallest_candidate_covering_100pct") or 0)

    return Phase5AGateResult(
        passed=not reasons,
        reasons=reasons,
        actual_max_tokens=report_max,
        report_max_tokens=report_max,
        audit_max_tokens=0,
        selected_max_length=selected_max_length,
        samples_over_max_length=0,
        max_audit_over_total_tokens=0,
    )


def evaluate_phase5a_gate_with_audit(
    token_report: Mapping[str, Any],
    token_audit_rows: Sequence[Mapping[str, Any]],
) -> Phase5AGateResult:
    """``evaluate_phase5a_gate`` 的扩展版：额外做 audit 越限 / max 一致性硬校验。

    新增硬门槛：

    11. ``token_audit.jsonl`` 中 ``total_tokens > 2048`` 的样本数必须为 0；
        否则 ``TOKEN_AUDIT_OVER_MAX_LENGTH`` 计入 reasons 且 ``passed=False``。
    12. ``report_max_tokens == audit_max_tokens``；否则 ``TOKEN_REPORT_AUDIT_MAX_MISMATCH``
        计入 reasons 且 ``passed=False``。
    13. 最终 ``actual_max_tokens`` 取 ``audit_max_tokens``（与 token_audit 严格一致）。
    """
    base = evaluate_phase5a_gate(token_report)
    reasons: List[str] = list(base.reasons)

    audit_total_tokens = [int(r.get("total_tokens") or 0) for r in token_audit_rows]
    audit_max = max(audit_total_tokens) if audit_total_tokens else 0
    over_rows = [
        (int(r.get("total_tokens") or 0), str(r.get("sample_id") or ""))
        for r in token_audit_rows
        if int(r.get("total_tokens") or 0) > PHASE5A_MAX_TOKEN_LIMIT
    ]
    samples_over = len(over_rows)
    max_over = max((t for t, _ in over_rows), default=0)

    if samples_over > 0:
        reasons.append(
            f"TOKEN_AUDIT_OVER_MAX_LENGTH: samples_over_max_length={samples_over}, "
            f"max_audit_total_tokens={max_over}"
        )

    if base.report_max_tokens != audit_max:
        reasons.append(
            "TOKEN_REPORT_AUDIT_MAX_MISMATCH: "
            f"report_max_tokens={base.report_max_tokens} != audit_max_tokens={audit_max}"
        )

    return Phase5AGateResult(
        passed=not reasons,
        reasons=reasons,
        actual_max_tokens=audit_max if audit_max > 0 else base.report_max_tokens,
        report_max_tokens=base.report_max_tokens,
        audit_max_tokens=audit_max,
        selected_max_length=base.selected_max_length,
        samples_over_max_length=samples_over,
        max_audit_over_total_tokens=max_over,
    )


def select_smoke_samples(
    train_records: Sequence[Mapping[str, Any]],
    token_audit_rows: Sequence[Mapping[str, Any]],
    config: QLoRASmokeConfig,
    *,
    target_train_p95: float | None = None,
) -> SmokeSampleSelection:
    """从 train 样本中确定性挑选 smoke 子集。

    规则：

    1. 按 ``sample_id`` 升序取前 ``smoke_short_count`` 条；样本必须真实存在于
       ``train_records``；
    2. P95 候选必须同时满足：
       - ``token_audit_rows`` 中 ``split == "train"``；
       - ``sample_id`` 真实存在于 ``train_records``（不允许 audit 持有但
         train.jsonl 缺失的样本）；
    3. P95 tie-break 确定性：``delta`` 升序 → ``total_tokens`` 升序 →
       ``sample_id`` 升序；
    4. 最终总数必须严格等于 ``smoke_target_count``，否则由调用方在
       ``select_smoke_samples_strict`` 中抛 ``SMOKE_SELECTION_INCOMPLETE``。
    """
    if config.smoke_short_count < 0:
        raise ValueError(f"smoke_short_count 必须 >= 0，得到 {config.smoke_short_count}")
    if config.smoke_target_count <= 0:
        raise ValueError(f"smoke_target_count 必须 > 0，得到 {config.smoke_target_count}")

    # real train sample_id 集合
    actual_train_ids: set[str] = {str(r["sample_id"]) for r in train_records if r.get("sample_id")}
    audit_train_rows = [
        r for r in token_audit_rows
        if r.get("split") == "train" and r.get("sample_id") in actual_train_ids
    ]

    # 1) 按 sample_id 升序的 train 样本
    sorted_train = sorted(
        [r for r in train_records if r.get("sample_id")],
        key=lambda x: x["sample_id"],
    )
    short_ids: List[str] = []
    for r in sorted_train:
        sid = r["sample_id"]
        if sid in short_ids:
            continue
        short_ids.append(sid)
        if len(short_ids) >= config.smoke_short_count:
            break

    # 2) P95 目标：优先使用调用方传入；否则用 train 自身 P95（按 sample_id 升序）
    p95_target = target_train_p95
    if p95_target is None:
        train_totals = sorted(int(r.get("total_tokens") or 0) for r in audit_train_rows)
        if not train_totals:
            p95_target = 0.0
        else:
            n = len(train_totals)
            idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
            p95_target = float(train_totals[idx])

    # 3) P95 候选：仅在 audit_train_rows（已交集）中寻找
    p95_id: str | None = None
    p95_tokens: int = 0
    best_delta = math.inf
    best_total = math.inf
    best_sid: str | None = None
    if p95_target and p95_target > 0:
        candidates = sorted(
            (
                (str(r.get("sample_id")), int(r.get("total_tokens") or 0))
                for r in audit_train_rows
                if int(r.get("total_tokens") or 0) > 0
                and str(r.get("sample_id")) not in short_ids
            ),
            key=lambda x: (abs(x[1] - p95_target), x[1], x[0]),
        )
        if candidates:
            best_sid, best_total = candidates[0]
            p95_id = best_sid
            p95_tokens = best_total

    # 4) 合并去重（确定性顺序）
    merged: List[str] = list(short_ids)
    if p95_id and p95_id not in merged:
        merged.append(p95_id)

    return SmokeSampleSelection(
        sample_ids=merged,
        short_count=len(short_ids),
        p95_sample_id=p95_id,
        p95_total_tokens=p95_tokens,
    )


def select_smoke_samples_strict(
    train_records: Sequence[Mapping[str, Any]],
    token_audit_rows: Sequence[Mapping[str, Any]],
    config: QLoRASmokeConfig,
    *,
    target_train_p95: float | None = None,
) -> SmokeSampleSelection:
    """``select_smoke_samples`` 的硬约束版：总数必须严格等于 ``smoke_target_count``。

    任何不满足：
    - 任何选中的 sample_id 不在 train.jsonl ⇒ ``SMOKE_SAMPLE_NOT_IN_TRAIN``；
    - 数量不足 / 超出 ``smoke_target_count`` ⇒ ``SMOKE_SELECTION_INCOMPLETE``。
    """
    selection = select_smoke_samples(
        train_records,
        token_audit_rows,
        config,
        target_train_p95=target_train_p95,
    )

    actual_train_ids: set[str] = {str(r["sample_id"]) for r in train_records if r.get("sample_id")}
    missing_in_train = [sid for sid in selection.sample_ids if sid not in actual_train_ids]
    if missing_in_train:
        raise TrainingGateError(
            f"SMOKE_SAMPLE_NOT_IN_TRAIN: {len(missing_in_train)} 个 sample_id 不在 train.jsonl，"
            f"前 5 个={missing_in_train[:5]}"
        )

    if len(selection.sample_ids) != config.smoke_target_count:
        raise TrainingGateError(
            "SMOKE_SELECTION_INCOMPLETE: "
            f"len(sample_ids)={len(selection.sample_ids)} != smoke_target_count={config.smoke_target_count}"
        )

    return selection


def build_inference_prompt(prompt: str) -> List[Dict[str, str]]:
    """构造最小推理用的 messages 列表。"""
    if not prompt or not prompt.strip():
        raise ValueError("inference prompt 不能为空")
    return [{"role": "user", "content": prompt}]


def load_token_report(report_path: Path) -> Dict[str, Any]:
    """读取 ``token_report.json``；缺失或非法 JSON 抛 ``TrainingGateError``。"""
    if not report_path.exists():
        raise TrainingGateError(f"token_report.json 不存在：{report_path}")
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise TrainingGateError(f"无法解析 token_report.json: {e}") from e


def load_token_audit_rows(audit_path: Path) -> List[Dict[str, Any]]:
    """读取 ``token_audit.jsonl``；缺失或解析失败抛 ``TrainingGateError``。"""
    if not audit_path.exists():
        raise TrainingGateError(f"token_audit.jsonl 不存在：{audit_path}")
    try:
        return load_jsonl_records(audit_path)
    except Exception as e:
        raise TrainingGateError(f"无法解析 token_audit.jsonl: {e}") from e


def write_smoke_config(
    config: QLoRASmokeConfig,
    output_path: Path,
    *,
    logger: logging.Logger | None = None,
    model_revision: Mapping[str, Any] | None = None,
    selection: SmokeSampleSelection | None = None,
) -> None:
    """把 ``QLoRASmokeConfig`` 序列化为 ``smoke_config.json``。

    输出不含 timestamp / VRAM / loss 等运行期数据；同 config + 同 model_revision +
    同 selection ⇒ byte-for-byte 一致。

    可选参数：

    - ``model_revision``：写入 ``model.requested_revision`` / ``model.resolved_revision`` /
      ``model.load_revision`` / ``model.revision_pinned``；
    - ``selection``：写入 ``smoke.selected_sample_ids`` / ``smoke.p95_sample_id`` /
      ``smoke.p95_total_tokens``。
    """
    log = logger or get_logger("data_pipeline.training")
    model_block: Dict[str, Any] = {
        "model_name": config.model_name,
        "requested_revision": None,
        "resolved_revision": None,
        "load_revision": None,
        "revision_pinned": False,
    }
    if model_revision:
        model_block["requested_revision"] = model_revision.get("requested_revision")
        model_block["resolved_revision"] = model_revision.get("resolved_revision")
        model_block["load_revision"] = model_revision.get("load_revision")
        model_block["revision_pinned"] = bool(model_revision.get("revision_pinned"))

    smoke_block: Dict[str, Any] = {
        "short_count": config.smoke_short_count,
        "target_count": config.smoke_target_count,
        "inference_prompt": config.inference_prompt,
        "inference_max_new_tokens": config.inference_max_new_tokens,
        "selected_sample_ids": [],
        "p95_sample_id": None,
        "p95_total_tokens": 0,
    }
    if selection is not None:
        smoke_block["selected_sample_ids"] = list(selection.sample_ids)
        smoke_block["p95_sample_id"] = selection.p95_sample_id
        smoke_block["p95_total_tokens"] = int(selection.p95_total_tokens)

    payload: Dict[str, Any] = {
        "model": model_block,
        "revision": config.revision,
        "max_length": config.max_length,
        "quantization": {
            "quant_type": config.quant_type,
            "compute_dtype": config.compute_dtype,
            "double_quant": config.double_quant,
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
        },
        "smoke": smoke_block,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from ..utils import atomic_write_json
    atomic_write_json(output_path, payload)
    log.info("Wrote smoke_config.json to %s", output_path)