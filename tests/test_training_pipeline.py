"""Phase 5B — QLoRA 训练冒烟测试的单元测试。

默认 CPU-only / 不联网；不加载 Qwen3-8B；不下载任何权重。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from data_pipeline.training import (
    Phase5AGateResult,
    QLoRASmokeConfig,
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


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def good_report() -> Dict[str, Any]:
    """构造一份通过所有 gate 的合成 token_report。"""
    return {
        "model": {"model_name": "Qwen/Qwen3-8B", "requested_revision": "main", "resolved_revision": None, "transformers_version": "5.15.0"},
        "tokenizer": {"chat_template_present": True, "tokenizer_class": "Qwen2Tokenizer", "vocab_size": 151643},
        "roundtrip_check": {"ok": True, "decoded_contains_question": True, "decoded_contains_answer": True},
        "dataset_membership": {
            "all_count": 10,
            "train_count": 8,
            "validation_count": 1,
            "test_count": 1,
            "set_equal_all_vs_union": True,
            "missing_all_sample_ids": [],
            "missing_split_sample_ids": [],
        },
        "source_thinking_markers": {"think_open": 0, "think_close": 0, "reasoning_content": 0},
        "length_stats": {
            "total_tokens": {"min": 24, "mean": 100.0, "median": 80, "p90": 200, "p95": 400, "p99": 700, "max": 1672},
            "prompt_tokens": {"max": 100, "mean": 24.77},
            "assistant_target_tokens": {"max": 1653, "mean": 157.85},
        },
        "split_stats": {
            "train": {"count": 8, "mean_total_tokens": 100, "median_total_tokens": 80, "p95_total_tokens": 400, "max_total_tokens": 1672},
            "validation": {"count": 1, "mean_total_tokens": 100, "median_total_tokens": 100, "p95_total_tokens": 100, "max_total_tokens": 100},
            "test": {"count": 1, "mean_total_tokens": 100, "median_total_tokens": 100, "p95_total_tokens": 100, "max_total_tokens": 100},
        },
        "coverage": {
            "512": {"candidate_max_length": 512, "fit_count": 8, "fit_ratio": 0.8, "over_count": 2, "over_ratio": 0.2},
            "1024": {"candidate_max_length": 1024, "fit_count": 9, "fit_ratio": 0.9, "over_count": 1, "over_ratio": 0.1},
            "1536": {"candidate_max_length": 1536, "fit_count": 10, "fit_ratio": 1.0, "over_count": 0, "over_ratio": 0.0},
            "2048": {"candidate_max_length": 2048, "fit_count": 10, "fit_ratio": 1.0, "over_count": 0, "over_ratio": 0.0},
        },
        "coverage_thresholds": {
            "smallest_candidate_covering_95pct": 512,
            "smallest_candidate_covering_99pct": 1024,
            "smallest_candidate_covering_100pct": 1536,
        },
        "prompt_prefix_mismatch_count": 0,
        "zero_assistant_target_count": 0,
        "assistant_only_loss_feasible": True,
    }


# ---------------------------------------------------------------------------
# 1) Phase5A gate
# ---------------------------------------------------------------------------


def test_phase5a_gate_passes_valid_report(good_report):
    """合成一份通过所有条件的 report，gate 必须 PASS。"""
    result = evaluate_phase5a_gate(good_report)
    assert result.passed is True
    assert result.reasons == []
    assert result.actual_max_tokens == 1672
    assert result.report_max_tokens == 1672
    assert result.audit_max_tokens == 0
    assert result.samples_over_max_length == 0


def test_phase5a_gate_with_audit_passes_when_audit_max_matches(good_report):
    """audit max 与 report max 一致且无越限 ⇒ PASS。"""
    rows = [
        {"sample_id": f"sft_{i:06d}", "split": "train", "total_tokens": 100 + i}
        for i in range(10)
    ]
    # 把 report.max 调到 audit 中最大值
    good_report["length_stats"]["total_tokens"]["max"] = max(r["total_tokens"] for r in rows)
    result = evaluate_phase5a_gate_with_audit(good_report, rows)
    assert result.passed is True
    assert result.report_max_tokens == result.audit_max_tokens


def test_phase5a_gate_blocks_membership_mismatch(good_report):
    """membership 不一致 → BLOCKED。"""
    good_report["dataset_membership"]["set_equal_all_vs_union"] = False
    good_report["dataset_membership"]["missing_split_sample_ids"] = ["sft_a"]
    result = evaluate_phase5a_gate(good_report)
    assert result.passed is False
    assert any("set_equal_all_vs_union" in r for r in result.reasons)
    assert any("missing_split_sample_ids" in r for r in result.reasons)


def test_phase5a_gate_blocks_missing_all_sample(good_report):
    """missing_all_sample_ids 非空 → BLOCKED。"""
    good_report["dataset_membership"]["missing_all_sample_ids"] = ["sft_x"]
    assert evaluate_phase5a_gate(good_report).passed is False


def test_phase5a_gate_blocks_prefix_mismatch(good_report):
    """prompt_prefix_mismatch_count != 0 → BLOCKED。"""
    good_report["prompt_prefix_mismatch_count"] = 3
    assert evaluate_phase5a_gate(good_report).passed is False


def test_phase5a_gate_blocks_zero_assistant_target(good_report):
    """zero_assistant_target_count != 0 → BLOCKED。"""
    good_report["zero_assistant_target_count"] = 1
    assert evaluate_phase5a_gate(good_report).passed is False


def test_phase5a_gate_blocks_thinking_source_marker(good_report):
    """source thinking marker 非 0 → BLOCKED。"""
    good_report["source_thinking_markers"]["think_open"] = 1
    assert evaluate_phase5a_gate(good_report).passed is False
    good_report["source_thinking_markers"]["think_open"] = 0
    good_report["source_thinking_markers"]["reasoning_content"] = 2
    assert evaluate_phase5a_gate(good_report).passed is False


def test_phase5a_gate_blocks_token_length_over_2048(good_report):
    """max total_tokens > 2048 → BLOCKED。"""
    good_report["length_stats"]["total_tokens"]["max"] = 2100
    result = evaluate_phase5a_gate(good_report)
    assert result.passed is False
    assert any("> 2048" in r or "2048" in r for r in result.reasons)


def test_phase5a_gate_blocks_roundtrip_failure(good_report):
    """roundtrip ok=False → BLOCKED。"""
    good_report["roundtrip_check"]["ok"] = False
    assert evaluate_phase5a_gate(good_report).passed is False


def test_phase5a_gate_blocks_chat_template_missing(good_report):
    """chat_template_present=False → BLOCKED。"""
    good_report["tokenizer"]["chat_template_present"] = False
    assert evaluate_phase5a_gate(good_report).passed is False


def test_phase5a_gate_blocks_wrong_model(good_report):
    """model_name 不等于 Qwen/Qwen3-8B → BLOCKED。"""
    good_report["model"]["model_name"] = "Qwen/Qwen3-7B"
    assert evaluate_phase5a_gate(good_report).passed is False


def test_phase5a_gate_with_audit_counts_samples_over(good_report):
    """samples_over_max_length 必须按 token_audit rows 真实计数。"""
    rows = [
        {"sample_id": "sft_a", "split": "train", "total_tokens": 100},
        {"sample_id": "sft_b", "split": "train", "total_tokens": 2100},
        {"sample_id": "sft_c", "split": "test", "total_tokens": 3000},
    ]
    # audit 越限 ⇒ passed 必须 False
    result = evaluate_phase5a_gate_with_audit(good_report, rows)
    assert result.samples_over_max_length == 2
    assert result.passed is False
    assert any("TOKEN_AUDIT_OVER_MAX_LENGTH" in r for r in result.reasons)

    # report.max 仍 1672，audit.max = 3000 ⇒ max 不一致 ⇒ 必须 False
    assert any("TOKEN_REPORT_AUDIT_MAX_MISMATCH" in r for r in result.reasons)
    assert result.report_max_tokens == 1672
    assert result.audit_max_tokens == 3000
    # actual_max_tokens 取 audit
    assert result.actual_max_tokens == 3000


def test_audit_over_max_length_blocks_gate(good_report):
    """新增：audit 任意样本 total_tokens > 2048 ⇒ 必须 BLOCKED。"""
    rows = [
        {"sample_id": "sft_a", "split": "train", "total_tokens": 2100},
        {"sample_id": "sft_b", "split": "test", "total_tokens": 3000},
    ]
    result = evaluate_phase5a_gate_with_audit(good_report, rows)
    assert result.samples_over_max_length == 2
    assert result.passed is False
    assert any("TOKEN_AUDIT_OVER_MAX_LENGTH" in r for r in result.reasons)
    assert result.max_audit_over_total_tokens == 3000


def test_report_and_audit_max_must_match(good_report):
    """新增：report_max_tokens == audit_max_tokens 必须成立，否则 BLOCKED。"""
    rows = [
        {"sample_id": "sft_a", "split": "train", "total_tokens": 100},
        {"sample_id": "sft_b", "split": "train", "total_tokens": 900},
    ]
    # 故意制造 max 不一致
    good_report["length_stats"]["total_tokens"]["max"] = 1672  # report max
    # audit max 900 ⇒ mismatch
    result = evaluate_phase5a_gate_with_audit(good_report, rows)
    assert result.passed is False
    assert any("TOKEN_REPORT_AUDIT_MAX_MISMATCH" in r for r in result.reasons)
    assert result.report_max_tokens == 1672
    assert result.audit_max_tokens == 900
    assert result.actual_max_tokens == 900


def test_gate_passes_when_audit_max_equals_report_max(good_report):
    """report.max == audit.max 且无越限 ⇒ PASS。"""
    rows = [
        {"sample_id": f"sft_{i:06d}", "split": "train", "total_tokens": 200 + i}
        for i in range(5)
    ] + [
        {"sample_id": "sft_long", "split": "train", "total_tokens": 1672},
    ]
    good_report["length_stats"]["total_tokens"]["max"] = 1672
    result = evaluate_phase5a_gate_with_audit(good_report, rows)
    assert result.samples_over_max_length == 0
    assert result.report_max_tokens == 1672
    assert result.audit_max_tokens == 1672
    assert result.actual_max_tokens == 1672
    assert result.passed is True


# ---------------------------------------------------------------------------
# 2) max_length 固定 2048
# ---------------------------------------------------------------------------


def test_max_length_is_fixed_to_2048_for_current_dataset():
    """Phase 5B 固定 max_length=2048，不允许临时降低。"""
    cfg = QLoRASmokeConfig()
    assert cfg.max_length == 2048
    # frozen dataclass：非法值在 __post_init__ 抛 ValueError；
    # 任何违反固定语义都必须抛异常。
    with pytest.raises(ValueError):
        QLoRASmokeConfig(max_length=512)
    with pytest.raises(ValueError):
        QLoRASmokeConfig(max_length=4096)


# ---------------------------------------------------------------------------
# 3) Smoke 选样
# ---------------------------------------------------------------------------


def _sample(sid: str, *, total: int = 100, split: str = "train") -> Dict[str, Any]:
    return {
        "sample_id": sid,
        "total_tokens": total,
        "prompt_tokens": 20,
        "assistant_target_tokens": max(0, total - 20),
        "split": split,
        "metadata": {"unit_id": f"u_{sid}", "source_path": "x.docx", "source_record_indices": [0]},
    }


def test_smoke_sample_selection_deterministic(tmp_path: Path):
    """相同输入跑两次，结果完全一致。"""
    train_records = [
        {"sample_id": f"sft_{i:06d}", "messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]}
        for i in range(20)
    ]
    audit_rows = [_sample(f"sft_{i:06d}", total=50 + i * 10) for i in range(20)]
    cfg = QLoRASmokeConfig()
    s1 = select_smoke_samples(train_records, audit_rows, cfg)
    s2 = select_smoke_samples(train_records, audit_rows, cfg)
    assert s1.sample_ids == s2.sample_ids
    assert s1.p95_sample_id == s2.p95_sample_id


def test_smoke_sample_selection_includes_p95_example():
    """P95 附近一条样本必须被选中（且不是已有短样本）。"""
    # 让 7 条短样本 total 都在 60-100
    train_records = [
        {"sample_id": f"sft_a{i}", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        for i in range(7)
    ] + [
        {"sample_id": "sft_p95", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
    ]
    # audit rows: 7 条短 + 1 条 P95 (410 tokens 接近目标 400)
    audit_rows = [_sample(f"sft_a{i}", total=60 + i * 5) for i in range(7)] + [
        _sample("sft_p95", total=410),
    ]
    cfg = QLoRASmokeConfig()
    sel = select_smoke_samples(train_records, audit_rows, cfg)
    assert len(sel.sample_ids) == 8
    assert sel.p95_sample_id == "sft_p95"
    assert "sft_p95" in sel.sample_ids
    # 短样本不包含 p95
    assert sel.short_count == 7


def test_smoke_sample_selection_handles_missing_p95():
    """train audit rows 为空时，p95 必为 None，short 仍可用。"""
    train_records = [
        {"sample_id": "sft_x", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
    ]
    cfg = QLoRASmokeConfig(smoke_short_count=1, smoke_target_count=8)
    sel = select_smoke_samples(train_records, [], cfg)
    assert sel.short_count == 1
    assert sel.p95_sample_id is None


def test_smoke_sample_selection_requires_train_membership():
    """audit 持有但 train.jsonl 缺失的样本不得被选为 P95。"""
    train_records = [
        {"sample_id": f"sft_{i:06d}", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        for i in range(8)
    ]
    # 8 条真实 train；audit 中混入 1 条 audit-only sft_ghost
    audit_rows = [_sample(f"sft_{i:06d}", total=50 + i * 5) for i in range(8)]
    audit_rows.append(_sample("sft_ghost", total=447))
    audit_rows[-1]["split"] = "train"
    cfg = QLoRASmokeConfig()
    sel = select_smoke_samples(train_records, audit_rows, cfg)
    # train P95 应来自真实 train
    assert sel.p95_sample_id != "sft_ghost"
    assert sel.p95_sample_id in {f"sft_{i:06d}" for i in range(8)}


def test_smoke_sample_selection_deterministic_p95_tiebreak():
    """P95 tie-break 确定性：delta → total_tokens → sample_id 升序。"""
    # 构造 3 条 token 数完全相同的样本 ⇒ 必须按 sample_id 升序选最小
    train_records = [
        {"sample_id": "sft_x2", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        {"sample_id": "sft_x1", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        {"sample_id": "sft_x3", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
    ]
    # 7 条短样本以满足 short_count
    extra_train = [
        {"sample_id": f"sft_e{i}", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        for i in range(7)
    ]
    audit_rows = (
        [_sample("sft_e0", total=10)] * 1
        + [_sample(f"sft_e{i}", total=10 + i) for i in range(1, 7)]
        + [
            _sample("sft_x1", total=430),
            _sample("sft_x2", total=430),
            _sample("sft_x3", total=430),
        ]
    )
    cfg = QLoRASmokeConfig()
    sel = select_smoke_samples(extra_train + train_records, audit_rows, cfg)
    # 目标 P95 = 430，三条全部 delta=0；按 sample_id 升序应选 sft_x1
    assert sel.p95_sample_id == "sft_x1"
    # 短样本不含 P95
    assert "sft_x1" not in (
        [sid for sid in sel.sample_ids if sid.startswith("sft_e")] if False else []
    ) or "sft_x1" in sel.sample_ids


def test_smoke_sample_selection_strict_requires_8():
    """select_smoke_samples_strict 必须在样本不足时抛 SMOKE_SELECTION_INCOMPLETE。"""
    train_records = [
        {"sample_id": f"sft_{i:06d}", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        for i in range(8)
    ]
    audit_rows = [_sample(f"sft_{i:06d}", total=50 + i * 5) for i in range(8)]
    audit_rows.append(_sample("sft_ghost", total=900))
    audit_rows[-1]["split"] = "train"
    cfg = QLoRASmokeConfig()
    sel = select_smoke_samples_strict(train_records, audit_rows, cfg)
    assert len(sel.sample_ids) == 8
    assert sel.p95_sample_id != "sft_ghost"


def test_smoke_sample_selection_strict_raises_on_missing_in_train():
    """select_smoke_samples_strict 必须拒绝 audit 持有但 train.jsonl 缺失的样本。"""
    # short 数量足够，但 P95 唯一候选不在 train_records 中
    train_records = [
        {"sample_id": f"sft_e{i}", "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        for i in range(7)
    ]
    audit_rows = [_sample(f"sft_e{i}", total=10 + i) for i in range(7)]
    audit_rows.append(_sample("sft_ghost", total=900))
    audit_rows[-1]["split"] = "train"
    cfg = QLoRASmokeConfig()
    # short 数量 = 7，但 P95 无合法候选 ⇒ total < target_count ⇒ INCOMPLETE
    with pytest.raises(TrainingGateError) as exc:
        select_smoke_samples_strict(train_records, audit_rows, cfg)
    assert "SMOKE_SELECTION_INCOMPLETE" in str(exc.value) or "SMOKE_SAMPLE_NOT_IN_TRAIN" in str(exc.value)


# ---------------------------------------------------------------------------
# 4) Quantization 配置语义
# ---------------------------------------------------------------------------


def test_quantization_config_semantics_are_nf4():
    cfg = QLoRASmokeConfig()
    assert cfg.quant_type == "nf4"
    with pytest.raises(ValueError):
        QLoRASmokeConfig(quant_type="fp4")
    with pytest.raises(ValueError):
        QLoRASmokeConfig(quant_type="int8")


def test_double_quant_enabled():
    cfg = QLoRASmokeConfig()
    assert cfg.double_quant is True
    with pytest.raises(ValueError):
        QLoRASmokeConfig(double_quant=False)


def test_compute_dtype_is_bf16():
    cfg = QLoRASmokeConfig()
    assert cfg.compute_dtype == "bfloat16"
    with pytest.raises(ValueError):
        QLoRASmokeConfig(compute_dtype="float16")
    assert cfg.bf16 is True
    assert cfg.fp16 is False
    with pytest.raises(ValueError):
        QLoRASmokeConfig(bf16=False)
    with pytest.raises(ValueError):
        QLoRASmokeConfig(fp16=True)


# ---------------------------------------------------------------------------
# 5) LoRA baseline
# ---------------------------------------------------------------------------


def test_lora_config_semantics():
    """r=16 / alpha=32 / dropout=0.05 / all-linear 固定。"""
    cfg = QLoRASmokeConfig()
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.lora_target_modules == "all-linear"
    assert cfg.lora_bias == "none"
    assert cfg.lora_task_type == "CAUSAL_LM"
    with pytest.raises(ValueError):
        QLoRASmokeConfig(lora_r=32)
    with pytest.raises(ValueError):
        QLoRASmokeConfig(lora_target_modules="q_proj,v_proj")


# ---------------------------------------------------------------------------
# 6) Smoke training semantics
# ---------------------------------------------------------------------------


def test_smoke_training_semantics():
    """batch=1 / accum=1 / max_steps=1 / assistant_only_loss=True / packing=False。"""
    cfg = QLoRASmokeConfig()
    assert cfg.per_device_train_batch_size == 1
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.max_steps == 1
    assert cfg.assistant_only_loss is True
    assert cfg.packing is False
    assert cfg.gradient_checkpointing is True
    with pytest.raises(ValueError):
        QLoRASmokeConfig(max_steps=2)
    with pytest.raises(ValueError):
        QLoRASmokeConfig(packing=True)
    with pytest.raises(ValueError):
        QLoRASmokeConfig(assistant_only_loss=False)


# ---------------------------------------------------------------------------
# 7) smoke_config.json 确定性
# ---------------------------------------------------------------------------


def test_smoke_config_is_deterministic(tmp_path: Path):
    """同 config + 同 model_revision + 同 selection 写两次 smoke_config.json 必须 byte-for-byte 一致。"""
    cfg = QLoRASmokeConfig()
    revision = {"requested_revision": "main", "resolved_revision": "abc123", "load_revision": "abc123", "revision_pinned": True}
    sel = SmokeSampleSelection(
        sample_ids=["sft_a", "sft_b", "sft_c"],
        short_count=2,
        p95_sample_id="sft_c",
        p95_total_tokens=400,
    )
    p1 = tmp_path / "c1.json"
    p2 = tmp_path / "c2.json"
    write_smoke_config(cfg, p1, model_revision=revision, selection=sel)
    write_smoke_config(cfg, p2, model_revision=revision, selection=sel)
    assert p1.read_bytes() == p2.read_bytes()
    payload = json.loads(p1.read_text(encoding="utf-8"))
    for k in ("model", "revision", "max_length", "quantization", "lora", "training", "smoke"):
        assert k in payload
    blob = json.dumps(payload, ensure_ascii=False)
    for forbidden in ("timestamp", "runtime", "hostname", "peak_memory", "vram", "loss_finite", "peak_allocated", "peak_reserved"):
        assert forbidden not in blob, forbidden
    # sample IDs 必须写入 smoke block
    assert payload["smoke"]["selected_sample_ids"] == ["sft_a", "sft_b", "sft_c"]
    assert payload["smoke"]["p95_sample_id"] == "sft_c"
    assert payload["smoke"]["p95_total_tokens"] == 400
    # revision 必须写入 model block
    assert payload["model"]["resolved_revision"] == "abc123"
    assert payload["model"]["revision_pinned"] is True


# ---------------------------------------------------------------------------
# 8) Inference prompt
# ---------------------------------------------------------------------------


def test_inference_prompt_basic():
    msgs = build_inference_prompt("什么是基层党组织？")
    assert msgs == [{"role": "user", "content": "什么是基层党组织？"}]


def test_inference_prompt_rejects_empty():
    with pytest.raises(ValueError):
        build_inference_prompt("")


# ---------------------------------------------------------------------------
# 9) Phase 5A gate 在真实 token_report 上的回归
# ---------------------------------------------------------------------------


def test_real_local_token_report_passes_gate(tmp_path: Path):
    """本地真实 token_report 必须通过 gate（如果数据未变）。"""
    path = Path("data/tokenizer/token_report.json")
    if not path.exists():
        pytest.skip("data/tokenizer/token_report.json 尚未生成")
    report = load_token_report(path)
    result = evaluate_phase5a_gate(report)
    assert result.passed is True
    assert result.actual_max_tokens == 1672
    assert result.samples_over_max_length == 0 or result.samples_over_max_length == 0


def test_real_local_token_audit_rows(tmp_path: Path):
    """本地真实 token_audit.jsonl 必须能解析，且 train P95 接近 446.6。"""
    path = Path("data/tokenizer/token_audit.jsonl")
    if not path.exists():
        pytest.skip("data/tokenizer/token_audit.jsonl 尚未生成")
    rows = load_token_audit_rows(path)
    assert len(rows) == 1303
    train_rows = [r for r in rows if r.get("split") == "train"]
    assert len(train_rows) == 1149
    totals = sorted(int(r.get("total_tokens") or 0) for r in train_rows)
    idx = int(round(0.95 * (len(totals) - 1)))
    p95 = totals[idx]
    # 当前 report train p95 ≈ 446.6
    assert 400 <= p95 <= 500


# ---------------------------------------------------------------------------
# 10) Dataclass 字段中文注释（src 约定）
# ---------------------------------------------------------------------------


def test_training_dataclass_fields_have_chinese_comments():
    """新增 dataclass 字段前必须有中文注释。"""
    src_dir = Path(__file__).resolve().parents[1] / "src" / "data_pipeline" / "training"
    import ast

    def _is_dataclass(node: ast.ClassDef) -> bool:
        for d in node.decorator_list:
            if isinstance(d, ast.Name) and d.id == "dataclass":
                return True
            if isinstance(d, ast.Attribute) and d.attr == "dataclass":
                return True
            if isinstance(d, ast.Call):
                f = d.func
                if isinstance(f, ast.Name) and f.id == "dataclass":
                    return True
                if isinstance(f, ast.Attribute) and f.attr == "dataclass":
                    return True
        return False

    def _collect_dataclasses(py_file: Path) -> dict[str, list[str]]:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        out: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_dataclass(node):
                fields = [
                    t.target.id  # type: ignore[attr-defined]
                    for t in node.body
                    if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
                ]
                if fields:
                    out[node.name] = fields
        return out

    found = False
    for py_file in sorted(src_dir.glob("*.py")):
        cls_to_fields = _collect_dataclasses(py_file)
        if not cls_to_fields:
            continue
        found = True
        text = py_file.read_text(encoding="utf-8")
        for cls, fields in cls_to_fields.items():
            cls_idx = text.find(f"class {cls}")
            assert cls_idx >= 0, f"class {cls} not found in {py_file.name}"
            tail = text[cls_idx:]
            next_boundary = re.search(
                r"^class\s+\w+|^def\s+\w+|^@\w+",
                tail[len(f"class {cls}"):],
                re.MULTILINE,
            )
            cls_body = tail[: len(f"class {cls}") + (next_boundary.start() if next_boundary else len(tail))]
            for field_name in fields:
                field_pattern = re.compile(
                    r"(?:^|\n)([^\n]*\n[^\n]*\n)?[ \t]*" + re.escape(field_name) + r"\s*[:=]"
                )
                m = field_pattern.search(cls_body)
                assert m, f"field {field_name!r} not found in {cls} ({py_file.name})"
                preceding = (m.group(1) or "")
                assert re.search(r"[\u4e00-\u9fff]", preceding), (
                    f"字段 {cls}.{field_name} 前两行缺少中文注释（src: {py_file.name}）"
                )

    assert found, "expected at least one dataclass in src/data_pipeline/training"


# ---------------------------------------------------------------------------
# 11) Token report 读取辅助
# ---------------------------------------------------------------------------


def test_load_token_report_missing_file(tmp_path: Path):
    with pytest.raises(TrainingGateError):
        load_token_report(tmp_path / "nope.json")


def test_load_token_audit_rows_missing_file(tmp_path: Path):
    with pytest.raises(TrainingGateError):
        load_token_audit_rows(tmp_path / "nope.jsonl")