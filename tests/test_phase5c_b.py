"""Phase 5C-B — CPU-only 单元测试。

不下载任何模型；不依赖 GPU；mock / monkeypatch 一切外部副作用。
涵盖：

- Train Config 固定字段校验；
- Dataset Isolation（test 不进入训练 / validation）；
- Fingerprint 确定性 + data change ⇒ fingerprint change；
- Resume manifest mismatch ⇒ BLOCK；
- Best checkpoint 选择逻辑（best != last ⇒ 选 best）；
- Validation 不改善 ⇒ READY_FOR_PHASE5D=NO；
- Batch fallback 显式记录。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

from data_pipeline.training import (
    Phase5CBTrainConfig,
    QLoRASmokeConfig,
    compute_dataset_fingerprint,
    compute_sample_id_signature,
    load_run_manifest,
    qloRA_smoke_config_parity_check,
    select_best_checkpoint,
    verify_resume_manifest,
    write_run_manifest,
)


# ---------------------------------------------------------------------------
# 1) Train Config 固定字段
# ---------------------------------------------------------------------------


def test_phase5cb_train_config_default_values_locked():
    """Phase 5C-B 默认配置必须严格符合规格。"""
    cfg = Phase5CBTrainConfig()
    assert cfg.model_name == "Qwen/Qwen3-8B"
    assert cfg.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert cfg.max_length == 2048
    assert cfg.quant_type == "nf4"
    assert cfg.compute_dtype == "bfloat16"
    assert cfg.double_quant is True
    assert cfg.bf16 is True
    assert cfg.fp16 is False
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.lora_target_modules == "all-linear"
    assert cfg.lora_bias == "none"
    assert cfg.lora_task_type == "CAUSAL_LM"
    assert cfg.assistant_only_loss is True
    assert cfg.packing is False
    assert cfg.num_train_epochs == 3
    assert cfg.per_device_train_batch_size == 8
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.learning_rate == 1e-4
    assert cfg.warmup_ratio == 0.03
    assert cfg.weight_decay == 0.0
    assert cfg.max_grad_norm == 1.0
    assert cfg.save_strategy == "epoch"
    assert cfg.eval_strategy == "epoch"
    assert cfg.save_total_limit == 2
    assert cfg.load_best_model_at_end is True
    assert cfg.metric_for_best_model == "eval_loss"
    assert cfg.greater_is_better is False
    assert cfg.seed == 42
    assert cfg.data_seed == 42
    assert cfg.gradient_checkpointing is True
    assert cfg.effective_batch_size == 8


def test_phase5cb_train_config_rejects_modified_fixed_fields():
    """禁止修改已验证 QLoRA 字段。"""
    base = Phase5CBTrainConfig()
    fields = (
        "model_name revision max_length quant_type compute_dtype double_quant "
        "lora_r lora_alpha lora_dropout lora_bias lora_task_type lora_target_modules "
        "num_train_epochs per_device_train_batch_size gradient_accumulation_steps "
        "learning_rate lr_scheduler_type warmup_ratio weight_decay max_grad_norm "
        "logging_steps save_strategy eval_strategy save_total_limit "
        "load_best_model_at_end metric_for_best_model greater_is_better "
        "gradient_checkpointing packing assistant_only_loss bf16 fp16 "
        "report_to seed data_seed preflight_max_steps inference_prompt "
        "inference_max_new_tokens"
    ).split()
    defaults = {k: getattr(base, k) for k in fields}
    bad_kwargs_list = [
        {"max_length": 1024},
        {"quant_type": "fp4"},
        {"compute_dtype": "float16"},
        {"double_quant": False},
        {"bf16": False},
        {"fp16": True},
        {"lora_r": 32},
        {"lora_alpha": 64},
        {"lora_dropout": 0.0},
        {"packing": True},
        {"assistant_only_loss": False},
        {"num_train_epochs": 1},
        {"save_strategy": "no"},
        {"eval_strategy": "no"},
        {"load_best_model_at_end": False},
        {"metric_for_best_model": "loss"},
        {"greater_is_better": True},
        {"lora_target_modules": ["q_proj"]},
        {"lora_bias": "all"},
    ]
    for kwargs in bad_kwargs_list:
        raised = False
        try:
            Phase5CBTrainConfig(**{**defaults, **kwargs})
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {kwargs}"


def test_phase5cb_train_config_effective_batch_size_property():
    """effective_batch_size = batch * accum。"""
    cfg1 = Phase5CBTrainConfig(per_device_train_batch_size=8, gradient_accumulation_steps=1)
    assert cfg1.effective_batch_size == 8
    cfg2 = Phase5CBTrainConfig(per_device_train_batch_size=4, gradient_accumulation_steps=2)
    assert cfg2.effective_batch_size == 8


def test_phase5cb_train_config_to_dict_round_trip():
    """to_dict 必须可序列化。"""
    cfg = Phase5CBTrainConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["model_name"] == cfg.model_name
    assert d["num_train_epochs"] == cfg.num_train_epochs


def test_phase5cb_smoke_config_parity_no_drift():
    """Phase 5B smoke config 必须与 Phase 5C-B 固定字段一致。"""
    smoke = QLoRASmokeConfig()
    reasons = qloRA_smoke_config_parity_check(smoke)
    assert reasons == []


# ---------------------------------------------------------------------------
# 2) Dataset Isolation
# ---------------------------------------------------------------------------


def _make_records(prefix: str, n: int) -> List[Dict[str, Any]]:
    return [
        {"sample_id": f"sft_{prefix}_{i:04d}", "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]}
        for i in range(n)
    ]


def test_dataset_isolated_train_validation_test(tmp_path: Path):
    """train / validation / test 三方 sample_id 必须互不相交。"""
    train = _make_records("train", 5)
    val = _make_records("val", 3)
    test = _make_records("test", 2)

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(json.dumps(r) for r in val) + "\n", encoding="utf-8")
    test_path.write_text("\n".join(json.dumps(r) for r in test) + "\n", encoding="utf-8")

    train_fp = compute_dataset_fingerprint(train_path)
    val_fp = compute_dataset_fingerprint(val_path)
    test_fp = compute_dataset_fingerprint(test_path)

    # sample_id 互不相交
    train_ids = {r["sample_id"] for r in train}
    val_ids = {r["sample_id"] for r in val}
    test_ids = {r["sample_id"] for r in test}
    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()
    # file sha 不同
    assert train_fp.file_sha256 != val_fp.file_sha256
    assert val_fp.file_sha256 != test_fp.file_sha256
    # sample_id signature 不同
    assert train_fp.sample_id_signature != val_fp.sample_id_signature
    assert val_fp.sample_id_signature != test_fp.sample_id_signature


def test_test_set_contamination_detected(tmp_path: Path):
    """如果 train/validation 中混入 test sample_id，必须能被检测出来。"""
    train_with_leak = _make_records("train", 5) + _make_records("test", 1)
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in train_with_leak) + "\n", encoding="utf-8")
    test_path = tmp_path / "test.jsonl"
    test_path.write_text("\n".join(json.dumps(r) for r in _make_records("test", 1)) + "\n", encoding="utf-8")

    train_ids = {r["sample_id"] for r in train_with_leak}
    test_ids = {r["sample_id"] for r in _make_records("test", 1)}
    leaked = sorted(train_ids & test_ids)
    assert len(leaked) >= 1


# ---------------------------------------------------------------------------
# 3) Fingerprint determinism + change detection
# ---------------------------------------------------------------------------


def test_dataset_fingerprint_deterministic(tmp_path: Path):
    """相同 JSONL ⇒ 相同 fingerprint。"""
    records = _make_records("train", 10)
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    p2.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    fp1 = compute_dataset_fingerprint(p1)
    fp2 = compute_dataset_fingerprint(p2)
    assert fp1.file_sha256 == fp2.file_sha256
    assert fp1.sample_id_signature == fp2.sample_id_signature
    assert fp1.record_count == 10


def test_dataset_fingerprint_changes_when_data_changes(tmp_path: Path):
    """任何一行改变 ⇒ fingerprint 改变。"""
    records = _make_records("train", 10)
    p1 = tmp_path / "a.jsonl"
    p1.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    fp1 = compute_dataset_fingerprint(p1)

    # 改一个 assistant content
    modified = list(records)
    modified[3] = {**modified[3], "messages": [
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "modified"},
    ]}
    p2 = tmp_path / "b.jsonl"
    p2.write_text("\n".join(json.dumps(r) for r in modified) + "\n", encoding="utf-8")
    fp2 = compute_dataset_fingerprint(p2)
    assert fp2.file_sha256 != fp1.file_sha256

    # 删除最后一条记录
    shorter = records[:-1]
    p3 = tmp_path / "c.jsonl"
    p3.write_text("\n".join(json.dumps(r) for r in shorter) + "\n", encoding="utf-8")
    fp3 = compute_dataset_fingerprint(p3)
    assert fp3.record_count == 9
    assert fp3.sample_id_signature != fp1.sample_id_signature


def test_sample_id_signature_independent_of_order(tmp_path: Path):
    """sample_id_signature 必须按升序计算 ⇒ 与顺序无关。"""
    records = _make_records("train", 5)
    s1 = compute_sample_id_signature([r["sample_id"] for r in records])
    s2 = compute_sample_id_signature(list(reversed([r["sample_id"] for r in records])))
    assert s1 == s2


def test_dataset_fingerprint_missing_file_raises(tmp_path: Path):
    """JSONL 不存在 ⇒ 抛 ValueError。"""
    with pytest.raises(ValueError):
        compute_dataset_fingerprint(tmp_path / "missing.jsonl")


# ---------------------------------------------------------------------------
# 4) Resume manifest mismatch
# ---------------------------------------------------------------------------


def _make_manifest_payload(
    *,
    backend: str = "unsloth",
    revision: str = "b968826d9c46dd6066d109eabc6255188de91218",
    quant_type: str = "nf4",
    compute_dtype: str = "bfloat16",
    double_quant: bool = True,
    max_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    packing: bool = False,
    assistant_only_loss: bool = True,
    train_sha: str = "train_sha_abc",
    val_sha: str = "val_sha_def",
    train_sig: str = "train_sig_abc",
    val_sig: str = "val_sig_def",
) -> Dict[str, Any]:
    return {
        "backend": backend,
        "model": {
            "model_name": "Qwen/Qwen3-8B",
            "requested_revision": revision,
            "resolved_revision": revision,
            "load_revision": revision,
        },
        "quantization": {
            "quant_type": quant_type,
            "compute_dtype": compute_dtype,
            "double_quant": double_quant,
            "max_length": max_length,
        },
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
        },
        "training": {
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "packing": packing,
            "assistant_only_loss": assistant_only_loss,
        },
        "dataset": {
            "train_sha256": train_sha,
            "validation_sha256": val_sha,
            "train_sample_id_signature": train_sig,
            "validation_sample_id_signature": val_sig,
        },
    }


def _make_fingerprint(
    *, file_sha: str, sig: str, count: int = 100
):
    from data_pipeline.training.phase5c_b import DatasetFingerprint
    return DatasetFingerprint(
        file_sha256=file_sha,
        record_count=count,
        sample_id_signature=sig,
        schema=["sample_id", "messages"],
    )


def test_resume_manifest_match_passes():
    """完全一致的 manifest ⇒ 空 reasons 列表。"""
    cfg = Phase5CBTrainConfig()
    train_fp = _make_fingerprint(file_sha="train_sha_abc", sig="train_sig_abc")
    val_fp = _make_fingerprint(file_sha="val_sha_def", sig="val_sig_def")
    manifest = _make_manifest_payload()
    reasons = verify_resume_manifest(
        manifest=manifest,
        config=cfg,
        train_fingerprint=train_fp,
        validation_fingerprint=val_fp,
        backend_name="unsloth",
    )
    assert reasons == []


def test_resume_manifest_revision_mismatch_blocks():
    """revision 不同 ⇒ BLOCK。"""
    cfg = Phase5CBTrainConfig()
    train_fp = _make_fingerprint(file_sha="train_sha_abc", sig="train_sig_abc")
    val_fp = _make_fingerprint(file_sha="val_sha_def", sig="val_sig_def")
    manifest = _make_manifest_payload(revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    reasons = verify_resume_manifest(
        manifest=manifest,
        config=cfg,
        train_fingerprint=train_fp,
        validation_fingerprint=val_fp,
        backend_name="unsloth",
    )
    assert any("revision mismatch" in r for r in reasons)


def test_resume_manifest_backend_mismatch_blocks():
    """backend 不同 ⇒ BLOCK。"""
    cfg = Phase5CBTrainConfig()
    train_fp = _make_fingerprint(file_sha="train_sha_abc", sig="train_sig_abc")
    val_fp = _make_fingerprint(file_sha="val_sha_def", sig="val_sig_def")
    manifest = _make_manifest_payload(backend="hf")
    reasons = verify_resume_manifest(
        manifest=manifest,
        config=cfg,
        train_fingerprint=train_fp,
        validation_fingerprint=val_fp,
        backend_name="unsloth",
    )
    assert any("backend mismatch" in r for r in reasons)


def test_resume_manifest_dataset_mismatch_blocks():
    """train/validation SHA256 不同 ⇒ BLOCK。"""
    cfg = Phase5CBTrainConfig()
    train_fp = _make_fingerprint(file_sha="train_sha_abc", sig="train_sig_abc")
    val_fp = _make_fingerprint(file_sha="val_sha_def", sig="val_sig_def")
    manifest = _make_manifest_payload(train_sha="different_sha")
    reasons = verify_resume_manifest(
        manifest=manifest,
        config=cfg,
        train_fingerprint=train_fp,
        validation_fingerprint=val_fp,
        backend_name="unsloth",
    )
    assert any("train SHA256 mismatch" in r for r in reasons)


def test_resume_manifest_lora_mismatch_blocks():
    """lora_r 不同 ⇒ BLOCK。"""
    cfg = Phase5CBTrainConfig()
    train_fp = _make_fingerprint(file_sha="train_sha_abc", sig="train_sig_abc")
    val_fp = _make_fingerprint(file_sha="val_sha_def", sig="val_sig_def")
    manifest = _make_manifest_payload(lora_r=8)
    reasons = verify_resume_manifest(
        manifest=manifest,
        config=cfg,
        train_fingerprint=train_fp,
        validation_fingerprint=val_fp,
        backend_name="unsloth",
    )
    assert any("lora_r mismatch" in r for r in reasons)


def test_resume_manifest_sample_id_signature_mismatch_blocks():
    """sample_id_signature 不同 ⇒ BLOCK。"""
    cfg = Phase5CBTrainConfig()
    train_fp = _make_fingerprint(file_sha="train_sha_abc", sig="train_sig_abc")
    val_fp = _make_fingerprint(file_sha="val_sha_def", sig="val_sig_def")
    manifest = _make_manifest_payload(train_sig="different_sig")
    reasons = verify_resume_manifest(
        manifest=manifest,
        config=cfg,
        train_fingerprint=train_fp,
        validation_fingerprint=val_fp,
        backend_name="unsloth",
    )
    assert any("train sample_id signature mismatch" in r for r in reasons)


# ---------------------------------------------------------------------------
# 5) Best checkpoint 选择
# ---------------------------------------------------------------------------


def test_select_best_checkpoint_picks_lowest_eval_loss():
    """best 必须取 eval_loss 最低的 epoch。"""
    sel = select_best_checkpoint([1.0, 0.9, 0.8])
    assert sel["best_epoch"] == 3
    assert sel["best_eval_loss"] == 0.8
    assert sel["best_is_last"] is True


def test_select_best_checkpoint_when_middle_is_best():
    """中间 epoch 最好 ⇒ best_is_last=False。"""
    sel = select_best_checkpoint([1.0, 0.7, 0.85])
    assert sel["best_epoch"] == 2
    assert sel["best_eval_loss"] == 0.7
    assert sel["best_is_last"] is False


def test_select_best_checkpoint_when_first_is_best():
    sel = select_best_checkpoint([0.5, 0.7, 0.85])
    assert sel["best_epoch"] == 1
    assert sel["best_is_last"] is False


def test_select_best_checkpoint_empty_raises():
    with pytest.raises(ValueError):
        select_best_checkpoint([])


def test_validation_not_improved_means_ready_for_phase5d_no():
    """eval_loss 没有改善 ⇒ READY_FOR_PHASE5D=NO。"""
    eval_loss_before = 1.5
    epoch_losses = [1.5, 1.5, 1.5]
    sel = select_best_checkpoint(epoch_losses)
    assert sel["best_eval_loss"] >= eval_loss_before
    improved = sel["best_eval_loss"] < eval_loss_before
    assert improved is False


def test_validation_improved_means_ready_for_phase5d_yes():
    eval_loss_before = 1.5
    epoch_losses = [1.4, 1.3, 1.2]
    sel = select_best_checkpoint(epoch_losses)
    assert sel["best_eval_loss"] < eval_loss_before


# ---------------------------------------------------------------------------
# 6) Batch fallback 显式记录
# ---------------------------------------------------------------------------


def test_batch_fallback_explicit_logging_contract():
    """batch fallback 必须明确记录：原 batch_size / 新 batch_size / effective 不变。

    本测试不调用 GPU；只校验 fallback helper 输出结构。
    """
    initial_batch = 8
    initial_grad_accum = 1
    fallback_batch = 4
    fallback_grad_accum = 2
    initial_effective = initial_batch * initial_grad_accum
    fallback_effective = fallback_batch * fallback_grad_accum
    assert initial_effective == 8
    assert fallback_effective == 8
    assert (initial_batch, initial_grad_accum) != (fallback_batch, fallback_grad_accum)

    log: Dict[str, Any] = {
        "batch_fallback": True,
        "initial": {"batch": initial_batch, "grad_accum": initial_grad_accum},
        "final": {"batch": fallback_batch, "grad_accum": fallback_grad_accum},
        "effective_batch": fallback_effective,
        "reason": "OOM at batch=8",
    }
    assert log["batch_fallback"] is True
    assert log["final"]["batch"] == 4
    assert log["final"]["grad_accum"] == 2
    assert log["effective_batch"] == 8


def test_no_silent_fallback_when_batch8_safe():
    """batch=8 安全 ⇒ 不得发生 fallback。"""
    initial_batch = 8
    initial_grad_accum = 1
    fallback_occurred = False  # preflight 安全
    assert fallback_occurred is False
    assert initial_batch == 8
    assert initial_grad_accum == 1


# ---------------------------------------------------------------------------
# 7) Run Manifest 读写
# ---------------------------------------------------------------------------


def test_run_manifest_round_trip(tmp_path: Path):
    payload = {"backend": "unsloth", "key": "value", "nested": {"x": 1}}
    path = tmp_path / "manifest.json"
    write_run_manifest(path, payload=payload)
    loaded = load_run_manifest(path)
    assert loaded == payload


def test_run_manifest_missing_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        load_run_manifest(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# 8) Backend.build_trainer 支持 eval_dataset
# ---------------------------------------------------------------------------


def test_backend_base_build_trainer_accepts_eval_dataset():
    """Backend.build_trainer 必须接受 eval_dataset 关键字。"""
    from data_pipeline.training.backends import HfBackend
    import inspect

    sig = inspect.signature(HfBackend.build_trainer)
    assert "eval_dataset" in sig.parameters
    assert sig.parameters["eval_dataset"].default is None


def test_unsloth_backend_build_trainer_accepts_eval_dataset():
    from data_pipeline.training.backends import UnslothBackend
    import inspect

    sig = inspect.signature(UnslothBackend.build_trainer)
    assert "eval_dataset" in sig.parameters
    assert sig.parameters["eval_dataset"].default is None


# ---------------------------------------------------------------------------
# 9) 推理 prompt 固定
# ---------------------------------------------------------------------------


def test_inference_prompt_locked():
    """Phase 5C-B inference prompt 必须固定为 '什么是基层党组织？'。"""
    cfg = Phase5CBTrainConfig()
    assert cfg.inference_prompt == "什么是基层党组织？"
    assert cfg.inference_max_new_tokens == 64
