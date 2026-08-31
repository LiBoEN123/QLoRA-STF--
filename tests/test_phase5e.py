"""Phase 5E — Release & Project Closeout CPU 单元测试。

不依赖 GPU / 不下载模型。运行：

    .venv/Scripts/python.exe -m pytest tests/test_phase5e.py -q
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# 0) helpers — fake safetensors adapter fixtures
# ---------------------------------------------------------------------------


def _make_fake_safetensors(
    path: Path,
    *,
    tensors: Dict[str, List[float]],
) -> None:
    """写入最小 fake safetensors。

    只保证 ``safetensors.safe_open`` 能读到 tensors；值不重要，仅用于
    数量 + shape 的 signature。
    """
    import numpy as np  # type: ignore
    from safetensors.numpy import save_file  # type: ignore

    np_tensors: Dict[str, Any] = {}
    for name, vals in tensors.items():
        np_tensors[name] = np.array(vals, dtype=np.float32).reshape(-1, 1)
    save_file(np_tensors, str(path))


def _make_fake_adapter_dir(
    base: Path,
    *,
    n_lora_layers: int = 4,
    tensor_values: List[float] | None = None,
    base_filename: str = "adapter_model.safetensors",
) -> Path:
    """构造一个最简单的 LoRA adapter 目录（含 *必需* 的 safetensors）。

    返回目录路径。
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    # 每个 LoRA layer = 2 tensors (A/B)
    tensors = {}
    base_val = tensor_values or [0.1, 0.2, 0.3]
    for i in range(n_lora_layers):
        tensors[f"layer_{i}.lora_A"] = list(base_val)
        tensors[f"layer_{i}.lora_B"] = list(base_val)
    _make_fake_safetensors(base / base_filename, tensors=tensors)
    # 配套 token / chat template 等最小文件
    (base / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3-8B",
                "peft_type": "LORA",
                "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "task_type": "CAUSAL_LM",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (base / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen3Tokenizer"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (base / "chat_template.jinja").write_text(
        "{% for m in messages %}<|{{m.role}}|{{m.content}}|>{% endfor %}",
        encoding="utf-8",
    )
    (base / "README.md").write_text("# fake adapter\n", encoding="utf-8")
    return base


# ---------------------------------------------------------------------------
# 1) Phase 5D gate (against local artifact)
# ---------------------------------------------------------------------------


def test_phase5d_report_exists_and_pass():
    """Phase 5D 真实 report 存在且 status/code/ready 满足 PASS。"""
    from data_pipeline.training.phase5e import (
        PHASE5D_ARTIFACT_DIR, validate_phase5d_report,
    )

    if not (PHASE5D_ARTIFACT_DIR / "phase5d_report.json").exists():
        pytest.skip("Phase 5D report not present")
    v = validate_phase5d_report()
    assert v.pass_ is True, v.reasons
    assert v.status == "PASS"
    assert v.code == "PASS"
    assert v.ready_for_phase5e is True
    assert v.test_count == 79
    assert v.predictions_rows == 79
    assert v.finetuned_test_loss < v.base_test_loss


def test_phase5d_report_fields_complete():
    """validate_phase5d_report 必须填齐所有 Phase 5D 报告关键字段。"""
    from data_pipeline.training.phase5e import (
        PHASE5D_ARTIFACT_DIR, validate_phase5d_report,
    )

    if not (PHASE5D_ARTIFACT_DIR / "phase5d_report.json").exists():
        pytest.skip("Phase 5D report not present")
    v = validate_phase5d_report()
    assert v.eval_protocol_signature
    assert v.model_name == "Qwen/Qwen3-8B"
    assert v.model_revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert v.tokenizer_signature == (
        "4ac4fed6d76f2cddf5ee756990b7bd4dc40655f96e4c6be460dc3299e73651c0"
    )
    assert v.expected_lora_layers == 252
    assert v.protocol_frozen is True
    assert v.generation_max_new_tokens > 0


def test_phase5d_blocked_when_report_missing(tmp_path: Path):
    """Phase 5D 缺失 ⇒ validate_phase5d_report 必返回 pass_=False。"""
    from data_pipeline.training.phase5e import validate_phase5d_report

    v = validate_phase5d_report(phase5d_dir=tmp_path / "no_phase5d")
    assert v.pass_ is False
    assert v.ready_for_phase5e is False
    assert v.status == "BLOCKED"
    assert any("missing" in r.lower() for r in v.reasons)


def test_phase5d_blocked_when_test_count_drift(tmp_path: Path):
    """构造伪 report（test_count != 79）⇒ pass_=False。"""
    from data_pipeline.training.phase5e import validate_phase5d_report

    base = tmp_path / "p5d"
    base.mkdir()
    # minimal manifest
    (base / "eval_manifest.json").write_text(
        json.dumps(
            {
                "protocol_frozen": True,
                "eval_protocol_signature": "x",
                "model_name": "Qwen/Qwen3-8B",
                "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "tokenizer_signature": (
                    "4ac4fed6d76f2cddf5ee756990b7bd4dc40655f96e4c6be460dc3299e73651c0"
                ),
                "generation_max_new_tokens": 1610,
                "test_jsonl_sha256": "x",
                "test_sample_id_signature": "y",
                "best_adapter_signature_sha256": "z",
                "expected_lora_layers": 252,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (base / "base_test_metrics.json").write_text(
        json.dumps({"n": 79, "test_loss": 1.9, "mean_char_f1": 0.27}),
        encoding="utf-8",
    )
    (base / "finetuned_test_metrics.json").write_text(
        json.dumps({"n": 79, "test_loss": 1.2, "mean_char_f1": 0.37}),
        encoding="utf-8",
    )
    (base / "paired_metrics.json").write_text(
        json.dumps(
            {"char_f1": {"n": 79}, "rouge_l_f1": {"n": 79}}, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (base / "phase5d_report.json").write_text(
        json.dumps(
            {
                "status": "PASS", "code": "PASS",
                "ready_for_phase5e": True,
                "test_count": 50,  # 故意 != 79
                "base_test_loss": 1.9,
                "finetuned_test_loss": 1.2,
                "test_loss_improvement": 0.7,
                "test_loss_improvement_pct": 36.8,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # predictions file 行数 != 79
    pred_path = base / "test_predictions.jsonl"
    pred_path.write_text(
        "\n".join(
            json.dumps({"sample_id": f"x{i}"}, ensure_ascii=False) for i in range(50)
        )
        + "\n",
        encoding="utf-8",
    )
    v = validate_phase5d_report(phase5d_dir=base)
    assert v.pass_ is False
    assert any("test_count" in r for r in v.reasons)


# ---------------------------------------------------------------------------
# 2) Release manifest
# ---------------------------------------------------------------------------


def test_release_manifest_has_required_fields():
    """release_manifest 必须包含全部硬约束字段。"""
    from data_pipeline.training.phase5e import (
        PHASE5E_BASE_MODEL_NAME,
        PHASE5E_BASE_REVISION,
        PHASE5E_TOKENIZER_SIGNATURE,
        Phase5DVerification,
        AdapterProvenance,
        build_release_manifest,
    )

    dummy_p5d = Phase5DVerification(
        phase5d_report_path="/tmp",
        status="PASS", code="PASS", ready_for_phase5e=True,
        test_count=79, predictions_rows=79,
        base_test_loss=1.9, finetuned_test_loss=1.2,
        test_loss_improvement=0.7, test_loss_improvement_pct=36.8,
        protocol_frozen=True, eval_protocol_signature="a" * 64,
        model_name="Qwen/Qwen3-8B",
        model_revision="b968826d9c46dd6066d109eabc6255188de91218",
        tokenizer_signature=PHASE5E_TOKENIZER_SIGNATURE,
        generation_max_new_tokens=1610,
        test_jsonl_sha256="x", test_sample_id_signature="y",
        best_adapter_signature_sha256="z", expected_lora_layers=252,
        pass_=True,
    )
    from data_pipeline.training.phase5e import AdapterProvenance
    prov = AdapterProvenance(
        source_path="/tmp/src",
        checkpoint_path="/tmp/ckpt",
        source_signature="a" * 64,
        checkpoint_signature="a" * 64,
        n_tensors=504,
        equal=True,
    )
    payload = build_release_manifest(
        git_commit="abc",
        git_dirty=False,
        phase5d_verification=dummy_p5d,
        adapter_provenance=prov,
    )
    # 必备字段
    assert payload["schema_version"] == "phase5e.v1"
    assert payload["release_type"] == "lora_adapter"
    assert payload["project_name"]
    assert payload["release_created_at"]
    # base
    assert payload["base_model"]["model_name"] == PHASE5E_BASE_MODEL_NAME
    assert payload["base_model"]["revision"] == PHASE5E_BASE_REVISION
    assert payload["base_model"]["quant_type"] == "nf4"
    assert payload["base_model"]["double_quant"] is True
    assert payload["base_model"]["compute_dtype"] == "bfloat16"
    assert payload["base_model"]["load_in_4bit"] is True
    # tokenizer
    assert payload["tokenizer"]["signature"] == PHASE5E_TOKENIZER_SIGNATURE
    # lora
    assert payload["lora"]["r"] == 16
    assert payload["lora"]["alpha"] == 32
    assert payload["lora"]["dropout"] == 0.05
    assert payload["lora"]["bias"] == "none"
    assert payload["lora"]["target_modules"] == "all-linear"
    assert payload["lora"]["lora_layers"] == 252
    assert payload["lora"]["trainable_params"] > 0
    # training
    assert payload["training"]["epochs"] == 3
    assert payload["training"]["global_steps"] == 432
    assert payload["training"]["batch_size"] == 8
    assert payload["training"]["effective_batch"] == 8
    # dataset
    assert payload["dataset"]["train_count"] == 1149
    assert payload["dataset"]["validation_count"] == 75
    assert payload["dataset"]["test_count"] == 79
    assert payload["dataset"]["test_sha256"]
    # validation
    assert payload["validation"]["best_epoch"] == 2
    assert payload["validation"]["best_checkpoint"] == "checkpoint-288"
    # held_out_test
    assert payload["held_out_test"]["finetuned_test_loss"] < payload["held_out_test"]["base_test_loss"]
    # adapter
    assert payload["adapter"]["tensor_count"] == 504
    assert payload["adapter"]["equal_to_checkpoint_288"] is True
    # phase5d
    assert payload["phase5d"]["phase5d_code"] == "PASS"


def test_release_manifest_base_revision_is_exact():
    """base model revision 必须严格 == pinned commit SHA。"""
    from data_pipeline.training.phase5e import PHASE5E_BASE_REVISION
    assert PHASE5E_BASE_REVISION == "b968826d9c46dd6066d109eabc6255188de91218"
    assert len(PHASE5E_BASE_REVISION) == 40


def test_release_manifest_release_type_is_lora_adapter():
    """release_type 必须是 'lora_adapter'，禁止 merged / gguf / 等。"""
    from data_pipeline.training.phase5e import (
        PHASE5E_BASE_MODEL_NAME,
        PHASE5E_BASE_REVISION,
        PHASE5E_TOKENIZER_SIGNATURE,
        AdapterProvenance,
        Phase5DVerification,
        build_release_manifest,
    )
    dummy_p5d = Phase5DVerification(
        phase5d_report_path="/tmp", status="PASS", code="PASS",
        ready_for_phase5e=True, test_count=79, predictions_rows=79,
        base_test_loss=1.9, finetuned_test_loss=1.2,
        test_loss_improvement=0.7, test_loss_improvement_pct=36.8,
        protocol_frozen=True, eval_protocol_signature="x" * 64,
        model_name=PHASE5E_BASE_MODEL_NAME,
        model_revision=PHASE5E_BASE_REVISION,
        tokenizer_signature=PHASE5E_TOKENIZER_SIGNATURE,
        generation_max_new_tokens=1610,
        test_jsonl_sha256="x", test_sample_id_signature="y",
        best_adapter_signature_sha256="z", expected_lora_layers=252,
        pass_=True,
    )
    prov = AdapterProvenance(
        source_path="/tmp/src", checkpoint_path="/tmp/ckpt",
        source_signature="a" * 64, checkpoint_signature="a" * 64,
        n_tensors=504, equal=True,
    )
    payload = build_release_manifest(
        git_commit="x", git_dirty=False,
        phase5d_verification=dummy_p5d,
        adapter_provenance=prov,
    )
    assert payload["release_type"] == "lora_adapter"
    assert "merged" not in payload["release_type"]
    assert "gguf" not in payload["release_type"]


# ---------------------------------------------------------------------------
# 3) Adapter copy signature equality
# ---------------------------------------------------------------------------


def test_copy_release_adapter_preserves_signature(tmp_path: Path):
    """copy_release_adapter 必须产生与 source tensor signature 一致的新副本。"""
    from data_pipeline.training.phase5e import (
        copy_release_adapter,
        verify_best_adapter_provenance,
    )

    src = _make_fake_adapter_dir(tmp_path / "src", n_lora_layers=4)
    dst_root = tmp_path / "release"
    dst_root.mkdir()
    target = copy_release_adapter(src, dst_root)
    # verify_best_adapter_provenance 直接看 src 内部 safetensors — 改用
    # 等价手段：source & release 两个 safetensors 的 tensor signature 一致
    from data_pipeline.training.phase5c_b import compute_adapter_signature
    import torch  # type: ignore
    sig_src = compute_adapter_signature(
        src / "adapter_model.safetensors", torch=torch, prefer_dtype="bfloat16",
    )
    sig_dst = compute_adapter_signature(
        target / "adapter_model.safetensors", torch=torch, prefer_dtype="bfloat16",
    )
    assert sig_src.n_tensors == sig_dst.n_tensors
    assert sig_src.signature == sig_dst.signature
    assert sig_src.n_tensors == 8  # 4 layers × 2 tensors


def test_copy_release_adapter_does_not_delete_source(tmp_path: Path):
    """copy_release_adapter 必须不删除源 best_adapter。"""
    from data_pipeline.training.phase5e import copy_release_adapter

    src = _make_fake_adapter_dir(tmp_path / "src", n_lora_layers=2)
    src_main = src / "adapter_model.safetensors"
    assert src_main.exists()
    copy_release_adapter(src, tmp_path / "release")
    assert src_main.exists(), "source must not be deleted"


def test_copy_release_adapter_works_when_destination_exists(tmp_path: Path):
    """如果目标 adapter 目录已存在，copy_release_adapter 仍然 idempotent。"""
    from data_pipeline.training.phase5e import copy_release_adapter

    src = _make_fake_adapter_dir(tmp_path / "src", n_lora_layers=2)
    dst_root = tmp_path / "release"
    dst_root.mkdir()
    copy_release_adapter(src, dst_root)
    # 第二次执行：必须成功且 source / dest  signature 一致
    target = copy_release_adapter(src, dst_root)
    assert (target / "adapter_model.safetensors").exists()


# ---------------------------------------------------------------------------
# 4) Checksums
# ---------------------------------------------------------------------------


def test_checksums_correct_file_passes(tmp_path: Path):
    """生成 inventory → 写 checksums → 重新 verify ⇒ PASS。"""
    from data_pipeline.training.phase5e import (
        build_checksums, build_file_inventory,
        verify_checksums, write_checksums,
    )

    (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bbb", encoding="utf-8")
    inv = build_file_inventory(tmp_path)
    write_checksums(tmp_path / "checksums.sha256", build_checksums(inv))
    ok, reasons = verify_checksums(tmp_path / "checksums.sha256")
    assert ok, reasons
    assert reasons == []


def test_checksums_modified_file_fails(tmp_path: Path):
    """checksum 写完后篡改文件 ⇒ verify_checksums 必返回 ok=False。"""
    from data_pipeline.training.phase5e import (
        build_checksums, build_file_inventory,
        verify_checksums, write_checksums,
    )

    target = tmp_path / "a.txt"
    target.write_text("aaa", encoding="utf-8")
    inv = build_file_inventory(tmp_path)
    write_checksums(tmp_path / "checksums.sha256", build_checksums(inv))
    # tamper
    target.write_text("aaaa", encoding="utf-8")
    ok, reasons = verify_checksums(tmp_path / "checksums.sha256")
    assert not ok
    assert any("mismatch" in r for r in reasons)


def test_checksums_missing_file_fails(tmp_path: Path):
    """checksums 列出某文件，但文件已被删除 ⇒ verify_checksums 必返回 ok=False。"""
    from data_pipeline.training.phase5e import (
        build_checksums, build_file_inventory,
        verify_checksums, write_checksums,
    )

    (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bbb", encoding="utf-8")
    inv = build_file_inventory(tmp_path)
    write_checksums(tmp_path / "checksums.sha256", build_checksums(inv))
    # delete
    (tmp_path / "b.txt").unlink()
    ok, reasons = verify_checksums(tmp_path / "checksums.sha256")
    assert not ok
    assert any("missing" in r for r in reasons)


def test_checksums_format_is_sha256sum_compatible(tmp_path: Path):
    """checksums 行格式必须为 ``<sha>  <relpath>``。"""
    from data_pipeline.training.phase5e import build_checksums, build_file_inventory

    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    inv = build_file_inventory(tmp_path)
    body = build_checksums(inv)
    assert body.endswith("\n")
    for line in body.strip().splitlines():
        assert re.match(r"^[0-9a-f]{64}  \S+$", line)


# ---------------------------------------------------------------------------
# 5) No merged model artifacts
# ---------------------------------------------------------------------------


def test_is_forbidden_release_artifact_detects_merged():
    from data_pipeline.training.phase5e import is_forbidden_release_artifact

    assert is_forbidden_release_artifact("merged_model.bin")
    assert is_forbidden_release_artifact("pytorch_model-00001-of-00002.bin")
    assert is_forbidden_release_artifact("model-00001-of-00002.safetensors")
    assert is_forbidden_release_artifact("something.gguf")
    # not forbidden
    assert not is_forbidden_release_artifact("adapter_model.safetensors")
    assert not is_forbidden_release_artifact("tokenizer.json")


def test_forbidden_patterns_excludes_gguf_merged_etc():
    """FORBIDDEN_RELEASE_ARTIFACT_PATTERNS 列出覆盖 merge / gguf / 等。"""
    from data_pipeline.training.phase5e import FORBIDDEN_RELEASE_ARTIFACT_PATTERNS

    pats = " ".join(FORBIDDEN_RELEASE_ARTIFACT_PATTERNS)
    assert "gguf" in pats
    assert "merged" in pats
    assert "pytorch_model" in pats


# ---------------------------------------------------------------------------
# 6) Release inventory
# ---------------------------------------------------------------------------


def test_release_inventory_required_files(tmp_path: Path):
    """build_file_inventory 必须能枚举 release 目录中所有稳定文件。"""
    from data_pipeline.training.phase5e import build_file_inventory

    release = tmp_path / "release"
    release.mkdir()
    (release / "adapter").mkdir()
    (release / "adapter" / "adapter_model.safetensors").write_bytes(b"\x00\x01\x02")
    (release / "MODEL_CARD.md").write_text("# mc\n", encoding="utf-8")
    (release / "release_manifest.json").write_text(
        json.dumps({"x": 1}), encoding="utf-8",
    )
    (release / "evaluation_summary.json").write_text(
        json.dumps({"x": 1}), encoding="utf-8",
    )
    (release / "training_summary.json").write_text(
        json.dumps({"x": 1}), encoding="utf-8",
    )
    (release / "RELEASE_README.md").write_text("# readme\n", encoding="utf-8")
    (release / "checksums.sha256").write_text("dummy\n", encoding="utf-8")
    inv = build_file_inventory(release)
    paths = sorted(item["path"] for item in inv)
    expected = [
        "MODEL_CARD.md",
        "RELEASE_README.md",
        "adapter/adapter_model.safetensors",
        "checksums.sha256",
        "evaluation_summary.json",
        "release_manifest.json",
        "training_summary.json",
    ]
    assert paths == expected
    for item in inv:
        assert "sha256" in item
        assert len(item["sha256"]) == 64
        assert item["size_bytes"] > 0


# ---------------------------------------------------------------------------
# 7) Final gate
# ---------------------------------------------------------------------------


def test_final_gate_pass_when_all_pass(tmp_path: Path):
    from data_pipeline.training.phase5e import (
        AdapterProvenance, FinalGateResult, Phase5DVerification,
        final_gate,
    )
    p5d = Phase5DVerification(
        phase5d_report_path="/tmp", status="PASS", code="PASS",
        ready_for_phase5e=True, test_count=79, predictions_rows=79,
        base_test_loss=1.9, finetuned_test_loss=1.2,
        test_loss_improvement=0.7, test_loss_improvement_pct=36.8,
        protocol_frozen=True, eval_protocol_signature="x",
        model_name="Qwen/Qwen3-8B",
        model_revision="b968826d9c46dd6066d109eabc6255188de91218",
        tokenizer_signature="t",
        generation_max_new_tokens=1610,
        test_jsonl_sha256="x", test_sample_id_signature="y",
        best_adapter_signature_sha256="z", expected_lora_layers=252,
        pass_=True,
    )
    prov = AdapterProvenance(
        source_path="/s", checkpoint_path="/c",
        source_signature="x", checkpoint_signature="x",
        n_tensors=504, equal=True,
    )
    manifest_path = tmp_path / "release_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    res = final_gate(
        phase5d_verification=p5d,
        adapter_provenance=prov,
        release_manifest_path=manifest_path,
        checksums_verified=True,
        fresh_reload_passed=True,
        inference_non_empty=3, inference_total=3,
    )
    assert res.code == "PASS"
    assert res.project_complete is True


def test_final_gate_block_on_test_loss(tmp_path: Path):
    from data_pipeline.training.phase5e import (
        AdapterProvenance, Phase5DVerification, final_gate,
    )
    p5d = Phase5DVerification(
        phase5d_report_path="/tmp", status="PASS", code="PASS",
        ready_for_phase5e=True, test_count=79, predictions_rows=79,
        base_test_loss=1.2, finetuned_test_loss=1.9,  # 退化
        test_loss_improvement=-0.7, test_loss_improvement_pct=-57.6,
        protocol_frozen=True, eval_protocol_signature="x",
        model_name="Qwen/Qwen3-8B",
        model_revision="b968826d9c46dd6066d109eabc6255188de91218",
        tokenizer_signature="t",
        generation_max_new_tokens=1610,
        test_jsonl_sha256="x", test_sample_id_signature="y",
        best_adapter_signature_sha256="z", expected_lora_layers=252,
        pass_=True,
    )
    prov = AdapterProvenance(
        source_path="/s", checkpoint_path="/c",
        source_signature="x", checkpoint_signature="x",
        n_tensors=504, equal=True,
    )
    manifest_path = tmp_path / "x.json"
    manifest_path.write_text("{}", encoding="utf-8")
    res = final_gate(
        phase5d_verification=p5d,
        adapter_provenance=prov,
        release_manifest_path=manifest_path,
        checksums_verified=True,
        fresh_reload_passed=True,
        inference_non_empty=3, inference_total=3,
    )
    assert res.code == "BLOCKED"
    assert res.project_complete is False


def test_final_gate_block_on_adapter_provenance_fail(tmp_path: Path):
    from data_pipeline.training.phase5e import (
        AdapterProvenance, Phase5DVerification, final_gate,
    )
    p5d = Phase5DVerification(
        phase5d_report_path="/tmp", status="PASS", code="PASS",
        ready_for_phase5e=True, test_count=79, predictions_rows=79,
        base_test_loss=1.9, finetuned_test_loss=1.2,
        test_loss_improvement=0.7, test_loss_improvement_pct=36.8,
        protocol_frozen=True, eval_protocol_signature="x",
        model_name="Qwen/Qwen3-8B",
        model_revision="b968826d9c46dd6066d109eabc6255188de91218",
        tokenizer_signature="t",
        generation_max_new_tokens=1610,
        test_jsonl_sha256="x", test_sample_id_signature="y",
        best_adapter_signature_sha256="z", expected_lora_layers=252,
        pass_=True,
    )
    prov = AdapterProvenance(
        source_path="/s", checkpoint_path="/c",
        source_signature="x", checkpoint_signature="y",
        n_tensors=504, equal=False, error="mismatch",
    )
    manifest_path = tmp_path / "x.json"
    manifest_path.write_text("{}", encoding="utf-8")
    res = final_gate(
        phase5d_verification=p5d,
        adapter_provenance=prov,
        release_manifest_path=manifest_path,
        checksums_verified=True,
        fresh_reload_passed=True,
        inference_non_empty=3, inference_total=3,
    )
    assert res.code == "BLOCKED"
    assert any("PROVENANCE" in r for r in res.reasons)


def test_final_gate_block_on_inference(tmp_path: Path):
    from data_pipeline.training.phase5e import (
        AdapterProvenance, Phase5DVerification, final_gate,
    )
    p5d = Phase5DVerification(
        phase5d_report_path="/tmp", status="PASS", code="PASS",
        ready_for_phase5e=True, test_count=79, predictions_rows=79,
        base_test_loss=1.9, finetuned_test_loss=1.2,
        test_loss_improvement=0.7, test_loss_improvement_pct=36.8,
        protocol_frozen=True, eval_protocol_signature="x",
        model_name="Qwen/Qwen3-8B",
        model_revision="b968826d9c46dd6066d109eabc6255188de91218",
        tokenizer_signature="t",
        generation_max_new_tokens=1610,
        test_jsonl_sha256="x", test_sample_id_signature="y",
        best_adapter_signature_sha256="z", expected_lora_layers=252,
        pass_=True,
    )
    prov = AdapterProvenance(
        source_path="/s", checkpoint_path="/c",
        source_signature="x", checkpoint_signature="x",
        n_tensors=504, equal=True,
    )
    manifest_path = tmp_path / "x.json"
    manifest_path.write_text("{}", encoding="utf-8")
    res = final_gate(
        phase5d_verification=p5d,
        adapter_provenance=prov,
        release_manifest_path=manifest_path,
        checksums_verified=True,
        fresh_reload_passed=True,
        inference_non_empty=2, inference_total=3,  # 不全 non-empty
    )
    assert res.code == "BLOCKED"
    assert any("INFERENCE" in r for r in res.reasons)


# ---------------------------------------------------------------------------
# 8) Phase 5E module: no training
# ---------------------------------------------------------------------------


def test_phase5e_module_does_not_export_training():
    """phase5e 模块不得导出任何训练入口。"""
    from data_pipeline.training import phase5e
    forbidden = {"train", "trainer_train", "optimizer_step", "loss_backward"}
    assert forbidden.isdisjoint(set(getattr(phase5e, "__all__", [])))


def test_phase5e_module_does_not_import_trainer():
    """phase5e 模块 source 不得 import trainer / optimizer / scheduler。"""
    import ast
    p = Path("src/data_pipeline/training/phase5e.py")
    source = p.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # 移除 docstring 节点，避免误命中叙述性文字。
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
    cleaned = ast.unparse(tree)
    for needle in (
        "from trl import",
        "trainer.train",
        "optimizer.step",
        "scheduler.step",
        "loss.backward",
        "model.train(",
    ):
        assert needle not in cleaned, f"phase5e.py 禁止 {needle!r}"


def test_phase5e_module_does_not_use_gpu_model_loading():
    """phase5e 模块不得 import transformers / peft / torch model。"""
    import ast
    p = Path("src/data_pipeline/training/phase5e.py")
    source = p.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
    cleaned = ast.unparse(tree)
    for needle in ("from transformers import", "from peft import", "from unsloth import"):
        assert needle not in cleaned, f"phase5e.py 禁止 {needle!r}"


def test_finalizer_does_not_call_train():
    """finalize_qlora_release.py 不得出现 trainer.train / model.train() 调用。"""
    import ast
    p = Path("scripts/finalize_qlora_release.py")
    source = p.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
    cleaned = ast.unparse(tree)
    for needle in ("trainer.train", "optimizer.step", "loss.backward",
                   ".backward()", "scheduler.step"):
        assert needle not in cleaned, f"finalizer 禁止 {needle!r}"


# ---------------------------------------------------------------------------
# 9) Best adapter provenance
# ---------------------------------------------------------------------------


def test_verify_best_adapter_provenance_passes_with_equal(tmp_path: Path):
    """source 与 checkpoint 数值一致 ⇒ equal=True。"""
    from data_pipeline.training.phase5e import verify_best_adapter_provenance

    src = tmp_path / "src"
    _make_fake_adapter_dir(src, n_lora_layers=3)
    # 复制到 checkpoint
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    shutil.copy2(
        src / "adapter_model.safetensors",
        ckpt / "adapter_model.safetensors",
    )
    prov = verify_best_adapter_provenance(
        best_adapter_path=src / "adapter_model.safetensors",
        checkpoint_path=ckpt / "adapter_model.safetensors",
    )
    assert prov.equal is True
    assert prov.n_tensors == 6  # 3 layers × 2
    assert prov.source_signature == prov.checkpoint_signature


def test_verify_best_adapter_provenance_fails_when_diff(tmp_path: Path):
    """source 与 checkpoint 数值不一致 ⇒ equal=False。"""
    from data_pipeline.training.phase5e import verify_best_adapter_provenance

    src = tmp_path / "src"
    _make_fake_adapter_dir(src, n_lora_layers=3, tensor_values=[1.0, 2.0, 3.0])
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _make_fake_safetensors(
        ckpt / "adapter_model.safetensors",
        tensors={
            "layer_0.lora_A": [9.0, 9.0, 9.0],
            "layer_0.lora_B": [9.0, 9.0, 9.0],
            "layer_1.lora_A": [9.0, 9.0, 9.0],
            "layer_1.lora_B": [9.0, 9.0, 9.0],
            "layer_2.lora_A": [9.0, 9.0, 9.0],
            "layer_2.lora_B": [9.0, 9.0, 9.0],
        },
    )
    prov = verify_best_adapter_provenance(
        best_adapter_path=src / "adapter_model.safetensors",
        checkpoint_path=ckpt / "adapter_model.safetensors",
    )
    assert prov.equal is False


def test_verify_best_adapter_provenance_fails_when_missing(tmp_path: Path):
    """source 或 checkpoint 缺失 ⇒ equal=False 且有 error。"""
    from data_pipeline.training.phase5e import verify_best_adapter_provenance

    prov = verify_best_adapter_provenance(
        best_adapter_path=tmp_path / "missing" / "x.safetensors",
        checkpoint_path=tmp_path / "missing" / "y.safetensors",
    )
    assert prov.equal is False
    assert prov.error is not None


# ---------------------------------------------------------------------------
# 10) Training / Evaluation summary builders
# ---------------------------------------------------------------------------


def test_training_summary_reads_real_artifacts():
    """build_training_summary 必须能从真实 phase5c_b artifacts 读出。"""
    from data_pipeline.training.phase5e import build_training_summary

    if not (Path("data/training/phase5c_b/run_manifest.json")).exists():
        pytest.skip("phase5c_b artifacts not present")
    s = build_training_summary()
    assert s["training"]["epochs"] == 3
    assert s["training"]["global_steps"] >= 1
    assert s["lora"]["r"] == 16
    assert s["lora"]["alpha"] == 32
    assert s["lora"]["dropout"] == 0.05
    assert s["validation"]["best_epoch"] == 2
    assert s["validation"]["best_checkpoint"] == "checkpoint-288"


def test_evaluation_summary_reads_real_artifacts():
    """build_evaluation_summary 必须能从真实 phase5d artifacts 读出。"""
    from data_pipeline.training.phase5e import build_evaluation_summary

    if not (Path("data/training/phase5d/phase5d_report.json")).exists():
        pytest.skip("phase5d artifacts not present")
    s = build_evaluation_summary()
    assert s["test_count"] == 79
    assert s["phase5d_code"] == "PASS"
    assert s["phase5d_status"] == "PASS"
    assert s["ready_for_phase5e"] is True
    assert s["loss_improvement"] > 0
    assert s["loss_improvement_pct"] > 0
    assert s["paired_char_f1"]["n"] == 79
    assert s["paired_rouge_l_f1"]["n"] == 79
    assert s["finetuned"]["mean_char_f1"] > s["base"]["mean_char_f1"]
    assert s["finetuned"]["mean_rouge_l_f1"] > s["base"]["mean_rouge_l_f1"]


def test_training_summary_missing_artifact_raises():
    from data_pipeline.training.phase5e import build_training_summary
    with pytest.raises(Exception):
        build_training_summary(
            run_manifest_path=Path("/tmp/none.json"),
            final_report_path=Path("/tmp/none2.json"),
        )


def test_evaluation_summary_missing_artifact_raises(tmp_path: Path):
    from data_pipeline.training.phase5e import build_evaluation_summary
    with pytest.raises(Exception):
        build_evaluation_summary(phase5d_dir=tmp_path / "no_phase5d")


# ---------------------------------------------------------------------------
# 11) Phase 5E module exported names
# ---------------------------------------------------------------------------


def test_phase5e_exports_required_symbols():
    """phase5e 模块必须导出关键的 dataclass + builder 名称。"""
    from data_pipeline.training import phase5e
    required = {
        "validate_phase5d_report",
        "verify_best_adapter_provenance",
        "build_release_manifest",
        "build_training_summary",
        "build_evaluation_summary",
        "build_file_inventory",
        "build_checksums",
        "verify_checksums",
        "copy_release_adapter",
        "final_gate",
        "write_release_manifest",
        "write_training_summary",
        "write_evaluation_summary",
        "write_checksums",
        "PHASE5E_BASE_MODEL_NAME",
        "PHASE5E_BASE_REVISION",
        "PHASE5E_TOKENIZER_SIGNATURE",
        "PHASE5E_BEST_ADAPTER_SOURCE",
        "PHASE5E_BEST_CHECKPOINT_SOURCE",
        "PHASE5E_RELEASE_ROOT",
        "PHASE5E_RUNTIME_ARTIFACT_DIR",
        "Phase5DVerification",
        "AdapterProvenance",
        "FinalGateResult",
        "ReleaseManifest",
        "FORBIDDEN_RELEASE_ARTIFACT_PATTERNS",
        "is_forbidden_release_artifact",
    }
    exported = set(getattr(phase5e, "__all__", []))
    missing = required - exported
    assert not missing, f"phase5e 缺少导出: {sorted(missing)}"