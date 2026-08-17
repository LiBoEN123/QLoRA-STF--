"""Phase 4 — SFT 数据集构造与划分的单元测试。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from data_pipeline.sft import run_sft_pipeline
from data_pipeline.sft.models import SFTBuildConfig, SFTSample
from data_pipeline.sft.pipeline import (
    _bucket_to_split,
    _build_sample,
    _dedup_and_conflicts,
    _filter_faq_units,
    _make_sample_id,
    _make_split_group_key,
    _validate_sample_messages,
    _validate_unit_basic,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _faq_unit(
    unit_id: str,
    question: str,
    answer: str,
    *,
    unit_type: str = "faq",
    faq_confidence: str = "high",
    source_doc_ids=None,
    source_record_indices=None,
    source_path: str = "test.docx",
    file_name: str = "test.docx",
    file_sha256: str = "deadbeef",
    structure_path=None,
    detected_section: str | None = None,
    metadata=None,
) -> dict:
    """构造一个测试用 SemanticUnit 风格 dict。"""
    return {
        "unit_id": unit_id,
        "unit_type": unit_type,
        "source_path": source_path,
        "file_name": file_name,
        "source_type": "docx",
        "file_sha256": file_sha256,
        "structure_path": structure_path or [],
        "detected_section": detected_section,
        "source_doc_ids": source_doc_ids or ["doc_1"],
        "source_record_indices": source_record_indices or [0],
        "text": f"问题：{question}\n\n答案：{answer}",
        "question_text": question,
        "answer_text": answer,
        "raw_question_text": question,
        "faq_confidence": faq_confidence,
        "char_count": len(question) + len(answer),
        "metadata": metadata or {},
    }


@pytest.fixture
def tmp_outputs(tmp_path: Path) -> dict[str, Path]:
    """一组临时输出路径。"""
    return {
        "input": tmp_path / "semantic_units.jsonl",
        "all": tmp_path / "all.jsonl",
        "train": tmp_path / "train.jsonl",
        "validation": tmp_path / "validation.jsonl",
        "test": tmp_path / "test.jsonl",
        "rejected": tmp_path / "rejected.jsonl",
        "report": tmp_path / "dataset_report.json",
    }


def _write_units(path: Path, units: list[dict]) -> None:
    """把 unit dict 列表写入 JSONL。"""
    path.write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in units) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# FAQ → messages
# ---------------------------------------------------------------------------


def test_faq_to_messages_basic():
    """FAQ 应被转成 user + assistant 两条 message。"""
    unit = _faq_unit("u1", "什么是基层党组织？", "基层党组织是党在社会基层组织中的战斗堡垒。")
    sample = _build_sample(unit)
    assert len(sample.messages) == 2
    assert sample.messages[0]["role"] == "user"
    assert sample.messages[1]["role"] == "assistant"


def test_no_question_or_answer_prefix():
    """不应自动给 user/assistant content 加上'问题：'/'答案：'前缀。"""
    unit = _faq_unit("u1", "什么是党课？", "党课是中国共产党对党员进行教育的课程。")
    sample = _build_sample(unit)
    assert not sample.messages[0]["content"].startswith("问题：")
    assert not sample.messages[1]["content"].startswith("答案：")


def test_original_text_preserved_in_messages():
    """user/assistant content 应严格等于 ``question_text.strip()`` / ``answer_text.strip()``。"""
    unit = _faq_unit("u1", "  党员应履行哪些义务？  ", "  党员应……  ")
    sample = _build_sample(unit)
    assert sample.messages[0]["content"] == "党员应履行哪些义务？"
    assert sample.messages[1]["content"] == "党员应……"


def test_metadata_kept_outside_messages():
    """metadata 应在 messages 之外，且不污染 content。"""
    unit = _faq_unit(
        "u1", "Q?", "A.",
        source_path="abc.docx",
        source_doc_ids=["doc_x"],
        source_record_indices=[7],
    )
    sample = _build_sample(unit)
    assert sample.metadata["source_path"] == "abc.docx"
    assert sample.metadata["source_doc_ids"] == ["doc_x"]
    assert sample.metadata["source_record_indices"] == [7]
    for m in sample.messages:
        assert "source_path" not in m["content"]
        assert "doc_x" not in m["content"]


# ---------------------------------------------------------------------------
# 只消费 FAQ
# ---------------------------------------------------------------------------


def test_only_faq_units_pass_filter():
    """非 FAQ 类型（heading / list_group / paragraph_group / subheading）应被过滤掉。"""
    units = [
        _faq_unit("u1", "Q?", "A."),
        _faq_unit("u2", "Q2?", "A2."),
        _faq_unit("u3", "Q3?", "A3.", unit_type="heading"),
        _faq_unit("u4", "Q4?", "A4.", unit_type="list_group"),
        _faq_unit("u5", "Q5?", "A5.", unit_type="paragraph_group"),
        _faq_unit("u6", "Q6?", "A6.", unit_type="subheading"),
    ]
    faqs = _filter_faq_units(units, "medium")
    assert len(faqs) == 2
    assert {u["unit_id"] for u in faqs} == {"u1", "u2"}


def test_confidence_filter():
    """low 不应进入默认 medium 阈值。"""
    units = [
        _faq_unit("h", "Q1", "A1", faq_confidence="high"),
        _faq_unit("m", "Q2", "A2", faq_confidence="medium"),
        _faq_unit("l", "Q3", "A3", faq_confidence="low"),
    ]
    faqs = _filter_faq_units(units, "medium")
    assert {u["unit_id"] for u in faqs} == {"h", "m"}
    faqs = _filter_faq_units(units, "high")
    assert {u["unit_id"] for u in faqs} == {"h"}


# ---------------------------------------------------------------------------
# 字段校验
# ---------------------------------------------------------------------------


def test_empty_question_rejected():
    """question 为空 → missing reason。"""
    unit = _faq_unit("u1", "", "answer")
    reasons = _validate_unit_basic(unit)
    assert "empty_question" in reasons


def test_empty_answer_rejected():
    """answer 为空 → missing reason。"""
    unit = _faq_unit("u1", "question", "")
    reasons = _validate_unit_basic(unit)
    assert "empty_answer" in reasons


def test_whitespace_only_rejected():
    """只含空白的 question / answer 也应被识别为 empty。"""
    unit = _faq_unit("u1", "   \n  ", "answer")
    assert "empty_question" in _validate_unit_basic(unit)
    unit = _faq_unit("u1", "question", "  \t  ")
    assert "empty_answer" in _validate_unit_basic(unit)


def test_invalid_confidence_rejected():
    """非法置信度字符串。"""
    unit = _faq_unit("u1", "Q", "A", faq_confidence="unknown")
    assert "invalid_confidence" in _validate_unit_basic(unit)


def test_missing_traceability_rejected():
    """缺 source_doc_ids / source_record_indices。"""
    unit = _faq_unit("u1", "Q", "A", source_doc_ids=[], source_record_indices=[])
    # fixture 会在空列表时回退成默认值，直接覆盖 unit 字段
    unit["source_doc_ids"] = []
    unit["source_record_indices"] = []
    reasons = _validate_unit_basic(unit)
    assert "missing_traceability" in reasons


def test_missing_unit_id_rejected():
    """空 unit_id。"""
    unit = _faq_unit("", "Q", "A")
    assert "missing_unit_id" in _validate_unit_basic(unit)


# ---------------------------------------------------------------------------
# sample_id 稳定
# ---------------------------------------------------------------------------


def test_sample_id_stable():
    """同一 (unit_id, question, answer) 两次构造应得到相同 sample_id。"""
    s1 = _make_sample_id("u1", "Q", "A")
    s2 = _make_sample_id("u1", "Q", "A")
    assert s1 == s2
    assert s1.startswith("sft_")
    assert len(s1) == 4 + 24  # "sft_" + 24 hex


def test_sample_id_changes_with_input():
    """不同输入应得到不同 sample_id。"""
    a = _make_sample_id("u1", "Q1", "A")
    b = _make_sample_id("u1", "Q2", "A")
    c = _make_sample_id("u2", "Q1", "A")
    d = _make_sample_id("u1", "Q1", "B")
    assert len({a, b, c, d}) == 4


def test_pipeline_sample_id_stable_across_runs(tmp_outputs: dict[str, Path]):
    """同一输入 + 同 config 跑两次，每个 sample_id 必须一致。"""
    units = [
        _faq_unit(f"u{i}", f"问题{i}", f"答案{i}") for i in range(10)
    ]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    ids_run1 = [json.loads(l)["sample_id"] for l in tmp_outputs["all"].read_text(encoding="utf-8").splitlines() if l]
    # 第二次
    run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    ids_run2 = [json.loads(l)["sample_id"] for l in tmp_outputs["all"].read_text(encoding="utf-8").splitlines() if l]
    assert ids_run1 == ids_run2


# ---------------------------------------------------------------------------
# Exact Dedup
# ---------------------------------------------------------------------------


def test_exact_duplicate_collapse(tmp_outputs: dict[str, Path]):
    """question + answer 完全相同的两条 unit，accepted 只保留 1 条；其余 rejected。"""
    units = [
        _faq_unit("u1", "什么是A？", "A 的定义。"),
        _faq_unit("u2", "什么是A？", "A 的定义。"),  # exact dup
    ]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    result = run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    assert result.accepted_samples == 1
    assert result.duplicate_samples == 1
    assert result.rejected_samples == 1

    rejected = [json.loads(l) for l in tmp_outputs["rejected"].read_text(encoding="utf-8").splitlines() if l]
    assert "duplicate_sample" in rejected[0]["reject_reasons"]
    assert "canonical_sample_id" in rejected[0]


def test_exact_duplicate_canonical_stable(tmp_outputs: dict[str, Path]):
    """相同 (q, a) 的多条样本，canonical 必须稳定（按 sample_id 升序）。"""
    # 故意让 unit_id 顺序与 sample_id 顺序相反
    units = [
        _faq_unit("zzz", "Q?", "A."),
        _faq_unit("aaa", "Q?", "A."),
        _faq_unit("mmm", "Q?", "A."),
    ]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    accepted = [json.loads(l) for l in tmp_outputs["all"].read_text(encoding="utf-8").splitlines() if l]
    assert len(accepted) == 1
    # canonical 应是 sample_id 字典序最小的（即 unit_id 最小的 aaa 对应）
    all_sids = []
    for s in (
        [_make_sample_id("zzz", "Q?", "A."), _make_sample_id("aaa", "Q?", "A."), _make_sample_id("mmm", "Q?", "A.")]
    ):
        all_sids.append(s)
    canonical = min(all_sids)
    assert accepted[0]["sample_id"] == canonical


def test_exact_duplicate_uses_normalized_text():
    """内容带不同空白 / 换行也应被识别为相同。"""
    a = _faq_unit("u1", "什么是A？", "A 的定义。")
    b = _faq_unit("u2", "  什么是A？\n  ", "  A 的定义。\n  ")
    pairs = [(a, _build_sample(a)), (b, _build_sample(b))]
    accepted, rejected, dup, _, _ = _dedup_and_conflicts(pairs)
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert dup == 1


# ---------------------------------------------------------------------------
# Question Conflict
# ---------------------------------------------------------------------------


def test_question_conflict_excluded(tmp_outputs: dict[str, Path]):
    """同问不同答 ⇒ 全部进入 rejected，不进入训练集。"""
    units = [
        _faq_unit("u1", "什么是A？", "答案 1"),
        _faq_unit("u2", "什么是A？", "答案 2"),
    ]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    result = run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    assert result.accepted_samples == 0
    assert result.rejected_samples == 2
    assert result.conflicting_question_groups == 1
    assert result.conflicting_samples == 2

    rejected = [json.loads(l) for l in tmp_outputs["rejected"].read_text(encoding="utf-8").splitlines() if l]
    assert all("conflicting_answer" in r["reject_reasons"] for r in rejected)


def test_conflict_with_dup(tmp_outputs: dict[str, Path]):
    """同问不同答 + 同时存在完全重复：先 dedup，再冲突；最终全 rejected。"""
    units = [
        _faq_unit("u1", "什么是A？", "答案 1"),
        _faq_unit("u2", "什么是A？", "答案 1"),  # dup of u1
        _faq_unit("u3", "什么是A？", "答案 2"),
    ]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    result = run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    # accepted = 0；rejected = 1 (dup) + 2 (conflict) = 3
    assert result.accepted_samples == 0
    assert result.rejected_samples == 3
    assert result.duplicate_samples == 1
    assert result.conflicting_question_groups == 1
    assert result.conflicting_samples == 2


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------


def test_split_deterministic_same_input(tmp_outputs: dict[str, Path]):
    """同输入 + 同 config 跑两次，split 完全一致。"""
    units = [_faq_unit(f"u{i}", f"问题{i}", f"答案{i}") for i in range(50)]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()

    def _run_once():
        run_sft_pipeline(
            input_path=tmp_outputs["input"],
            output_all=tmp_outputs["all"],
            output_train=tmp_outputs["train"],
            output_validation=tmp_outputs["validation"],
            output_test=tmp_outputs["test"],
            output_rejected=tmp_outputs["rejected"],
            output_report=tmp_outputs["report"],
            config=cfg,
        )
        t = [json.loads(l)["sample_id"] for l in tmp_outputs["train"].read_text(encoding="utf-8").splitlines() if l]
        v = [json.loads(l)["sample_id"] for l in tmp_outputs["validation"].read_text(encoding="utf-8").splitlines() if l]
        s = [json.loads(l)["sample_id"] for l in tmp_outputs["test"].read_text(encoding="utf-8").splitlines() if l]
        return sorted(t), sorted(v), sorted(s)

    a = _run_once()
    b = _run_once()
    assert a == b


def test_split_group_key_does_not_split_same_question():
    """同 normalized question 必须进入同一 split。"""
    cfg = SFTBuildConfig()
    samples = []
    for q in ("什么是A？", "什么是 B ？", "什么是C？", "什么是D？"):
        for variant in (q, f"  {q}  ", f"{q}\n"):
            u = _faq_unit(f"u_{hashlib.md5(variant.encode()).hexdigest()[:6]}", variant, "A.")
            samples.append((u, _build_sample(u)))
    splits = {}
    from data_pipeline.sft.pipeline import _assign_splits
    splits = _assign_splits(samples, cfg)
    # 每个 question 出现的位置应当一致
    by_q: dict[str, set] = {}
    for name, ss in splits.items():
        for s in ss:
            q = s.messages[0]["content"]
            by_q.setdefault(q, set()).add(name)
    for q, parts in by_q.items():
        assert len(parts) == 1, f"Question {q!r} spans multiple splits: {parts}"


def test_split_actual_ratios_close_to_target(tmp_outputs: dict[str, Path]):
    """1000 个不同问题 → 实际 train/val/test 比例应接近 90/5/5。"""
    units = [_faq_unit(f"u{i:04d}", f"问题{i}", f"答案{i}") for i in range(1000)]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    result = run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    total = result.splits["train"] + result.splits["validation"] + result.splits["test"]
    assert total == 1000
    train_ratio = result.splits["train"] / total
    val_ratio = result.splits["validation"] / total
    test_ratio = result.splits["test"] / total
    # 允许 ±0.03 的偏差
    assert 0.87 <= train_ratio <= 0.93
    assert 0.02 <= val_ratio <= 0.08
    assert 0.02 <= test_ratio <= 0.08


def test_split_salt_changes_assignment(tmp_outputs: dict[str, Path]):
    """改变 split_salt 应让划分结果改变（说明划分是真正由 salt 驱动）。"""
    units = [_faq_unit(f"u{i}", f"问题{i}", f"答案{i}") for i in range(50)]
    _write_units(tmp_outputs["input"], units)

    def _train_ids(salt: str) -> list[str]:
        cfg = SFTBuildConfig(split_salt=salt)
        run_sft_pipeline(
            input_path=tmp_outputs["input"],
            output_all=tmp_outputs["all"],
            output_train=tmp_outputs["train"],
            output_validation=tmp_outputs["validation"],
            output_test=tmp_outputs["test"],
            output_rejected=tmp_outputs["rejected"],
            output_report=tmp_outputs["report"],
            config=cfg,
        )
        return [json.loads(l)["sample_id"] for l in tmp_outputs["train"].read_text(encoding="utf-8").splitlines() if l]

    a = set(_train_ids("salt-A"))
    b = set(_train_ids("salt-B"))
    # 至少有一些样本在不同 salt 下进入了不同 split
    assert a != b


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_no_cross_split_leakage(tmp_outputs: dict[str, Path]):
    """所有 split 之间 normalized question / (q+a) 必须互不重叠。"""
    units = [_faq_unit(f"u{i}", f"问题{i}", f"答案{i}") for i in range(200)]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    report = json.loads(tmp_outputs["report"].read_text(encoding="utf-8"))
    assert report["leakage"]["cross_split_question"] == 0
    assert report["leakage"]["cross_split_exact_sample"] == 0


# ---------------------------------------------------------------------------
# SFT messages 校验
# ---------------------------------------------------------------------------


def test_sample_messages_validation():
    """构造出的 sample 必须满足 2 messages + user/assistant 顺序 + content 非空。"""
    unit = _faq_unit("u1", "Q?", "A.")
    sample = _build_sample(unit)
    assert _validate_sample_messages(sample) == []


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_accounting_input_equals_accepted_plus_rejected(tmp_outputs: dict[str, Path]):
    """input_faq_units = accepted + rejected。"""
    units = [
        _faq_unit("u1", "Q1", "A1"),
        _faq_unit("u2", "Q1", "A1"),  # dup
        _faq_unit("u3", "Q2", "A2"),
        _faq_unit("u4", "Q3", "A3"),
        _faq_unit("u5", "Q3", "A4"),  # conflict with u4
        _faq_unit("u6", "", "A6"),    # empty question
        _faq_unit("u7", "Q7", ""),    # empty answer
    ]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    result = run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    assert result.input_faq_units == 7
    assert result.accepted_samples + result.rejected_samples == result.input_faq_units


def test_accounting_accepted_equals_train_plus_val_plus_test(tmp_outputs: dict[str, Path]):
    """accepted = train + validation + test。"""
    units = [_faq_unit(f"u{i}", f"问题{i}", f"答案{i}") for i in range(30)]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    result = run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    assert result.accepted_samples == (
        result.splits["train"] + result.splits["validation"] + result.splits["test"]
    )


# ---------------------------------------------------------------------------
# 原子输出
# ---------------------------------------------------------------------------


def test_outputs_all_present_and_parseable(tmp_outputs: dict[str, Path]):
    """Pipeline 完成后 6 个输出文件都应存在且可解析。"""
    units = [_faq_unit(f"u{i}", f"问题{i}", f"答案{i}") for i in range(10)]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    for path in (
        tmp_outputs["all"],
        tmp_outputs["train"],
        tmp_outputs["validation"],
        tmp_outputs["test"],
        tmp_outputs["rejected"],
        tmp_outputs["report"],
    ):
        assert path.exists()
        # JSONL：每行应可解析
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    json.loads(line)
        else:
            json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 配置文件 / 边界
# ---------------------------------------------------------------------------


def test_config_min_confidence_must_be_high_or_medium():
    """非法 confidence 应抛错。"""
    with pytest.raises(ValueError):
        SFTBuildConfig(min_confidence="low")
    with pytest.raises(ValueError):
        SFTBuildConfig(min_confidence="none")


def test_config_ratios_must_sum_to_one():
    """比例之和必须为 1。"""
    with pytest.raises(ValueError):
        SFTBuildConfig(train_ratio=0.5, validation_ratio=0.3, test_ratio=0.1)
    with pytest.raises(ValueError):
        SFTBuildConfig(train_ratio=0.0, validation_ratio=0.5, test_ratio=0.5)
    with pytest.raises(ValueError):
        SFTBuildConfig(train_ratio=-0.1, validation_ratio=0.5, test_ratio=0.6)


def test_config_split_salt_must_be_nonempty():
    """split_salt 不能为空。"""
    with pytest.raises(ValueError):
        SFTBuildConfig(split_salt="")


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------


def test_traceability_metadata_complete(tmp_outputs: dict[str, Path]):
    """accepted sample 的 metadata 必须保留完整可追溯信息。"""
    units = [
        _faq_unit(
            "u_x",
            "什么是A？",
            "A 的定义。",
            source_path="x/y.docx",
            file_name="y.docx",
            file_sha256="abc",
            source_doc_ids=["doc_1", "doc_2"],
            source_record_indices=[3, 4],
            structure_path=["第一章", "第一节"],
            detected_section="一、设置",
            metadata={"question_record_index": 3, "answer_record_count": 1, "number_style": "arabic_dot"},
        ),
    ]
    _write_units(tmp_outputs["input"], units)
    cfg = SFTBuildConfig()
    run_sft_pipeline(
        input_path=tmp_outputs["input"],
        output_all=tmp_outputs["all"],
        output_train=tmp_outputs["train"],
        output_validation=tmp_outputs["validation"],
        output_test=tmp_outputs["test"],
        output_rejected=tmp_outputs["rejected"],
        output_report=tmp_outputs["report"],
        config=cfg,
    )
    accepted = [json.loads(l) for l in tmp_outputs["all"].read_text(encoding="utf-8").splitlines() if l]
    assert len(accepted) == 1
    meta = accepted[0]["metadata"]
    for k in (
        "unit_id",
        "source_path",
        "file_name",
        "file_sha256",
        "source_doc_ids",
        "source_record_indices",
        "faq_confidence",
        "structure_path",
        "detected_section",
        "question_record_index",
        "answer_record_count",
        "number_style",
    ):
        assert k in meta, f"missing metadata key: {k}"


# ---------------------------------------------------------------------------
# sample_id 与 messages 字段中文注释的源码约定
# ---------------------------------------------------------------------------


def test_sft_models_have_chinese_field_comments():
    """新增 dataclass 字段前必须有中文注释（源代码约定）。"""
    src_dir = Path(__file__).resolve().parents[1] / "src" / "data_pipeline" / "sft"
    targets = {
        "models.py": ["SFTBuildConfig", "SFTSample"],
    }
    for fname, class_names in targets.items():
        text = (src_dir / fname).read_text(encoding="utf-8")
        for cls in class_names:
            # 找类定义范围（不严格 AST，简单 grep 即可）
            cls_idx = text.find(f"class {cls}")
            assert cls_idx >= 0, f"class {cls} not found in {fname}"
            # 类后到下一个顶级 def/class 之前
            tail = text[cls_idx:]
            # 找下一个 class / def 边界
            next_boundary = re.search(r"^class\s+\w+|^def\s+\w+|^@\w+", tail[len(f"class {cls}"):], re.MULTILINE)
            cls_body = tail[: len(f"class {cls}") + (next_boundary.start() if next_boundary else len(tail))]
            for field_name in re.findall(r"^\s{4}(\w+)\s*[:=]", cls_body, re.MULTILINE):
                # 该字段前两行内必须有中文字符
                field_pattern = re.compile(
                    r"(?:^|\n)([^\n]*\n[^\n]*\n)?[ \t]*" + re.escape(field_name) + r"\s*[:=]"
                )
                m = field_pattern.search(cls_body)
                assert m, f"field {field_name!r} not found in {cls}"
                preceding = (m.group(1) or "")
                assert re.search(r"[\u4e00-\u9fff]", preceding), (
                    f"字段 {cls}.{field_name} 前两行缺少中文注释（src: {fname}）"
                )
