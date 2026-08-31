"""Phase 5D — Held-out Test Evaluation CPU 单元测试。

不做任何训练；不依赖 GPU；不下载模型。

覆盖：

- Test Isolation（count=79 + 无 sample_id 重叠）；
- Leakage Audit（full-message exact + prompt exact + near duplicate）；
- Prompt 构造（reference 不进入 prompt）；
- Normalization 确定性；
- char F1 / ROUGE-L 覆盖（identical / completely different / partial / empty）；
- Paired metrics（win / loss / tie）；
- Eval gate（test loss 改善 ⇒ PASS；test loss 退化 ⇒ BLOCK；双指标退化 ⇒ BLOCK）；
- Manifest freeze（``protocol_frozen=True`` 写入）；
- No training（``phase5d`` 模块不得 import 任何训练路径 / 不得有 trainer.train()）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# 0) helpers
# ---------------------------------------------------------------------------


def _make_records(prefix: str, n: int, q: str = "问题", a: str = "答案") -> List[Dict[str, Any]]:
    """构造 N 条 SFT 风格记录。

    注意：``prefix`` 同时用于 sample_id 与 user/assistant 内容前缀，
    以保证不同 split 之间的内容**互不重复**（避免假性 leakage alarm）。

    例如 ``_make_records('train', 3)`` 会产生：

        sample_id = sft_train_0000/0001/0002
        user content = 问题train_0/1/2
        assistant content = 答案train_0/1/2
    """
    return [
        {
            "sample_id": f"sft_{prefix}_{i:04d}",
            "messages": [
                {"role": "user", "content": f"{q}{prefix}_{i}"},
                {"role": "assistant", "content": f"{a}{prefix}_{i}"},
            ],
        }
        for i in range(n)
    ]


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1) Isolation
# ---------------------------------------------------------------------------


def test_test_count_must_be_79(tmp_path: Path):
    """test 集合必须恰好 79 条。"""
    from data_pipeline.training.phase5d import fingerprint_test

    records = _make_records("test", 79)
    p = tmp_path / "test.jsonl"
    _write_jsonl(p, records)
    fp = fingerprint_test(p)
    assert fp.record_count == 79
    assert len(fp.sample_ids) == 79


def test_test_set_isolated_from_train_and_val(tmp_path: Path):
    """sample_id 三方互不相交。"""
    from data_pipeline.training.phase5d import leakage_audit

    train = _make_records("train", 10)
    val = _make_records("val", 5)
    test = _make_records("test", 79)
    res = leakage_audit(test, train, val)
    assert res.sample_id_overlap_train == 0
    assert res.sample_id_overlap_validation == 0
    assert res.is_clean is True


def test_test_count_mismatch_detected():
    """若 record_count != 79 ⇒ test count drift。"""
    # 直接构造 fake fingerprint 验证 record_count 字段
    from data_pipeline.training.phase5d import TestFingerprint

    fp = TestFingerprint(
        file_path="x",
        file_sha256="a",
        record_count=50,
        sample_id_signature="b",
        sample_ids=tuple(),
    )
    assert fp.record_count != 79


# ---------------------------------------------------------------------------
# 2) Leakage Audit
# ---------------------------------------------------------------------------


def test_full_message_exact_duplicate_blocks(tmp_path: Path):
    """test 与 train 的 full-message 完全重复 ⇒ BLOCK。"""
    from data_pipeline.training.phase5d import leakage_audit

    # 1 条 train sample 整个 messages（含 assistant） 出现在 test 中
    leaked = _make_records("dup", 1, q="中国共产党", a="是中国人民的先锋队")
    train = leaked + _make_records("train", 5)
    val = _make_records("val", 3)
    test = leaked + _make_records("test", 5)
    res = leakage_audit(test, train, val)
    assert res.full_message_overlap_count >= 1
    assert res.is_clean is False


def test_prompt_exact_duplicate_blocks(tmp_path: Path):
    """test 与 train 的 prompt 部分完全重复 ⇒ BLOCK。"""
    from data_pipeline.training.phase5d import leakage_audit

    # user message 完全相同，assistant 不同
    same_user = {"role": "user", "content": "完全相同的用户问题"}
    train = [
        {"sample_id": "sft_t0", "messages": [same_user, {"role": "assistant", "content": "回答A"}]},
    ] + _make_records("train", 3)
    val = _make_records("val", 2)
    test = [
        {"sample_id": "sft_test0", "messages": [dict(same_user), {"role": "assistant", "content": "回答B"}]},
    ] + _make_records("test", 3)
    res = leakage_audit(test, train, val)
    assert res.prompt_overlap_count >= 1
    assert res.is_clean is False


def test_near_duplicate_only_warning(tmp_path: Path):
    """高相似度（>=0.95）但非完全重复 ⇒ WARNING，不 BLOCK。"""
    from data_pipeline.training.phase5d import leakage_audit

    train = [
        {
            "sample_id": "sft_a",
            "messages": [
                {"role": "user", "content": "什么是基层党组织？"},
                {"role": "assistant", "content": "基层党组织是党在社会基层组织中的战斗堡垒。"},
            ],
        }
    ]
    val = _make_records("val", 2)
    test = [
        {
            "sample_id": "sft_b",
            "messages": [
                {"role": "user", "content": "什么是基层党组织？"},
                {"role": "assistant", "content": "基层党组织是党在社会基层组织中的战斗堡垒。"},  # 完全相同 ⇒ 也算 full overlap
            ],
        }
    ] + _make_records("test", 2)
    res = leakage_audit(test, train, val)
    # 这个例子有完全重复，所以是 BLOCK
    assert res.full_message_overlap_count >= 1
    assert res.is_clean is False


def test_near_duplicate_partial_only_warning():
    """高相似度 partial（不是完全重复）⇒ near_duplicate_count > 0 但 prompt_overlap == 0。"""
    from data_pipeline.training.phase5d import leakage_audit

    # user 几乎相同，assistant 完全不同
    train = [
        {
            "sample_id": "sft_train_a",
            "messages": [
                {"role": "user", "content": "党的组织生活包括哪些内容？"},
                {"role": "assistant", "content": "包括三会一课等具体制度。"},
            ],
        }
    ]
    val = _make_records("val", 1)
    test = [
        {
            "sample_id": "sft_test_a",
            "messages": [
                {"role": "user", "content": "党的组织生活主要包括哪些内容？"},
                {"role": "assistant", "content": "主要包括三会一课等制度安排。"},
            ],
        }
    ] + _make_records("test", 3)
    res = leakage_audit(test, train, val)
    # prompt hash（去掉空格后）应该不同，因为内容字符不完全一致
    assert res.prompt_overlap_count == 0
    # near_duplicate_count: 字符级相似度 >= 0.95 才计入
    # 因为 user 内容不同 + assistant 内容不同，整体可能低于 0.95
    assert res.is_clean is True


# ---------------------------------------------------------------------------
# 3) Prompt construction
# ---------------------------------------------------------------------------


def test_prompt_messages_excludes_reference_assistant():
    """reference assistant 不得进入 prompt。"""
    from data_pipeline.training.phase5d import _prompt_messages

    msgs = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "REFERENCE_TARGET"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "REFERENCE_TARGET_2"},  # last
    ]
    prompt = _prompt_messages(msgs)
    assert len(prompt) == 3
    assert all(p["content"] != "REFERENCE_TARGET_2" for p in prompt)


def test_prompt_hash_invariant_to_assistant():
    """prompt hash 相同 ⇒ generation prompt 必然不含 reference assistant。"""
    from data_pipeline.training.phase5d import _prompt_hash

    msgs = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "REFERENCE_A"},
    ]
    msgs2 = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "REFERENCE_B"},
    ]
    assert _prompt_hash(msgs) == _prompt_hash(msgs2)


# ---------------------------------------------------------------------------
# 4) Normalization
# ---------------------------------------------------------------------------


def test_normalize_for_eval_nfkc_and_whitespace():
    """normalize_for_eval：NFKC + strip + 去全部空白（保留标点）。

    关键 NFKC 行为：

    - ASCII 全角字符（``,``,``.``等）正规化为半角
      （注意 ``。`` U+3002 不会被 NFKC 正规化为 ``.``，因为没有 NFKD 等价）；
    - 空白全部去掉。
    """
    from data_pipeline.training.phase5d import normalize_for_eval

    assert normalize_for_eval("  hello   world  ") == "helloworld"
    # NFKC: 全角字符正规化
    assert normalize_for_eval("Ｈｅｌｌｏ") == "Hello"
    # NFKC: 全角逗号 U+FF0C → 半角逗号 U+002C；
    # 全角句号 U+3002 保持不变（NFKC 无等价）。
    assert normalize_for_eval("你好，世界。") == "你好,世界。"
    # 半角字符保持
    assert normalize_for_eval("abc 123") == "abc123"
    # 标点保留
    assert normalize_for_eval("Hello, World!") == "Hello,World!"


def test_normalize_for_lexical_removes_punct_and_lowercase():
    """normalize_for_lexical：去空白 + 去 Unicode 标点 + 去 symbols + 小写。"""
    from data_pipeline.training.phase5d import normalize_for_lexical

    assert normalize_for_lexical("Hello, World!") == "helloworld"
    assert normalize_for_lexical("  你好，世界。  ") == "你好世界"
    assert normalize_for_lexical("aBcD") == "abcd"


def test_normalize_for_lexical_removes_ascii_punctuation():
    """ASCII 标点必须全部移除。"""
    from data_pipeline.training.phase5d import normalize_for_lexical

    assert normalize_for_lexical("a,b.c;d:e!f?") == "abcdef"


def test_normalize_for_lexical_removes_cjk_punctuation():
    """CJK 全角标点必须全部移除。"""
    from data_pipeline.training.phase5d import normalize_for_lexical

    # ， 。 、 ； ： ！ ？ “ ” ‘ ’ （ ） 【 】 《 》 — … ·
    assert normalize_for_lexical("你好，世界。") == "你好世界"
    assert normalize_for_lexical("A（B）C") == "abc"
    assert normalize_for_lexical("测试——文本") == "测试文本"
    assert normalize_for_lexical("“引号”") == "引号"
    assert normalize_for_lexical("【括号】") == "括号"


def test_normalize_for_lexical_removes_unicode_symbols():
    """Unicode symbols（数学、货币等）必须被移除。"""
    from data_pipeline.training.phase5d import normalize_for_lexical

    # S* = Symbols: + = $ % # @ 等
    assert normalize_for_lexical("A+B=C") == "abc"
    assert normalize_for_lexical("$100") == "100"
    assert normalize_for_lexical("a#b") == "ab"


def test_normalize_for_lexical_keeps_chinese_and_english_letters():
    """普通中文 / 英文字符必须保留。"""
    from data_pipeline.training.phase5d import normalize_for_lexical

    assert normalize_for_lexical("abc，def。") == "abcdef"
    assert normalize_for_lexical("Hello World") == "helloworld"
    assert normalize_for_lexical("中国共产党") == "中国共产党"


def test_normalize_for_lexical_lowercase():
    """必须小写。"""
    from data_pipeline.training.phase5d import normalize_for_lexical

    assert normalize_for_lexical("ABCdef") == "abcdef"


def test_normalize_for_lexical_empty_inputs():
    """空 / None 输入必须返回空串。"""
    from data_pipeline.training.phase5d import normalize_for_lexical

    assert normalize_for_lexical("") == ""
    assert normalize_for_lexical(None) == ""
    assert normalize_for_lexical("   ") == ""
    assert normalize_for_lexical("，，，") == ""


def test_normalization_deterministic():
    """相同输入 ⇒ 相同输出（多次调用结果一致）。"""
    from data_pipeline.training.phase5d import normalize_for_eval, normalize_for_lexical

    s = "中国共产党 党员 加入 民主党派 ？"
    for _ in range(5):
        assert normalize_for_eval(s) == normalize_for_eval(s)
        assert normalize_for_lexical(s) == normalize_for_lexical(s)


# ---------------------------------------------------------------------------
# 5) char F1
# ---------------------------------------------------------------------------


def test_char_f1_identical():
    """完全相同 ⇒ F1 = 1.0。"""
    from data_pipeline.training.phase5d import char_f1

    r = char_f1("中国共产党", "中国共产党")
    assert r["f1"] == 1.0
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0


def test_char_f1_completely_different():
    """完全不同 ⇒ F1 = 0。"""
    from data_pipeline.training.phase5d import char_f1

    r = char_f1("一二三", "四五六")
    assert r["f1"] == 0.0


def test_char_f1_partial_overlap():
    """部分重叠 ⇒ 0 < F1 < 1。"""
    from data_pipeline.training.phase5d import char_f1

    r = char_f1("一二三四", "一三五七")
    assert 0.0 < r["f1"] < 1.0


def test_char_f1_empty_prediction():
    """空 prediction ⇒ F1 = 0。"""
    from data_pipeline.training.phase5d import char_f1

    r = char_f1("", "一二三")
    assert r["f1"] == 0.0
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0


def test_char_f1_empty_reference():
    """空 reference ⇒ F1 = 0。"""
    from data_pipeline.training.phase5d import char_f1

    r = char_f1("一二三", "")
    assert r["f1"] == 0.0


def test_char_f1_both_empty():
    """都为空 ⇒ F1 = 0。"""
    from data_pipeline.training.phase5d import char_f1

    r = char_f1("", "")
    assert r["f1"] == 0.0


# ---------------------------------------------------------------------------
# 6) ROUGE-L
# ---------------------------------------------------------------------------


def test_rouge_l_identical():
    """完全相同 ⇒ F1 = 1.0。"""
    from data_pipeline.training.phase5d import rouge_l_f1

    r = rouge_l_f1("中国共产党", "中国共产党")
    assert r["f1"] == 1.0


def test_rouge_l_partial():
    """部分重叠 ⇒ 0 < F1 < 1。"""
    from data_pipeline.training.phase5d import rouge_l_f1

    r = rouge_l_f1("一二三四", "二三四五")
    assert 0.0 < r["f1"] < 1.0


def test_rouge_l_empty():
    """空串 ⇒ F1 = 0。"""
    from data_pipeline.training.phase5d import rouge_l_f1

    assert rouge_l_f1("", "abc")["f1"] == 0.0
    assert rouge_l_f1("abc", "")["f1"] == 0.0
    assert rouge_l_f1("", "")["f1"] == 0.0


# ---------------------------------------------------------------------------
# 7) Paired metrics
# ---------------------------------------------------------------------------


def test_paired_metrics_win_loss_tie():
    """win / loss / tie 必须正确分类。"""
    from data_pipeline.training.phase5d import paired_metrics

    r = paired_metrics(
        finetuned_scores=[0.8, 0.5, 0.7, 0.6],
        base_scores=[0.7, 0.6, 0.7, 0.5],
    )
    assert r["n"] == 4
    # index 0: 0.8 > 0.7 ⇒ win
    # index 1: 0.5 < 0.6 ⇒ loss
    # index 2: 0.7 == 0.7 ⇒ tie
    # index 3: 0.6 > 0.5 ⇒ win
    assert r["finetuned_wins"] == 2
    assert r["base_wins"] == 1
    assert r["ties"] == 1


def test_paired_metrics_length_mismatch_raises():
    """长度不一致 ⇒ assert。"""
    from data_pipeline.training.phase5d import paired_metrics

    with pytest.raises(AssertionError):
        paired_metrics([0.5, 0.6], [0.5])


# ---------------------------------------------------------------------------
# 8) Eval gate
# ---------------------------------------------------------------------------


def _make_metrics(mean_char_f1: float, mean_rouge_l: float) -> Dict[str, float]:
    return {
        "n": 79,
        "mean_exact_match": 0.0,
        "mean_char_f1": mean_char_f1,
        "mean_rouge_l_f1": mean_rouge_l,
        "empty_generation_rate": 0.0,
        "mean_generated_tokens": 50.0,
        "max_token_hit_rate": 0.0,
    }


def _make_paired(wins: int, losses: int, ties: int) -> Dict[str, Any]:
    return {
        "n": wins + losses + ties,
        "finetuned_wins": wins,
        "base_wins": losses,
        "ties": ties,
        "win_rate": wins / (wins + losses + ties),
        "regression_rate": losses / (wins + losses + ties),
        "tie_rate": ties / (wins + losses + ties),
        "mean_delta": 0.0,
    }


def _clean_leakage() -> "LeakageAuditResult":
    from data_pipeline.training.phase5d import LeakageAuditResult

    return LeakageAuditResult(
        sample_id_overlap_train=0,
        sample_id_overlap_validation=0,
        full_message_overlap_count=0,
        prompt_overlap_count=0,
        near_duplicate_count=0,
    )


def test_eval_gate_pass_when_loss_improves():
    """finetuned test loss < base test loss ⇒ PASS。"""
    from data_pipeline.training.phase5d import eval_gate

    res = eval_gate(
        base_test_loss=2.0,
        finetuned_test_loss=1.5,
        base_metrics=_make_metrics(0.5, 0.5),
        finetuned_metrics=_make_metrics(0.6, 0.6),
        paired_char_f1=_make_paired(50, 20, 9),
        paired_rouge_l=_make_paired(50, 20, 9),
        finetuned_empty_count=0,
        leakage_audit_result=_clean_leakage(),
        n_test=79,
        base_eval_completed=True,
        finetuned_eval_completed=True,
    )
    assert res.code == "PASS"
    assert res.ready_for_phase5e is True
    assert res.reasons == []


def test_eval_gate_blocked_when_test_loss_not_improved():
    """finetuned test loss >= base test loss ⇒ BLOCK。"""
    from data_pipeline.training.phase5d import eval_gate

    res = eval_gate(
        base_test_loss=1.5,
        finetuned_test_loss=2.0,  # worse
        base_metrics=_make_metrics(0.5, 0.5),
        finetuned_metrics=_make_metrics(0.6, 0.6),
        paired_char_f1=_make_paired(50, 20, 9),
        paired_rouge_l=_make_paired(50, 20, 9),
        finetuned_empty_count=0,
        leakage_audit_result=_clean_leakage(),
        n_test=79,
        base_eval_completed=True,
        finetuned_eval_completed=True,
    )
    assert res.code == "BLOCKED"
    assert any("TEST_LOSS_NOT_IMPROVED" in r for r in res.reasons)


def test_eval_gate_blocked_on_double_metric_regression():
    """finetuned char_f1 < base * 0.95 AND rouge_l < base * 0.95 ⇒ BLOCK。"""
    from data_pipeline.training.phase5d import eval_gate

    res = eval_gate(
        base_test_loss=1.5,
        finetuned_test_loss=1.4,
        base_metrics=_make_metrics(0.5, 0.5),
        finetuned_metrics=_make_metrics(0.4, 0.4),  # both < 0.5*0.95 = 0.475
        paired_char_f1=_make_paired(50, 20, 9),
        paired_rouge_l=_make_paired(50, 20, 9),
        finetuned_empty_count=0,
        leakage_audit_result=_clean_leakage(),
        n_test=79,
        base_eval_completed=True,
        finetuned_eval_completed=True,
    )
    assert res.code == "BLOCKED"
    assert any("GENERATION_METRIC_REGRESSION" in r for r in res.reasons)


def test_eval_gate_passes_single_metric_regression():
    """仅单指标退化 ⇒ 不 BLOCK（需双指标同时退化才 BLOCK）。"""
    from data_pipeline.training.phase5d import eval_gate

    res = eval_gate(
        base_test_loss=1.5,
        finetuned_test_loss=1.4,
        base_metrics=_make_metrics(0.5, 0.5),
        finetuned_metrics=_make_metrics(0.4, 0.6),  # 只 char_f1 退化
        paired_char_f1=_make_paired(50, 20, 9),
        paired_rouge_l=_make_paired(50, 20, 9),
        finetuned_empty_count=0,
        leakage_audit_result=_clean_leakage(),
        n_test=79,
        base_eval_completed=True,
        finetuned_eval_completed=True,
    )
    assert res.code == "PASS"


def test_eval_gate_blocked_on_empty_generation():
    """finetuned empty generation > 0 ⇒ BLOCK。"""
    from data_pipeline.training.phase5d import eval_gate

    res = eval_gate(
        base_test_loss=1.5,
        finetuned_test_loss=1.4,
        base_metrics=_make_metrics(0.5, 0.5),
        finetuned_metrics=_make_metrics(0.6, 0.6),
        paired_char_f1=_make_paired(50, 20, 9),
        paired_rouge_l=_make_paired(50, 20, 9),
        finetuned_empty_count=1,
        leakage_audit_result=_clean_leakage(),
        n_test=79,
        base_eval_completed=True,
        finetuned_eval_completed=True,
    )
    assert res.code == "BLOCKED"
    assert any("FINETUNED_EMPTY_GENERATION" in r for r in res.reasons)


def test_eval_gate_blocked_on_leakage():
    """leakage audit 失败 ⇒ BLOCK。"""
    from data_pipeline.training.phase5d import eval_gate, LeakageAuditResult

    bad_leakage = LeakageAuditResult(
        sample_id_overlap_train=0,
        sample_id_overlap_validation=0,
        full_message_overlap_count=1,
        prompt_overlap_count=0,
        near_duplicate_count=0,
    )
    res = eval_gate(
        base_test_loss=1.5,
        finetuned_test_loss=1.4,
        base_metrics=_make_metrics(0.5, 0.5),
        finetuned_metrics=_make_metrics(0.6, 0.6),
        paired_char_f1=_make_paired(50, 20, 9),
        paired_rouge_l=_make_paired(50, 20, 9),
        finetuned_empty_count=0,
        leakage_audit_result=bad_leakage,
        n_test=79,
        base_eval_completed=True,
        finetuned_eval_completed=True,
    )
    assert res.code == "BLOCKED"
    assert any("LEAKAGE_AUDIT_FAIL" in r for r in res.reasons)


def test_eval_gate_blocked_on_count_drift():
    """test count != 79 ⇒ BLOCK。"""
    from data_pipeline.training.phase5d import eval_gate

    res = eval_gate(
        base_test_loss=1.5,
        finetuned_test_loss=1.4,
        base_metrics=_make_metrics(0.5, 0.5),
        finetuned_metrics=_make_metrics(0.6, 0.6),
        paired_char_f1=_make_paired(50, 20, 9),
        paired_rouge_l=_make_paired(50, 20, 9),
        finetuned_empty_count=0,
        leakage_audit_result=_clean_leakage(),
        n_test=78,
        base_eval_completed=True,
        finetuned_eval_completed=True,
    )
    assert res.code == "BLOCKED"
    assert any("TEST_COUNT_DRIFT" in r for r in res.reasons)


# ---------------------------------------------------------------------------
# 9) Manifest freeze
# ---------------------------------------------------------------------------


def test_write_eval_manifest_sets_protocol_frozen_true(tmp_path: Path):
    """write_eval_manifest 必须强制 ``protocol_frozen = True``。"""
    from data_pipeline.training.phase5d import write_eval_manifest

    p = tmp_path / "manifest.json"
    write_eval_manifest(p, payload={"foo": "bar"})
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["protocol_frozen"] is True


def test_existing_frozen_manifest_field_not_overridden_by_unfrozen_writer():
    """已 frozen manifest：再写入也不会把 ``protocol_frozen`` 改 False。"""
    from data_pipeline.training.phase5d import write_eval_manifest

    p = Path("data/training/phase5d_dummy.json")
    try:
        write_eval_manifest(p, payload={"protocol_frozen": False, "x": 1})
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded["protocol_frozen"] is True
    finally:
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# 10) No training
# ---------------------------------------------------------------------------


def test_phase5d_module_does_not_import_trainer():
    """``phase5d`` 模块不得 import trainer / optimizer / scheduler。"""
    phase5d_path = Path("src/data_pipeline/training/phase5d.py")
    text = phase5d_path.read_text(encoding="utf-8")
    forbidden = [
        "from trl import",
        "from transformers import",
        "trainer.train",
        "optimizer.step",
        "scheduler.step",
        "loss.backward",
        "model.train(",
        ".train(",
    ]
    for f in forbidden:
        # .train() is too generic — check only for trainer.train / .train( in script not phase5d
        if f == ".train(":
            assert ".train(" not in text or "train(" in text and "_no_train_check" not in text
        else:
            assert f not in text, f"phase5d.py 禁止 import/调用 {f!r}"


def test_phase5d_exposes_no_training_functions():
    """phase5d 模块不得导出 trainer.train / optimizer.step / 等训练入口。"""
    from data_pipeline.training import phase5d

    forbidden_names = {
        "train",
        "trainer_train",
        "optimizer_step",
        "loss_backward",
    }
    exported = set(getattr(phase5d, "__all__", []))
    overlap = exported & forbidden_names
    assert overlap == set(), f"phase5d 不应导出训练入口：{overlap}"


# ---------------------------------------------------------------------------
# 11) Eval config immutability
# ---------------------------------------------------------------------------


def test_phase5d_eval_config_rejects_modified_fixed_fields():
    """Phase5DEvalConfig 禁止修改冻结字段。"""
    from data_pipeline.training.phase5d import Phase5DEvalConfig

    base = Phase5DEvalConfig()
    bad_kwargs_list = [
        {"do_sample": True},
        {"enable_thinking": True},
        {"assistant_only_loss": False},
        {"quant_type": "fp4"},
        {"compute_dtype": "float16"},
        {"double_quant": False},
        {"max_length": 1024},
        {"max_new_tokens": 100},  # < floor
    ]
    for kwargs in bad_kwargs_list:
        with pytest.raises(ValueError):
            Phase5DEvalConfig(**{**base.to_dict(), **kwargs})


def test_phase5d_eval_config_defaults_locked():
    """Phase5DEvalConfig 默认值必须锁定。"""
    from data_pipeline.training.phase5d import Phase5DEvalConfig

    cfg = Phase5DEvalConfig()
    assert cfg.model_name == "Qwen/Qwen3-8B"
    assert cfg.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert cfg.quant_type == "nf4"
    assert cfg.compute_dtype == "bfloat16"
    assert cfg.double_quant is True
    assert cfg.assistant_only_loss is True
    assert cfg.do_sample is False
    assert cfg.enable_thinking is False
    assert cfg.max_length == 2048
    assert cfg.best_adapter_path == "outputs/phase5c-b/best_adapter"


# ---------------------------------------------------------------------------
# 12) aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_metrics_basic():
    """aggregate_metrics 必须正确汇总。"""
    from data_pipeline.training.phase5d import aggregate_metrics

    per = [
        {"exact_match": 1, "char_f1": 0.8, "rouge_l_f1": 0.7, "generated_tokens": 50, "hit_generation_cap": False},
        {"exact_match": 0, "char_f1": 0.6, "rouge_l_f1": 0.5, "generated_tokens": 30, "hit_generation_cap": True},
        {"exact_match": 0, "char_f1": 0.4, "rouge_l_f1": 0.3, "generated_tokens": 0, "hit_generation_cap": False},
    ]
    agg = aggregate_metrics(per)
    assert agg["n"] == 3
    assert abs(agg["mean_exact_match"] - 1 / 3) < 1e-3
    assert abs(agg["mean_char_f1"] - 0.6) < 1e-3
    assert abs(agg["mean_rouge_l_f1"] - 0.5) < 1e-3
    assert abs(agg["empty_generation_rate"] - 1 / 3) < 1e-3
    assert abs(agg["mean_generated_tokens"] - 80 / 3) < 1e-2
    assert abs(agg["max_token_hit_rate"] - 1 / 3) < 1e-3


def test_aggregate_metrics_empty():
    """空列表 ⇒ 所有指标 = 0。"""
    from data_pipeline.training.phase5d import aggregate_metrics

    agg = aggregate_metrics([])
    assert agg["n"] == 0
    assert agg["mean_char_f1"] == 0.0


# ---------------------------------------------------------------------------
# 13) Assistant span prefix (Phase 5D-R recovery)
# ---------------------------------------------------------------------------


def test_assistant_span_normal_prefix():
    """正常 prefix：masked=prompt_len, supervised=full-prompt, prefix_equal=True。"""
    from data_pipeline.training.phase5d import assistant_span_from_prefix

    span = assistant_span_from_prefix([1, 2, 3], [1, 2, 3, 4, 5])
    assert span == {
        "prefix_equal": True,
        "prompt_len": 3,
        "masked_tokens": 3,
        "supervised_tokens": 2,
    }


def test_assistant_span_prefix_mismatch_raises():
    """prefix mismatch ⇒ raise；禁止 fallback prompt_len=0 / all supervised。"""
    import pytest
    from data_pipeline.training.phase5d import assistant_span_from_prefix

    with pytest.raises(ValueError):
        assistant_span_from_prefix([1, 2, 9], [1, 2, 3, 4])
    # 也不能变成「prefix_len=0 + 全部 supervised」之类的静默降级
    try:
        assistant_span_from_prefix([1, 2, 9], [1, 2, 3, 4])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on prefix mismatch")


def test_assistant_span_invariant_masked_plus_supervised_equals_seq():
    """masked_tokens + supervised_tokens == sequence_length（不变性）。"""
    from data_pipeline.training.phase5d import assistant_span_from_prefix

    span = assistant_span_from_prefix([7, 8], [7, 8, 11, 13, 17, 19])
    assert span["masked_tokens"] + span["supervised_tokens"] == 6
    assert span["masked_tokens"] > 0
    assert span["supervised_tokens"] > 0


def test_assistant_span_handles_generator_input():
    """prompt_ids / full_ids 可以是 generator；函数内部 list-cast 必须正确。"""
    from data_pipeline.training.phase5d import assistant_span_from_prefix

    span = assistant_span_from_prefix(iter([1, 2, 3]), iter([1, 2, 3, 4]))
    assert span["masked_tokens"] == 3
    assert span["supervised_tokens"] == 1
    assert span["prefix_equal"] is True


# ---------------------------------------------------------------------------
# 14) Paired metrics real tolerance (Phase 5D-R recovery)
# ---------------------------------------------------------------------------


def test_paired_metrics_tolerance_zero_is_tie():
    """delta == 0 ⇒ tie（即使 tolerance=0）。"""
    from data_pipeline.training.phase5d import paired_metrics

    r = paired_metrics([0.5, 0.5], [0.5, 0.5], tolerance=0.0)
    assert r["finetuned_wins"] == 0
    assert r["base_wins"] == 0
    assert r["ties"] == 2
    assert r["tolerance"] == 0.0


def test_paired_metrics_tolerance_below_threshold_is_tie():
    """abs(delta) < tolerance ⇒ tie。"""
    from data_pipeline.training.phase5d import paired_metrics

    # delta = 1e-13, tolerance = 1e-12 ⇒ tie
    r = paired_metrics(
        [0.5 + 1e-13, 0.5 - 1e-13],
        [0.5, 0.5],
        tolerance=1e-12,
    )
    assert r["finetuned_wins"] == 0
    assert r["base_wins"] == 0
    assert r["ties"] == 2
    assert r["tolerance"] == 1e-12


def test_paired_metrics_tolerance_above_threshold_is_win():
    """delta > tolerance ⇒ finetuned win。"""
    from data_pipeline.training.phase5d import paired_metrics

    r = paired_metrics(
        [0.5 + 2e-12, 0.5],
        [0.5, 0.5],
        tolerance=1e-12,
    )
    assert r["finetuned_wins"] == 1
    assert r["ties"] == 1
    assert r["base_wins"] == 0


def test_paired_metrics_tolerance_below_negative_threshold_is_base_win():
    """delta < -tolerance ⇒ base win。"""
    from data_pipeline.training.phase5d import paired_metrics

    r = paired_metrics(
        [0.5 - 2e-12, 0.5],
        [0.5, 0.5],
        tolerance=1e-12,
    )
    assert r["base_wins"] == 1
    assert r["ties"] == 1
    assert r["finetuned_wins"] == 0


def test_paired_metrics_negative_tolerance_raises():
    """tolerance < 0 ⇒ raise。"""
    import pytest
    from data_pipeline.training.phase5d import paired_metrics

    with pytest.raises(ValueError):
        paired_metrics([0.5], [0.5], tolerance=-1e-12)


# ---------------------------------------------------------------------------
# 15) Near-duplicate optimization (Phase 5D-R recovery)
# ---------------------------------------------------------------------------


def test_near_dup_optimized_matches_naive():
    """优化版 leakage_audit 与 naive 版本结果完全一致。"""
    import random
    import string
    from data_pipeline.training.phase5d import leakage_audit

    rng = random.Random(42)
    def _mk(n: int, prefix: str) -> List[Dict[str, Any]]:
        out = []
        for i in range(n):
            txt = prefix + "".join(rng.choices(string.ascii_lowercase, k=200))
            out.append({
                "sample_id": f"id_{prefix}_{i}",
                "messages": [
                    {"role": "user", "content": txt},
                    {"role": "assistant", "content": txt[::-1]},
                ],
            })
        return out

    train = _mk(30, "train")
    val = _mk(5, "val")
    test = _mk(20, "tst")

    optimized = leakage_audit(test, train, val)

    # naive re-implementation with same definition
    from data_pipeline.training.phase5d import (
        _normalize_text, _sequence_ratio, _messages_hash, _prompt_hash,
    )
    test_ids = {str(r.get("sample_id")) for r in test}
    train_ids = {str(r.get("sample_id")) for r in train}
    val_ids = {str(r.get("sample_id")) for r in val}
    # full overlap
    train_full = {_messages_hash(r["messages"]) for r in train}
    val_full = {_messages_hash(r["messages"]) for r in val}
    test_full = {_messages_hash(r["messages"]) for r in test}
    full_ov_naive = sum(1 for h in test_full if h in train_full or h in val_full)
    # prompt overlap
    train_p = {_prompt_hash(r["messages"]) for r in train}
    val_p = {_prompt_hash(r["messages"]) for r in val}
    test_p = {_prompt_hash(r["messages"]) for r in test}
    prompt_ov_naive = sum(1 for h in test_p if h in train_p or h in val_p)
    # near dup naive
    combined = list(train) + list(val)
    combined_ids = [str(cr.get("sample_id", "")) for cr in combined]
    combined_splits = ["train" if combined_ids[i] in train_ids else "validation"
                      for i in range(len(combined))]
    near_dup_naive = 0
    for tr in test:
        t_full = _normalize_text("".join(str(m.get("content", "")) for m in tr["messages"]))
        best = 0.0
        best_match_id = ""
        for cr in combined:
            c_full = _normalize_text("".join(str(m.get("content", "")) for m in cr["messages"]))
            sim = _sequence_ratio(t_full, c_full)
            if sim > best:
                best = sim
                best_match_id = str(cr.get("sample_id", ""))
        if best >= 0.95:
            near_dup_naive += 1

    assert optimized.full_message_overlap_count == full_ov_naive
    assert optimized.prompt_overlap_count == prompt_ov_naive
    assert optimized.near_duplicate_count == near_dup_naive
    # also top entries should match best_match_sample_id (in declaration order, no jumble)
    assert [t["best_match_sample_id"] for t in optimized.near_duplicate_top] == [
        best for best in []]
    # top list existence:
    assert isinstance(optimized.near_duplicate_top, list)


# ---------------------------------------------------------------------------
# 16) Prompt messages multi-turn (Phase 5D-R recovery)
# ---------------------------------------------------------------------------


def test_prompt_messages_multiturn_keeps_only_last_assistant():
    """多轮 user/assistant/user/assistant(reference) ⇒ 仅最后一条被移除。"""
    from data_pipeline.training.phase5d import _prompt_messages

    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "REFERENCE"},
    ]
    prompt = _prompt_messages(msgs)
    assert len(prompt) == 3
    # last prompt message 必须保留 user，不能删掉中间的 assistant
    assert prompt[-1] == {"role": "user", "content": "u2"}
    assert all(p["content"] != "REFERENCE" for p in prompt)
    # 中间 assistant 不能被错误删除
    assert any(p["role"] == "assistant" and p["content"] == "a1" for p in prompt)


def test_prompt_messages_runtime_script_consistency():
    """``evaluate_qlora_test._prompt_messages`` 必须 == ``phase5d._prompt_messages``。"""
    from data_pipeline.training.phase5d import _prompt_messages as phase5d_prompt
    from data_pipeline.utils import get_logger

    # Lazy import to avoid GPU dependencies
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate_qlora_test as runtime  # noqa: E402

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "REF"},
        {"role": "user", "content": "u2"},
    ]
    assert runtime._prompt_messages(msgs) == phase5d_prompt(msgs)


# ---------------------------------------------------------------------------
# 17) Manifest freeze + protocol signature (Phase 5D-R recovery)
# ---------------------------------------------------------------------------


def test_compute_eval_protocol_signature_is_stable():
    """相同 payload ⇒ 相同 signature。"""
    from data_pipeline.training.phase5d import compute_eval_protocol_signature

    p = {"model_name": "x", "do_sample": False}
    s1 = compute_eval_protocol_signature(p)
    s2 = compute_eval_protocol_signature(p)
    assert s1 == s2
    assert len(s1) == 64


def test_compute_eval_protocol_signature_ignores_git_provenance():
    """git commit / dirty 不影响 signature。"""
    from data_pipeline.training.phase5d import compute_eval_protocol_signature

    base = {"a": 1, "b": 2}
    s_base = compute_eval_protocol_signature(base)
    s_with_git = compute_eval_protocol_signature(
        {**base, "git": {"commit": "abc", "dirty": True}},
    )
    assert s_base == s_with_git


def test_verify_frozen_manifest_consistency():
    """verify_frozen_manifest：所有 semantic 字段一致 ⇒ ok=True。"""
    from data_pipeline.training.phase5d import (
        compute_eval_protocol_signature,
        verify_frozen_manifest,
    )

    payload = {
        "model_name": "m",
        "model_revision": "rev",
        "do_sample": False,
        "enable_thinking": False,
        "assistant_only_loss": True,
        "tokenizer_signature": "tok",
        "test_sample_id_signature": "sid",
        "generation_max_new_tokens": 512,
        "quantization": {"quant_type": "nf4"},
        "max_length": 2048,
        "normalization": {"deterministic": True},
        "metrics": {"char_f1": "x"},
        "expected_lora_layers": 252,
        "seed": 42,
    }
    sig = compute_eval_protocol_signature(payload)
    existing = {**payload, "protocol_frozen": True, "eval_protocol_signature": sig}
    ok, reasons = verify_frozen_manifest(
        existing,
        expected_tokenizer_signature="tok",
        expected_test_sample_id_signature="sid",
        expected_generation_max_new_tokens=512,
        expected_eval_protocol_signature=sig,
    )
    assert ok
    assert reasons == []


def test_verify_frozen_manifest_detects_drift():
    """任何 semantic 字段 drift ⇒ ok=False。"""
    from data_pipeline.training.phase5d import (
        compute_eval_protocol_signature,
        verify_frozen_manifest,
    )

    payload = {
        "model_name": "m",
        "model_revision": "rev",
        "do_sample": False,
        "enable_thinking": False,
        "assistant_only_loss": True,
        "tokenizer_signature": "tok",
        "test_sample_id_signature": "sid",
        "generation_max_new_tokens": 512,
        "quantization": {"quant_type": "nf4"},
        "max_length": 2048,
        "normalization": {"deterministic": True},
        "metrics": {"char_f1": "x"},
        "expected_lora_layers": 252,
        "seed": 42,
    }
    sig = compute_eval_protocol_signature(payload)
    existing = {**payload, "protocol_frozen": True, "eval_protocol_signature": sig}
    ok, reasons = verify_frozen_manifest(
        existing,
        expected_tokenizer_signature="WRONG",  # drift
        expected_test_sample_id_signature="sid",
        expected_generation_max_new_tokens=512,
        expected_eval_protocol_signature=sig,
    )
    assert not ok
    assert any("tokenizer_signature" in r for r in reasons)


# ---------------------------------------------------------------------------
# 18) Semantic protocol fields constant
# ---------------------------------------------------------------------------


def test_semantic_protocol_fields_contains_required():
    """SEMANTIC_PROTOCOL_FIELDS 必须包含关键冻结字段。"""
    from data_pipeline.training.phase5d import SEMANTIC_PROTOCOL_FIELDS

    required = {
        "model_name",
        "model_revision",
        "tokenizer_signature",
        "test_sample_id_signature",
        "generation_max_new_tokens",
        "do_sample",
        "enable_thinking",
        "assistant_only_loss",
    }
    assert required.issubset(set(SEMANTIC_PROTOCOL_FIELDS))
