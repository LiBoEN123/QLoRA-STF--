"""Phase 5A — Qwen3 tokenizer 审计的单元测试。

默认 ``pytest -q`` 不联网，使用 ``FakeTokenizer`` 验证纯逻辑；
集成测试（需要 Hugging Face 网络）由 ``RUN_QWEN_TOKENIZER_INTEGRATION=1`` 显式开启。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from data_pipeline.tokenization import run_tokenizer_audit
from data_pipeline.tokenization.models import (
    DEFAULT_CANDIDATE_MAX_LENGTHS,
    TokenizerAuditConfig,
)
from data_pipeline.tokenization.pipeline import (
    _audit_samples,
    _build_split_membership,
    _coverage,
    _describe_tokenizer,
    _length_stats,
    _load_tokenizer,
    _longest_samples,
    _percentile,
    _render_synthetic,
    _scan_source_thinking_markers,
    _smallest_candidate_covering,
    _split_stats,
    _token_lengths,
    _verify_split_membership,
    TokenizerAuditError,
)


# ---------------------------------------------------------------------------
# Fake tokenizer（最小化，足以覆盖审计代码所需接口）
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """极简 tokenizer 替代品；不模拟完整 transformers 接口。

    结构模仿 Qwen3 ChatML：

    - full: ``<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n{assistant}<|im_end|>\\n``
    - prompt（``add_generation_prompt=True``）：``<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n``

    两者共享同一前缀 → ``full[:len(prompt)] == prompt`` 天然成立。
    """

    def __init__(self, eos_token: str = "<eos>", bos_token: str = "<bos>") -> None:
        self.chat_template = "{% for m in messages %}<|im_start|>{{ m['role'] }}\\n{{ m['content'] }}<|im_end|>\\n{% endfor %}"
        self.name_or_path = "fake/qwen3"
        self.vocab_size = 100
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = "<pad>"
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.model_max_length = 32768
        self._piece_to_id: Dict[str, int] = {}

    def _id(self, piece: str) -> int:
        if piece not in self._piece_to_id:
            self._piece_to_id[piece] = len(self._piece_to_id) + 10
        return self._piece_to_id[piece]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        **_: Any,
    ) -> Any:
        del enable_thinking
        pieces: List[str] = []
        # 仅渲染 messages[:1] 跟渲染全部 messages 走相同 user 段，确保 prefix property 成立
        render_messages = messages
        for idx, m in enumerate(render_messages):
            pieces.append(f"<|im_start|>{m['role']}\n")
            pieces.append(m["content"])
            if idx == len(render_messages) - 1 and not add_generation_prompt:
                # 末尾 assistant 在 full 模式下追加 <|im_end|>\n
                pieces.append("<|im_end|>\n")
            else:
                pieces.append("<|im_end|>\n")

        if not tokenize:
            return "".join(pieces)
        return [self._id(p) for p in pieces]

    def decode(self, ids: List[int], *, skip_special_tokens: bool = False) -> str:
        # 不精确还原 piece；只保证 synthetic 内容可识别
        return "测试问题 测试答案"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _sft_sample(
    sample_id: str,
    question: str,
    answer: str,
    *,
    unit_id: str | None = None,
    source_path: str | None = None,
    source_record_indices: list[int] | None = None,
) -> dict:
    """构造 SFT 风格 dict。"""
    return {
        "sample_id": sample_id,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "unit_id": unit_id or f"unit_{sample_id[-6:]}",
            "source_path": source_path or "test/doc.docx",
            "source_record_indices": source_record_indices or [0],
        },
    }


@pytest.fixture
def tmp_sft_dir(tmp_path: Path) -> Path:
    """在 tmp_path 下铺一份最小 SFT 数据集（all + 三 split + 一致 sample_id）。"""
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    samples = []
    for i in range(10):
        q = f"问题{i}"
        a = "答案" * (i + 1)
        samples.append(_sft_sample(f"sft_{i:024d}", q, a))

    # 划分：train 6 / validation 2 / test 2（人为顺序）
    train = samples[0:6]
    val = samples[6:8]
    test = samples[8:10]

    def _write(name: str, rows: list[dict]) -> None:
        path = sft_dir / name
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    _write("train.jsonl", train)
    _write("validation.jsonl", val)
    _write("test.jsonl", test)
    _write("all.jsonl", samples)
    return sft_dir


# ---------------------------------------------------------------------------
# 1. split membership
# ---------------------------------------------------------------------------


def test_split_membership_is_complete(tmp_sft_dir: Path):
    """all sample_id ⊆ train ∪ validation ∪ test，且互斥。"""
    membership, counts = _build_split_membership(tmp_sft_dir)
    expected_sids = {f"sft_{i:024d}" for i in range(10)}
    assert set(membership.keys()) == expected_sids
    assert counts == {"train": 6, "validation": 2, "test": 2}
    # 互斥：每个 sample 只属于一个 split
    assert len(set(membership.values())) == 3


def test_split_membership_detects_duplicate_across_splits(tmp_path: Path):
    """同一样本 id 出现在 train 与 test → 报错。"""
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    dup = _sft_sample("sft_dup", "Q?", "A.")
    (sft_dir / "train.jsonl").write_text(json.dumps(dup, ensure_ascii=False) + "\n", encoding="utf-8")
    (sft_dir / "validation.jsonl").write_text("\n", encoding="utf-8")
    (sft_dir / "test.jsonl").write_text(json.dumps(dup, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(TokenizerAuditError):
        _build_split_membership(sft_dir)


def test_verify_split_membership_missing_split(tmp_path: Path):
    """all 中存在 split 不含的 sample_id → 报错。"""
    all_records = [_sft_sample("sft_x", "Q?", "A.")]
    membership = {}  # 没有任何 split
    with pytest.raises(TokenizerAuditError):
        _verify_split_membership(all_records, membership)


# ---------------------------------------------------------------------------
# 2. prompt prefix match + assistant target token count
# ---------------------------------------------------------------------------


def test_prompt_is_prefix_of_full_sequence():
    """FakeTokenizer: full 序列的 prompt token 必须等于 prompt-only 序列。"""
    tok = FakeTokenizer()
    messages = [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}]
    full, prompt = _token_lengths(tok, messages)
    assert full[: len(prompt)] == prompt
    assert len(full) > len(prompt)


def test_assistant_token_count_is_suffix_length():
    """assistant_target_tokens = full - prompt（fake tokenizer 中成立）。"""
    tok = FakeTokenizer()
    messages = [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}]
    full, prompt = _token_lengths(tok, messages)
    assert len(full) - len(prompt) > 0


def test_audit_samples_records_prefix_match(tmp_sft_dir: Path):
    """_audit_samples 应当填齐字段并按 sample_id 升序。"""
    membership, _ = _build_split_membership(tmp_sft_dir)
    records = [
        json.loads(l)
        for l in (tmp_sft_dir / "all.jsonl").read_text(encoding="utf-8").splitlines()
        if l
    ]
    rows = _audit_samples(FakeTokenizer(), records, membership)
    assert [r["sample_id"] for r in rows] == sorted(r["sample_id"] for r in rows)
    for r in rows:
        assert r["prompt_prefix_match"] is True
        assert r["assistant_target_tokens"] > 0
        assert r["total_tokens"] == r["prompt_tokens"] + r["assistant_target_tokens"]


def test_invalid_sample_messages_fail():
    """messages 结构非法（不是 user/assistant 两轮）→ 抛错。"""
    bad = {"sample_id": "sft_bad", "metadata": {}}
    with pytest.raises(TokenizerAuditError):
        _audit_samples(FakeTokenizer(), [bad], {"sft_bad": "train"})


# ---------------------------------------------------------------------------
# 3. coverage + percentiles
# ---------------------------------------------------------------------------


def test_coverage_statistics():
    """给定 [100,500,600,1100,2100]，检查 512/1024/2048/4096 coverage。"""
    vs = [100, 500, 600, 1100, 2100]
    cov_512 = _coverage(vs, 512)
    assert cov_512["fit_count"] == 2  # 100, 500
    assert cov_512["fit_ratio"] == 0.4
    cov_1024 = _coverage(vs, 1024)
    assert cov_1024["fit_count"] == 3  # 100, 500, 600
    assert cov_1024["fit_ratio"] == 0.6
    cov_2048 = _coverage(vs, 2048)
    assert cov_2048["fit_count"] == 4  # 100, 500, 600, 1100；2100 不通过
    cov_4096 = _coverage(vs, 4096)
    assert cov_4096["fit_count"] == 5


def test_percentiles():
    """100/200/.../1000 上检验 median/p90/p95/p99/max。"""
    vs = list(range(100, 1001, 100))  # 10 个：100..1000
    s = sorted(vs)
    assert _percentile(s, 0.5) == pytest.approx(550.0)
    assert _percentile(s, 0.90) == pytest.approx(910.0)
    assert _percentile(s, 0.95) == pytest.approx(955.0)
    assert _percentile(s, 0.99) == pytest.approx(991.0)
    assert s[-1] == 1000


def test_length_stats_complete_fields():
    """长度统计必须含 min/mean/median/p90/p95/p99/max。"""
    stats = _length_stats([1, 2, 3, 4, 5])
    for k in ("min", "mean", "median", "p90", "p95", "p99", "max"):
        assert k in stats
    assert stats["min"] == 1
    assert stats["max"] == 5


def test_smallest_candidate_covering_thresholds():
    """95% / 99% / 100% 候选解析正确。"""
    vs = [100, 200, 300, 800, 900, 1500]
    cands = (512, 1024, 1536, 2048)
    # 95% ⇒ 至少 6*0.95=5.7 ⇒ fit>=6 ⇒ 1536 cover 全部
    assert _smallest_candidate_covering(vs, cands, 0.95) == 1536
    # 99% ⇒ fit>=6 ⇒ 1536
    assert _smallest_candidate_covering(vs, cands, 0.99) == 1536
    # 100% ⇒ fit>=6 ⇒ 1536
    assert _smallest_candidate_covering(vs, cands, 1.0) == 1536


def test_smallest_candidate_returns_none_when_unreachable():
    """候选全部不覆盖 → None。"""
    vs = [5000, 6000, 7000]
    cands = (512, 1024, 2048)
    assert _smallest_candidate_covering(vs, cands, 1.0) is None


# ---------------------------------------------------------------------------
# 4. longest samples
# ---------------------------------------------------------------------------


def test_longest_samples_sorted_descending():
    """longest 20 必须按 total_tokens 降序。"""
    rows = [
        {"sample_id": f"sft_{i}", "split": "train", "total_tokens": v,
         "prompt_tokens": v // 2, "assistant_target_tokens": v - v // 2,
         "metadata": {"unit_id": f"u_{i}", "source_path": "x.docx", "source_record_indices": [i]}}
        for i, v in enumerate([100, 5000, 200, 300, 50])
    ]
    longest = _longest_samples(rows, k=3)
    assert [r["total_tokens"] for r in longest] == [5000, 300, 200]
    # 顺序对 sample_id 兜底
    assert longest[1]["sample_id"] == "sft_3"
    # 仅保留定位 metadata，不含 user/assistant 文本
    for r in longest:
        assert "messages" not in r
        assert "content" not in r
        for k in ("unit_id", "source_path", "source_record_indices"):
            assert k in r


# ---------------------------------------------------------------------------
# 5. per-split stats
# ---------------------------------------------------------------------------


def test_split_stats_keys():
    """split 统计必须含 train/validation/test 三组，每组含 count/mean/median/p95/max。"""
    rows = [
        {"sample_id": "a", "split": "train", "total_tokens": 100, "prompt_tokens": 10, "assistant_target_tokens": 90},
        {"sample_id": "b", "split": "train", "total_tokens": 200, "prompt_tokens": 10, "assistant_target_tokens": 190},
        {"sample_id": "c", "split": "validation", "total_tokens": 150, "prompt_tokens": 10, "assistant_target_tokens": 140},
        {"sample_id": "d", "split": "test", "total_tokens": 250, "prompt_tokens": 10, "assistant_target_tokens": 240},
    ]
    stats = _split_stats(rows)
    assert set(stats.keys()) == {"train", "validation", "test"}
    for name, s in stats.items():
        for k in ("count", "mean_total_tokens", "median_total_tokens", "p95_total_tokens", "max_total_tokens"):
            assert k in s, (name, k)
    assert stats["train"]["count"] == 2
    assert stats["train"]["max_total_tokens"] == 200
    assert stats["test"]["count"] == 1
    assert stats["test"]["max_total_tokens"] == 250


# ---------------------------------------------------------------------------
# 6. thinking marker 扫描
# ---------------------------------------------------------------------------


def test_source_thinking_marker_detection():
    """source 中出现 <think> / </think> / reasoning_content 必须被计数。"""
    records = [
        {"messages": [{"role": "user", "content": "正常内容"}, {"role": "assistant", "content": "<think>...</think>答案"}]},
        {"messages": [{"role": "user", "content": "<think>A"}, {"role": "assistant", "content": "B"}]},
        {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y", "reasoning_content": "..."}]},
        {"messages": [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "ok"}]},
    ]
    counters = _scan_source_thinking_markers(records)
    assert counters["think_open"] >= 1
    assert counters["think_close"] >= 1
    assert counters["reasoning_content"] == 1


# ---------------------------------------------------------------------------
# 7. tokenizer metadata + chat template presence
# ---------------------------------------------------------------------------


def test_describe_tokenizer_requires_chat_template():
    """chat_template 为空 → 抛错。"""
    tok = FakeTokenizer()
    tok.chat_template = ""
    cfg = TokenizerAuditConfig()
    with pytest.raises(TokenizerAuditError):
        _describe_tokenizer(tok, cfg, transformers_version="4.51.0")


def test_describe_tokenizer_records_metadata():
    """正常 chat_template 时元数据字段齐。"""
    tok = FakeTokenizer()
    cfg = TokenizerAuditConfig()
    info = _describe_tokenizer(tok, cfg, transformers_version="4.51.0")
    for k in (
        "model_name", "requested_revision", "resolved_revision", "transformers_version",
        "tokenizer_class", "tokenizer_name_or_path", "vocab_size",
        "bos_token", "bos_token_id", "eos_token", "eos_token_id",
        "pad_token", "pad_token_id", "model_max_length", "chat_template_present",
    ):
        assert k in info
    assert info["tokenizer_class"] == "FakeTokenizer"
    assert info["chat_template_present"] is True


# ---------------------------------------------------------------------------
# 8. synthetic template render
# ---------------------------------------------------------------------------


def test_synthetic_render_does_not_modify_template():
    """_render_synthetic 只观察、不修改 chat_template。"""
    tok = FakeTokenizer()
    before = tok.chat_template
    result = _render_synthetic(tok)
    assert tok.chat_template == before
    assert result["thinking_mode"] == "disabled"
    assert result["template_manually_modified"] is False
    assert "测试问题" in result["synthetic_rendered_full"]
    assert "测试答案" in result["synthetic_rendered_full"]


# ---------------------------------------------------------------------------
# 9. end-to-end pipeline (Fake tokenizer)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_tokenizer_factory(monkeypatch):
    """把 ``_load_tokenizer`` / ``_import_transformers`` 替换掉，避免单元测试依赖真实 transformers。"""
    def _factory(model_name: str, revision: str) -> FakeTokenizer:
        return FakeTokenizer()

    class _StubTransformers:
        __version__ = "4.51.0"

    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._load_tokenizer",
        _factory,
    )
    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._import_transformers",
        lambda: _StubTransformers(),
    )
    return _factory


def test_tokenizer_audit_deterministic(tmp_sft_dir: Path, fake_tokenizer_factory, monkeypatch):
    """相同输入 + 同 config 跑两次 → 输出 byte-for-byte 一致。"""
    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._describe_model_config",
        lambda cfg: {"model_type": "qwen3", "max_position_embeddings": 32768, "rope_scaling": None, "load_error": None},
    )

    out1 = tmp_sft_dir.parent / "out1"
    out2 = tmp_sft_dir.parent / "out2"
    cfg = TokenizerAuditConfig(real_sample_check_per_split=2)

    def _run(out_dir: Path) -> None:
        run_tokenizer_audit(
            sft_dir=tmp_sft_dir,
            output_audit_jsonl=out_dir / "token_audit.jsonl",
            output_report_json=out_dir / "token_report.json",
            config=cfg,
        )

    _run(out1)
    _run(out2)

    audit1 = (out1 / "token_audit.jsonl").read_bytes()
    audit2 = (out2 / "token_audit.jsonl").read_bytes()
    report1 = (out1 / "token_report.json").read_bytes()
    report2 = (out2 / "token_report.json").read_bytes()
    assert audit1 == audit2
    assert report1 == report2


def test_assistant_only_loss_feasible(tmp_sft_dir: Path, fake_tokenizer_factory, monkeypatch):
    """fake tokenizer 满足 prefix match + assistant_target > 0 ⇒ feasible。"""
    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._describe_model_config",
        lambda cfg: {"model_type": "qwen3", "max_position_embeddings": 32768, "rope_scaling": None, "load_error": None},
    )
    out = tmp_sft_dir.parent / "feasible"
    cfg = TokenizerAuditConfig(real_sample_check_per_split=2)
    report = run_tokenizer_audit(
        sft_dir=tmp_sft_dir,
        output_audit_jsonl=out / "token_audit.jsonl",
        output_report_json=out / "token_report.json",
        config=cfg,
    )
    assert report["prompt_prefix_mismatch_count"] == 0
    assert report["zero_assistant_target_count"] == 0
    assert report["assistant_only_loss_feasible"] is True


def test_assistant_only_loss_not_feasible_on_prefix_mismatch(tmp_sft_dir: Path, monkeypatch):
    """伪造一个会破坏 prefix 的 tokenizer：assistant 部分改了字符。"""
    class BadTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False, enable_thinking=False, **_):
            # 故意在 full 序列最前面插入一个 prompt 中没有的 token，破坏 prefix
            full_ids: List[int] = [9999]
            for i, m in enumerate(messages):
                full_ids.append(hash((i, m["role"], m["content"])) % 1000 + 10)
            if add_generation_prompt:
                # prompt 走与父类完全相同的逻辑，自然不会包含 9999
                return super().apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                )
            return full_ids if tokenize else "<full>"

    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._load_tokenizer",
        lambda mn, rv: BadTokenizer(),
    )
    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._describe_model_config",
        lambda cfg: {"model_type": "qwen3", "max_position_embeddings": 32768, "rope_scaling": None, "load_error": None},
    )
    class _StubTransformers:
        __version__ = "4.51.0"
    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._import_transformers",
        lambda: _StubTransformers(),
    )

    out = tmp_sft_dir.parent / "infeasible"
    report = run_tokenizer_audit(
        sft_dir=tmp_sft_dir,
        output_audit_jsonl=out / "token_audit.jsonl",
        output_report_json=out / "token_report.json",
        config=TokenizerAuditConfig(real_sample_check_per_split=2),
    )
    assert report["prompt_prefix_mismatch_count"] > 0
    assert report["assistant_only_loss_feasible"] is False


def test_report_contains_model_and_tokenizer_metadata(tmp_sft_dir: Path, fake_tokenizer_factory, monkeypatch):
    """report 必须含 model + tokenizer 全部元数据字段。"""
    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._describe_model_config",
        lambda cfg: {"model_type": "qwen3", "max_position_embeddings": 32768, "rope_scaling": None, "load_error": None},
    )
    out = tmp_sft_dir.parent / "meta"
    report = run_tokenizer_audit(
        sft_dir=tmp_sft_dir,
        output_audit_jsonl=out / "token_audit.jsonl",
        output_report_json=out / "token_report.json",
        config=TokenizerAuditConfig(),
    )
    for k in ("model", "tokenizer", "length_stats", "split_stats", "coverage",
              "coverage_thresholds", "longest_samples", "synthetic_template",
              "source_thinking_markers", "assistant_only_loss_feasible"):
        assert k in report
    for k in ("tokenizer_class", "vocab_size", "chat_template_present", "bos_token", "eos_token", "pad_token"):
        assert k in report["tokenizer"]
    for k in ("model_name", "requested_revision", "resolved_revision", "transformers_version"):
        assert k in report["model"]


def test_report_contains_split_stats(tmp_sft_dir: Path, fake_tokenizer_factory, monkeypatch):
    """split_stats 必须 train/validation/test 都出现。"""
    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._describe_model_config",
        lambda cfg: {"model_type": "qwen3", "max_position_embeddings": 32768, "rope_scaling": None, "load_error": None},
    )
    out = tmp_sft_dir.parent / "split_stats"
    report = run_tokenizer_audit(
        sft_dir=tmp_sft_dir,
        output_audit_jsonl=out / "token_audit.jsonl",
        output_report_json=out / "token_report.json",
        config=TokenizerAuditConfig(),
    )
    for name in ("train", "validation", "test"):
        s = report["split_stats"][name]
        assert s["count"] >= 0
        assert "max_total_tokens" in s


def test_no_real_text_written_to_longest_samples(tmp_sft_dir: Path, fake_tokenizer_factory, monkeypatch):
    """longest samples 中不得含 user/assistant 原文。"""
    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._describe_model_config",
        lambda cfg: {"model_type": "qwen3", "max_position_embeddings": 32768, "rope_scaling": None, "load_error": None},
    )
    out = tmp_sft_dir.parent / "longest"
    report = run_tokenizer_audit(
        sft_dir=tmp_sft_dir,
        output_audit_jsonl=out / "token_audit.jsonl",
        output_report_json=out / "token_report.json",
        config=TokenizerAuditConfig(),
    )
    for r in report["longest_samples"]:
        for forbidden in ("content", "messages", "question_text", "answer_text"):
            assert forbidden not in r
        for k in ("unit_id", "source_path", "source_record_indices"):
            assert k in r


def test_audit_jsonl_does_not_include_input_ids(tmp_sft_dir: Path, fake_tokenizer_factory, monkeypatch):
    """token_audit.jsonl 每行必须不含 input_ids / attention_mask / labels。"""
    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._describe_model_config",
        lambda cfg: {"model_type": "qwen3", "max_position_embeddings": 32768, "rope_scaling": None, "load_error": None},
    )
    out = tmp_sft_dir.parent / "rows"
    run_tokenizer_audit(
        sft_dir=tmp_sft_dir,
        output_audit_jsonl=out / "token_audit.jsonl",
        output_report_json=out / "token_report.json",
        config=TokenizerAuditConfig(),
    )
    for line in (out / "token_audit.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        rec = json.loads(line)
        for forbidden in ("input_ids", "attention_mask", "labels"):
            assert forbidden not in rec
        # 只保留定位 metadata
        meta = rec["metadata"]
        for forbidden in ("content", "messages", "question_text", "answer_text"):
            assert forbidden not in meta


def test_dataset_membership_set_equality(tmp_sft_dir: Path, fake_tokenizer_factory, monkeypatch):
    """dataset_membership.set_equal_all_vs_union 必须为 True。"""
    monkeypatch.setattr(
        "data_pipeline.tokenization.pipeline._describe_model_config",
        lambda cfg: {"model_type": "qwen3", "max_position_embeddings": 32768, "rope_scaling": None, "load_error": None},
    )
    out = tmp_sft_dir.parent / "mem"
    report = run_tokenizer_audit(
        sft_dir=tmp_sft_dir,
        output_audit_jsonl=out / "token_audit.jsonl",
        output_report_json=out / "token_report.json",
        config=TokenizerAuditConfig(),
    )
    assert report["dataset_membership"]["set_equal_all_vs_union"] is True
    assert report["dataset_membership"]["all_count"] == 10


# ---------------------------------------------------------------------------
# 10. Config 边界
# ---------------------------------------------------------------------------


def test_config_candidate_lengths_must_be_positive():
    with pytest.raises(ValueError):
        TokenizerAuditConfig(candidate_max_lengths=())
    with pytest.raises(ValueError):
        TokenizerAuditConfig(candidate_max_lengths=(0,))


def test_config_real_sample_check_non_negative():
    with pytest.raises(ValueError):
        TokenizerAuditConfig(real_sample_check_per_split=-1)


def test_config_default_candidate_lengths():
    """默认候选长度必须包含 512/1024/1536/2048/3072/4096。"""
    cfg = TokenizerAuditConfig()
    for v in (512, 1024, 1536, 2048, 3072, 4096):
        assert v in cfg.candidate_max_lengths
    assert cfg.candidate_max_lengths == DEFAULT_CANDIDATE_MAX_LENGTHS


# ---------------------------------------------------------------------------
# 11. 类字段中文注释（src 约定）
# ---------------------------------------------------------------------------


def test_tokenization_dataclasses_have_chinese_field_comments():
    """新增 dataclass 字段前必须有中文注释。"""
    src_dir = Path(__file__).resolve().parents[1] / "src" / "data_pipeline" / "tokenization"
    import ast

    def _collect_dataclasses(py_file: Path) -> dict[str, list[str]]:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        out: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                is_dc = any(
                    (isinstance(d, ast.Name) and d.id == "dataclass")
                    or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                    for d in node.decorator_list
                )
                if not is_dc:
                    continue
                fields = [
                    t.target.id  # type: ignore[attr-defined]
                    for t in node.body
                    if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
                ]
                if fields:
                    out[node.name] = fields
        return out

    for py_file in sorted(src_dir.glob("*.py")):
        cls_to_fields = _collect_dataclasses(py_file)
        if not cls_to_fields:
            continue
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


# ---------------------------------------------------------------------------
# 12. integration（默认跳过；RUN_QWEN_TOKENIZER_INTEGRATION=1 时启动）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_QWEN_TOKENIZER_INTEGRATION") != "1",
    reason="需显式环境变量 RUN_QWEN_TOKENIZER_INTEGRATION=1 才运行真实 Qwen3 tokenizer",
)
def test_qwen3_tokenizer_integration(tmp_path: Path):
    """真实 Qwen3 tokenizer + 真实 data/sft 跑一次 end-to-end。

    默认跳过；CI 不应因网络抖动失败。
    """
    from data_pipeline.utils import load_jsonl_records  # noqa: F401
    sft_dir = Path("data/sft")
    if not (sft_dir / "all.jsonl").exists():
        pytest.skip("data/sft 尚未生成；先运行 python scripts/build_sft_dataset.py")
    out = tmp_path / "integration"
    report = run_tokenizer_audit(
        sft_dir=sft_dir,
        output_audit_jsonl=out / "token_audit.jsonl",
        output_report_json=out / "token_report.json",
        config=TokenizerAuditConfig(),
    )
    # 真实数据下 assistant_only_loss_feasible 必须 True
    assert report["assistant_only_loss_feasible"] is True
    assert report["prompt_prefix_mismatch_count"] == 0


