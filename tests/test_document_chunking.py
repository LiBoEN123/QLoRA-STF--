"""Tests for the Phase 3 chunking pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data_pipeline.chunking import (
    ChunkingConfig,
    detect_question_candidate,
    pair_questions_with_answers,
    restore_source_order,
    run_chunking_pipeline,
    split_long_text,
    strip_question_number,
)
from data_pipeline.chunking.faq import STRONG_QUESTION_WORDS


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_record(
    idx: int,
    text: str,
    *,
    source_path: str = "test.docx",
    structure_type: str = "paragraph",
    doc_id: str | None = None,
    detected_section: str | None = None,
    detected_heading: str | None = None,
    page_number: int | None = None,
    sheet_name: str | None = None,
    file_sha256: str = "abc123",
) -> dict:
    return {
        "doc_id": doc_id or f"doc_{idx:04d}",
        "source_type": "docx",
        "source_path": source_path,
        "file_name": "test.docx",
        "record_index": idx,
        "page_number": page_number,
        "sheet_name": sheet_name,
        "section": None,
        "title": None,
        "text": text,
        "clean_text": text,
        "structure_type": structure_type,
        "structure_level": 6,
        "detected_section": detected_section,
        "detected_heading": detected_heading,
        "quality": {},
        "cleaning_actions": [],
        "metadata": {},
        "file_sha256": file_sha256,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_source_order_restores_by_record_index_not_doc_id():
    """doc_id 顺序与 record_index 顺序必须不一致，依然恢复正确原文顺序。"""
    recs = [
        _make_record(1, "一、党组织设置", structure_type="heading"),
        _make_record(2, "1.什么是党组织？", structure_type="list_item"),
        _make_record(3, "党组织是...", structure_type="paragraph"),
        _make_record(4, "2.如何成立？", structure_type="list_item"),
        _make_record(5, "向上级申请...", structure_type="paragraph"),
    ]
    # 反转 doc_id 顺序
    for r in recs:
        r["doc_id"] = "doc_" + hashlib.sha256(r["text"].encode()).hexdigest()[:24]

    ordered = restore_source_order(recs)
    indices = [r["record_index"] for r in ordered]
    assert indices == [1, 2, 3, 4, 5]


def test_source_order_preserves_stability_for_same_key():
    """相同排序键的 records 必须保持输入顺序（稳定排序）。"""
    a = _make_record(0, "A", structure_type="paragraph", doc_id="doc_AAAA0000000000000000aaaa")
    b = _make_record(0, "B", structure_type="paragraph", doc_id="doc_BBBB0000000000000000bbbb")
    c = _make_record(0, "C", structure_type="paragraph", doc_id="doc_CCCC0000000000000000cccc")
    ordered = restore_source_order([a, b, c])
    texts = [r["text"] for r in ordered]
    assert texts == ["A", "B", "C"]


def test_source_order_does_not_use_python_hash():
    """排序结果不应依赖 Python hash()——同一进程两次调用结果完全相同。"""
    recs = [_make_record(i, f"段落{i}", structure_type="paragraph") for i in range(10)]
    o1 = restore_source_order(recs)
    o2 = restore_source_order(recs)
    assert [r["doc_id"] for r in o1] == [r["doc_id"] for r in o2]


# ---------------------------------------------------------------------------
# Question number stripping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1.什么是党组织？", "什么是党组织？"),
        ("23、发展党员需要哪些程序", "发展党员需要哪些程序"),
        ("（1）从单位名称的角度考量", "从单位名称的角度考量"),
        ("① 第一项", "第一项"),
        ("一、基层党组织的设置", "基层党组织的设置"),
        ("（一）基本原则", "基本原则"),
        ("普通段落内容", "普通段落内容"),
        ("党的章程第3条规定", "党的章程第3条规定"),  # 行内数字不能动
    ],
)
def test_strip_question_number(raw, expected):
    assert strip_question_number(raw) == expected


# ---------------------------------------------------------------------------
# Question detection
# ---------------------------------------------------------------------------


def test_detect_high_confidence_with_question_mark():
    cand = detect_question_candidate({"clean_text": "1.什么是党组织？"})
    assert cand.is_candidate is True
    assert cand.confidence == "high"


def test_detect_medium_confidence_numbered_no_mark():
    cand = detect_question_candidate({"clean_text": "12. 发展党员需要哪些程序"})
    assert cand.is_candidate is True
    assert cand.confidence == "medium"


def test_detect_paren_list_not_question():
    """（1）... 即使看起来像编号，也是答案内部枚举，不是问题。"""
    cand = detect_question_candidate({"clean_text": "（1）从单位名称的角度考量"})
    assert cand.is_candidate is False


def test_detect_circled_number_with_mark():
    cand = detect_question_candidate({"clean_text": "① 什么事？"})
    assert cand.is_candidate is True
    assert cand.confidence == "high"


def test_detect_normal_paragraph_not_question():
    cand = detect_question_candidate({"clean_text": "党的基层组织应当加强教育管理。"})
    assert cand.is_candidate is False


def test_detect_chinese_numbered_with_mark():
    cand = detect_question_candidate({"clean_text": "一、基层党组织的设置？"})
    # 中文编号 + 问号 → medium
    assert cand.is_candidate is True
    assert cand.confidence == "medium"


def test_detect_loose_question_word_no_prefix_is_medium():
    """纯段落问句（无编号 + 强疑问词 + 问号）→ medium（已提升，不再走 low + recovery）。"""
    cand = detect_question_candidate({"clean_text": "什么是党组织？"})
    assert cand.is_candidate is True
    assert cand.confidence == "medium"


def test_question_word_set_includes_expected_words():
    for w in ("什么", "如何", "怎么", "哪些", "何时", "为什么", "怎么办"):
        assert w in STRONG_QUESTION_WORDS


# ---------------------------------------------------------------------------
# FAQ pairing
# ---------------------------------------------------------------------------


def test_pair_one_question_one_answer():
    recs = [
        _make_record(1, "一、党组织设置", structure_type="heading", detected_section="一、党组织设置", detected_heading="一、党组织设置"),
        _make_record(2, "1.什么是党组织？", structure_type="list_item", detected_section="一、党组织设置"),
        _make_record(3, "党组织是...", structure_type="paragraph", detected_section="一、党组织设置"),
        _make_record(4, "二、其他", structure_type="heading", detected_section="二、其他", detected_heading="二、其他"),
    ]
    paired, unpaired = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 1
    assert paired[0].question_text == "什么是党组织？"
    assert "党组织是" in paired[0].answer_text
    assert paired[0].confidence == "high"
    assert len(unpaired) == 0


def test_pair_one_question_multiple_answers():
    recs = [
        _make_record(1, "1.什么是党组织？", structure_type="list_item"),
        _make_record(2, "第一段...", structure_type="paragraph"),
        _make_record(3, "第二段...", structure_type="paragraph"),
        _make_record(4, "（1）子项", structure_type="list_item"),
    ]
    paired, _ = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 1
    a = paired[0]
    assert a.answer_record_indices == [2, 3, 4]
    assert "第一段" in a.answer_text and "第二段" in a.answer_text and "（1）子项" in a.answer_text


def test_pair_one_question_answer_with_list_not_split():
    """答案内部 list_items 不会被识别成新问题。"""
    recs = [
        _make_record(1, "1.党员有哪些义务？", structure_type="list_item"),
        _make_record(2, "应当：", structure_type="paragraph"),
        _make_record(3, "（1）学习理论", structure_type="list_item"),
        _make_record(4, "（2）执行决定", structure_type="list_item"),
        _make_record(5, "（3）遵守纪律", structure_type="list_item"),
    ]
    paired, _ = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 1
    assert "（1）学习理论" in paired[0].answer_text
    assert "（3）遵守纪律" in paired[0].answer_text


def test_pair_consecutive_faqs_do_not_swallow_each_other():
    recs = [
        _make_record(1, "1.什么是A？", structure_type="list_item"),
        _make_record(2, "A答案。", structure_type="paragraph"),
        _make_record(3, "2.什么是B？", structure_type="list_item"),
        _make_record(4, "B答案。", structure_type="paragraph"),
        _make_record(5, "3.什么是C？", structure_type="list_item"),
        _make_record(6, "C答案。", structure_type="paragraph"),
    ]
    paired, _ = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 3
    questions = [p.question_text for p in paired]
    assert questions == ["什么是A？", "什么是B？", "什么是C？"]
    answers = [p.answer_text for p in paired]
    assert "A答案" in answers[0] and "B" not in answers[0]
    assert "B答案" in answers[1] and "C" not in answers[1]


def test_pair_heading_breaks_faq():
    recs = [
        _make_record(1, "1.问题？", structure_type="list_item"),
        _make_record(2, "答案。", structure_type="paragraph"),
        _make_record(3, "二、新章节", structure_type="heading", detected_section="二、新章节", detected_heading="二、新章节"),
        _make_record(4, "该段不应被绑定。", structure_type="paragraph"),
    ]
    paired, _ = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 1
    assert "答案。" in paired[0].answer_text
    assert "该段不应被绑定" not in paired[0].answer_text


def test_pair_unpaired_question_recorded():
    """识别为问题但无答案（紧接着 heading） → unpaired。"""
    recs = [
        _make_record(1, "1.问题？", structure_type="list_item"),
        _make_record(2, "二、新章节", structure_type="heading", detected_section="二、新章节", detected_heading="二、新章节"),
    ]
    paired, unpaired = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 0
    assert len(unpaired) == 1


def test_pair_low_confidence_not_auto_paired():
    """没有问号、也没有编号的纯陈述句（即使含疑问词）不会被识别为候选。"""
    recs = [
        _make_record(1, "如何做好发展党员工作", structure_type="paragraph"),
        _make_record(2, "党组织是...", structure_type="paragraph"),
    ]
    paired, unpaired = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 0
    # 重构后只保留 high / medium 两个真实候选级别；无问号的陈述句直接不入审计。
    assert len(unpaired) == 0


def test_pair_numbered_answer_list_not_treated_as_questions():
    """答案里的（1）（2）（3）... 不会被识别成三个问题。"""
    recs = [
        _make_record(1, "1.党员有哪些义务？", structure_type="list_item"),
        _make_record(2, "申请入党需要：", structure_type="paragraph"),
        _make_record(3, "（1）年满十八周岁", structure_type="list_item"),
        _make_record(4, "（2）承认党的纲领", structure_type="list_item"),
        _make_record(5, "（3）愿意参加党的活动", structure_type="list_item"),
    ]
    paired, _ = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 1


def test_pair_no_mark_numbered_with_qword_is_medium():
    recs = [
        _make_record(1, "12. 发展党员需要哪些程序", structure_type="list_item"),
        _make_record(2, "发展党员一般...", structure_type="paragraph"),
    ]
    paired, _ = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 1
    assert paired[0].confidence == "medium"


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


def test_split_under_limit_returns_single_segment():
    text = "短文本。"
    out = split_long_text(text, max_chars=100)
    assert len(out) == 1
    assert out[0].text == text


def test_split_at_paragraph_boundary():
    text = "段落一。" + ("x" * 50) + "\n\n" + "段落二。" + ("y" * 50)
    out = split_long_text(text, max_chars=80)
    assert len(out) >= 2
    assert all(len(seg.text) <= 80 for seg in out)


def test_split_falls_back_to_hard_split():
    text = "x" * 1000  # 无任何边界
    out = split_long_text(text, max_chars=100)
    assert all(len(seg.text) <= 100 for seg in out)


def test_split_with_overlap():
    text = "句子一。\n\n" + "句子二。" * 30
    out = split_long_text(text, max_chars=80, overlap_chars=20)
    assert len(out) >= 2


# ---------------------------------------------------------------------------
# Pipeline end-to-end
# ---------------------------------------------------------------------------


def _sample_records() -> list[dict]:
    return [
        _make_record(1, "一、党组织设置", structure_type="heading",
                     detected_section="一、党组织设置", detected_heading="一、党组织设置"),
        _make_record(2, "1.什么是党组织？", structure_type="list_item",
                     detected_section="一、党组织设置"),
        _make_record(3, "党组织是党的基层组织。", structure_type="paragraph",
                     detected_section="一、党组织设置"),
        _make_record(4, "（1）依据党章", structure_type="list_item",
                     detected_section="一、党组织设置"),
        _make_record(5, "2.如何成立？", structure_type="list_item",
                     detected_section="一、党组织设置"),
        _make_record(6, "向上级申请。", structure_type="paragraph",
                     detected_section="一、党组织设置"),
        _make_record(7, "二、任务", structure_type="heading",
                     detected_section="二、任务", detected_heading="二、任务"),
        _make_record(8, "应当完成任务。", structure_type="paragraph",
                     detected_section="二、任务"),
        _make_record(9, "坚持党的领导。", structure_type="paragraph",
                     detected_section="二、任务"),
    ]


def test_pipeline_produces_units_and_chunks(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, _sample_records())
    out_units = tmp_path / "units.jsonl"
    out_chunks = tmp_path / "chunks.jsonl"
    out_report = tmp_path / "report.json"

    result = run_chunking_pipeline(
        inp, out_units, out_chunks, out_report,
        config=ChunkingConfig(min_chars=50, target_chars=300, max_chars=500, overlap_chars=50),
    )
    assert result.input_records == 9
    assert result.semantic_units >= 2  # FAQ + heading + paragraph groups
    assert result.chunks > 0
    assert out_units.exists()
    assert out_chunks.exists()
    assert out_report.exists()


def test_pipeline_does_not_cross_heading_boundary(tmp_path: Path):
    """正文 A + 跨 heading 的正文 B 不应出现在同一 chunk。"""
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, _sample_records())
    out_units = tmp_path / "units.jsonl"
    out_chunks = tmp_path / "chunks.jsonl"
    out_report = tmp_path / "report.json"
    run_chunking_pipeline(inp, out_units, out_chunks, out_report,
                          config=ChunkingConfig(min_chars=50, target_chars=300, max_chars=500))
    lines = out_chunks.read_text(encoding="utf-8").splitlines()
    for line in lines:
        c = json.loads(line)
        if c["chunk_type"] != "text":
            continue
        # 任何一个 text chunk 不应同时含有「一、党组织设置」与「二、任务」两个 heading
        if "一、党组织设置" in c["text"] and "二、任务" in c["text"]:
            # 允许二者在 structure_path 但不应在同一个 chunk 文本里
            pytest.fail(f"chunk 跨越 heading：{c['chunk_id']}")


def test_pipeline_faq_one_per_question(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, _sample_records())
    out_units = tmp_path / "units.jsonl"
    out_chunks = tmp_path / "chunks.jsonl"
    out_report = tmp_path / "report.json"
    run_chunking_pipeline(inp, out_units, out_chunks, out_report,
                          config=ChunkingConfig(min_chars=50, target_chars=300, max_chars=500))
    unit_lines = out_units.read_text(encoding="utf-8").splitlines()
    faq_units = [json.loads(l) for l in unit_lines if json.loads(l)["unit_type"] == "faq"]
    assert len(faq_units) == 2  # 1.什么是党组织 + 2.如何成立
    qtexts = [u["question_text"] for u in faq_units]
    assert qtexts == ["什么是党组织？", "如何成立？"]


def test_pipeline_short_faq_not_merged(tmp_path: Path):
    """短 FAQ 不与下一道 FAQ 合并。"""
    recs = [
        _make_record(1, "1.党龄从何时算？", structure_type="list_item"),
        _make_record(2, "从预备期满算起。", structure_type="paragraph"),
        _make_record(3, "2.何时转正？", structure_type="list_item"),
        _make_record(4, "一年后。", structure_type="paragraph"),
    ]
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, recs)
    out_units = tmp_path / "units.jsonl"
    out_chunks = tmp_path / "chunks.jsonl"
    out_report = tmp_path / "report.json"
    run_chunking_pipeline(
        inp, out_units, out_chunks, out_report,
        config=ChunkingConfig(min_chars=500, target_chars=800, max_chars=1200),
    )
    chunk_lines = out_chunks.read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(l) for l in chunk_lines]
    faq_chunks = [c for c in chunks if c["chunk_type"] == "faq"]
    assert len(faq_chunks) == 2


def test_pipeline_long_paragraph_splits(tmp_path: Path):
    """超长 paragraph 必须被切分成多个 chunk，每段 <= max_chars + overlap。"""
    long_text = "句子一。" * 50  # 200 chars
    long_text += "\n\n"
    long_text += "句子二。" * 200  # 800 chars
    long_text += "\n\n"
    long_text += "句子三。" * 50
    recs = [_make_record(1, long_text, structure_type="paragraph")]
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, recs)
    out_units = tmp_path / "units.jsonl"
    out_chunks = tmp_path / "chunks.jsonl"
    out_report = tmp_path / "report.json"
    cfg = ChunkingConfig(min_chars=50, target_chars=200, max_chars=400, overlap_chars=50)
    run_chunking_pipeline(inp, out_units, out_chunks, out_report, config=cfg)
    chunks = [json.loads(l) for l in out_chunks.read_text(encoding="utf-8").splitlines()]
    assert len(chunks) >= 2
    for c in chunks:
        # overlap chars 允许超出，但不超过 max + overlap
        assert c["char_count"] <= cfg.max_chars + cfg.overlap_chars


def test_pipeline_deterministic(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, _sample_records())
    cfg = ChunkingConfig()

    def _run(tag: str):
        run_chunking_pipeline(
            inp,
            tmp_path / f"u_{tag}.jsonl",
            tmp_path / f"c_{tag}.jsonl",
            tmp_path / f"r_{tag}.json",
            config=cfg,
        )

    _run("a")
    _run("b")
    for stem in ("u_", "c_", "r_"):
        for tag in ("a", "b"):
            (tmp_path / f"{stem}{tag}.jsonl").read_bytes() if stem != "r_" else None
    a_units = (tmp_path / "u_a.jsonl").read_bytes()
    b_units = (tmp_path / "u_b.jsonl").read_bytes()
    a_chunks = (tmp_path / "c_a.jsonl").read_bytes()
    b_chunks = (tmp_path / "c_b.jsonl").read_bytes()
    a_rep = (tmp_path / "r_a.json").read_bytes()
    b_rep = (tmp_path / "r_b.json").read_bytes()
    assert a_units == b_units
    assert a_chunks == b_chunks
    assert a_rep == b_rep


def test_pipeline_unit_ids_stable(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, _sample_records())
    out_units = tmp_path / "units.jsonl"
    out_chunks = tmp_path / "chunks.jsonl"
    out_report = tmp_path / "report.json"
    run_chunking_pipeline(inp, out_units, out_chunks, out_report,
                          config=ChunkingConfig())
    units = [json.loads(l) for l in out_units.read_text(encoding="utf-8").splitlines()]
    seen = set()
    for u in units:
        assert u["unit_id"] not in seen, f"unit_id 重复：{u['unit_id']}"
        seen.add(u["unit_id"])
        assert u["unit_id"].startswith("unit_")
        assert len(u["unit_id"]) == len("unit_") + 24


def test_pipeline_chunk_ids_stable(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, _sample_records())
    out_units = tmp_path / "units.jsonl"
    out_chunks = tmp_path / "chunks.jsonl"
    out_report = tmp_path / "report.json"
    run_chunking_pipeline(inp, out_units, out_chunks, out_report,
                          config=ChunkingConfig())
    chunks = [json.loads(l) for l in out_chunks.read_text(encoding="utf-8").splitlines()]
    seen = set()
    for c in chunks:
        assert c["chunk_id"] not in seen, f"chunk_id 重复：{c['chunk_id']}"
        seen.add(c["chunk_id"])
        assert c["chunk_id"].startswith("chunk_")
        assert len(c["chunk_id"]) == len("chunk_") + 24


def test_pipeline_traceability(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, _sample_records())
    out_units = tmp_path / "units.jsonl"
    out_chunks = tmp_path / "chunks.jsonl"
    out_report = tmp_path / "report.json"
    run_chunking_pipeline(inp, out_units, out_chunks, out_report,
                          config=ChunkingConfig())
    units = [json.loads(l) for l in out_units.read_text(encoding="utf-8").splitlines()]
    chunks = [json.loads(l) for l in out_chunks.read_text(encoding="utf-8").splitlines()]
    # 每个 chunk 必须能追溯到 doc_id 和 record_indices
    for c in chunks:
        assert c["source_doc_ids"]
        assert all(isinstance(x, str) for x in c["source_doc_ids"])
        assert c["source_record_indices"]
    for u in units:
        assert u["source_doc_ids"]
        assert u["source_record_indices"]


def test_config_validation():
    with pytest.raises(ValueError):
        ChunkingConfig(min_chars=200, target_chars=100, max_chars=300)
    with pytest.raises(ValueError):
        ChunkingConfig(overlap_chars=1500, max_chars=1000)
    with pytest.raises(ValueError):
        ChunkingConfig(faq_min_confidence="bogus")


# ---------------------------------------------------------------------------
# 重构测试：统一的 QuestionDetector
# - paragraph FAQ（无编号 + 短 + 问号 + 疑问词 → medium）
# - sticky-number FAQ（开头 1~3 位数字紧接中文 → medium）
# - 正常年份 / 年龄 / 金额不能被 strip
# - paragraph question 必须终止上一题的答案范围
# - no_answer 必须自然落入 unpaired，不能伪造答案
# ---------------------------------------------------------------------------


def test_detect_paragraph_question_is_medium():
    """纯段落短问句无编号 → medium（不再经过 low 后处理）。"""
    cand = detect_question_candidate(
        {"clean_text": "如何确定基层党组织的名称？"}
    )
    assert cand.is_candidate is True
    assert cand.confidence == "medium"


def test_detect_sticky_number_is_medium():
    """开头 1~3 位数字紧接中文、剥离后是问句 → medium。"""
    cand = detect_question_candidate(
        {"clean_text": "87哪些单位一般不设立党组？"}
    )
    assert cand.is_candidate is True
    assert cand.confidence == "medium"
    # 解析后的 question_text 应该剥离粘连题号；raw 保留原文。
    assert cand.question_text == "哪些单位一般不设立党组？"
    assert cand.raw_question_text == "87哪些单位一般不设立党组？"


@pytest.mark.parametrize(
    "text, preserved_prefix",
    [
        ("2026年党员教育工作重点是什么？", "2026"),
        ("18岁以上人员能否申请入党？", "18"),
        ("100万元党费如何管理？", "100"),
    ],
)
def test_detect_does_not_strip_year_age_money(text, preserved_prefix):
    """正常数字开头的段落问题不能 strip 掉数字。"""
    cand = detect_question_candidate({"clean_text": text})
    assert cand.is_candidate is True
    assert cand.confidence == "medium"
    assert cand.question_text == text
    assert cand.raw_question_text == text
    assert preserved_prefix in cand.question_text


def test_pair_paragraph_question_terminates_previous():
    """中间夹一道段落问句时，前面那道 FAQ 的答案必须在这里终止。"""
    recs = [
        _make_record(1, "1.什么是A？", structure_type="list_item"),
        _make_record(2, "A第一段。", structure_type="paragraph"),
        _make_record(3, "如何办理B？", structure_type="paragraph"),
        _make_record(4, "B答案。", structure_type="paragraph"),
        _make_record(5, "2.什么是C？", structure_type="list_item"),
        _make_record(6, "C答案。", structure_type="paragraph"),
    ]
    paired, _ = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 3
    a, b, c = paired
    # A 的答案只到 A第一段，不能吞 B。
    assert "A第一段" in a.answer_text
    assert "如何办理B" not in a.answer_text
    assert "B答案" not in a.answer_text
    # B 是被恢复的段落 FAQ。
    assert b.question_text == "如何办理B？"
    assert "B答案" in b.answer_text
    # C 是结构化 FAQ，与 B 不串扰。
    assert c.question_text == "什么是C？"
    assert "C答案" in c.answer_text


def test_pair_no_answer_does_not_invent():
    """段落问句紧接 heading → 无答案，不能伪造。"""
    recs = [
        _make_record(1, "如何办理该业务？", structure_type="paragraph"),
        _make_record(
            2,
            "二、新章节",
            structure_type="heading",
            detected_section="二、新章节",
            detected_heading="二、新章节",
        ),
    ]
    paired, unpaired = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 0
    # detector 把它识别为 medium，但因为无答案，归入 unpaired。
    assert len(unpaired) == 1
    assert unpaired[0].confidence == "medium"
    assert unpaired[0].reason  # 非空，能溯源


def test_pair_sticky_number_gets_correct_strip_and_pair():
    recs = [
        _make_record(1, "87哪些单位一般不设立党组？", structure_type="paragraph"),
        _make_record(2, "（1）领导机关中的党员领导成员不足3人的。", structure_type="list_item"),
        _make_record(3, "（2）与党的机关合并设立或者合署办公的。", structure_type="list_item"),
    ]
    paired, _ = pair_questions_with_answers(recs, min_confidence="medium")
    assert len(paired) == 1
    p = paired[0]
    assert p.confidence == "medium"
    assert p.question_text == "哪些单位一般不设立党组？"
    assert p.raw_question_text == "87哪些单位一般不设立党组？"
    assert "（1）" in p.answer_text and "（2）" in p.answer_text


def test_pipeline_runs_with_real_data_and_produces_faqs(tmp_path: Path):
    """端到端跑 Phase 2 → Phase 3 真实数据，FAQ 总量大致不退化。"""
    real_input = Path("data/processed/documents_clean.jsonl").resolve()
    if not real_input.exists():
        pytest.skip("真实清洗数据缺失，跳过回归")
    out_units = tmp_path / "units.jsonl"
    out_chunks = tmp_path / "chunks.jsonl"
    out_report = tmp_path / "report.json"
    result = run_chunking_pipeline(
        real_input,
        out_units,
        out_chunks,
        out_report,
        config=ChunkingConfig(),
    )
    units = [
        json.loads(l) for l in out_units.read_text(encoding="utf-8").splitlines()
    ]
    faq_units = [u for u in units if u["unit_type"] == "faq"]
    # 重构不要求与旧版 SHA256 严格一致；FAQ 总量应在原 1327 附近。
    assert len(faq_units) >= 1300
    # 不能出现明显膨胀（避免误配）。
    assert len(faq_units) <= 1400
    # 高置信度 FAQ 必须保留绝大多数原 1318 条结构化结果。
    high = sum(1 for u in faq_units if u.get("faq_confidence") == "high")
    assert high >= 1300