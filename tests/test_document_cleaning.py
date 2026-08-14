"""Tests for the Phase 2 document cleaning pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_pipeline.cleaning import (
    dedup,
    normalizer,
    pipeline,
    quality,
    structure,
)
from data_pipeline.cleaning.pipeline import (
    CleaningConfig,
    QualityConfig,
    run_cleaning_pipeline,
)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def test_normalize_unicode_nfkc_collapses_fullwidth():
    text = "ＡＢＣ１２３"  # fullwidth letters/digits
    out, changed = normalizer.normalize_unicode(text)
    assert out == "ABC123"
    assert changed is True


def test_normalize_unicode_strips_invisible_chars():
    text = "\ufeff\u200b中\u200c国\u200d共\u00ad产\u2060党\u2028"
    out, changed = normalizer.normalize_unicode(text)
    assert out == "中国共产党"
    assert changed is True


def test_normalize_unicode_strips_control_chars():
    text = "中\x00\x01\x02\x07\x08国\x0b\x0c党"
    out, changed = normalizer.normalize_unicode(text)
    assert out == "中国党"
    assert changed is True


def test_normalize_unicode_preserves_chinese_punctuation():
    text = "中国共产党是中国工人阶级的先锋队，同时是中国人民和中华民族的先锋队。"
    out, _ = normalizer.normalize_unicode(text)
    assert out == text  # no-op for normal Chinese


def test_normalize_whitespace_nbsp_to_space():
    text = "党\u00a0务"
    out, changed = normalizer.normalize_whitespace(text)
    assert out == "党 务"
    assert changed is True


def test_normalize_whitespace_collapses_spaces():
    text = "党务      工作"
    out, changed = normalizer.normalize_whitespace(text)
    assert out == "党务 工作"
    assert changed is True


def test_normalize_whitespace_trims_lines_and_preserves_paragraphs():
    text = "第一行   \n  第二行  \n\n\n\n\n第三行"
    out, _ = normalizer.normalize_whitespace(text)
    # No trailing whitespace on lines, multiple blank lines collapsed to two.
    assert "第一行\n第二行" in out
    assert "第二行\n" in out
    assert "\n\n\n\n" not in out


def test_normalize_punctuation_collapses_runs():
    text = "党的工作。。。非常好！！！真的？？？"
    out, changed = normalizer.normalize_punctuation(text)
    assert out == "党的工作。非常好！真的？"
    assert changed is True


def test_normalize_punctuation_preserves_ellipsis_and_emdash():
    text = "重要意义……我们要——继续前进"
    out, changed = normalizer.normalize_punctuation(text)
    assert out == text
    assert changed is False


def test_normalize_full_records_actions():
    text = "  党\u00a0\u00a0务。。。   "
    out, actions = normalizer.normalize_full(text)
    assert "unicode_nfkc" in actions
    assert "collapse_spaces" in actions
    assert "collapse_punctuation" in actions
    assert out == "党 务。"


def test_normalize_full_empty():
    out, actions = normalizer.normalize_full("")
    assert out == ""
    assert actions == []


# ---------------------------------------------------------------------------
# Line break repair
# ---------------------------------------------------------------------------


def test_repair_line_breaks_merges_wrapped_chinese():
    text = "中国共产党是中国工人阶级的先锋\n队，同时是中国人民和中华民族的先锋队。"
    out, changed = normalizer.repair_line_breaks(text)
    assert out == "中国共产党是中国工人阶级的先锋队，同时是中国人民和中华民族的先锋队。"
    assert changed is True


def test_repair_line_breaks_keeps_headings():
    text = "第一章 总则\n本章规定……"
    out, changed = normalizer.repair_line_breaks(text)
    # The chapter title stays on its own line.
    assert out.startswith("第一章 总则\n")
    assert changed is False  # nothing merged


def test_repair_line_breaks_keeps_list_items():
    text = "（1）第一项\n（2）第二项"
    out, _ = normalizer.repair_line_breaks(text)
    assert out == "（1）第一项\n（2）第二项"


def test_repair_line_breaks_keeps_blank_paragraphs():
    text = "第一段内容，没有句号\n\n第二段内容。"
    out, _ = normalizer.repair_line_breaks(text)
    # Blank line must NOT be collapsed; first line not ending in 。！ should
    # merge, but we preserve the blank between paragraphs.
    assert "\n\n" in out


def test_repair_line_breaks_keeps_terminal_punctuation_boundary():
    text = "第一段结束了。\n第二章 新章节"
    out, _ = normalizer.repair_line_breaks(text)
    assert out == "第一段结束了。\n第二章 新章节"


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_type, expected_level",
    [
        ("第一章 总则", "chapter", 1),
        ("第十一章", "chapter", 1),
        ("第一节 基层党组织", "section", 2),
        ("一、基层党组织的设置", "heading", 3),
        ("（一）基本原则", "subheading", 4),
        ("1. 坚持党的领导", "list_item", 5),
        ("1、坚持党的领导", "list_item", 5),
        ("（1）第一项", "list_item", 5),
        ("① 第一项", "list_item", 5),
        ("普通段落内容。", "paragraph", 6),
    ],
)
def test_detect_structure(text, expected_type, expected_level):
    info = structure.detect_structure(text)
    assert info.structure_type == expected_type
    assert info.structure_level == expected_level


def test_detect_structure_propagates_section_context():
    ctx = structure.SectionContext()
    ctx.update(structure.detect_structure("第一章 总则"))
    assert ctx.detected_section() == "第一章 总则"

    ctx.update(structure.detect_structure("这是第一个普通段落。"))
    assert ctx.detected_section() == "第一章 总则"

    ctx.update(structure.detect_structure("第二章 任务"))
    assert ctx.detected_section() == "第二章 任务"

    ctx.update(structure.detect_structure("这是新章节的第一个段落。"))
    assert ctx.detected_section() == "第二章 任务"


def test_detect_structure_table_kind_from_metadata():
    info = structure.detect_structure("| a | b |\n| 1 | 2 |", metadata={"kind": "table"})
    assert info.structure_type == "table"


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------


def test_quality_normal_chinese():
    q = quality.evaluate_quality("中国共产党是中国工人阶级的先锋队" * 5)
    assert q.is_empty is False
    assert q.is_too_short is False
    assert q.is_too_long is False
    assert q.possible_garbled is False
    assert q.chinese_char_ratio > 0.9


def test_quality_empty_text_flagged():
    q = quality.evaluate_quality("")
    assert q.is_empty is True
    assert "empty_after_cleaning" in q.flags


def test_quality_too_short_paragraph():
    q = quality.evaluate_quality("党")  # 1 char, paragraph
    assert q.is_too_short is True
    assert "too_short" in q.flags


def test_quality_heading_exempt_from_short():
    q = quality.evaluate_quality("第一章", exempt_short=True)
    assert q.is_too_short is False


def test_quality_too_long_text():
    q = quality.evaluate_quality("中" * (quality.MAX_TEXT_LENGTH + 10))
    assert q.is_too_long is True


def test_quality_symbol_ratio_garbled():
    q = quality.evaluate_quality("##########%%%%%%&&&&&&")
    assert q.possible_garbled is True


def test_quality_chinese_punctuation_not_counted_as_symbol():
    # 。  not in noise punctuation, should not push symbol_ratio over threshold.
    text = "中国共产党是中国工人阶级的先锋队。" * 5
    q = quality.evaluate_quality(text)
    assert q.symbol_ratio < 0.1


def test_quality_long_run_garbled():
    text = "啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊"
    q = quality.evaluate_quality(text)
    assert q.possible_garbled is True


def test_detect_header_footer_conservative():
    texts = ["党的工作"] * 10 + ["这是普通段落内容。"] * 5
    flagged = quality.detect_header_footer_candidates(texts, min_repeats=5, min_ratio=0.3)
    assert "党的工作" in flagged


def test_detect_header_footer_does_not_kill_normal_high_frequency():
    # Many unique paragraphs each mention "中国共产党" but each is unique.
    texts = [f"这是第{i}段关于中国共产党的重要论述。" for i in range(20)]
    flagged = quality.detect_header_footer_candidates(texts)
    assert flagged == {}


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_exact_dedup_keeps_first_occurrence():
    records = [
        ("a", "中国共产党章程"),
        ("b", "基层党务工作"),
        ("c", "中国共产党章程"),
        ("d", "基层党务工作"),
    ]
    dups = dedup.find_exact_duplicates(records)
    assert {d.duplicate_doc_id for d in dups} == {"c", "d"}
    assert {d.canonical_doc_id for d in dups} == {"a", "b"}


def test_exact_dedup_after_whitespace_normalization():
    records = [
        ("a", "党的工作。"),
        ("b", "党 的 工 作。"),
    ]
    dups = dedup.find_exact_duplicates(records)
    assert {d.duplicate_doc_id for d in dups} == {"b"}


def test_simhash_same_text_zero_distance():
    a = dedup.simhash("中国共产党是中国工人阶级的先锋队")
    b = dedup.simhash("中国共产党是中国工人阶级的先锋队")
    assert dedup.hamming_distance(a, b) == 0


def test_simhash_near_similar_small_distance():
    # Two texts that differ by only a few characters should have small
    # Hamming distance (not asserted to a specific value to stay robust).
    a = dedup.simhash("中国共产党是中国工人阶级的先锋队，同时是中国人民和中华民族的先锋队。")
    b = dedup.simhash("中国共产党是中国工人阶级的先锋队，同时也是中国人民和中华民族的先锋队。")
    assert dedup.hamming_distance(a, b) <= 8


def test_simhash_unrelated_large_distance():
    a = dedup.simhash("中国共产党是中国工人阶级的先锋队")
    b = dedup.simhash("Today is a beautiful day to study Python programming.")
    assert dedup.hamming_distance(a, b) > 20


def test_simhash_is_deterministic_across_runs():
    text = "中文测试：稳定 hash。" * 3
    a = dedup.simhash(text)
    b = dedup.simhash(text)
    assert a == b


def test_near_dedup_bucketing_finds_similar():
    base = "中国共产党是中国工人阶级的先锋队，同时是中国人民和中华民族的先锋队。重要论述。重要论述。"
    records = [
        ("a", base),
        ("b", base[:-1] + "。"),  # tiny diff
    ]
    dups = dedup.find_near_duplicates(records, max_distance=8, bucket_bits=16)
    assert {d.duplicate_doc_id for d in dups} == {"b"}


def test_deduplicate_runs_both_layers():
    records = [
        ("a", "完全相同的段落内容。"),
        ("b", "完全相同的段落内容。"),  # exact duplicate
        ("c", "中国共产党是中国工人阶级的先锋队。"),
        ("d", "中国共产党是中国工人阶级的先锋队，同时是中国人民和中华民族的先锋队。"),
    ]
    exact, near = dedup.deduplicate(records, max_distance=8)
    assert {d.duplicate_doc_id for d in exact} == {"b"}
    # near dedup skips already-exact entries; behavior depends on simhash
    assert isinstance(near, list)


# ---------------------------------------------------------------------------
# Pipeline end-to-end
# ---------------------------------------------------------------------------


def _build_records(pairs: list[tuple[str, str]]) -> list[dict]:
    return [
        {
            "doc_id": f"doc_{i:04d}",
            "source_type": "text",
            "source_path": "a.txt",
            "file_name": "a.txt",
            "record_index": i,
            "text": t,
        }
        for i, t in enumerate(pairs)
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_pipeline_accounting(tmp_path: Path):
    records = _build_records([
        "  党\u00a0\u00a0务   工作。",
        "第一章 总则",
        "中国共产党",
        "##########%%%%%%&&&&",
        "   ",
        "党的工作",  # too short
    ])
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, records)
    out_clean = tmp_path / "clean.jsonl"
    out_rejected = tmp_path / "rejected.jsonl"
    out_dups = tmp_path / "dupes.jsonl"
    out_report = tmp_path / "report.json"

    result = run_cleaning_pipeline(
        inp,
        out_clean,
        out_rejected,
        out_dups,
        out_report,
        config=CleaningConfig(min_length=5),
    )
    assert result.input_records == 6
    # input = clean + rejected + duplicates (no dups in this sample)
    assert result.output_records + result.rejected_records + result.exact_duplicates + result.near_duplicates == result.input_records

    assert out_clean.exists()
    assert out_rejected.exists()
    assert out_dups.exists()
    assert out_report.exists()

    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["input_records"] == 6
    assert "structure" in report
    assert "text_length" in report


def test_pipeline_preserves_original_text(tmp_path: Path):
    original = "  党\u00a0\u00a0务   工作。"
    records = _build_records([original])
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, records)
    out_clean = tmp_path / "clean.jsonl"
    run_cleaning_pipeline(
        inp, out_clean,
        tmp_path / "rejected.jsonl",
        tmp_path / "dupes.jsonl",
        tmp_path / "report.json",
    )
    line = out_clean.read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    assert rec["text"] == original
    assert rec["clean_text"] != original  # actually changed
    assert "unicode_nfkc" in rec["cleaning_actions"]


def test_pipeline_deterministic(tmp_path: Path):
    records = _build_records([
        "中国共产党是中国工人阶级的先锋队。",
        "中国共产党是中国工人阶级的先锋队",  # near duplicate
        "完全相同段落。",
        "完全相同段落。",  # exact duplicate
    ])
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, records)

    r1 = run_cleaning_pipeline(
        inp, tmp_path / "c1.jsonl",
        tmp_path / "r1.jsonl", tmp_path / "d1.jsonl", tmp_path / "rep1.json",
    )
    r2 = run_cleaning_pipeline(
        inp, tmp_path / "c2.jsonl",
        tmp_path / "r2.jsonl", tmp_path / "d2.jsonl", tmp_path / "rep2.json",
    )
    assert r1.input_records == r2.input_records
    assert r1.output_records == r2.output_records
    assert r1.rejected_records == r2.rejected_records
    assert r1.exact_duplicates == r2.exact_duplicates
    assert r1.near_duplicates == r2.near_duplicates
    # Output bytes must match.
    assert (tmp_path / "c1.jsonl").read_bytes() == (tmp_path / "c2.jsonl").read_bytes()
    assert (tmp_path / "d1.jsonl").read_bytes() == (tmp_path / "d2.jsonl").read_bytes()
    assert (tmp_path / "rep1.json").read_bytes() == (tmp_path / "rep2.json").read_bytes()


def test_pipeline_dedup_drops_exact(tmp_path: Path):
    records = _build_records(["完全相同段落。"] * 3)
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, records)
    out_clean = tmp_path / "clean.jsonl"
    out_dups = tmp_path / "dupes.jsonl"
    result = run_cleaning_pipeline(
        inp, out_clean,
        tmp_path / "rejected.jsonl", out_dups,
        tmp_path / "report.json",
    )
    assert result.exact_duplicates == 2
    assert result.output_records == 1
    dup_lines = out_dups.read_text(encoding="utf-8").splitlines()
    assert len(dup_lines) == 2


def test_pipeline_classifies_headings_correctly(tmp_path: Path):
    records = _build_records([
        "第一章 总则",
        "第一节 基层党组织",
        "一、基层党组织的设置",
        "（一）基本原则",
        "1. 坚持党的领导",
        "（1）第一项",
        "① 第一项",
        "普通段落内容。",
    ])
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, records)
    out_clean = tmp_path / "clean.jsonl"
    run_cleaning_pipeline(
        inp, out_clean,
        tmp_path / "rejected.jsonl",
        tmp_path / "dupes.jsonl",
        tmp_path / "report.json",
    )
    by_id = {json.loads(line)["doc_id"]: json.loads(line) for line in out_clean.read_text(encoding="utf-8").splitlines()}
    types = [by_id[f"doc_{i:04d}"]["structure_type"] for i in range(8)]
    assert types == [
        "chapter",
        "section",
        "heading",
        "subheading",
        "list_item",
        "list_item",
        "list_item",
        "paragraph",
    ]


def test_pipeline_section_propagation(tmp_path: Path):
    records = _build_records([
        "第一章 总则",
        "这是第一个普通段落。",
        "另一个段落。",
        "第二章 任务",
        "新章节的段落。",
    ])
    inp = tmp_path / "in.jsonl"
    _write_jsonl(inp, records)
    out_clean = tmp_path / "clean.jsonl"
    run_cleaning_pipeline(
        inp, out_clean,
        tmp_path / "rejected.jsonl",
        tmp_path / "dupes.jsonl",
        tmp_path / "report.json",
    )
    by_id = {json.loads(line)["doc_id"]: json.loads(line) for line in out_clean.read_text(encoding="utf-8").splitlines()}
    sections = [by_id[f"doc_{i:04d}"]["detected_section"] for i in range(5)]
    assert sections == [
        "第一章 总则",
        "第一章 总则",
        "第一章 总则",
        "第二章 任务",
        "第二章 任务",
    ]