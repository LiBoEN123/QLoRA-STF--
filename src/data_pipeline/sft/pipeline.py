"""Phase 4 SFT 数据集构造与确定性划分流水线。

调用链（唯一）::

    run_sft_pipeline
        ↓
    load_jsonl_records
        ↓
    _filter_faq_units          # 只保留 unit_type == "faq"
        ↓
    _build_samples             # FAQ → SFTSample + 字段校验
        ↓
    _dedup_and_conflicts        # exact dedup + 同问不同答冲突
        ↓
    _assign_splits             # 按 split_group_key 确定性划分
        ↓
    _check_leakage              # 跨 split 题目/样本泄漏检查
        ↓
    write outputs (atomic_write_jsonl/json)
        ↓
    build report

本模块不复用 Phase 3 的 FAQ detector / pairer，只消费已配对的
``SemanticUnit.unit_type == "faq"`` 数据。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from ..utils import atomic_write_json, atomic_write_jsonl, get_logger, load_jsonl_records
from .models import (
    DEFAULT_SPLIT_SALT,
    SFTSample,
    SFTBuildConfig,
    SPLIT_BUCKET_TOTAL,
)

# 置信度大小关系；用于把字符串映射为可比较整数。
_CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


# ---------------------------------------------------------------------------
# 顶层数据
# ---------------------------------------------------------------------------


@dataclass
class SFTBuildResult:
    """Phase 4 一次完整运行的汇总结果。"""

    # Phase 4 输入的 semantic_units.jsonl 路径
    input_path: Path

    # 全量 accepted SFT 样本输出路径
    output_all: Path

    # 训练集输出路径
    output_train: Path

    # 验证集输出路径
    output_validation: Path

    # 测试集输出路径
    output_test: Path

    # 被剔除样本输出路径
    output_rejected: Path

    # 数据集质量报告输出路径
    output_report: Path

    # semantic_units.jsonl 总记录数（含所有 unit_type）
    input_units: int = 0

    # 其中所有 unit_type == "faq" 的数量（不再受 confidence 影响）
    input_faq_units: int = 0

    # 通过 min_confidence 阈值、进入下一步构造的 FAQ 数量
    eligible_faq_units: int = 0

    # 低于 min_confidence 阈值、被置信度过滤掉的 FAQ 数量
    filtered_by_confidence: int = 0

    # 最终进入 all.jsonl 的 accepted 样本数
    accepted_samples: int = 0

    # 进入 rejected.jsonl 的样本数（含 below_min_confidence / dedup / conflict / 字段校验失败）
    rejected_samples: int = 0

    # exact q+a 重复而被剔除的样本数（canonical 仍计入 accepted）
    duplicate_samples: int = 0

    # 同问不同答冲突的题目组数
    conflicting_question_groups: int = 0

    # 同问不同答冲突被整组剔除的样本数
    conflicting_samples: int = 0

    # train / validation / test 各自分到的样本数
    splits: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _stable_hash(*parts: str) -> str:
    """对若干字符串拼接后计算 SHA256，返回完整 64 位 hex。"""
    payload = "\u241F".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_for_compare(text: str) -> str:
    """只用于比较/去重 key 的最小规范化：strip + 折叠普通空白。

    不用于修改训练内容；训练内容仍使用 ``question_text.strip()`` /
    ``answer_text.strip()`` 的原文。
    """
    if not text:
        return ""
    # 把任意空白（含 \r\n\t）折叠为单空格
    return re.sub(r"\s+", " ", text.strip())


def _make_sample_id(unit_id: str, question: str, answer: str) -> str:
    """根据 unit_id + 问题 + 答案生成稳定 sample_id（24 位 hex）。"""
    digest = _stable_hash(unit_id, question, answer)
    return f"sft_{digest[:24]}"


def _make_split_group_key(normalized_question: str) -> str:
    """对 ``normalized_question`` 独立求 SHA256，得到 split_group_key。

    这样所有同问样本会被分到同一 split，不会因 sample_id 不同而分散。
    """
    return _stable_hash("q", normalized_question)


def _bucket_for(group_key: str, salt: str) -> int:
    """按 SHA256(group_key + salt) 映射到 0..SPLIT_BUCKET_TOTAL-1 的桶。"""
    digest = _stable_hash(group_key, salt)
    return int(digest[:8], 16) % SPLIT_BUCKET_TOTAL


def _compute_split_boundaries(config: SFTBuildConfig, total: int = SPLIT_BUCKET_TOTAL) -> Tuple[int, int]:
    """根据 config 中的 ratio 计算 bucket 边界。

    返回 ``(train_end, validation_end)``：

    - ``[0, train_end)``        → train
    - ``[train_end, validation_end)`` → validation
    - ``[validation_end, total)``     → test

    边界由 ``round(ratio * total)`` 计算；与 config 完全一致，
    不再依赖任何硬编码业务常量。
    """
    train_end = round(config.train_ratio * total)
    validation_end = round((config.train_ratio + config.validation_ratio) * total)
    # 防御性夹紧：保证 train_end <= validation_end <= total
    train_end = max(0, min(total, train_end))
    validation_end = max(train_end, min(total, validation_end))
    return train_end, validation_end


def _bucket_to_split(bucket: int, config: SFTBuildConfig) -> str:
    """把 bucket 整数映射到 train / validation / test；边界来自 config。"""
    train_end, validation_end = _compute_split_boundaries(config)
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "test"


# ---------------------------------------------------------------------------
# Sample 构造与字段校验
# ---------------------------------------------------------------------------


def _validate_unit_basic(unit: Dict[str, Any]) -> List[str]:
    """对单个 SemanticUnit dict 做基础字段校验，返回 reject reason 列表。"""
    reasons: List[str] = []
    if not unit.get("unit_id"):
        reasons.append("missing_unit_id")
    q = unit.get("question_text")
    a = unit.get("answer_text")
    if not isinstance(q, str) or not q.strip():
        reasons.append("empty_question")
    if not isinstance(a, str) or not a.strip():
        reasons.append("empty_answer")
    conf = unit.get("faq_confidence")
    if conf not in _CONFIDENCE_RANK:
        reasons.append("invalid_confidence")
    if not unit.get("source_doc_ids"):
        reasons.append("missing_traceability")
    if not unit.get("source_record_indices"):
        reasons.append("missing_traceability")
    return reasons


def _build_sample(unit: Dict[str, Any]) -> SFTSample:
    """把 SemanticUnit dict 转成 SFTSample。"""
    question = unit["question_text"].strip()
    answer = unit["answer_text"].strip()
    sample_id = _make_sample_id(unit["unit_id"], question, answer)

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    metadata: Dict[str, Any] = {
        "unit_id": unit.get("unit_id"),
        "source_path": unit.get("source_path"),
        "file_name": unit.get("file_name"),
        "file_sha256": unit.get("file_sha256"),
        "source_doc_ids": list(unit.get("source_doc_ids") or []),
        "source_record_indices": list(unit.get("source_record_indices") or []),
        "faq_confidence": unit.get("faq_confidence"),
        "structure_path": list(unit.get("structure_path") or []),
        "detected_section": unit.get("detected_section"),
    }

    # 透传 Phase 3 写入的可选字段
    for k in ("question_record_index", "answer_record_count", "number_style"):
        if k in (unit.get("metadata") or {}):
            metadata[k] = unit["metadata"][k]

    return SFTSample(sample_id=sample_id, messages=messages, metadata=metadata)


def _validate_sample_messages(sample: SFTSample) -> List[str]:
    """校验 messages 的结构；不通过则返回 reject reason 列表。"""
    reasons: List[str] = []
    msgs = sample.messages
    if len(msgs) != 2:
        reasons.append("invalid_role_order")
        return reasons
    if msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
        reasons.append("invalid_role_order")
    for m in msgs:
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            reasons.append("empty_content")
    return reasons


# ---------------------------------------------------------------------------
# 过滤：只保留 FAQ，并按最低置信度筛
# ---------------------------------------------------------------------------


def _filter_faq_units(
    units: List[Dict[str, Any]],
    min_confidence: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """从所有 semantic units 中分出 FAQ，并按 ``min_confidence`` 切分。

    返回 ``(eligible_units, below_min_confidence_units)``：

    - ``eligible_units``：unit_type=="faq" 且 faq_confidence ≥ min_confidence；
    - ``below_min_confidence_units``：unit_type=="faq" 但 faq_confidence 不足，
      这些不会被静默丢弃，会在 pipeline 中写入 rejected.jsonl。
    """
    threshold = _CONFIDENCE_RANK[min_confidence]
    eligible: List[Dict[str, Any]] = []
    below: List[Dict[str, Any]] = []
    for u in units:
        if u.get("unit_type") != "faq":
            continue
        conf = u.get("faq_confidence")
        rank = _CONFIDENCE_RANK.get(conf, 0)
        if rank < threshold:
            below.append(u)
            continue
        eligible.append(u)
    return eligible, below


# ---------------------------------------------------------------------------
# Exact dedup + 同问不同答冲突
# ---------------------------------------------------------------------------


def _dedup_and_conflicts(
    accepted_pairs: List[Tuple[Dict[str, Any], SFTSample]],
) -> Tuple[
    List[Tuple[Dict[str, Any], SFTSample]],
    List[Dict[str, Any]],
    int,
    int,
    int,
]:
    """对 FAQ 单元做：

    1. exact dedup：相同 (normalized_q, normalized_a) ⇒ 保留 1 条，按 sample_id 升序。
    2. 冲突检测：相同 normalized_q 但不同 normalized_a ⇒ 该组所有单元均剔除。

    返回 ``(accepted_pairs, rejected_records, duplicate_count,
    conflicting_sample_count, conflicting_group_count)``。

    冲突处理原则：宁可全部剔除，不让冲突监督信号进入训练集。
    """
    # 先把每个 (u, s) 附上 (norm_q, norm_a)，便于分组
    enriched: List[Tuple[Dict[str, Any], SFTSample, str, str]] = []
    for u, s in accepted_pairs:
        question = (u.get("question_text") or "").strip()
        answer = (u.get("answer_text") or "").strip()
        norm_q = _normalize_for_compare(question)
        norm_a = _normalize_for_compare(answer)
        enriched.append((u, s, norm_q, norm_a))

    # 按 (norm_q, norm_a) 分组做 dedup。
    by_qa: Dict[Tuple[str, str], List[Tuple[Dict[str, Any], SFTSample]]] = defaultdict(list)
    for u, s, q, a in enriched:
        by_qa[(q, a)].append((u, s))

    dedup_accepted: List[Tuple[Dict[str, Any], SFTSample]] = []
    rejected: List[Dict[str, Any]] = []
    duplicate_count = 0

    for (q, a), group in by_qa.items():
        # 同 (q, a) 内按 sample_id 升序，保留首条作 canonical。
        group_sorted = sorted(group, key=lambda x: x[1].sample_id)
        canonical_u, canonical_s = group_sorted[0]
        dedup_accepted.append((canonical_u, canonical_s))
        for dup_u, dup_s in group_sorted[1:]:
            rejected.append({
                "unit_id": dup_u.get("unit_id"),
                "question_text": dup_u.get("question_text"),
                "answer_text": dup_u.get("answer_text"),
                "reject_reasons": ["duplicate_sample"],
                "canonical_sample_id": canonical_s.sample_id,
                "sample_id": dup_s.sample_id,
            })
            duplicate_count += 1

    # 按 norm_q 分组做冲突检测：同问不同答 ⇒ 整组剔除。
    by_q: Dict[str, List[Tuple[Dict[str, Any], SFTSample]]] = defaultdict(list)
    for u, s in dedup_accepted:
        by_q[_normalize_for_compare((u.get("question_text") or "").strip())].append((u, s))

    final_accepted: List[Tuple[Dict[str, Any], SFTSample]] = []
    conflicting_group_count = 0
    conflicting_sample_count = 0
    for q, group in by_q.items():
        distinct_answers = {_normalize_for_compare((u.get("answer_text") or "").strip()) for u, _ in group}
        if len(distinct_answers) > 1:
            conflicting_group_count += 1
            for u, s in group:
                rejected.append({
                    "unit_id": u.get("unit_id"),
                    "question_text": u.get("question_text"),
                    "answer_text": u.get("answer_text"),
                    "reject_reasons": ["conflicting_answer"],
                    "sample_id": s.sample_id,
                })
                conflicting_sample_count += 1
            continue
        # 单 answer 的组：保持稳定顺序
        group_sorted = sorted(group, key=lambda x: x[1].sample_id)
        final_accepted.extend(group_sorted)

    return final_accepted, rejected, duplicate_count, conflicting_sample_count, conflicting_group_count


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------


def _assign_splits(
    accepted: List[Tuple[Dict[str, Any], SFTSample]],
    config: SFTBuildConfig,
) -> Dict[str, List[SFTSample]]:
    """按 ``split_group_key = SHA256(normalized_question)`` 确定性划分。"""
    splits: Dict[str, List[SFTSample]] = {"train": [], "validation": [], "test": []}
    for _, sample in accepted:
        q = sample.messages[0]["content"]
        group_key = _make_split_group_key(_normalize_for_compare(q))
        bucket = _bucket_for(group_key, config.split_salt)
        splits[_bucket_to_split(bucket, config)].append(sample)
    for k in splits:
        splits[k].sort(key=lambda s: s.sample_id)
    return splits


# ---------------------------------------------------------------------------
# 泄漏检查
# ---------------------------------------------------------------------------


def _check_no_leakage(splits: Dict[str, List[SFTSample]]) -> Tuple[int, int]:
    """检查 normalized question / (q+a) 是否跨 split 共现。"""
    q_to_splits: Dict[str, set] = defaultdict(set)
    qa_to_splits: Dict[Tuple[str, str], set] = defaultdict(set)
    for name, samples in splits.items():
        for s in samples:
            q = _normalize_for_compare(s.messages[0]["content"])
            a = _normalize_for_compare(s.messages[1]["content"])
            q_to_splits[q].add(name)
            qa_to_splits[(q, a)].add(name)
    cross_q = sum(1 for v in q_to_splits.values() if len(v) > 1)
    cross_qa = sum(1 for v in qa_to_splits.values() if len(v) > 1)
    return cross_q, cross_qa


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _length_stats(values: List[int]) -> Dict[str, float]:
    """返回 min/mean/median/p95/max 长度统计；空序列返回全 0。"""
    if not values:
        return {"min": 0, "mean": 0, "median": 0, "p95": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    return {
        "min": s[0],
        "mean": round(mean(s), 2),
        "median": s[n // 2],
        "p95": s[p95_idx],
        "max": s[-1],
    }


def build_report(
    input_units: int,
    input_faq_units: int,
    eligible_faq_units: int,
    filtered_by_confidence: int,
    accepted_samples: List[SFTSample],
    rejected_samples: List[Dict[str, Any]],
    duplicate_samples: int,
    conflicting_question_groups: int,
    conflicting_samples: int,
    splits: Dict[str, List[SFTSample]],
    config: SFTBuildConfig,
) -> Dict[str, Any]:
    """构造 ``dataset_report.json`` 报告内容。"""
    accepted_count = len(accepted_samples)
    rejected_count = len(rejected_samples)

    conf_counter: Counter = Counter()
    q_lengths: List[int] = []
    a_lengths: List[int] = []
    invalid_role = 0
    empty_content = 0
    unique_unit_ids: set = set()
    unique_doc_ids: set = set()
    unique_paths: set = set()

    for s in accepted_samples:
        msgs = s.messages
        if len(msgs) != 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
            invalid_role += 1
        for m in msgs:
            if not (isinstance(m.get("content"), str) and m["content"].strip()):
                empty_content += 1

        q_lengths.append(len(msgs[0]["content"]))
        a_lengths.append(len(msgs[1]["content"]))

        meta = s.metadata
        conf_counter[meta.get("faq_confidence")] += 1
        if meta.get("unit_id"):
            unique_unit_ids.add(meta["unit_id"])
        unique_doc_ids.update(meta.get("source_doc_ids") or [])
        if meta.get("source_path"):
            unique_paths.add(meta["source_path"])

    cross_q, cross_qa = _check_no_leakage(splits)

    train_n = len(splits["train"])
    valid_n = len(splits["validation"])
    test_n = len(splits["test"])
    total_split = max(1, train_n + valid_n + test_n)

    train_end, validation_end = _compute_split_boundaries(config)

    return {
        "input_units": input_units,
        "input_faq_units": input_faq_units,
        "eligible_faq_units": eligible_faq_units,
        "filtered_by_confidence": filtered_by_confidence,
        "accepted_samples": accepted_count,
        "rejected_samples": rejected_count,
        "duplicate_samples": duplicate_samples,
        "conflicting_question_groups": conflicting_question_groups,
        "conflicting_samples": conflicting_samples,
        "confidence": {
            "high": conf_counter.get("high", 0),
            "medium": conf_counter.get("medium", 0),
        },
        "splits": {
            "train": train_n,
            "validation": valid_n,
            "test": test_n,
        },
        "split_ratios_actual": {
            "train": round(train_n / total_split, 4),
            "validation": round(valid_n / total_split, 4),
            "test": round(test_n / total_split, 4),
        },
        "split_ratios_target": {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        },
        "split_bucket_boundaries": {
            "bucket_total": SPLIT_BUCKET_TOTAL,
            "train_end": train_end,
            "validation_end": validation_end,
        },
        "split_salt": config.split_salt,
        "question_length": _length_stats(q_lengths),
        "answer_length": _length_stats(a_lengths),
        "messages": {
            "invalid_role_order": invalid_role,
            "empty_content": empty_content,
        },
        "leakage": {
            "cross_split_question": cross_q,
            "cross_split_exact_sample": cross_qa,
        },
        "traceability": {
            "unique_unit_ids": len(unique_unit_ids),
            "unique_source_doc_ids": len(unique_doc_ids),
            "unique_source_paths": len(unique_paths),
        },
        "min_confidence": config.min_confidence,
    }


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------


def serialize_sample(sample: SFTSample) -> str:
    """把 SFTSample 序列化为 JSONL 行。"""
    payload = {
        "sample_id": sample.sample_id,
        "messages": sample.messages,
        "metadata": sample.metadata,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def serialize_rejected(record: Dict[str, Any]) -> str:
    """把 rejected 记录序列化为 JSONL 行。"""
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_sft_pipeline(
    input_path: Path,
    output_all: Path,
    output_train: Path,
    output_validation: Path,
    output_test: Path,
    output_rejected: Path,
    output_report: Path,
    *,
    config: Optional[SFTBuildConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> SFTBuildResult:
    """运行 Phase 4 SFT 数据集构造与划分流水线。"""
    log = logger or get_logger("data_pipeline.sft")
    cfg = config or SFTBuildConfig()

    result = SFTBuildResult(
        input_path=input_path,
        output_all=output_all,
        output_train=output_train,
        output_validation=output_validation,
        output_test=output_test,
        output_rejected=output_rejected,
        output_report=output_report,
    )

    log.info("Phase 4 SFT pipeline started")
    log.info("Input: %s", input_path)
    log.info("Config: min_confidence=%s, ratios=(%s,%s,%s), split_salt=%s",
             cfg.min_confidence, cfg.train_ratio, cfg.validation_ratio, cfg.test_ratio, cfg.split_salt)

    # 1) 读取
    units = load_jsonl_records(input_path)
    result.input_units = len(units)
    log.info("Loaded %d semantic units", len(units))

    # 2) 只保留 FAQ；按 min_confidence 切分为 eligible / below_min_confidence
    faq_units_all = [u for u in units if u.get("unit_type") == "faq"]
    result.input_faq_units = len(faq_units_all)
    faq_units, below_min = _filter_faq_units(units, cfg.min_confidence)
    result.eligible_faq_units = len(faq_units)
    result.filtered_by_confidence = len(below_min)
    log.info(
        "FAQ units: total=%d, eligible=%d, below_min_confidence=%d",
        result.input_faq_units, result.eligible_faq_units, result.filtered_by_confidence,
    )

    # 3) 构造 sample + 字段校验
    accepted_pairs: List[Tuple[Dict[str, Any], SFTSample]] = []
    rejected: List[Dict[str, Any]] = []

    # 3a) 先把低于置信度的 FAQ 写入 rejected.jsonl，避免静默消失
    for u in below_min:
        rejected.append({
            "unit_id": u.get("unit_id"),
            "question_text": u.get("question_text"),
            "answer_text": u.get("answer_text"),
            "faq_confidence": u.get("faq_confidence"),
            "required_min_confidence": cfg.min_confidence,
            "reject_reasons": ["below_min_confidence"],
        })

    for u in faq_units:
        reasons = _validate_unit_basic(u)
        if reasons:
            rejected.append({
                "unit_id": u.get("unit_id"),
                "question_text": u.get("question_text"),
                "answer_text": u.get("answer_text"),
                "reject_reasons": reasons,
            })
            continue
        sample = _build_sample(u)
        msg_reasons = _validate_sample_messages(sample)
        if msg_reasons:
            rejected.append({
                "unit_id": u.get("unit_id"),
                "question_text": u.get("question_text"),
                "answer_text": u.get("answer_text"),
                "sample_id": sample.sample_id,
                "reject_reasons": msg_reasons,
            })
            continue
        accepted_pairs.append((u, sample))

    log.info("Field-valid samples: %d; rejected: %d", len(accepted_pairs), len(rejected))

    # 4) Dedup + 冲突（按 unit_id 重建关联，避免 id() 不稳定）
    sample_by_unit_id: Dict[str, SFTSample] = {u["unit_id"]: s for u, s in accepted_pairs}
    (
        final_pairs,
        more_rejected,
        duplicate_count,
        conflicting_sample_count,
        conflicting_group_count,
    ) = _dedup_and_conflicts(accepted_pairs)
    final_pairs_full: List[Tuple[Dict[str, Any], SFTSample]] = [
        (u, sample_by_unit_id[u["unit_id"]]) for u, _ in final_pairs
    ]

    rejected.extend(more_rejected)
    result.duplicate_samples = duplicate_count
    result.conflicting_samples = conflicting_sample_count
    result.conflicting_question_groups = conflicting_group_count

    log.info(
        "Dedup: %d duplicate(s) collapsed; conflicts: %d sample(s) in %d group(s) removed",
        duplicate_count, conflicting_sample_count, conflicting_group_count,
    )

    # 5) 划分
    splits = _assign_splits(final_pairs_full, cfg)
    for k, v in splits.items():
        result.splits[k] = len(v)
    log.info("Splits: train=%d, validation=%d, test=%d",
             len(splits["train"]), len(splits["validation"]), len(splits["test"]))

    # 6) 泄漏检查（硬约束）
    cross_q, cross_qa = _check_no_leakage(splits)
    if cross_q != 0 or cross_qa != 0:
        raise RuntimeError(
            f"跨 split 泄漏：question={cross_q}, exact_sample={cross_qa}（必须为 0）"
        )
    log.info("Leakage check: cross_split_question=%d, cross_split_exact_sample=%d (both 0)", cross_q, cross_qa)

    # 7) 排序与写出
    all_samples = [s for _, s in final_pairs_full]
    all_samples_sorted = sorted(all_samples, key=lambda s: s.sample_id)
    accepted_count = len(all_samples_sorted)
    rejected_count = len(rejected)

    # 拒绝样本也按稳定顺序：先 reject_reasons 字符串排序，再按 unit_id / sample_id
    rejected_sorted = sorted(
        rejected,
        key=lambda r: (sorted(r.get("reject_reasons") or []), r.get("unit_id") or "", r.get("sample_id") or ""),
    )

    result.accepted_samples = accepted_count
    result.rejected_samples = rejected_count

    atomic_write_jsonl(output_all, [serialize_sample(s) for s in all_samples_sorted])
    atomic_write_jsonl(output_train, [serialize_sample(s) for s in splits["train"]])
    atomic_write_jsonl(output_validation, [serialize_sample(s) for s in splits["validation"]])
    atomic_write_jsonl(output_test, [serialize_sample(s) for s in splits["test"]])
    atomic_write_jsonl(output_rejected, [serialize_rejected(r) for r in rejected_sorted])

    # 8) 报告
    report = build_report(
        input_units=result.input_units,
        input_faq_units=result.input_faq_units,
        eligible_faq_units=result.eligible_faq_units,
        filtered_by_confidence=result.filtered_by_confidence,
        accepted_samples=all_samples_sorted,
        rejected_samples=rejected_sorted,
        duplicate_samples=duplicate_count,
        conflicting_question_groups=conflicting_group_count,
        conflicting_samples=conflicting_sample_count,
        splits=splits,
        config=cfg,
    )
    atomic_write_json(output_report, report)

    log.info(
        "Wrote all=%d, train=%d, validation=%d, test=%d, rejected=%d, report=%s",
        accepted_count, len(splits["train"]), len(splits["validation"]), len(splits["test"]),
        rejected_count, output_report,
    )

    return result
