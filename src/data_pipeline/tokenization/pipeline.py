"""Phase 5A — Qwen3 tokenizer 审计流水线。

调用链::

    run_tokenizer_audit
        ↓
    _build_split_membership          # 读 train/val/test → sample_id → split
        ↓
    _verify_split_membership         # all.jsonl 每个 sample 恰好属于一个 split
        ↓
    _load_tokenizer                  # AutoTokenizer（CPU only，trust_remote_code=False）
        ↓
    _describe_tokenizer              # 收集 tokenizer 元数据 + chat template 可用性
        ↓
    _scan_source_thinking_markers    # 扫描原始 messages 是否含 <think> 痕迹
        ↓
    _render_synthetic_template       # 用 synthetic messages 验证 Qwen3 非思考模板
        ↓
    _audit_samples                   # 逐 sample 计算 token 长度与 prefix match
        ↓
    _summarize_lengths               # 长度统计 + coverage + longest 20
        ↓
    _build_report                    # 汇总为 token_report.json
        ↓
    atomic_write_jsonl/json

Phase 5A 严格：
- 不加载 ``AutoModelForCausalLM``；不初始化 CUDA。
- 不手工拼 ChatML；唯一真值是 ``tokenizer.apply_chat_template``。
- 不做 truncation / padding。
- 不修改 SFT messages；不改写上游 Phase 1/2/3/4 数据。
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..utils import atomic_write_json, atomic_write_jsonl, get_logger, load_jsonl_records
from .models import MINIMUM_TRANSFORMERS_VERSION, TokenizerAuditConfig


# Qwen3 chat template 在 generation prompt 下可能预填空的 thinking scaffold。
# 这里只观察，不做任何 replace / 删除。
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# 当前 SFT schema 中不存在，但保留扫描以防御上游意外引入。
REASONING_CONTENT_KEY = "reasoning_content"


# ---------------------------------------------------------------------------
# 顶层异常
# ---------------------------------------------------------------------------


class TokenizerAuditError(RuntimeError):
    """Phase 5A 任意硬性条件不满足时抛出。"""


# ---------------------------------------------------------------------------
# Split membership
# ---------------------------------------------------------------------------


def _build_split_membership(sft_dir: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    """读 train/val/test.jsonl，建立 ``sample_id -> split`` 映射。

    返回 ``(membership, counts)``；counts 仅用于日志。
    """
    membership: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for split_name, filename in (
        ("train", "train.jsonl"),
        ("validation", "validation.jsonl"),
        ("test", "test.jsonl"),
    ):
        path = sft_dir / filename
        records = load_jsonl_records(path)
        counts[split_name] = len(records)
        for r in records:
            sid = r.get("sample_id")
            if not sid:
                raise TokenizerAuditError(f"{path} 中存在缺 sample_id 的记录")
            if sid in membership:
                raise TokenizerAuditError(
                    f"sample_id {sid} 同时出现在 {membership[sid]} 与 {split_name}"
                )
            membership[sid] = split_name
    return membership, counts


def _verify_split_membership(
    all_records: Sequence[Mapping[str, Any]],
    membership: Mapping[str, str],
) -> List[str]:
    """校验 ``all.jsonl`` 中每个 sample_id 恰好存在于一个 split。

    返回属于 split 的 sample 列表；任何缺失 / 重复都会抛 ``TokenizerAuditError``。
    """
    seen: Dict[str, str] = {}
    for r in all_records:
        sid = r.get("sample_id")
        if not sid:
            raise TokenizerAuditError("all.jsonl 中存在缺 sample_id 的记录")
        if sid not in membership:
            raise TokenizerAuditError(f"sample_id {sid} 不在 train/validation/test 任一 split")
        split = membership[sid]
        if sid in seen and seen[sid] != split:
            raise TokenizerAuditError(f"sample_id {sid} 在 all.jsonl 出现多次")
        seen[sid] = split
    return [r for r in all_records]


# ---------------------------------------------------------------------------
# Tokenizer 加载
# ---------------------------------------------------------------------------


def _import_transformers() -> Any:
    """延迟 import transformers；缺包时给出明确提示。"""
    try:
        import transformers  # type: ignore
    except ImportError as e:
        raise TokenizerAuditError(
            "未安装 transformers；Phase 5A 需要 transformers>=4.51.0。"
            "请运行 `pip install 'transformers>=4.51.0'`。"
        ) from e
    return transformers


def _check_transformers_version(transformers_module: Any) -> str:
    """校验 transformers 版本是否满足最低要求。"""
    from packaging.version import Version  # type: ignore

    current = Version(transformers_module.__version__)
    required = Version(MINIMUM_TRANSFORMERS_VERSION)
    if current < required:
        raise TokenizerAuditError(
            f"transformers 版本过低：当前 {current}，需要 >= {required}"
        )
    return str(current)


def _load_tokenizer(model_name: str, revision: str) -> Any:
    """通过 ``AutoTokenizer.from_pretrained`` 加载 tokenizer（CPU only）。

    绝对不加载 ``AutoModelForCausalLM``，不调用 ``torch.cuda``。
    """
    transformers = _import_transformers()
    AutoTokenizer = transformers.AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=True,
            trust_remote_code=False,
        )
    except Exception as e:
        raise TokenizerAuditError(
            f"加载 tokenizer 失败：model_name={model_name}, revision={revision}, err={e}"
        ) from e
    return tokenizer


def _resolve_revision(tokenizer: Any) -> str | None:
    """尽力从 tokenizer 元数据中解析 commit hash。

    不同 transformers 版本下字段位置不完全一致；找不到就返回 ``None``。
    """
    for attr_chain in (
        ("init_kwargs",),
        ("_from_pipeline",),
    ):
        obj = tokenizer
        ok = True
        for name in attr_chain:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if not ok:
            continue
        if isinstance(obj, Mapping):
            revision = obj.get("revision")
            if isinstance(revision, str) and revision:
                return revision

    # 部分版本把 commit_sha 放在 name_or_path 旁的子对象
    try:
        info = getattr(tokenizer, "model_max_length", None)
        # 尝试访问 backend_tokenizer 或同名属性
        for attr in ("backend_tokenizer", "sp_model"):
            if hasattr(tokenizer, attr):
                sub = getattr(tokenizer, attr)
                if isinstance(sub, Mapping):
                    rev = sub.get("revision")
                    if isinstance(rev, str) and rev:
                        return rev
    except Exception:
        return None
    return None


def _describe_tokenizer(tokenizer: Any, config: TokenizerAuditConfig, transformers_version: str) -> Dict[str, Any]:
    """收集 tokenizer 元数据用于 report。"""
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_present = bool(chat_template) if chat_template else False

    # resolved_revision：尽力解析；拿不到就 None
    resolved_revision = _resolve_revision(tokenizer)

    info: Dict[str, Any] = {
        "model_name": config.model_name,
        "requested_revision": config.revision,
        "resolved_revision": resolved_revision,
        "transformers_version": transformers_version,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "chat_template_present": chat_template_present,
    }

    # 失败：chat template 不可用
    if not chat_template_present:
        raise TokenizerAuditError("CHAT_TEMPLATE_MISSING: tokenizer.chat_template 为空")

    return info


def _describe_model_config(config: TokenizerAuditConfig) -> Dict[str, Any]:
    """读取 ``AutoConfig`` 提供的模型上下文信息，只读不改。"""
    transformers = _import_transformers()
    try:
        model_config = transformers.AutoConfig.from_pretrained(
            config.model_name,
            revision=config.revision,
            trust_remote_code=False,
        )
    except Exception as e:
        return {
            "model_type": None,
            "max_position_embeddings": None,
            "rope_scaling": None,
            "load_error": str(e),
        }
    return {
        "model_type": getattr(model_config, "model_type", None),
        "max_position_embeddings": getattr(model_config, "max_position_embeddings", None),
        "rope_scaling": getattr(model_config, "rope_scaling", None),
        "load_error": None,
    }


# ---------------------------------------------------------------------------
# Thinking marker 扫描
# ---------------------------------------------------------------------------


def _scan_source_thinking_markers(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """扫描原始 messages 的 user/assistant content 是否出现 thinking 痕迹。"""
    counter = {
        "think_open": 0,
        "think_close": 0,
        "reasoning_content": 0,
    }
    for r in records:
        msgs = r.get("messages") or []
        for m in msgs:
            content = m.get("content")
            if isinstance(content, str):
                if THINK_OPEN in content:
                    counter["think_open"] += content.count(THINK_OPEN)
                if THINK_CLOSE in content:
                    counter["think_close"] += content.count(THINK_CLOSE)
            if m.get(REASONING_CONTENT_KEY):
                counter["reasoning_content"] += 1
    return counter


# ---------------------------------------------------------------------------
# Synthetic template 渲染
# ---------------------------------------------------------------------------


_SYNTHETIC_MESSAGES_FULL: List[Dict[str, str]] = [
    {"role": "user", "content": "测试问题。"},
    {"role": "assistant", "content": "测试答案。"},
]


def _render_synthetic(tokenizer: Any) -> Dict[str, Any]:
    """用 synthetic messages 渲染 Qwen3 non-thinking 模板。

    完整渲染 + generation-prompt 渲染 + 是否出现 thinking scaffold。
    """
    full_text = tokenizer.apply_chat_template(
        _SYNTHETIC_MESSAGES_FULL,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        _SYNTHETIC_MESSAGES_FULL[:1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {
        "synthetic_messages": _SYNTHETIC_MESSAGES_FULL,
        "synthetic_rendered_full": full_text,
        "synthetic_rendered_prompt": prompt_text,
        "thinking_mode": "disabled",
        "synthetic_full_has_think_open": THINK_OPEN in full_text,
        "synthetic_full_has_think_close": THINK_CLOSE in full_text,
        "synthetic_prompt_has_think_open": THINK_OPEN in prompt_text,
        "synthetic_prompt_has_think_close": THINK_CLOSE in prompt_text,
        "template_manually_modified": False,
    }


# ---------------------------------------------------------------------------
# Token 长度审计
# ---------------------------------------------------------------------------


def _token_lengths(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> Tuple[List[int], List[int]]:
    """对单条样本做 full + prompt tokenize，返回 ``(full_ids, prompt_ids)``。

    严格 ``truncation=False``，``padding=False``；完全 raw token 序列。

    transformers 不同版本可能返回：

    - ``list[int]``；
    - ``torch.Tensor``（无 torch 环境不会走这里）；
    - ``BatchEncoding`` 字典（含 ``input_ids`` key）；
    - 元素类型为 tensor / numpy。

    本函数统一规整为 ``list[int]``。
    """
    full_ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.apply_chat_template(
        list(messages[:1]),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_ids = _extract_input_ids(full_ids)
    prompt_ids = _extract_input_ids(prompt_ids)
    return full_ids, prompt_ids


def _extract_input_ids(value: Any) -> List[int]:
    """把 transformers ``apply_chat_template(tokenize=True)`` 的返回值规整为 ``list[int]``。"""
    # BatchEncoding / dict：取 input_ids
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    # tensor：调用 .tolist()（依赖 torch）；无 torch 不会到这里
    if hasattr(value, "tolist") and not isinstance(value, list):
        value = value.tolist()
    # 嵌套 list（如 [[ids...]]）：展一层
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    # numpy：转 list
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.ndarray):
            value = value.tolist()
    except ImportError:
        pass
    return [int(x) for x in value]


def _audit_samples(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    membership: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """逐 sample 计算 token 长度与 prefix match，返回 audit rows。"""
    rows: List[Dict[str, Any]] = []
    for r in records:
        sid = r["sample_id"]
        messages = r.get("messages") or []
        # 防御性：必须是 2 条 user/assistant
        if len(messages) != 2 or messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise TokenizerAuditError(f"sample {sid} 的 messages 结构非法")

        full_ids, prompt_ids = _token_lengths(tokenizer, messages)
        prefix_match = list(full_ids[: len(prompt_ids)]) == list(prompt_ids)
        assistant_target_tokens = max(0, len(full_ids) - len(prompt_ids)) if prefix_match else 0
        meta = r.get("metadata") or {}
        rows.append({
            "sample_id": sid,
            "split": membership[sid],
            "total_tokens": len(full_ids),
            "prompt_tokens": len(prompt_ids),
            "assistant_target_tokens": assistant_target_tokens,
            "prompt_prefix_match": prefix_match,
            "metadata": {
                "unit_id": meta.get("unit_id"),
                "source_path": meta.get("source_path"),
                "source_record_indices": list(meta.get("source_record_indices") or []),
            },
        })
    rows.sort(key=lambda x: x["sample_id"])
    return rows


# ---------------------------------------------------------------------------
# 统计：长度 / coverage / longest / per-split
# ---------------------------------------------------------------------------


def _percentile(sorted_values: Sequence[int], p: float) -> float:
    """线性插值式百分位；空序列返回 0。"""
    if not sorted_values:
        return 0
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    idx = p * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def _length_stats(values: Sequence[int]) -> Dict[str, float]:
    """min/mean/median/p90/p95/p99/max；空序列返回全 0。"""
    if not values:
        return {"min": 0, "mean": 0.0, "median": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    s = sorted(values)
    return {
        "min": s[0],
        "mean": round(mean(s), 2),
        "median": _percentile(s, 0.5),
        "p90": _percentile(s, 0.90),
        "p95": _percentile(s, 0.95),
        "p99": _percentile(s, 0.99),
        "max": s[-1],
    }


def _coverage(values: Sequence[int], candidate: int) -> Dict[str, float]:
    """计算 ``<= candidate`` 的 fit 比例。"""
    total = max(1, len(values))
    fit = sum(1 for v in values if v <= candidate)
    return {
        "candidate_max_length": candidate,
        "fit_count": fit,
        "fit_ratio": round(fit / total, 4),
        "over_count": total - fit,
        "over_ratio": round((total - fit) / total, 4),
    }


def _smallest_candidate_covering(values: Sequence[int], candidates: Sequence[int], target_ratio: float) -> int | None:
    """返回首个 fit_ratio >= target_ratio 的候选长度；找不到为 ``None``。"""
    total = max(1, len(values))
    for c in sorted(set(candidates)):
        fit = sum(1 for v in values if v <= c)
        if fit / total >= target_ratio:
            return c
    return None


def _longest_samples(rows: Sequence[Mapping[str, Any]], k: int = 20) -> List[Dict[str, Any]]:
    """返回 ``total_tokens`` 降序的前 k 条；只保留定位 metadata。"""
    sorted_rows = sorted(rows, key=lambda x: (-x["total_tokens"], x["sample_id"]))
    out: List[Dict[str, Any]] = []
    for r in sorted_rows[:k]:
        meta = r.get("metadata") or {}
        out.append({
            "sample_id": r["sample_id"],
            "split": r["split"],
            "total_tokens": r["total_tokens"],
            "prompt_tokens": r["prompt_tokens"],
            "assistant_target_tokens": r["assistant_target_tokens"],
            "unit_id": meta.get("unit_id"),
            "source_path": meta.get("source_path"),
            "source_record_indices": meta.get("source_record_indices"),
        })
    return out


def _split_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """按 split 计算 count + mean/p95/max 的轻量统计。"""
    by_split: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
    for r in rows:
        by_split[r["split"]].append(r["total_tokens"])

    out: Dict[str, Any] = {}
    for name, vs in by_split.items():
        s = sorted(vs)
        out[name] = {
            "count": len(s),
            "mean_total_tokens": round(mean(s), 2) if s else 0.0,
            "median_total_tokens": _percentile(s, 0.5) if s else 0,
            "p95_total_tokens": _percentile(s, 0.95) if s else 0,
            "max_total_tokens": s[-1] if s else 0,
        }
    return out


# ---------------------------------------------------------------------------
# Tokenizer roundtrip 基础验证
# ---------------------------------------------------------------------------


def _verify_roundtrip(tokenizer: Any) -> Dict[str, Any]:
    """decode → 保留关键子串即可，不要求 byte-for-byte 一致。"""
    full_text = tokenizer.apply_chat_template(
        _SYNTHETIC_MESSAGES_FULL,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    full_ids = tokenizer.apply_chat_template(
        _SYNTHETIC_MESSAGES_FULL,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    decoded = tokenizer.decode(full_ids, skip_special_tokens=False)
    has_question = "测试问题" in decoded
    has_answer = "测试答案" in decoded
    return {
        "decoded_contains_question": has_question,
        "decoded_contains_answer": has_answer,
        "decoded_text": decoded,
        "rendered_text": full_text,
        "ok": has_question and has_answer,
    }


def _verify_real_samples(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    per_split: int,
) -> Dict[str, Any]:
    """对每个 split 抽 ``per_split`` 条做内部 sanity；结果不入 report 主字段。"""
    by_split: Dict[str, List[Mapping[str, Any]]] = {"train": [], "validation": [], "test": []}
    for r in rows:
        by_split[r["split"]].append(r)

    summary: Dict[str, Any] = {"per_split_check": {}}
    total_checked = 0
    total_failed = 0
    failed_ids: List[str] = []
    for name, items in by_split.items():
        subset = items[:per_split]
        sub_failed: List[str] = []
        for it in subset:
            total_checked += 1
            if not it["prompt_prefix_match"] or it["assistant_target_tokens"] <= 0:
                sub_failed.append(it["sample_id"])
                total_failed += 1
                failed_ids.append(it["sample_id"])
        summary["per_split_check"][name] = {
            "checked": len(subset),
            "failed": len(sub_failed),
            "failed_sample_ids": sub_failed,
        }
    summary["checked_count"] = total_checked
    summary["failed_count"] = total_failed
    summary["failed_sample_ids"] = failed_ids
    return summary


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_tokenizer_audit(
    sft_dir: Path,
    output_audit_jsonl: Path,
    output_report_json: Path,
    *,
    config: TokenizerAuditConfig | None = None,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """运行 Phase 5A tokenizer 审计流水线。"""
    log = logger or get_logger("data_pipeline.tokenization")
    cfg = config or TokenizerAuditConfig()

    # 显式声明：审计阶段不应需要 GPU
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        log.info("CUDA_VISIBLE_DEVICES=%s（Phase 5A 不使用 GPU）", os.environ["CUDA_VISIBLE_DEVICES"])

    transformers_module = _import_transformers()
    transformers_version = _check_transformers_version(transformers_module)
    log.info("transformers=%s", transformers_version)

    # 1) split membership
    membership, split_counts = _build_split_membership(sft_dir)
    log.info(
        "Split counts: train=%d, validation=%d, test=%d",
        split_counts.get("train", 0), split_counts.get("validation", 0), split_counts.get("test", 0),
    )

    # 2) load all.jsonl
    all_records = load_jsonl_records(sft_dir / "all.jsonl")
    if not all_records:
        raise TokenizerAuditError(f"{sft_dir / 'all.jsonl'} 为空")

    # 3) verify membership
    verified = _verify_split_membership(all_records, membership)
    log.info("Verified %d samples against split membership", len(verified))

    # 4) load tokenizer
    tokenizer = _load_tokenizer(cfg.model_name, cfg.revision)
    log.info("Loaded tokenizer: %s", type(tokenizer).__name__)

    # 5) tokenizer metadata
    tokenizer_info = _describe_tokenizer(tokenizer, cfg, transformers_version)

    # 6) model config（可选）
    model_config_info = _describe_model_config(cfg)

    # 7) source thinking markers
    thinking_counters = _scan_source_thinking_markers(verified)

    # 8) synthetic template render
    synthetic = _render_synthetic(tokenizer)

    # 9) roundtrip
    roundtrip = _verify_roundtrip(tokenizer)
    if not roundtrip["ok"]:
        raise TokenizerAuditError("synthetic roundtrip 失败：decode 后丢失关键子串")

    # 10) audit samples
    rows = _audit_samples(tokenizer, verified, membership)

    # 11) 统计
    total_tokens = [r["total_tokens"] for r in rows]
    prompt_tokens = [r["prompt_tokens"] for r in rows]
    assistant_target_tokens = [r["assistant_target_tokens"] for r in rows]

    coverage = {str(c): _coverage(total_tokens, c) for c in cfg.candidate_max_lengths}
    coverage_thresholds = {
        "smallest_candidate_covering_95pct": _smallest_candidate_covering(total_tokens, cfg.candidate_max_lengths, 0.95),
        "smallest_candidate_covering_99pct": _smallest_candidate_covering(total_tokens, cfg.candidate_max_lengths, 0.99),
        "smallest_candidate_covering_100pct": _smallest_candidate_covering(total_tokens, cfg.candidate_max_lengths, 1.0),
    }

    summary_lengths = {
        "total_tokens": _length_stats(total_tokens),
        "prompt_tokens": _length_stats(prompt_tokens),
        "assistant_target_tokens": _length_stats(assistant_target_tokens),
    }

    split_stats = _split_stats(rows)

    # 12) assistant-only loss 可行性
    prompt_prefix_mismatch = sum(1 for r in rows if not r["prompt_prefix_match"])
    zero_assistant_target = sum(1 for r in rows if r["assistant_target_tokens"] <= 0)
    assistant_only_loss_feasible = (
        prompt_prefix_mismatch == 0 and zero_assistant_target == 0
    )

    # 13) longest 20
    longest = _longest_samples(rows, k=20)

    # 14) real sample per-split verify
    real_sample_check = _verify_real_samples(tokenizer, rows, cfg.real_sample_check_per_split)

    # 15) accounting
    set_all = {r["sample_id"] for r in verified}
    set_split = set(membership.keys())
    set_equal = set_all == set_split

    # 16) audit jsonl 输出
    output_audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(
        output_audit_jsonl,
        [json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for r in rows],
    )

    # 17) report
    report: Dict[str, Any] = {
        "audit_version": 1,
        "model": {
            "model_name": tokenizer_info["model_name"],
            "requested_revision": tokenizer_info["requested_revision"],
            "resolved_revision": tokenizer_info["resolved_revision"],
            "transformers_version": tokenizer_info["transformers_version"],
            "model_config": model_config_info,
        },
        "tokenizer": {
            "tokenizer_class": tokenizer_info["tokenizer_class"],
            "tokenizer_name_or_path": tokenizer_info["tokenizer_name_or_path"],
            "vocab_size": tokenizer_info["vocab_size"],
            "bos_token": tokenizer_info["bos_token"],
            "bos_token_id": tokenizer_info["bos_token_id"],
            "eos_token": tokenizer_info["eos_token"],
            "eos_token_id": tokenizer_info["eos_token_id"],
            "pad_token": tokenizer_info["pad_token"],
            "pad_token_id": tokenizer_info["pad_token_id"],
            "model_max_length": tokenizer_info["model_max_length"],
            "chat_template_present": tokenizer_info["chat_template_present"],
        },
        "thinking_mode": synthetic["thinking_mode"],
        "synthetic_template": {
            "messages": synthetic["synthetic_messages"],
            "rendered_full": synthetic["synthetic_rendered_full"],
            "rendered_prompt": synthetic["synthetic_rendered_prompt"],
            "full_has_think_open": synthetic["synthetic_full_has_think_open"],
            "full_has_think_close": synthetic["synthetic_full_has_think_close"],
            "prompt_has_think_open": synthetic["synthetic_prompt_has_think_open"],
            "prompt_has_think_close": synthetic["synthetic_prompt_has_think_close"],
            "template_manually_modified": synthetic["template_manually_modified"],
        },
        "roundtrip_check": {
            "ok": roundtrip["ok"],
            "decoded_contains_question": roundtrip["decoded_contains_question"],
            "decoded_contains_answer": roundtrip["decoded_contains_answer"],
        },
        "dataset_membership": {
            "all_count": len(verified),
            "train_count": split_counts.get("train", 0),
            "validation_count": split_counts.get("validation", 0),
            "test_count": split_counts.get("test", 0),
            "set_equal_all_vs_union": set_equal,
            "missing_split_sample_ids": sorted(set_all - set_split),
            "missing_all_sample_ids": sorted(set_split - set_all),
        },
        "source_thinking_markers": thinking_counters,
        "length_stats": summary_lengths,
        "split_stats": split_stats,
        "coverage": coverage,
        "coverage_thresholds": coverage_thresholds,
        "prompt_prefix_mismatch_count": prompt_prefix_mismatch,
        "zero_assistant_target_count": zero_assistant_target,
        "assistant_only_loss_feasible": assistant_only_loss_feasible,
        "longest_samples": longest,
        "real_sample_check": real_sample_check,
        "candidate_max_lengths": list(cfg.candidate_max_lengths),
    }

    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_report_json, report)

    log.info("Wrote %d audit rows to %s", len(rows), output_audit_jsonl)
    log.info("Wrote token_report.json to %s", output_report_json)

    return report