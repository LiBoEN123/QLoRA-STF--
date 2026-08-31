"""Phase 5D — Held-out Test Evaluation。

不做任何训练；只：

1. 冻结 evaluation protocol（generation 参数 / max_new_tokens / tokenizer 等）；
2. test.jsonl 指纹 + leakage audit；
3. 加载 base model（4-bit / NF4 / BF16）+ best_adapter，构造 assistant-only
   teacher-forced loss + greedy generation；
4. 规范化 + char F1 + ROUGE-L + paired wins/ties；
5. 通过 ``PHASE5D_CODE = PASS`` gate。

职责：

- 协议相关：``Phase5DEvalConfig``、``fingerprint_test``、``frozen_protocol``；
- 数据相关：``leakage_audit``、``prompt_extraction``；
- 指标相关：``normalize_text``、``char_f1``、``rouge_l_f1``、``paired_metrics``；
- 决策相关：``eval_gate``。

不负责：

- GPU 模型加载（由 ``scripts/evaluate_qlora_test.py`` 处理）；
- tokenizer 实例化（由 ``evaluate_qlora_test.py`` 处理）。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..utils import atomic_write_json, get_logger, load_jsonl_records


_LOG = get_logger("data_pipeline.training.phase5d")


# ---------------------------------------------------------------------------
# 1) Evaluation config（frozen protocol）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase5DEvalConfig:
    """Phase 5D evaluation protocol 冻结配置。

    所有字段在 manifest 写入后立即 ``frozen = True``，禁止根据 test 结果修改。

    字段含义：

    - ``do_sample = False``：纯 greedy；不设置 temperature / top_p / top_k。
    - ``max_new_tokens``：由 reference assistant target token 长度 + buffer 计算，
      一旦写入立即冻结。
    - ``enable_thinking = False``：与 Phase 5C-B inference 一致。
    - ``assistant_only_loss = True``：teacher-forced loss 只算 assistant 段。
    - ``quant_type`` / ``compute_dtype`` / ``double_quant``：与 Phase 5C-B 一致。
    """

    # base 模型仓库名
    model_name: str = "Qwen/Qwen3-8B"
    # base 模型 pinned revision commit SHA
    revision: str = "b968826d9c46dd6066d109eabc6255188de91218"

    # 4-bit 量化类型（与 Phase 5C-B 一致，固定 NF4）
    quant_type: str = "nf4"
    # 量化张量计算 dtype（与 Phase 5C-B 一致，固定 bfloat16）
    compute_dtype: str = "bfloat16"
    # 是否启用 bitsandbytes double quant（与 Phase 5C-B 一致）
    double_quant: bool = True

    # best adapter 目录路径（Phase 5C-B 收口后产物）
    best_adapter_path: str = "outputs/phase5c-b/best_adapter"

    # 最大序列长度（与 Phase 5C-B 一致，固定 2048）
    max_length: int = 2048

    # 是否使用 assistant_only_loss（teacher-forced loss 只算 assistant 段）
    assistant_only_loss: bool = True

    # generation do_sample（必须 False，纯 greedy）
    do_sample: bool = False
    # generation enable_thinking（必须 False，与 Phase 5C-B inference 一致）
    enable_thinking: bool = False
    # generation 最大生成 token 数（由 reference + buffer 计算后冻结）
    max_new_tokens: int = 512
    # buffer 加到 max_reference_assistant_tokens 上的额外 token 数
    max_new_tokens_buffer: int = 64
    # max_new_tokens 下限
    max_new_tokens_floor: int = 256
    # max_new_tokens 上限
    max_new_tokens_ceiling: int = 2048

    # test JSONL 路径
    test_jsonl_path: str = "data/sft/test.jsonl"
    # train JSONL 路径（用于 leakage audit）
    train_jsonl_path: str = "data/sft/train.jsonl"
    # validation JSONL 路径（用于 leakage audit）
    validation_jsonl_path: str = "data/sft/validation.jsonl"

    # char n-gram 大小（保留字段，Phase 5D 当前用 char-level 指标）
    char_ngram_size: int = 3

    # 期望 tokenizer signature（与 Phase 5C-A/B 锁定一致）
    expected_tokenizer_signature: str = (
        "4ac4fed6d76f2cddf5ee756990b7bd4dc40655f96e4c6be460dc3299e73651c0"
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.do_sample is not False:
            raise ValueError("do_sample 必须为 False（greedy generation）")
        if self.enable_thinking is not False:
            raise ValueError("enable_thinking 必须为 False（Phase 5D 冻结）")
        if not self.assistant_only_loss:
            raise ValueError("assistant_only_loss 必须为 True")
        if self.quant_type != "nf4":
            raise ValueError(f"quant_type 必须为 'nf4'，得到 {self.quant_type!r}")
        if self.compute_dtype != "bfloat16":
            raise ValueError(f"compute_dtype 必须为 'bfloat16'，得到 {self.compute_dtype!r}")
        if not self.double_quant:
            raise ValueError("double_quant 必须为 True")
        if self.max_length != 2048:
            raise ValueError(f"max_length 必须为 2048，得到 {self.max_length}")
        if self.max_new_tokens < self.max_new_tokens_floor:
            raise ValueError(
                f"max_new_tokens={self.max_new_tokens} < floor={self.max_new_tokens_floor}"
            )
        if self.max_new_tokens > self.max_new_tokens_ceiling:
            raise ValueError(
                f"max_new_tokens={self.max_new_tokens} > ceiling={self.max_new_tokens_ceiling}"
            )


# ---------------------------------------------------------------------------
# 2) Test fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestFingerprint:
    """test.jsonl + ordered sample_id signature。

    - ``file_sha256``：整个 test.jsonl 的 SHA256；
    - ``record_count``：记录数（=79）；
    - ``sample_id_signature``：sample_id 升序拼接的 SHA256；
    - ``sample_ids``：升序 sample_id 列表。
    """

    # test.jsonl 绝对路径
    file_path: str
    # 整个 test.jsonl 文件的 SHA256
    file_sha256: str
    # 记录数（Phase 5D 固定 79）
    record_count: int
    # sample_id 升序拼接的 SHA256
    sample_id_signature: str
    # 升序 sample_id 元组
    sample_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_sha256": self.file_sha256,
            "record_count": int(self.record_count),
            "sample_id_signature": self.sample_id_signature,
            "sample_ids": list(self.sample_ids),
        }


def fingerprint_test(jsonl_path: Path) -> TestFingerprint:
    """计算 test.jsonl 指纹 + 升序 sample_id 列表。"""
    p = Path(jsonl_path)
    if not p.exists():
        raise ValueError(f"test.jsonl 缺失：{p}")
    records = load_jsonl_records(p)

    sha = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            sha.update(chunk)

    sids: List[str] = []
    for r in records:
        sid = r.get("sample_id")
        if sid is None:
            raise ValueError(f"record 缺少 sample_id：{p}")
        sids.append(str(sid))
    sids_sorted = sorted(sids)
    sig = hashlib.sha256()
    for s in sids_sorted:
        sig.update(s.encode("utf-8"))
        sig.update(b"\n")

    return TestFingerprint(
        file_path=str(p),
        file_sha256=sha.hexdigest(),
        record_count=len(records),
        sample_id_signature=sig.hexdigest(),
        sample_ids=tuple(sids_sorted),
    )


# ---------------------------------------------------------------------------
# 3) Leakage audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageAuditResult:
    """泄漏审计结果。

    字段：

    - ``sample_id_overlap_train``：test ∩ train 的 sample_id 个数（必须 = 0）。
    - ``sample_id_overlap_validation``：test ∩ val 的 sample_id 个数（必须 = 0）。
    - ``full_message_overlap_count``：规范化 full messages 完全重复数（必须 = 0）。
    - ``prompt_overlap_count``：规范化 prompt 完全重复数（必须 = 0，否则 BLOCK）。
    - ``near_duplicate_count``：高相似度（>=0.95）近似重复数（WARNING）。
    - ``near_duplicate_top``：top 匹配列表。
    """

    # test ∩ train sample_id 个数
    sample_id_overlap_train: int
    # test ∩ val sample_id 个数
    sample_id_overlap_validation: int
    # 规范化 full messages 重复数
    full_message_overlap_count: int
    # 规范化 prompt 重复数
    prompt_overlap_count: int
    # 高相似度近似重复数（>= 阈值，WARNING only）
    near_duplicate_count: int
    # top 近似重复匹配列表
    near_duplicate_top: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """test set 与 train/validation 完全隔离。"""
        return (
            self.sample_id_overlap_train == 0
            and self.sample_id_overlap_validation == 0
            and self.full_message_overlap_count == 0
            and self.prompt_overlap_count == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id_overlap_train": int(self.sample_id_overlap_train),
            "sample_id_overlap_validation": int(self.sample_id_overlap_validation),
            "full_message_overlap_count": int(self.full_message_overlap_count),
            "prompt_overlap_count": int(self.prompt_overlap_count),
            "near_duplicate_count": int(self.near_duplicate_count),
            "near_duplicate_top": list(self.near_duplicate_top),
            "is_clean": bool(self.is_clean),
        }


def _normalize_text(text: str) -> str:
    """规范化文本：NFKC + strip + collapse whitespace。"""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.strip()
    # collapse whitespace
    s = re.sub(r"\s+", "", s)  # 全部去空白用于 leakage audit hash
    return s


def assistant_span_from_prefix(
    prompt_ids: Sequence[int],
    full_ids: Sequence[int],
) -> Dict[str, Any]:
    """纯函数：从 ``prompt_ids`` / ``full_ids`` 推断 assistant 监督段。

    语义（与 Phase 5C-B teacher-forced loss 完全一致）：

    - ``prompt_ids`` = ``apply_chat_template(msgs[:-1], add_generation_prompt=True)``
      对应的 token 序列；
    - ``full_ids`` = ``apply_chat_template(msgs, add_generation_prompt=False)``
      对应的 token 序列；
    - 训练时 ``labels[i] = -100`` 当 ``i < len(prompt_ids)``，否则 ``= full_ids[i]``；
    - 因此：

      - ``masked_tokens = len(prompt_ids)``（prompt 部分 ⇒ ``-100``）
      - ``supervised_tokens = len(full_ids) - len(prompt_ids)``（assistant 部分）

    必须验证 ``full_ids[:len(prompt_ids)] == prompt_ids``。
    prefix mismatch ⇒ ``prefix_equal=False``，并 **raise** ``ValueError``；
    不允许静默降级为 ``prompt_len = 0`` / ``all supervised``。

    Returns
    -------
    dict
        - ``prefix_equal``: ``bool``（仅在正常路径为 True）
        - ``prompt_len``: ``int``
        - ``masked_tokens``: ``int``（= ``prompt_len``）
        - ``supervised_tokens``: ``int``（= ``full_len - prompt_len``）
    """
    prompt_list = [int(x) for x in prompt_ids]
    full_list = [int(x) for x in full_ids]
    if not (len(prompt_list) <= len(full_list)
            and full_list[: len(prompt_list)] == prompt_list):
        raise ValueError(
            "assistant_span_from_prefix: prefix mismatch: "
            f"len(prompt)={len(prompt_list)} len(full)={len(full_list)} "
            "且 full[:len(prompt)] != prompt"
        )
    return {
        "prefix_equal": True,
        "prompt_len": int(len(prompt_list)),
        "masked_tokens": int(len(prompt_list)),
        "supervised_tokens": int(len(full_list) - len(prompt_list)),
    }


def _messages_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    """规范化后完整 messages 的 SHA256。"""
    norm = json.dumps(
        [
            {"role": str(m.get("role", "")), "content": _normalize_text(m.get("content", ""))}
            for m in messages
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _prompt_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """构造 prompt messages：保留最后一个 assistant 之前的全部消息。

    对 SFT 数据：最后一条 = assistant target；构造 prompt 时
    messages[:-1] + add_generation_prompt=True 即可。
    """
    if not messages:
        return []
    return [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in messages[:-1]
    ]


def _prompt_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    """规范化 prompt messages 的 SHA256。

    规范化规则：NFKC + trim + collapse whitespace + 保留 role + content。
    """
    prompt = _prompt_messages(messages)
    norm = json.dumps(
        [
            {"role": p["role"], "content": _normalize_for_prompt(p["content"])}
            for p in prompt
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _normalize_for_prompt(text: str) -> str:
    """prompt hash 的规范化：NFKC + strip + collapse 内部空白（保留语义 token）。"""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _sequence_ratio(a: str, b: str) -> float:
    """无 difflib 依赖的字符级 SequenceMatcher-like 相似度。

    取 ``a`` / ``b`` 的最长公共子串长度比例，简单实现：
    ``2 * LCS / (len(a) + len(b))``。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # DP 计算 LCS 长度
    n, m = len(a), len(b)
    if n > 2000 or m > 2000:
        # 防止极端长字符串 O(n*m) 爆炸；用 set-based Jaccard 替代
        sa = set(a)
        sb = set(b)
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[m]
    return 2.0 * lcs / (n + m)


def leakage_audit(
    test_records: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    *,
    near_dup_threshold: float = 0.95,
    near_dup_top_n: int = 5,
) -> LeakageAuditResult:
    """执行 test set 与 train/validation 的全量泄漏审计。

    包含：

    1) sample_id 重叠检查（必须 = 0）；
    2) full messages 规范化后完全重复检查（必须 = 0）；
    3) prompt 规范化后完全重复检查（必须 = 0，否则 BLOCK）；
    4) 高相似度近似重复统计（>= near_dup_threshold，WARNING only）。

    性能优化（保持语义不变）：

    - 预计算 ``normalized_train_val`` / ``normalized_test``，不在 inner loop
      重复 ``_normalize_text(...)``；
    - 用 LCS-ratio 数学上界 ``2 * min(n, m) / (n + m) < threshold`` 做长度剪枝，
      对不可能达到 ``near_dup_threshold`` 的 pair 直接跳过 LCS DP。

    Returns
    -------
    LeakageAuditResult
    """
    test_ids = {str(r.get("sample_id")) for r in test_records}
    train_ids = {str(r.get("sample_id")) for r in train_records}
    val_ids = {str(r.get("sample_id")) for r in validation_records}

    sid_overlap_train = len(test_ids & train_ids)
    sid_overlap_val = len(test_ids & val_ids)

    # full-message hashes
    train_full_hashes = {_messages_hash(r["messages"]) for r in train_records}
    val_full_hashes = {_messages_hash(r["messages"]) for r in validation_records}
    test_full_hashes = {_messages_hash(r["messages"]) for r in test_records}
    full_overlap = sum(
        1 for h in test_full_hashes if h in train_full_hashes or h in val_full_hashes
    )

    # prompt hashes（仅 user/system 部分；不包含 reference assistant）
    train_prompt_hashes = {_prompt_hash(r["messages"]) for r in train_records}
    val_prompt_hashes = {_prompt_hash(r["messages"]) for r in validation_records}
    test_prompt_hashes = {_prompt_hash(r["messages"]) for r in test_records}
    prompt_overlap = sum(
        1 for h in test_prompt_hashes if h in train_prompt_hashes or h in val_prompt_hashes
    )

    # ------------------------------------------------------------------
    # Near duplicate (against train ∪ val, normalized full messages).
    # 预计算规范化字符串 + 长度上界剪枝。
    # ------------------------------------------------------------------
    combined = list(train_records) + list(validation_records)
    train_val_norm: List[str] = [
        _normalize_text("".join(str(m.get("content", "")) for m in cr["messages"]))
        for cr in combined
    ]
    combined_ids = [str(cr.get("sample_id", "")) for cr in combined]
    combined_splits = [
        "train" if combined_ids[i] in train_ids else "validation"
        for i in range(len(combined))
    ]

    near_dup_count = 0
    top: List[Dict[str, Any]] = []
    for tr in test_records:
        t_full = _normalize_text(
            "".join(str(m.get("content", "")) for m in tr["messages"])
        )
        n = len(t_full)
        if n == 0:
            continue
        best = 0.0
        best_match_id = ""
        best_split = ""
        for c_full, cid, csplit in zip(train_val_norm, combined_ids, combined_splits):
            m = len(c_full)
            if m == 0:
                continue
            # LCS ratio 上界 = 2 * min(n, m) / (n + m)；
            # 若上界 < threshold ⇒ 数学上不可能达到阈值，直接跳过。
            upper = 2.0 * min(n, m) / (n + m)
            if upper < near_dup_threshold:
                continue
            sim = _sequence_ratio(t_full, c_full)
            if sim > best:
                best = sim
                best_match_id = cid
                best_split = csplit
        if best >= near_dup_threshold:
            near_dup_count += 1
            if len(top) < near_dup_top_n:
                top.append(
                    {
                        "test_sample_id": str(tr.get("sample_id", "")),
                        "best_match_sample_id": best_match_id,
                        "best_match_split": best_split,
                        "similarity": round(best, 4),
                    }
                )

    return LeakageAuditResult(
        sample_id_overlap_train=sid_overlap_train,
        sample_id_overlap_validation=sid_overlap_val,
        full_message_overlap_count=full_overlap,
        prompt_overlap_count=prompt_overlap,
        near_duplicate_count=near_dup_count,
        near_duplicate_top=top,
    )


# ---------------------------------------------------------------------------
# 4) Normalization
# ---------------------------------------------------------------------------


# 用于生成评估的两种规范化（不依赖外部 LLM）。

def normalize_for_eval(text: str) -> str:
    """规范化：NFKC + strip + collapse whitespace。

    用于 Exact Match 和 char F1 / ROUGE-L。
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s


def _is_punct_or_symbol(ch: str) -> bool:
    """判断单字符是否为 punctuation / symbol / 空白。

    使用 ``unicodedata.category``：

    - ``P*`` = Unicode Punctuation（CJK 标点、ASCII 标点、引号、破折号、省略号等）
    - ``S*`` = Unicode Symbols（``+`` / ``=`` / 数学符号 / 货币符号等）
    - ``isspace()`` 覆盖 Unicode 各类空白（ASCII space、tab、CR/LF、NBSP、IDEOGRAPHIC SPACE 等）

    该实现不依赖源码字面量字符类，避免源码写入过程中 Unicode 字符
    被替换为 ``\\ufffd`` 导致正则失效。
    """
    if not ch:
        return False
    if ch.isspace():
        return True
    category = unicodedata.category(ch)
    return category.startswith("P") or category.startswith("S")


def normalize_for_lexical(text: str) -> str:
    """中文 lexical metric 规范化：

    - Unicode NFKC；
    - 去掉所有 Unicode 标点（``P*``）+ symbols（``S*``）+ 空白；
    - ASCII 转小写。

    使用 ``unicodedata.category`` 而非手写字符类正则，避免 Unicode
    字符在源码中被替换为 ``\\ufffd`` 造成静默失效。

    不依赖外部 NLP 库（无 jieba / 无 regex / 无第三方包）。
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = "".join(ch for ch in s if not _is_punct_or_symbol(ch))
    s = s.lower()
    return s


# ---------------------------------------------------------------------------
# 5) Metrics
# ---------------------------------------------------------------------------


def char_f1(prediction: str, reference: str) -> Dict[str, float]:
    """字符级 Precision / Recall / F1（基于 ``normalize_for_eval``）。

    空串预测 ⇒ P/R/F1 全 0；空串 reference ⇒ P/R/F1 全 0。
    """
    p = normalize_for_eval(prediction)
    r = normalize_for_eval(reference)
    if not p and not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not p or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if p == r:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # 用 Counter 计 char 重叠
    from collections import Counter

    pc = Counter(p)
    rc = Counter(r)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = overlap / sum(pc.values())
    recall = overlap / sum(rc.values())
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: Sequence[Any], b: Sequence[Any]) -> int:
    """O(n*m) LCS 长度（短串用，n,m <= 几 KB）。"""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    if n > 4000 or m > 4000:
        # 防止极端长字符串 O(n*m) 爆炸；用 set-based Jaccard 替代
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        union = len(sa | sb)
        return int(2 * inter * len(a) * len(b) / (union * (len(a) + len(b)))) if union else 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def rouge_l_f1(prediction: str, reference: str) -> Dict[str, float]:
    """字符级 ROUGE-L F1（基于 ``normalize_for_eval``）。

    ROUGE-L = ``2 * LCS / (|p| + |r|)``。
    """
    p = normalize_for_eval(prediction)
    r = normalize_for_eval(reference)
    if not p or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if p == r:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    lcs = _lcs_length(p, r)
    if lcs == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = lcs / len(p)
    recall = lcs / len(r)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match(prediction: str, reference: str) -> bool:
    """规范化后完全一致（``normalize_for_eval``）。"""
    return normalize_for_eval(prediction) == normalize_for_eval(reference)


def aggregate_metrics(per_sample: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """汇总 per-sample 指标。"""
    n = len(per_sample)
    if n == 0:
        return {
            "n": 0,
            "mean_exact_match": 0.0,
            "mean_char_f1": 0.0,
            "mean_rouge_l_f1": 0.0,
            "empty_generation_rate": 0.0,
            "mean_generated_tokens": 0.0,
            "max_token_hit_rate": 0.0,
        }
    em = sum(float(x.get("exact_match", 0)) for x in per_sample) / n
    cf1 = sum(float(x.get("char_f1", 0)) for x in per_sample) / n
    rl = sum(float(x.get("rouge_l_f1", 0)) for x in per_sample) / n
    empty = sum(1 for x in per_sample if int(x.get("generated_tokens", 0)) == 0) / n
    gen_tokens = [int(x.get("generated_tokens", 0)) for x in per_sample]
    mean_gt = sum(gen_tokens) / n
    cap_hit = sum(1 for x in per_sample if bool(x.get("hit_generation_cap", False))) / n
    return {
        "n": int(n),
        "mean_exact_match": round(em, 4),
        "mean_char_f1": round(cf1, 4),
        "mean_rouge_l_f1": round(rl, 4),
        "empty_generation_rate": round(empty, 4),
        "mean_generated_tokens": round(mean_gt, 2),
        "max_token_hit_rate": round(cap_hit, 4),
    }


# ---------------------------------------------------------------------------
# 6) Paired metrics
# ---------------------------------------------------------------------------


def paired_metrics(
    finetuned_scores: Sequence[float],
    base_scores: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """paired wins / regressions / ties（带 ``tolerance``）。

    对每个 sample：

    - ``delta = finetuned - base``；
    - ``delta > tolerance`` ⇒ finetuned win；
    - ``delta < -tolerance`` ⇒ base win（即 finetuned regression）；
    - 否则 ⇒ tie。

    默认 ``tolerance = 1e-12``，与 Phase 5D 既有 ``float`` 评分精度一致；
    但 ``tolerance`` 也参与 manifest 的 semantic protocol 字段。
    """
    assert len(finetuned_scores) == len(base_scores), "finetuned / base 长度不一致"
    if tolerance < 0:
        raise ValueError(f"tolerance 必须 >= 0，得到 {tolerance}")
    wins = 0
    losses = 0
    ties = 0
    deltas: List[float] = []
    for f, b in zip(finetuned_scores, base_scores):
        delta = float(f) - float(b)
        deltas.append(delta)
        if delta > tolerance:
            wins += 1
        elif delta < -tolerance:
            losses += 1
        else:
            ties += 1
    n = len(finetuned_scores)
    return {
        "n": int(n),
        "finetuned_wins": int(wins),
        "base_wins": int(losses),
        "ties": int(ties),
        "win_rate": round(wins / n, 4) if n else 0.0,
        "regression_rate": round(losses / n, 4) if n else 0.0,
        "tie_rate": round(ties / n, 4) if n else 0.0,
        "mean_delta": round(sum(deltas) / n, 4) if n else 0.0,
        "tolerance": float(tolerance),
    }


# ---------------------------------------------------------------------------
# 7) Eval gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalGateResult:
    """Phase 5D 收尾 gate 判定。"""

    # gate 状态码："PASS" 或 "BLOCKED"
    code: str
    # 是否满足 READY_FOR_PHASE5E
    ready_for_phase5e: bool
    # 未通过原因列表（空 = PASS）
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "ready_for_phase5e": bool(self.ready_for_phase5e),
            "reasons": list(self.reasons),
        }


def eval_gate(
    *,
    base_test_loss: float,
    finetuned_test_loss: float,
    base_metrics: Mapping[str, float],
    finetuned_metrics: Mapping[str, float],
    paired_char_f1: Mapping[str, Any],
    paired_rouge_l: Mapping[str, Any],
    finetuned_empty_count: int,
    leakage_audit_result: LeakageAuditResult,
    n_test: int,
    base_eval_completed: bool,
    finetuned_eval_completed: bool,
    regression_threshold: float = 0.95,
) -> EvalGateResult:
    """Phase 5D 收尾 gate。

    至少满足：

    - n_test == 79；
    - base_eval_completed / finetuned_eval_completed == True；
    - finetuned_test_loss < base_test_loss；
    - base/finetuned generation 均 79/79；
    - finetuned empty generation count == 0；
    - leakage audit clean；
    - 不出现双 lexical 指标 > 5% 整体退化
      （即 finetuned_char_f1 < base_char_f1 × 0.95 AND
        finetuned_rouge_l < base_rouge_l × 0.95）。

    Returns
    -------
    EvalGateResult
    """
    reasons: List[str] = []

    if n_test != 79:
        reasons.append(f"TEST_COUNT_DRIFT: n_test={n_test} != 79")

    if not base_eval_completed:
        reasons.append("BASE_EVAL_NOT_COMPLETED")
    if not finetuned_eval_completed:
        reasons.append("FINETUNED_EVAL_NOT_COMPLETED")

    if not math.isfinite(base_test_loss):
        reasons.append(f"BASE_TEST_LOSS_NON_FINITE: {base_test_loss}")
    if not math.isfinite(finetuned_test_loss):
        reasons.append(f"FINETUNED_TEST_LOSS_NON_FINITE: {finetuned_test_loss}")
    if math.isfinite(base_test_loss) and math.isfinite(finetuned_test_loss):
        if not (finetuned_test_loss < base_test_loss):
            reasons.append(
                f"TEST_LOSS_NOT_IMPROVED: finetuned={finetuned_test_loss} >= base={base_test_loss}"
            )

    if finetuned_empty_count > 0:
        reasons.append(f"FINETUNED_EMPTY_GENERATION: count={finetuned_empty_count}")

    if not leakage_audit_result.is_clean:
        reasons.append(
            f"LEAKAGE_AUDIT_FAIL: "
            f"sample_id_overlap_train={leakage_audit_result.sample_id_overlap_train}, "
            f"sample_id_overlap_validation={leakage_audit_result.sample_id_overlap_validation}, "
            f"full_message_overlap={leakage_audit_result.full_message_overlap_count}, "
            f"prompt_overlap={leakage_audit_result.prompt_overlap_count}"
        )

    # generation metrics 双指标 > 5% 整体退化
    base_cf1 = float(base_metrics.get("mean_char_f1", 0.0))
    base_rl = float(base_metrics.get("mean_rouge_l_f1", 0.0))
    ft_cf1 = float(finetuned_metrics.get("mean_char_f1", 0.0))
    ft_rl = float(finetuned_metrics.get("mean_rouge_l_f1", 0.0))
    if base_cf1 > 0 and ft_cf1 < base_cf1 * regression_threshold:
        if base_rl > 0 and ft_rl < base_rl * regression_threshold:
            reasons.append(
                f"GENERATION_METRIC_REGRESSION: "
                f"ft_char_f1={ft_cf1:.4f} < base*{regression_threshold:.2f}={base_cf1 * regression_threshold:.4f} "
                f"AND ft_rouge_l={ft_rl:.4f} < base*{regression_threshold:.2f}={base_rl * regression_threshold:.4f}"
            )

    if reasons:
        return EvalGateResult(
            code="BLOCKED",
            ready_for_phase5e=False,
            reasons=reasons,
        )
    return EvalGateResult(
        code="PASS",
        ready_for_phase5e=True,
        reasons=[],
    )


# ---------------------------------------------------------------------------
# 8) Manifest 写入
# ---------------------------------------------------------------------------


def write_eval_manifest(
    path: Path,
    *,
    payload: Mapping[str, Any],
    logger: Any = None,
) -> None:
    """原子写入 ``eval_manifest.json``，并强制 ``protocol_frozen = True``。

    同时计算并写入 ``eval_protocol_signature``：基于 semantic protocol 关键字段
    （排除 git provenance 与 runtime 测试样本级字段）的 SHA256。
    """
    log = logger or _LOG
    body = dict(payload)
    body["protocol_frozen"] = True
    body["eval_protocol_signature"] = compute_eval_protocol_signature(body)
    atomic_write_json(Path(path), body)
    log.info(
        "Wrote eval_manifest.json to %s (protocol_frozen=True, signature=%s)",
        path, body["eval_protocol_signature"][:16],
    )


# ---------------------------------------------------------------------------
# 9) Semantic evaluation protocol signature
# ---------------------------------------------------------------------------


# Semantic evaluation protocol 关键字段集合；其他字段不参与 signature。
SEMANTIC_PROTOCOL_FIELDS: Tuple[str, ...] = (
    "backend",
    "model_name",
    "model_revision",
    "best_adapter_path",
    "best_adapter_signature_sha256",
    "tokenizer_signature",
    "test_jsonl_path",
    "test_jsonl_sha256",
    "test_count",
    "test_sample_id_signature",
    "quantization",
    "max_length",
    "assistant_only_loss",
    "enable_thinking",
    "do_sample",
    "generation_max_new_tokens",
    "normalization",
    "metrics",
    "seed",
    "expected_lora_layers",
    "paired_metrics_tolerance",
    "resume_save_every",
    "near_dup_threshold",
)


def compute_eval_protocol_signature(
    manifest_body: Mapping[str, Any],
) -> str:
    """基于关键 semantic 字段计算 manifest 的稳定 signature。

    git provenance / dirty flag / runtime 测试 sample-level 字段不参与。
    """
    canonical: Dict[str, Any] = {}
    for k in SEMANTIC_PROTOCOL_FIELDS:
        if k in manifest_body:
            canonical[k] = manifest_body[k]
    canonical_text = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def verify_frozen_manifest(
    existing: Mapping[str, Any],
    *,
    expected_tokenizer_signature: str,
    expected_test_sample_id_signature: str,
    expected_generation_max_new_tokens: int,
    expected_eval_protocol_signature: str,
) -> Tuple[bool, List[str]]:
    """校验已存在 frozen manifest 的语义是否一致。

    Returns ``(ok, reasons)``；任一关键字段 drift ⇒ ``ok=False``。
    """
    reasons: List[str] = []
    if not existing.get("protocol_frozen"):
        reasons.append("existing manifest protocol_frozen != True")
    if existing.get("tokenizer_signature") != expected_tokenizer_signature:
        reasons.append("tokenizer_signature drift")
    if existing.get("test_sample_id_signature") != expected_test_sample_id_signature:
        reasons.append("test_sample_id_signature drift")
    if existing.get("generation_max_new_tokens") != expected_generation_max_new_tokens:
        reasons.append("generation_max_new_tokens drift")
    sig = existing.get("eval_protocol_signature")
    if sig != expected_eval_protocol_signature:
        reasons.append(
            f"eval_protocol_signature drift (got={sig[:16] if sig else None}, "
            f"expected={expected_eval_protocol_signature[:16]})"
        )
    return (len(reasons) == 0, reasons)


__all__ = [
    "Phase5DEvalConfig",
    "TestFingerprint",
    "LeakageAuditResult",
    "EvalGateResult",
    "fingerprint_test",
    "leakage_audit",
    "_normalize_text",
    "_normalize_for_prompt",
    "_prompt_messages",
    "_messages_hash",
    "_prompt_hash",
    "_sequence_ratio",
    "assistant_span_from_prefix",
    "normalize_for_eval",
    "normalize_for_lexical",
    "char_f1",
    "rouge_l_f1",
    "exact_match",
    "aggregate_metrics",
    "paired_metrics",
    "eval_gate",
    "write_eval_manifest",
    "compute_eval_protocol_signature",
    "verify_frozen_manifest",
    "SEMANTIC_PROTOCOL_FIELDS",
]
