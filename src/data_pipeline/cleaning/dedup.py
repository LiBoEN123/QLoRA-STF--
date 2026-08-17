"""Phase 2 清洗的去重原语。

提供两层去重：

* **Exact dedup** —— 对规范化后的 ``clean_text`` 取 SHA256。
* **Near dedup** —— 自实现的 64-bit SimHash + 顶位分桶，无外部依赖。

两层都是确定性的：相同输入在不同运行 / 不同进程下都得到相同结果。
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Exact 去重
# ---------------------------------------------------------------------------


def exact_dedup_key(text: str) -> str:
    """对 ``text`` 做轻量空白折叠后的 SHA256 十六进制摘要。"""
    normalized = re.sub(r"\s+", "", text or "")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExactDuplicate:
    """完全重复的 record 信息。

    Attributes:
        duplicate_doc_id: 被判为重复的 record 的 ``doc_id``。
        canonical_doc_id: 同一文本下首个出现的 canonical record 的 ``doc_id``。
        duplicate_type: 固定为 ``"exact"``。
        similarity: 固定为 ``1.0``。
    """

    duplicate_doc_id: str
    canonical_doc_id: str
    duplicate_type: str = "exact"
    similarity: float = 1.0


def find_exact_duplicates(
    records: Sequence[Tuple[str, str]],
) -> List[ExactDuplicate]:
    """返回 SHA256 key 不唯一的所有重复条目。

    ``records`` 是有序的 ``(doc_id, clean_text)`` 序列。
    每个唯一 key 的首次出现视为 canonical，后面的出现作为 ``ExactDuplicate`` 返回。

    顺序由调用方保证（排序输入）以获得完全确定性。
    """
    seen: Dict[str, str] = {}
    duplicates: List[ExactDuplicate] = []
    for doc_id, text in records:
        key = exact_dedup_key(text)
        canonical = seen.get(key)
        if canonical is None:
            seen[key] = doc_id
            continue
        duplicates.append(
            ExactDuplicate(
                duplicate_doc_id=doc_id,
                canonical_doc_id=canonical,
                duplicate_type="exact",
                similarity=1.0,
            )
        )
    return duplicates


# ---------------------------------------------------------------------------
# SimHash（64-bit）+ 分桶 near 去重
# ---------------------------------------------------------------------------


#: 非空白连续片段 → 一个 token。
_BIGRAM_RE = re.compile(r"(?s)\S+")


def _tokens(text: str) -> List[str]:
    """为 SimHash 对 ``text`` 做分词。

    使用确定性的字符 bigram 策略，无需任何 NLP 库。对 CJK 文本来说，
    这能在不引入 jieba 等分词器的情况下，给出可接受的近似词边界。

    对于长度 > 1 的 ASCII 片段，既会输出整个片段作为 token（保留英文单词），
    也会输出其中的字符 bigram。
    """
    if not text:
        return []
    out: List[str] = []
    for run in _BIGRAM_RE.findall(text):
        if len(run) == 1:
            out.append(run)
            continue
        # 对 CJK / 全角片段做 bigram。
        chars = list(run)
        for a, b in zip(chars, chars[1:]):
            out.append(a + b)
        # 同时把整个 ASCII 片段作为单个 token。
        if all(ord(c) < 128 and not c.isspace() for c in run):
            out.append(run.lower())
    return out


def _fnv1a_64(token: str) -> int:
    """``token`` 的确定性 64-bit FNV-1a 哈希。"""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    data = token.encode("utf-8")
    for byte in data:
        h ^= byte
        h = (h * prime) & mask
    return h


def simhash(text: str, n_bits: int = 64) -> int:
    """用确定性 FNV-1a 哈希计算 ``text`` 的 SimHash。

    返回一个无符号 ``n_bits`` 位整数。
    """
    mask = (1 << n_bits) - 1
    weights = [0] * n_bits
    for token in _tokens(text):
        h = _fnv1a_64(token)
        for i in range(n_bits):
            bit = (h >> i) & 1
            weights[i] += 1 if bit else -1
    out = 0
    for i in range(n_bits):
        if weights[i] > 0:
            out |= 1 << i
    return out & mask


def hamming_distance(a: int, b: int, n_bits: int = 64) -> int:
    """两个 SimHash 整数之间的 Hamming 距离。"""
    return bin((a ^ b) & ((1 << n_bits) - 1)).count("1")


# ---------------------------------------------------------------------------
# Near 去重（按顶位分桶）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NearDuplicate:
    """近似重复的 record 信息（SimHash 距离判定）。

    Attributes:
        duplicate_doc_id: 被判为近似重复的 record 的 ``doc_id``。
        canonical_doc_id: 与之最相似的 canonical record 的 ``doc_id``。
        duplicate_type: 固定为 ``"near"``。
        similarity: 相似度估计（0..1，越大越相似）。
        distance: SimHash Hamming 距离。
    """

    duplicate_doc_id: str
    canonical_doc_id: str
    duplicate_type: str = "near"
    similarity: float = 0.0
    distance: int = 0


def _bucket_key(simhash_value: int, top_bits: int) -> int:
    return simhash_value >> (64 - top_bits)


def find_near_duplicates(
    records: Sequence[Tuple[str, str]],
    *,
    max_distance: int = 3,
    bucket_bits: int = 16,
) -> List[NearDuplicate]:
    """使用 SimHash + 顶位分桶寻找近似重复。

    ``records`` 是有序的 ``(doc_id, clean_text)`` 序列。
    在同一桶内，第一个出现的指纹视为 canonical；在 ``max_distance`` 之内的
    后续指纹报告为 near-duplicate。

    该函数只在桶内两两比较，避免 O(N²)；分桶是 SimHash 的纯函数，因此仍然完全确定性。
    """
    indexed: List[Tuple[str, int]] = []
    for doc_id, text in records:
        indexed.append((doc_id, simhash(text)))

    # 按顶位分桶。
    buckets: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for doc_id, fp in indexed:
        buckets[_bucket_key(fp, bucket_bits)].append((doc_id, fp))

    duplicates: List[NearDuplicate] = []
    seen_doc_ids: set[str] = set()

    # 按确定性的桶序遍历。
    for bkey in sorted(buckets.keys()):
        members = buckets[bkey]
        canonical_in_bucket: Tuple[str, int] | None = None
        for doc_id, fp in members:
            if doc_id in seen_doc_ids:
                continue  # exact 去重已经处理过的 doc_id
            if canonical_in_bucket is None:
                canonical_in_bucket = (doc_id, fp)
                seen_doc_ids.add(doc_id)
                continue
            can_id, can_fp = canonical_in_bucket
            dist = hamming_distance(fp, can_fp)
            if dist <= max_distance:
                duplicates.append(
                    NearDuplicate(
                        duplicate_doc_id=doc_id,
                        canonical_doc_id=can_id,
                        duplicate_type="near",
                        similarity=1.0 - dist / 64.0,
                        distance=dist,
                    )
                )
                seen_doc_ids.add(doc_id)
            else:
                canonical_in_bucket = (doc_id, fp)
                seen_doc_ids.add(doc_id)
    return duplicates


def deduplicate(
    records: Sequence[Tuple[str, str]],
    *,
    max_distance: int = 3,
    enable_near: bool = True,
) -> Tuple[List[ExactDuplicate], List[NearDuplicate]]:
    """先跑 exact 再跑 near 去重；near 阶段会跳过已被 exact 移除的项。"""
    exact = find_exact_duplicates(records)
    exact_dropped = {d.duplicate_doc_id for d in exact}
    if not enable_near:
        return exact, []
    near = find_near_duplicates(
        [(d, t) for d, t in records if d not in exact_dropped],
        max_distance=max_distance,
    )
    return exact, near