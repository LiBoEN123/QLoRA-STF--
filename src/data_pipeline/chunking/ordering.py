"""恢复 Phase 2 clean records 的源文档顺序。

Phase 2 为保证确定性，按 ``doc_id`` 升序输出，但 ``doc_id`` 是基于
``SHA256(relative_path + page/sheet/record + text)`` 计算的，**不能**
反映真正的源文档顺序。Phase 3 在做语义恢复前必须先重建原文顺序。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def source_order_key(record: Dict[str, Any]) -> Tuple:
    """为单条 record 生成稳定排序键。

    优先级：``source_path → page_number → sheet_name → record_index → doc_id``。
    任何可能为 ``None`` 的字段都用 sentinel 值替代，保证不同输入下顺序稳定。
    """
    return (
        record.get("source_path", ""),
        record.get("page_number") if record.get("page_number") is not None else -1,
        record.get("sheet_name") or "",
        record.get("record_index") if record.get("record_index") is not None else -1,
        record.get("doc_id", ""),
    )


def restore_source_order(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 ``source_order_key`` 稳定排序，返回新列表（不修改入参）。

    使用 ``sorted``（Timsort）保证稳定排序：相同 key 的元素保持相对顺序。
    """
    return sorted(records, key=source_order_key)


def group_by_source(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按 ``source_path`` 分组；每个分组内部已经按 ``source_order_key`` 排序。"""
    ordered = restore_source_order(records)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in ordered:
        grouped.setdefault(rec.get("source_path", ""), []).append(rec)
    return grouped