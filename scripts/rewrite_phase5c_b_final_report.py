"""Phase 5C-B 收口 — 重建 ``final_report.json``。

不复跑训练；只读取已存在的 ``data/training/phase5c_b/final_report.json``，
按 Phase 5C-B 收口规范重写 **显存统计字段** 与 **adapter verification**，
**不修改**：

- training loss history / eval_loss
- validation before / best / best_epoch
- checkpoint 路径
- 4-bit / LoRA / tokenizer signature

只新增 / 重命名以下字段：

- 物理 tensor 占用峰值：``peak_allocated_bytes`` / ``peak_allocated_gib``
- CUDA allocator 保留字节数：``allocator_peak_reserved_bytes`` / ``allocator_peak_reserved_gib``
  （原 ``peak_reserved_*`` 字段保留为 deprecated 别名，保证向后兼容）
- 设备物理显存总量：``device_total_bytes`` / ``device_total_gib``
- ``memory_field_legend``：解释每个字段含义
- ``best_adapter_signature``：来自 ``best_adapter_verification.json``

退出码：

- 0：成功重建
- 10：源 final_report.json 缺失
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_pipeline.utils import atomic_write_json, get_logger  # noqa: E402

from data_pipeline.training import _bytes_to_gib, build_memory_field_legend  # noqa: E402


DEFAULT_FINAL_REPORT = Path("data/training/phase5c_b/final_report.json")
DEFAULT_VERIFY_REPORT = Path("data/training/phase5c_b/best_adapter_verification.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5C-B final_report.json 收口重建")
    p.add_argument("--final-report", default=str(DEFAULT_FINAL_REPORT), type=Path)
    p.add_argument("--verify-report", default=str(DEFAULT_VERIFY_REPORT), type=Path)
    return p.parse_args()


def _rewrite_memory(report: Dict[str, Any]) -> Dict[str, Any]:
    """重写 final_report.json 中的显存字段。

    处理三处：

    1) ``report["training"]``：原 peak_allocated_gib + peak_reserved_gib
       保留 + 增加 ``allocator_peak_reserved_*`` / ``device_total_*``。
    2) ``report["memory"]``：原 ``training_*`` + ``peak_*`` 改为新命名。
    3) ``report["memory_field_legend"]``：解释每个字段。
    """
    training_block = report.get("training", {})
    if not training_block:
        raise ValueError("final_report.json 缺少 'training' 块")

    peak_alloc_bytes = int(training_block.get("peak_allocated_bytes") or 0)
    if peak_alloc_bytes <= 0:
        raise ValueError("training.peak_allocated_bytes 必须 > 0")

    peak_reserved_bytes = int(
        training_block.get("peak_reserved_bytes")
        or training_block.get("allocator_peak_reserved_bytes")
        or 0
    )
    if peak_reserved_bytes <= 0:
        raise ValueError("training.peak_reserved_bytes 必须 > 0")

    # device_total：从 memory.before_model.total_bytes（最初模型加载前的设备总显存）
    mem_block = report.get("memory", {})
    before = mem_block.get("before_model", {}) if isinstance(mem_block, dict) else {}
    device_total = int(before.get("total_bytes") or 0)
    if device_total <= 0:
        # 后备：从 final_report 中读取（部分旧报告直接写 total_memory_bytes）
        env_block = report.get("environment", {}) or {}
        device_total = int(env_block.get("total_memory_bytes") or 0)
    if device_total <= 0:
        raise ValueError("无法从 final_report.json 恢复 device_total_bytes")

    # 1) 改写 training 块（保留 deprecated 别名）
    training_block["peak_allocated_bytes"] = peak_alloc_bytes
    training_block["peak_allocated_gib"] = _bytes_to_gib(peak_alloc_bytes)
    training_block["allocator_peak_reserved_bytes"] = peak_reserved_bytes
    training_block["allocator_peak_reserved_gib"] = _bytes_to_gib(peak_reserved_bytes)
    # deprecated aliases（向后兼容旧报告 reader）
    training_block["peak_reserved_bytes"] = peak_reserved_bytes
    training_block["peak_reserved_gib"] = _bytes_to_gib(peak_reserved_bytes)
    training_block["device_total_bytes"] = device_total
    training_block["device_total_gib"] = _bytes_to_gib(device_total)
    report["training"] = training_block

    # 2) 改写 memory 块
    mem_block = report.get("memory", {})
    if isinstance(mem_block, dict):
        # 保留 before/after 切片
        mem_block["training_peak_allocated_bytes"] = peak_alloc_bytes
        mem_block["training_peak_allocated_gib"] = _bytes_to_gib(peak_alloc_bytes)
        mem_block["training_allocator_peak_reserved_bytes"] = peak_reserved_bytes
        mem_block["training_allocator_peak_reserved_gib"] = _bytes_to_gib(peak_reserved_bytes)
        # deprecated alias
        mem_block["training_peak_reserved_bytes"] = peak_reserved_bytes
        mem_block["training_peak_reserved_gib"] = _bytes_to_gib(peak_reserved_bytes)
        mem_block["device_total_bytes"] = device_total
        mem_block["device_total_gib"] = _bytes_to_gib(device_total)

        # peak_allocated_within_80pct：物理占用率（按 peak_allocated / device_total）
        peak_alloc_pct = (peak_alloc_bytes / device_total) if device_total else 0.0
        mem_block["peak_allocated_within_80pct"] = bool(peak_alloc_pct <= 0.80)
        mem_block["vram_allocated_percentage"] = round(peak_alloc_pct * 100.0, 3)

        report["memory"] = mem_block

    # 3) legend
    report["memory_field_legend"] = build_memory_field_legend(device_total)

    # 收口元数据
    report["memory_audit"] = {
        "status": "CORRECTED",
        "reason": (
            "Phase 5C-B 收口：torch.cuda.max_memory_reserved() (= 35.408 GiB) > "
            "device total (= 31.842 GiB) 是 CUDA caching allocator 的合法行为 "
            "(virtual reservation / expandable segments / 池化复用)。"
            "报告口径修正：peak_reserved_gib → allocator_peak_reserved_gib，"
            "明确这是 allocator 保留字节数，NOT equivalent to physical VRAM consumption；"
            "判断 OOM 风险必须以 peak_allocated_gib (= 19.605 GiB) 为准。"
        ),
        "physical_vram_peak_gib": _bytes_to_gib(peak_alloc_bytes),
        "allocator_reserved_peak_gib": _bytes_to_gib(peak_reserved_bytes),
        "device_total_gib": _bytes_to_gib(device_total),
        "reserved_exceeds_total": bool(peak_reserved_bytes > device_total),
    }
    return report


def _attach_best_adapter_signature(report: Dict[str, Any], verify_path: Path) -> None:
    """把 best_adapter_verification.json 的 signature 挂到 final_report 上。"""
    if not verify_path.exists():
        report["best_adapter_signature"] = {
            "status": "VERIFY_REPORT_MISSING",
            "path": str(verify_path),
        }
        return
    try:
        verify = json.loads(verify_path.read_text(encoding="utf-8"))
    except Exception as e:
        report["best_adapter_signature"] = {
            "status": "VERIFY_REPORT_PARSE_FAIL",
            "error": str(e),
        }
        return
    report["best_adapter_signature"] = {
        "status": verify.get("status", "UNKNOWN"),
        "BEST_ADAPTER_MATCHES_CHECKPOINT_288": bool(
            verify.get("BEST_ADAPTER_MATCHES_CHECKPOINT_288", False)
        ),
        "best_signature": verify.get("best_signature"),
        "checkpoint_signature": verify.get("checkpoint_signature"),
        "n_tensors": verify.get("n_tensors"),
        "prefer_dtype": verify.get("prefer_dtype"),
        "verification_path": str(verify_path),
    }


def main() -> int:
    args = _parse_args()
    log = get_logger("rewrite_phase5c_b_final_report")

    if not args.final_report.exists():
        log.error("缺失：%s", args.final_report)
        return 10

    report = json.loads(args.final_report.read_text(encoding="utf-8"))
    original_train_losses = (report.get("training", {}) or {}).get("loss_history")
    original_validation = report.get("validation")

    log.info(
        "源 final_report.json: %d train steps, best_epoch=%s",
        len(original_train_losses) if isinstance(original_train_losses, list) else 0,
        (original_validation or {}).get("best_epoch"),
    )

    report = _rewrite_memory(report)
    _attach_best_adapter_signature(report, args.verify_report)

    # Safety: 确认 training/validation 数据未被改动
    assert (report.get("training", {}) or {}).get("loss_history") == original_train_losses, (
        "training.loss_history 被改动 ⇒ 立即中止"
    )
    assert report.get("validation") == original_validation, (
        "validation 数据被改动 ⇒ 立即中止"
    )

    atomic_write_json(args.final_report, report)
    log.info("已写回：%s", args.final_report)
    print(json.dumps({
        "status": "PASS",
        "output": str(args.final_report),
        "physical_vram_peak_gib": report["memory_audit"]["physical_vram_peak_gib"],
        "allocator_reserved_peak_gib": report["memory_audit"]["allocator_reserved_peak_gib"],
        "device_total_gib": report["memory_audit"]["device_total_gib"],
        "BEST_ADAPTER_MATCHES_CHECKPOINT_288": report.get(
            "best_adapter_signature", {}
        ).get("BEST_ADAPTER_MATCHES_CHECKPOINT_288"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
