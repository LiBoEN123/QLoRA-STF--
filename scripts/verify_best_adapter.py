"""Phase 5C-B 收口 — Best Adapter 一致性验证脚本。

不做任何训练；只读取 ``outputs/phase5c-b/best_adapter`` 与
``outputs/phase5c-b/checkpoints/checkpoint-288`` 的 adapter 权重
``adapter_model.safetensors``，按 ``tensor name / shape / bytes`` 升序拼接
计算 SHA256 signature，验证两者在 **数值** 上完全一致。

为什么不能直接比较文件 SHA256？

- TRL ``load_best_model_at_end=True`` 时 ``trainer.save_adapter`` 可能把
  权重重新保存为 ``float32``（即使原始 checkpoint 是 ``bfloat16``）；
- 这意味着文件级 SHA256 会假性不匹配；但 cast 到同一 dtype 后字节值相等。
- 因此本脚本强制 cast 到 ``bfloat16`` 再比较。

退出码：

- 0：signature 完全一致 ⇒ BEST_ADAPTER_MATCHES_CHECKPOINT_288 = TRUE
- 10：参数错误 / 文件不存在
- 20：signature 不一致 ⇒ BLOCKED
- 30：adapter 内部状态异常（如缺权重文件）
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

from data_pipeline.training import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_REVISION,
    compute_dataset_fingerprint,
    load_run_manifest,
    verify_best_adapter_matches_checkpoint,
)
from data_pipeline.utils import atomic_write_json, get_logger  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("outputs/phase5c-b")
DEFAULT_RUN_MANIFEST = Path("data/training/phase5c_b/run_manifest.json")
DEFAULT_OUTPUT_REPORT = Path("data/training/phase5c_b/best_adapter_verification.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5C-B best adapter 一致性验证")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), type=Path)
    p.add_argument("--run-manifest", default=str(DEFAULT_RUN_MANIFEST), type=Path)
    p.add_argument(
        "--output-report",
        default=str(DEFAULT_OUTPUT_REPORT),
        type=Path,
        help="验证结果 JSON 输出路径",
    )
    p.add_argument(
        "--best-epoch",
        type=int,
        default=2,
        help="expected best epoch；用于 cross-check 报告",
    )
    p.add_argument(
        "--prefer-dtype",
        default="bfloat16",
        choices=["bfloat16", "float32", "float16"],
        help="比较时把 tensor 强转到该 dtype；Phase 5C-B 固定 bfloat16",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    log = get_logger("verify_best_adapter")

    best_adapter_dir = args.output_root / "best_adapter"
    ckpt_dir = args.output_root / "checkpoints" / f"checkpoint-{int(144 * args.best_epoch)}"
    best_st = best_adapter_dir / "adapter_model.safetensors"
    ckpt_st = ckpt_dir / "adapter_model.safetensors"

    report: Dict[str, Any] = {
        "status": "BLOCKED",
        "best_adapter_matches_checkpoint_288": False,
        "backend": "unsloth",
        "expected_best_epoch": int(args.best_epoch),
    }

    if not best_st.exists():
        log.error("best_adapter 缺失：%s", best_st)
        report["error"] = f"best_adapter 缺失：{best_st}"
        atomic_write_json(args.output_report, report)
        return 30

    if not ckpt_st.exists():
        log.error("checkpoint adapter 缺失：%s", ckpt_st)
        report["error"] = f"checkpoint adapter 缺失：{ckpt_st}"
        atomic_write_json(args.output_report, report)
        return 30

    # Load torch + run_manifest
    try:
        import torch  # type: ignore  # noqa: F401
    except Exception as e:
        log.error("torch 不可用：%s", e)
        report["error"] = f"torch 不可用：{e}"
        atomic_write_json(args.output_report, report)
        return 10

    try:
        manifest = load_run_manifest(args.run_manifest)
    except Exception as e:
        log.warning("run_manifest 缺失/无法解析（%s）；继续做 signature 验证", e)
        manifest = {}

    # Verification
    try:
        result = verify_best_adapter_matches_checkpoint(
            best_adapter_path=best_st,
            checkpoint_path=ckpt_st,
            torch=torch,
            prefer_dtype=args.prefer_dtype,
        )
    except Exception as e:
        log.error("验证过程异常：%s", e)
        report["error"] = f"验证过程异常：{e}"
        atomic_write_json(args.output_report, report)
        return 30

    report.update(result)

    # Cross-check with manifest (best epoch)
    manifest_validation_block = manifest.get("validation", {})
    expected_best_epoch = manifest_validation_block.get("best_epoch", args.best_epoch)
    manifest_best_loss = manifest_validation_block.get("best_eval_loss")
    manifest_before = manifest_validation_block.get("before_training")
    report["cross_check"] = {
        "manifest_best_epoch": expected_best_epoch,
        "manifest_best_eval_loss": manifest_best_loss,
        "manifest_eval_loss_before_training": manifest_before,
        "expected_best_epoch_matches": bool(int(expected_best_epoch) == int(args.best_epoch)),
    }

    if result["equal"]:
        report["status"] = "PASS"
        report["best_adapter_matches_checkpoint_288"] = True
        report["BEST_ADAPTER_MATCHES_CHECKPOINT_288"] = True
        log.info(
            "PASS: best_adapter signature == checkpoint-288 signature; "
            "n_tensors=%d sig=%s",
            result["n_tensors"],
            result["best_signature"]["signature"][:16],
        )
        atomic_write_json(args.output_report, report)
        return 0
    else:
        report["status"] = "BLOCKED"
        report["error"] = (
            "best_adapter 与 checkpoint signature 不一致："
            f"best={result['best_signature']['signature'][:16]} "
            f"ckpt={result['checkpoint_signature']['signature'][:16]}"
        )
        log.error("BLOCKED: best_adapter != checkpoint-288")
        atomic_write_json(args.output_report, report)
        return 20


if __name__ == "__main__":
    sys.exit(main())
