"""Phase 4 CLI：构造 SFT 数据集并做 Train/Validation/Test 确定性划分。

直接消费 Phase 3 产出的 ``semantic_units.jsonl``，只取 ``unit_type == "faq"``，
不重新 detect / pair，不调用任何 LLM，不加载 tokenizer。

用法：

    python scripts/build_sft_dataset.py
    python scripts/build_sft_dataset.py --min-confidence high --train-ratio 0.8
    python scripts/build_sft_dataset.py --input data/chunks/semantic_units.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline.sft import run_sft_pipeline  # noqa: E402
from data_pipeline.sft.models import SFTBuildConfig  # noqa: E402
from data_pipeline.utils import get_logger  # noqa: E402


def _configure_logging(log_file: Path, log_level: str) -> None:
    """配置控制台 + 文件双输出 logger。"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = get_logger()
    root.setLevel(log_level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: 把 FAQ SemanticUnit 转成 SFT messages 并划分 Train/Val/Test。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunks" / "semantic_units.jsonl",
        help="Phase 3 输出的 semantic_units.jsonl。",
    )
    parser.add_argument(
        "--all-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft" / "all.jsonl",
        help="accepted SFT samples 全量 JSONL。",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft" / "train.jsonl",
        help="训练集 JSONL。",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft" / "validation.jsonl",
        help="验证集 JSONL。",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft" / "test.jsonl",
        help="测试集 JSONL。",
    )
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft" / "rejected.jsonl",
        help="被剔除的样本 JSONL。",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft" / "dataset_report.json",
        help="数据集质量报告 JSON。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=PROJECT_ROOT / "logs" / "build_sft_dataset.log",
        help="日志文件路径。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别（DEBUG / INFO / WARNING）。",
    )
    parser.add_argument(
        "--min-confidence",
        default="medium",
        choices=["high", "medium"],
        help="最低 FAQ 置信度。",
    )
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--split-salt",
        default="qlora-sft-phase4-v1",
        help="确定性划分盐值，必须固定才能复现 split。",
    )
    return parser.parse_args(argv)


def _print_summary(result, report: dict, args: argparse.Namespace) -> None:
    """按 README 要求的统计格式打印。"""
    print("=" * 60)
    print("Phase 4 SFT Dataset Construction Completed")
    print("=" * 60)
    print()
    print(f"Input:  {args.input.as_posix()}")
    print()
    print("Inputs")
    print(f"    Input semantic units:   {report['input_units']}")
    print(f"    Input FAQ units:        {report['input_faq_units']}")
    print()
    print("Counts")
    print(f"    Accepted:               {report['accepted_samples']}")
    print(f"    Rejected:               {report['rejected_samples']}")
    print()
    print("Quality")
    print(f"    Exact duplicate:        {report['duplicate_samples']}")
    print(f"    Conflicting questions:  {report['conflicting_question_groups']}")
    print(f"    Conflicting samples:    {report['conflicting_samples']}")
    print()
    print("Confidence (accepted)")
    print(f"    High:                   {report['confidence']['high']}")
    print(f"    Medium:                 {report['confidence']['medium']}")
    print()
    print("Splits")
    print(f"    Train:                  {report['splits']['train']}")
    print(f"    Validation:             {report['splits']['validation']}")
    print(f"    Test:                   {report['splits']['test']}")
    ra = report["split_ratios_actual"]
    print(f"    Actual ratios:          train={ra['train']}, validation={ra['validation']}, test={ra['test']}")
    print()
    print("Leakage")
    print(f"    Cross-split question:   {report['leakage']['cross_split_question']}")
    print(f"    Cross-split exact q+a:  {report['leakage']['cross_split_exact_sample']}")
    print()
    print("Outputs")
    print(f"    All:        {args.all_output.as_posix()}")
    print(f"    Train:      {args.train_output.as_posix()}")
    print(f"    Validation: {args.validation_output.as_posix()}")
    print(f"    Test:       {args.test_output.as_posix()}")
    print(f"    Rejected:   {args.rejected_output.as_posix()}")
    print(f"    Report:     {args.report.as_posix()}")
    print()
    print(f"See logs: {args.log_file.as_posix()}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_file, args.log_level)
    log = get_logger()

    cfg = SFTBuildConfig(
        min_confidence=args.min_confidence,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        split_salt=args.split_salt,
    )

    result = run_sft_pipeline(
        input_path=args.input,
        output_all=args.all_output,
        output_train=args.train_output,
        output_validation=args.validation_output,
        output_test=args.test_output,
        output_rejected=args.rejected_output,
        output_report=args.report,
        config=cfg,
        logger=log,
    )

    report = json.loads(args.report.read_text(encoding="utf-8"))
    _print_summary(result, report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
