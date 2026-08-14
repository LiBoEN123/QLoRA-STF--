"""Phase 2 文档清洗的 CLI 入口。

用法：

    python scripts/clean_documents.py                # 使用默认路径
    python scripts/clean_documents.py --input data/extracted/documents.jsonl \\
        --output data/processed/documents_clean.jsonl \\
        --rejected data/processed/rejected_documents.jsonl \\
        --duplicates data/processed/duplicates.jsonl \\
        --report data/processed/quality_report.json \\
        --log-file logs/clean_documents.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline.cleaning.pipeline import (  # noqa: E402
    CleaningConfig,
    QualityConfig,
    run_cleaning_pipeline,
)
from data_pipeline.utils import get_logger  # noqa: E402


def _configure_logging(log_file: Path, log_level: str) -> None:
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
        description="Phase 2: clean, normalize, filter, and deduplicate records.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "extracted" / "documents.jsonl",
        help="Phase 1 JSONL input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "documents_clean.jsonl",
        help="Destination JSONL for cleaned records.",
    )
    parser.add_argument(
        "--rejected",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "rejected_documents.jsonl",
        help="Destination JSONL for rejected records.",
    )
    parser.add_argument(
        "--duplicates",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "duplicates.jsonl",
        help="Destination JSONL for duplicates.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "quality_report.json",
        help="Destination quality report JSON.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=PROJECT_ROOT / "logs" / "clean_documents.log",
        help="Path to the run log file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=20_000)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.5)
    parser.add_argument(
        "--simhash-distance", type=int, default=3,
        help="Hamming distance threshold for near-duplicate detection.",
    )
    parser.add_argument(
        "--disable-near-dedup", action="store_true",
        help="Only run exact dedup, skip SimHash near-dedup.",
    )
    return parser.parse_args(argv)


def _print_summary(result, args: argparse.Namespace) -> None:
    stats = {
        "Input records": result.input_records,
        "Clean records": result.output_records,
        "Rejected records": result.rejected_records,
        "Exact duplicates": result.exact_duplicates,
        "Near duplicates": result.near_duplicates,
        "Retention rate": f"{(result.output_records / result.input_records) if result.input_records else 0:.2%}",
    }
    width = max(len(k) for k in stats)
    print("=" * 60)
    print("Phase 2 Document Cleaning Completed")
    print("=" * 60)
    print()
    print("Inputs:")
    print(f"    {result.input_path.as_posix()}")
    print()
    print("Outputs:")
    print(f"    Clean:       {args.output.as_posix()}")
    print(f"    Rejected:    {args.rejected.as_posix()}")
    print(f"    Duplicates:  {args.duplicates.as_posix()}")
    print(f"    Report:      {args.report.as_posix()}")
    print()
    for key, value in stats.items():
        print(f"{key.ljust(width)}: {value}")
    print()
    print("Structure counts:")
    for k, v in sorted(result.structure_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<12} {v}")
    print()
    print("Cleaning actions:")
    for k, v in sorted(result.cleaning_actions_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<24} {v}")
    print()
    print(f"See logs: {args.log_file.as_posix()}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_file, args.log_level)
    log = get_logger()

    cfg = CleaningConfig(
        min_length=args.min_length,
        max_length=args.max_length,
        max_symbol_ratio=args.max_symbol_ratio,
        simhash_distance=args.simhash_distance,
        enable_near_dedup=not args.disable_near_dedup,
        quality=QualityConfig(
            min_text_length=args.min_length,
            max_text_length=args.max_length,
            max_symbol_ratio=args.max_symbol_ratio,
        ),
    )

    result = run_cleaning_pipeline(
        input_path=args.input,
        output_clean=args.output,
        output_rejected=args.rejected,
        output_duplicates=args.duplicates,
        output_report=args.report,
        config=cfg,
        logger=log,
    )
    _print_summary(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())