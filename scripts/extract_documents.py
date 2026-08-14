"""CLI entry point for Phase 1 document extraction.

Usage:

    python scripts/extract_documents.py                # use defaults
    python scripts/extract_documents.py --input data/raw \\
        --output data/extracted/documents.jsonl \\
        --manifest data/extracted/manifest.json \\
        --log-file logs/extract_documents.log
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

from data_pipeline.extractor import run_pipeline  # noqa: E402
from data_pipeline.registry import supported_suffixes  # noqa: E402
from data_pipeline.utils import get_logger  # noqa: E402


def _configure_logging(log_file: Path, log_level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = get_logger()
    root.setLevel(log_level.upper())
    # Reset handlers so repeated calls (e.g. in tests) don't double-log.
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
        description="Phase 1: extract text from raw documents into a unified JSONL.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing raw source files (recursively scanned).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "extracted" / "documents.jsonl",
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "extracted" / "manifest.json",
        help="Destination manifest JSON file.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=PROJECT_ROOT / "logs" / "extract_documents.log",
        help="Path to the run log file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    return parser.parse_args(argv)


def _print_summary(result, suffix_counts: dict[str, int], log_file: Path) -> None:
    stats = {
        "Files found": len(result.outcomes),
        "Files succeeded": sum(1 for o in result.outcomes if o.status == "success"),
        "Files failed": sum(1 for o in result.outcomes if o.status == "failed"),
        "Files skipped": sum(1 for o in result.outcomes if o.status == "skipped"),
        "Records generated": result.total_records,
    }
    width = max(len(k) for k in stats)
    print("=" * 60)
    print("Document Extraction Completed")
    print("=" * 60)
    print()
    print("Input:")
    print(f"    {result.input_root.as_posix()}")
    print()
    print("Output:")
    print(f"    {result.output_jsonl.as_posix()}")
    print()
    print("Manifest:")
    print(f"    {result.manifest_path.as_posix()}")
    print()
    for key, value in stats.items():
        print(f"{key.ljust(width)}: {value}")
    print()
    print("By type:")
    for source_type, count in sorted(suffix_counts.items()):
        print(f"    {source_type:<14} {count}")
    warnings = sorted({w for o in result.outcomes for w in o.warnings})
    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            count = sum(1 for o in result.outcomes if w in o.warnings)
            print(f"    {w}: {count}")
    print()
    print("See logs:")
    print(f"    {log_file.as_posix()}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_file, args.log_level)
    log = get_logger()
    log.info("Supported formats: %s", ", ".join(supported_suffixes()))
    result = run_pipeline(
        input_root=args.input,
        output_jsonl=args.output,
        manifest_path=args.manifest,
        logger=log,
    )
    suffix_counts: dict[str, int] = {}
    for outcome in result.outcomes:
        if outcome.status != "success":
            continue
        suffix_counts[outcome.source_type] = (
            suffix_counts.get(outcome.source_type, 0) + outcome.records
        )
    _print_summary(result, suffix_counts, args.log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
