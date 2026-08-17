"""Phase 3 CLI：构建语义单元 + 结构感知 chunking。

用法：

    python scripts/build_chunks.py
    python scripts/build_chunks.py \\
        --input data/processed/documents_clean.jsonl \\
        --units-output data/chunks/semantic_units.jsonl \\
        --chunks-output data/chunks/chunks.jsonl \\
        --report data/chunks/chunk_report.json \\
        --log-file logs/build_chunks.log \\
        --min-chars 100 --target-chars 800 --max-chars 1200 --overlap-chars 100
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

from data_pipeline.chunking import ChunkingConfig, run_chunking_pipeline  # noqa: E402
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
        description="Phase 3: 恢复源序、识别 FAQ、构建语义单元与结构感知 chunking。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "documents_clean.jsonl",
        help="Phase 2 clean JSONL 输入。",
    )
    parser.add_argument(
        "--units-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunks" / "semantic_units.jsonl",
        help="semantic_units 输出 JSONL。",
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl",
        help="chunks 输出 JSONL。",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunks" / "chunk_report.json",
        help="chunk_report 输出 JSON。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=PROJECT_ROOT / "logs" / "build_chunks.log",
        help="日志文件路径。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别（DEBUG / INFO / WARNING）。",
    )
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--target-chars", type=int, default=800)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=100)
    parser.add_argument(
        "--faq-min-confidence",
        default="medium",
        choices=["low", "medium", "high"],
        help="FAQ 自动配对的最低置信度。",
    )
    return parser.parse_args(argv)


def _print_summary(result, report: dict, args: argparse.Namespace) -> None:
    width = max(len(k) for k in [
        "Input records",
        "Semantic units",
        "Chunks",
        "FAQ units",
        "Heading units",
        "Paragraph groups",
        "List groups",
        "High confidence FAQ",
        "Medium confidence FAQ",
        "Unpaired questions",
        "FAQ chunks",
        "Text chunks",
        "Heading chunks",
    ])
    faq = report.get("faq", {})
    ut = report.get("unit_types", {})
    ct = report.get("chunks_by_type", {})
    rows = [
        ("Input records", result.input_records),
        ("Semantic units", result.semantic_units),
        ("Chunks", result.chunks),
        ("FAQ units", ut.get("faq", 0)),
        ("Heading units", ut.get("heading", 0) + ut.get("subheading", 0) + ut.get("chapter", 0) + ut.get("section", 0)),
        ("Paragraph groups", ut.get("paragraph_group", 0)),
        ("List groups", ut.get("list_group", 0)),
        ("Standalone", ut.get("standalone", 0)),
        ("High confidence FAQ", faq.get("high_confidence", 0)),
        ("Medium confidence FAQ", faq.get("medium_confidence", 0)),
        ("Unpaired questions", faq.get("unpaired_questions", 0)),
        ("FAQ chunks", ct.get("faq", 0)),
        ("Text chunks", ct.get("text", 0)),
        ("Heading chunks", ct.get("heading", 0)),
    ]
    print("=" * 60)
    print("Phase 3 Semantic Unit & Chunking Completed")
    print("=" * 60)
    print()
    print("Inputs:")
    print(f"    {result.input_path.as_posix()}")
    print()
    print("Outputs:")
    print(f"    Units:      {args.units_output.as_posix()}")
    print(f"    Chunks:     {args.chunks_output.as_posix()}")
    print(f"    Report:     {args.report.as_posix()}")
    print()
    for k, v in rows:
        print(f"{k.ljust(width)}: {v}")
    print()
    print("Chunk length:")
    L = report.get("length", {})
    for k in ("min", "mean", "median", "p95", "max"):
        print(f"    {k:>6}: {L.get(k, 0)}")
    print()
    print("Splitting:")
    S = report.get("splitting", {})
    for k in ("split_units", "hard_splits", "overlap_chunks", "overlap_chars_total"):
        print(f"    {k:>20}: {S.get(k, 0)}")
    print()
    print(f"See logs: {args.log_file.as_posix()}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_file, args.log_level)
    log = get_logger()

    cfg = ChunkingConfig(
        min_chars=args.min_chars,
        target_chars=args.target_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        faq_min_confidence=args.faq_min_confidence,
    )

    result = run_chunking_pipeline(
        input_path=args.input,
        output_units=args.units_output,
        output_chunks=args.chunks_output,
        output_report=args.report,
        config=cfg,
        logger=log,
    )

    import json
    report = json.loads(args.report.read_text(encoding="utf-8"))
    _print_summary(result, report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())