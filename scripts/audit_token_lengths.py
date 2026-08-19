"""Phase 5A CLI — 用 Qwen3 tokenizer 审计 Phase 4 SFT 数据集。

不训练模型，不加载 ``AutoModelForCausalLM``，不初始化 GPU；
只通过 ``AutoTokenizer.apply_chat_template`` 做 token 长度与 prefix 审计。

用法::

    python scripts/audit_token_lengths.py

    # 自定义
    python scripts/audit_token_lengths.py \\
        --sft-dir data/sft \\
        --model-name Qwen/Qwen3-8B \\
        --revision main \\
        --output-audit data/tokenizer/token_audit.jsonl \\
        --output-report data/tokenizer/token_report.json \\
        --log-file logs/audit_token_lengths.log
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

from data_pipeline.tokenization import run_tokenizer_audit  # noqa: E402
from data_pipeline.tokenization.models import (  # noqa: E402
    DEFAULT_CANDIDATE_MAX_LENGTHS,
    DEFAULT_MODEL_NAME,
    DEFAULT_REAL_SAMPLE_CHECK_PER_SPLIT,
    DEFAULT_REVISION,
    TokenizerAuditConfig,
)
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
        description="Phase 5A: 用 Qwen3 tokenizer 审计 Phase 4 SFT 数据集 token 长度。",
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft",
        help="Phase 4 输出的 SFT 数据集目录（含 all/train/validation/test.jsonl）。",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face 模型仓库标识，默认 Qwen/Qwen3-8B。",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face 模型 revision，默认 main。",
    )
    parser.add_argument(
        "--candidate-max-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_MAX_LENGTHS),
        help="候选最大序列长度列表，用于统计覆盖率。",
    )
    parser.add_argument(
        "--real-sample-check-per-split",
        type=int,
        default=DEFAULT_REAL_SAMPLE_CHECK_PER_SPLIT,
        help="每个 split 抽几条做内部 sanity check。",
    )
    parser.add_argument(
        "--output-audit",
        type=Path,
        default=PROJECT_ROOT / "data" / "tokenizer" / "token_audit.jsonl",
        help="逐 sample 轻量审计 JSONL。",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "data" / "tokenizer" / "token_report.json",
        help="审计汇总报告 JSON。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=PROJECT_ROOT / "logs" / "audit_token_lengths.log",
        help="日志文件路径。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别（DEBUG / INFO / WARNING）。",
    )
    return parser.parse_args(argv)


def _print_summary(report: dict, args: argparse.Namespace) -> None:
    """按 README 要求的统计格式打印。"""
    print("=" * 60)
    print("Phase 5A Tokenizer Audit Completed")
    print("=" * 60)
    print()
    print(f"SFT dir:  {args.sft_dir.as_posix()}")
    print()
    m = report["model"]
    print("Model")
    print(f"    Name:              {m['model_name']}")
    print(f"    Requested rev:     {m['requested_revision']}")
    print(f"    Resolved rev:      {m['resolved_revision']}")
    print(f"    Transformers:      {m['transformers_version']}")
    print()
    t = report["tokenizer"]
    print("Tokenizer")
    print(f"    Class:             {t['tokenizer_class']}")
    print(f"    Vocab size:        {t['vocab_size']}")
    print(f"    BOS:               {t['bos_token']} ({t['bos_token_id']})")
    print(f"    EOS:               {t['eos_token']} ({t['eos_token_id']})")
    print(f"    PAD:               {t['pad_token']} ({t['pad_token_id']})")
    print(f"    Model max length:  {t['model_max_length']}")
    print(f"    Chat template:     {t['chat_template_present']}")
    print()
    dm = report["dataset_membership"]
    print("Dataset Membership")
    print(f"    All:               {dm['all_count']}")
    print(f"    Train:             {dm['train_count']}")
    print(f"    Validation:        {dm['validation_count']}")
    print(f"    Test:              {dm['test_count']}")
    print(f"    Set equal:         {dm['set_equal_all_vs_union']}")
    print()
    print("Length Stats (full tokens)")
    ls = report["length_stats"]["total_tokens"]
    print(f"    min={ls['min']} mean={ls['mean']} median={ls['median']} "
          f"p90={ls['p90']} p95={ls['p95']} p99={ls['p99']} max={ls['max']}")
    print()
    print("Coverage")
    for c, cov in report["coverage"].items():
        print(f"    {c:>5}: fit={cov['fit_count']} ({cov['fit_ratio']}), "
              f"over={cov['over_count']} ({cov['over_ratio']})")
    print()
    print("Coverage Thresholds")
    for k, v in report["coverage_thresholds"].items():
        print(f"    {k}: {v}")
    print()
    print("Assistant-only Loss")
    print(f"    Prompt prefix mismatch: {report['prompt_prefix_mismatch_count']}")
    print(f"    Zero assistant target:  {report['zero_assistant_target_count']}")
    print(f"    Feasible:               {report['assistant_only_loss_feasible']}")
    print()
    print("Source Thinking Markers")
    stm = report["source_thinking_markers"]
    print(f"    <think>:           {stm['think_open']}")
    print(f"    </think>:          {stm['think_close']}")
    print(f"    reasoning_content: {stm['reasoning_content']}")
    print()
    print(f"Outputs")
    print(f"    Audit:   {args.output_audit.as_posix()}")
    print(f"    Report:  {args.output_report.as_posix()}")
    print(f"See logs: {args.log_file.as_posix()}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_file, args.log_level)
    log = get_logger()

    cfg = TokenizerAuditConfig(
        model_name=args.model_name,
        revision=args.revision,
        candidate_max_lengths=tuple(args.candidate_max_lengths),
        real_sample_check_per_split=args.real_sample_check_per_split,
    )

    report = run_tokenizer_audit(
        sft_dir=args.sft_dir,
        output_audit_jsonl=args.output_audit,
        output_report_json=args.output_report,
        config=cfg,
        logger=log,
    )

    _print_summary(report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())