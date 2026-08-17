"""在目录树上运行 extractor 的编排层。"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import DocumentRecord, serialize_record
from .registry import EXTRACTORS, get_extractor
from .utils import (
    calculate_file_sha256,
    to_relative_path,
)


@dataclass
class FileOutcome:
    """单个文件的抽取结果。

    Attributes:
        source_path: 相对路径（POSIX 风格）。
        source_type: extractor 报告的类型标签。
        status: 处理状态，``success`` / ``failed`` / ``skipped`` 之一。
        records: 实际写入的 record 数。
        sha256: 文件内容 SHA256。
        warnings: 与该文件相关的告警列表。
        error_type: 失败时的异常类名。
        error: 失败时的异常字符串。
    """

    source_path: str
    source_type: str
    status: str  # "success" | "failed" | "skipped"
    records: int = 0
    sha256: str = ""
    warnings: list[str] = field(default_factory=list)
    error_type: str | None = None
    error: str | None = None


@dataclass
class PipelineResult:
    """整个目录树抽取的汇总结果。

    Attributes:
        input_root: 输入根目录绝对路径。
        output_jsonl: 写入的 JSONL 文件路径。
        manifest_path: 写入的 manifest 文件路径。
        outcomes: 每个文件的 :class:`FileOutcome`。
        log_path: 日志文件路径（可选）。
    """

    input_root: Path
    output_jsonl: Path
    manifest_path: Path
    outcomes: list[FileOutcome] = field(default_factory=list)
    log_path: Path | None = None

    @property
    def total_records(self) -> int:
        return sum(o.records for o in self.outcomes if o.status == "success")

    def to_manifest(self) -> dict[str, Any]:
        statistics = {
            "total_files": len(self.outcomes),
            "success_files": sum(1 for o in self.outcomes if o.status == "success"),
            "failed_files": sum(1 for o in self.outcomes if o.status == "failed"),
            "skipped_files": sum(1 for o in self.outcomes if o.status == "skipped"),
            "total_records": self.total_records,
        }
        source_types: dict[str, int] = {}
        for o in self.outcomes:
            if o.status != "success":
                continue
            source_types[o.source_type] = source_types.get(o.source_type, 0) + 1
        warnings_total: dict[str, int] = {}
        for o in self.outcomes:
            for w in o.warnings:
                warnings_total[w] = warnings_total.get(w, 0) + 1
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_root": self.input_root.as_posix(),
            "output_jsonl": self.output_jsonl.as_posix(),
            "statistics": statistics,
            "source_types": source_types,
            "warnings_summary": warnings_total,
            "files": [
                {
                    "source_path": o.source_path,
                    "source_type": o.source_type,
                    "status": o.status,
                    "records": o.records,
                    "sha256": o.sha256,
                    "warnings": o.warnings,
                    "error_type": o.error_type,
                    "error": o.error,
                }
                for o in self.outcomes
            ],
        }


def _atomic_write_jsonl(path: Path, lines: list[str]) -> None:
    """以原子方式写入 JSONL（临时文件 + os.replace）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for line in lines:
                fh.write(line)
                fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """以原子方式写入 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def run_pipeline(
    input_root: Path,
    output_jsonl: Path,
    manifest_path: Path,
    *,
    logger: logging.Logger | None = None,
) -> PipelineResult:
    """在 ``input_root`` 目录树上运行抽取并将 JSONL / manifest 写入磁盘。"""
    log = logger or logging.getLogger("data_pipeline")
    input_root = input_root.resolve()
    output_jsonl = output_jsonl.resolve()
    manifest_path = manifest_path.resolve()

    if not input_root.exists():
        log.warning("Input root does not exist: %s", input_root)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        empty_lines: list[str] = []
        _atomic_write_jsonl(output_jsonl, empty_lines)
        _atomic_write_json(manifest_path, {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_root": input_root.as_posix(),
            "output_jsonl": output_jsonl.as_posix(),
            "statistics": {
                "total_files": 0,
                "success_files": 0,
                "failed_files": 0,
                "skipped_files": 0,
                "total_records": 0,
            },
            "source_types": {},
            "warnings_summary": {},
            "files": [],
        })
        return PipelineResult(
            input_root=input_root,
            output_jsonl=output_jsonl,
            manifest_path=manifest_path,
        )

    files = sorted(p for p in input_root.rglob("*") if p.is_file())
    log.info("Found %d candidate files under %s", len(files), input_root)

    outcomes: list[FileOutcome] = []
    lines: list[str] = []
    records_by_type: dict[str, int] = {}

    for path in files:
        relative_path = to_relative_path(path, input_root)
        extractor = get_extractor(path)
        if extractor is None:
            outcomes.append(FileOutcome(
                source_path=relative_path,
                source_type="unknown",
                status="skipped",
                warnings=["unsupported_suffix"],
            ))
            log.info("Skipping (unsupported suffix): %s", relative_path)
            continue

        log.info("Extracting %s", relative_path)
        try:
            file_sha256 = calculate_file_sha256(path)
        except Exception as exc:  # pragma: no cover - filesystem errors
            outcomes.append(FileOutcome(
                source_path=relative_path,
                source_type=extractor.source_type,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
            ))
            log.error("Failed to hash %s: %s", relative_path, exc)
            continue

        try:
            extraction = extractor.extract(
                path,
                relative_path=relative_path,
                file_sha256=file_sha256,
            )
        except Exception as exc:
            outcomes.append(FileOutcome(
                source_path=relative_path,
                source_type=extractor.source_type,
                status="failed",
                sha256=file_sha256,
                error_type=type(exc).__name__,
                error=str(exc),
            ))
            log.error(
                "Failed to parse %s [%s]: %s",
                relative_path,
                type(exc).__name__,
                exc,
            )
            continue

        kept = 0
        for record in extraction.records:
            if not record.text or not record.text.strip():
                continue
            # Ensure SHA256 is populated even if the extractor forgot.
            if not record.file_sha256:
                record.file_sha256 = file_sha256
            # Ensure source_path is a relative POSIX path.
            record.source_path = relative_path
            lines.append(serialize_record(record))
            kept += 1
            records_by_type[extractor.source_type] = (
                records_by_type.get(extractor.source_type, 0) + 1
            )

        for record in extraction.records:
            if not record.text or not record.text.strip():
                extraction.warnings.append("empty_records_skipped")

        outcomes.append(FileOutcome(
            source_path=relative_path,
            source_type=extractor.source_type,
            status="success",
            records=kept,
            sha256=file_sha256,
            warnings=list(extraction.warnings),
        ))
        log.info("Extracted %d records from %s", kept, relative_path)
        if extraction.warnings:
            for w in extraction.warnings:
                log.warning("%s: %s", relative_path, w)

    _atomic_write_jsonl(output_jsonl, lines)
    result = PipelineResult(
        input_root=input_root,
        output_jsonl=output_jsonl,
        manifest_path=manifest_path,
        outcomes=outcomes,
    )
    _atomic_write_json(manifest_path, result.to_manifest())
    log.info(
        "Wrote %d records to %s and manifest to %s",
        result.total_records,
        output_jsonl,
        manifest_path,
    )
    return result
