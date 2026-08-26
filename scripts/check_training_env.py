"""Phase 5B 训练环境预检脚本。

仅做：

- 依赖 + 版本检查（torch / transformers / trl / peft / accelerate /
  datasets / bitsandbytes 共 7 项，缺一即 BLOCKED）。
- ``python -m pip check``；冲突即 BLOCKED（DEPENDENCY_CONFLICT）。
- CUDA availability + GPU 信息。
- BF16 支持。
- Blackwell (sm_120) 检查。
- 真实 BF16 matmul 计算测试。

不下载 Qwen3-8B 权重，不初始化训练。

输出：单行 JSON 到 stdout；详细日志写到 ``--log-file``。
退出码：

- 0：所有检查通过
- 2：依赖缺失（DEPENDENCY_MISSING）
- 3：CUDA / GPU 不可用
- 4：BF16 不支持
- 5：Blackwell 不兼容
- 6：CUDA 计算失败
- 7：依赖冲突（DEPENDENCY_CONFLICT）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_pipeline.utils import get_logger


# 必装依赖清单；缺一即 DEPENDENCY_MISSING，exit_code=2。
REQUIRED_DEPENDENCIES: List[str] = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "bitsandbytes",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5B 训练环境预检")
    p.add_argument("--log-file", default="logs/check_training_env.log", type=Path)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _safe_version(module_name: str) -> str | None:
    try:
        m = __import__(module_name)
    except Exception:
        return None
    return getattr(m, "__version__", None)


def _gather_versions() -> Dict[str, str | None]:
    out: Dict[str, str | None] = {"python": platform.python_version()}
    for dep in REQUIRED_DEPENDENCIES:
        out[dep] = _safe_version(dep)
    return out


def _torch_env() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import torch  # type: ignore
    except Exception as e:
        return {"installed": False, "error": str(e)}

    out["installed"] = True
    out["version"] = torch.__version__
    out["cuda_version"] = getattr(torch.version, "cuda", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        out["device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        out["compute_capability"] = list(cap)
        out["arch_list"] = list(torch.cuda.get_arch_list())
        out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        props = torch.cuda.get_device_properties(0)
        out["total_memory_bytes"] = int(props.total_memory)
    else:
        out["device_name"] = None
        out["compute_capability"] = None
        out["arch_list"] = []
        out["bf16_supported"] = False
        out["total_memory_bytes"] = 0
    return out


def _run_pip_check() -> Dict[str, Any]:
    """调 ``python -m pip check``，失败信息直接落入 ``error``。"""
    try:
        cp = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True, text=True, check=False, timeout=60,
        )
        return {
            "returncode": cp.returncode,
            "stdout": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
            "ok": cp.returncode == 0,
        }
    except Exception as e:
        return {"returncode": -1, "ok": False, "error": str(e)}


def _run_bf16_matmul() -> Dict[str, Any]:
    """执行 BF16 真实矩阵乘。"""
    try:
        import torch  # type: ignore
    except Exception as e:
        return {"ok": False, "error": f"torch 不可用: {e}"}

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA 不可用"}

    try:
        x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
        y = x @ x.T
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(y).all().item())
        return {"ok": finite, "shape": list(y.shape), "dtype": str(y.dtype)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _summarize_blackwell(torch_info: Dict[str, Any]) -> Dict[str, Any]:
    cap = torch_info.get("compute_capability")
    arch = torch_info.get("arch_list") or []
    if not cap:
        return {"sm_120_supported": None, "reason": "no CUDA"}
    is_sm120 = tuple(cap) == (12, 0)
    arch_has_sm120 = any("sm_120" in a or a == "12.0" for a in arch)
    return {
        "is_sm_120_gpu": is_sm120,
        "arch_list": arch,
        "arch_has_sm_120": arch_has_sm120,
        "sm_120_supported": (not is_sm120) or arch_has_sm120,
    }


def _missing_dependencies(versions: Dict[str, str | None]) -> List[str]:
    return [dep for dep in REQUIRED_DEPENDENCIES if not versions.get(dep)]


def _exit_code_from(report: Dict[str, Any]) -> int:
    """根据 report 字段统一计算 exit_code。"""
    missing = _missing_dependencies(report["versions"])
    if missing:
        report.setdefault("reasons", []).append(
            f"DEPENDENCY_MISSING: {missing}"
        )
        return 2
    if report["pip_check"]["returncode"] != 0:
        report.setdefault("reasons", []).append("DEPENDENCY_CONFLICT")
        return 7
    if not report["torch"]["cuda_available"]:
        report.setdefault("reasons", []).append("CUDA_UNAVAILABLE")
        return 3
    if not report["torch"]["bf16_supported"]:
        report.setdefault("reasons", []).append("BF16_UNSUPPORTED")
        return 4
    if report["blackwell"]["sm_120_supported"] is False:
        report.setdefault("reasons", []).append("BLACKWELL_INCOMPAT")
        return 5
    if not report["bf16_matmul"]["ok"]:
        report.setdefault("reasons", []).append("BF16_MATMUL_FAIL")
        return 6
    report.setdefault("reasons", []).append("PASS")
    return 0


def main() -> int:
    args = _parse_args()
    log = get_logger("check_training_env")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(args.log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    versions = _gather_versions()
    log.info("Versions: %s", versions)
    torch_info = _torch_env()
    log.info("Torch env: %s", torch_info)
    bf16_matmul = _run_bf16_matmul()
    log.info("BF16 matmul: %s", bf16_matmul)
    pip_check = _run_pip_check()
    log.info("pip check: %s", pip_check)
    blackwell = _summarize_blackwell(torch_info)
    log.info("Blackwell: %s", blackwell)

    report: Dict[str, Any] = {
        "versions": versions,
        "torch": torch_info,
        "blackwell": blackwell,
        "bf16_matmul": bf16_matmul,
        "pip_check": pip_check,
        "missing_dependencies": _missing_dependencies(versions),
    }

    report["exit_code"] = _exit_code_from(report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    log.info("FINAL exit_code=%d", report["exit_code"])
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())