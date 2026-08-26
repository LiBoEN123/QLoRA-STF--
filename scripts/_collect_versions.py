"""采集 Python 环境版本信息（被 build_phase5c_a_report 调用）。"""
import importlib
import json
import sys


def main():
    out = {}
    for name in ("torch", "transformers", "trl", "peft", "accelerate",
                 "datasets", "bitsandbytes", "unsloth", "unsloth_zoo",
                 "triton", "xformers"):
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__version__", None)
        except Exception as e:
            out[name] = f"NOT_AVAILABLE: {type(e).__name__}"
    try:
        import torch
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["cuda_version"] = torch.version.cuda
            out["device_name"] = torch.cuda.get_device_name(0)
            out["compute_capability"] = list(torch.cuda.get_device_capability(0))
            out["arch_list"] = list(torch.cuda.get_arch_list())
            out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            out["total_memory_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 3
            )
    except Exception as e:
        out["torch_error"] = f"{type(e).__name__}: {e}"
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()