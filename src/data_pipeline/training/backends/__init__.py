"""Phase 5C-A — Training backend registry.

Backend 是 Phase 5C 引入的新概念：把"加载 model + tokenizer / 应用 LoRA / 构建
trainer / 保存 adapter / 重载 + 推理"这一组共享边界从 ``smoke_train_qlora.py``
中抽出来，使 HF 和 Unsloth 两条路径都能走同一组 gate / 选样 / 上层语义。

设计原则：

- 默认 backend 是 ``hf``，与 Phase 5B 完全一致。
- Unsloth backend 必须 **lazy import**；普通环境（未安装 unsloth）不能因为
  ``import data_pipeline.training`` 触发 unsloth 的导入 / 副作用。
- ``resolve_backend(name)`` 返回 backend 实例；非法名称 ⇒ ``TrainingGateError``。
- 缺依赖 ⇒ ``TrainingGateError(UNSLOTH_DEPENDENCY_MISSING)``，消息清晰。
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Backend, BackendMetadata
from .hf import HfBackend
from .unsloth import UnslothBackend

__all__ = [
    "Backend",
    "BackendMetadata",
    "HfBackend",
    "UnslothBackend",
    "DEFAULT_BACKEND",
    "VALID_BACKENDS",
    "resolve_backend",
]


# 默认 backend；与 Phase 5B 保持完全一致。
DEFAULT_BACKEND = "hf"

# 当前合法 backend 列表；增加新 backend 时同步修改。
VALID_BACKENDS = ("hf", "unsloth")


def resolve_backend(name: str) -> Backend:
    """根据名称返回 backend 实例。

    - ``"hf"`` ⇒ :class:`HfBackend`（永远可用，不依赖 unsloth）。
    - ``"unsloth"`` ⇒ :class:`UnslothBackend`（lazy import；缺 unsloth ⇒ 清晰错误）。

    Parameters
    ----------
    name:
        backend 名称；非法 ⇒ :class:`TrainingGateError`。
    """
    if not isinstance(name, str):
        from ..pipeline import TrainingGateError
        raise TrainingGateError(f"backend 名称必须是 str，得到 {type(name).__name__}")

    key = name.strip().lower()
    if key == "hf":
        return HfBackend()
    if key == "unsloth":
        # UnslothBackend 构造时会 lazy import unsloth；缺依赖 ⇒ 清晰错误。
        return UnslothBackend()
    from ..pipeline import TrainingGateError
    raise TrainingGateError(
        f"INVALID_BACKEND: {name!r} 不在合法列表 {VALID_BACKENDS} 中"
    )


def backend_metadata(backend: Backend) -> Dict[str, Any]:
    """返回 backend 的可序列化 metadata（不触发 import）。"""
    meta = backend.metadata()
    return {
        "name": meta.name,
        "description": meta.description,
        "supports_revision_pin": bool(meta.supports_revision_pin),
        "supports_assistant_only_loss": bool(meta.supports_assistant_only_loss),
        "depends_on_unsloth": bool(meta.depends_on_unsloth),
    }