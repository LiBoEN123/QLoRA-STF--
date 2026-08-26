"""Phase 5C-A — Training backend 抽象接口。

backend 模块只负责 5 个边界：

1. ``load_model_and_tokenizer`` —— 4-bit 加载 base + tokenizer；返回 (model, tokenizer)。
2. ``apply_lora`` —— 在已加载 model 上注入 LoRA。
3. ``build_trainer`` —— 用 TRL SFTTrainer 封装；返回 trainer 实例。
4. ``save_adapter`` —— 保存 LoRA adapter + tokenizer。
5. ``reload_and_infer`` —— 释放旧 model → 重新 4-bit load → PeftModel.from_pretrained
   → 最小推理。

不负责：

- gate 校验（由 ``pipeline.py`` 处理）。
- 样本选择（由 ``pipeline.py`` 处理）。
- tokenizer signature（由 ``benchmarks`` 模块处理）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class BackendMetadata:
    """backend 静态元数据（不触发 import）。"""

    # backend 名称："hf" / "unsloth"
    name: str

    # 人类可读描述
    description: str

    # 是否支持精确 model revision pin
    supports_revision_pin: bool

    # 是否支持 TRL assistant_only_loss 路径
    supports_assistant_only_loss: bool

    # 是否依赖 unsloth（hf 永远 False）
    depends_on_unsloth: bool


class Backend:
    """Training backend 抽象接口。"""

    name: str = "abstract"

    def metadata(self) -> BackendMetadata:
        """返回 backend 静态元数据；不得触发重型 import。"""
        return BackendMetadata(
            name=self.name,
            description="abstract backend",
            supports_revision_pin=True,
            supports_assistant_only_loss=True,
            depends_on_unsloth=False,
        )

    # ------------------------------------------------------------------
    # 1) model + tokenizer
    # ------------------------------------------------------------------

    def load_model_and_tokenizer(
        self,
        *,
        model_name: str,
        revision: str,
        torch: Any,
        config: Any,
    ) -> Tuple[Any, Any]:
        """加载 4-bit base model + tokenizer。

        Parameters
        ----------
        model_name:
            仓库名，例如 ``"Qwen/Qwen3-8B"``。
        revision:
            必须 pin 的 commit SHA；backend 必须使用该 SHA，不能退回 ``main``。
        torch:
            已 import 的 ``torch``。
        config:
            :class:`QLoRASmokeConfig`，含 ``quant_type`` / ``compute_dtype`` /
            ``double_quant``。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 2) LoRA
    # ------------------------------------------------------------------

    def apply_lora(
        self,
        *,
        model: Any,
        config: Any,
    ) -> Any:
        """在已加载 model 上注入 LoRA，返回新 model（通常是 PeftModel 或 Unsloth 包装）。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 3) trainer
    # ------------------------------------------------------------------

    def build_trainer(
        self,
        *,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        config: Any,
        output_dir: Path,
    ) -> Any:
        """构造 TRL SFTTrainer（或等价物）；必须支持 ``trainer.train_dataset`` +
        ``trainer.data_collator`` 的公开 mask 校验。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 4) adapter save
    # ------------------------------------------------------------------

    def save_adapter(
        self,
        *,
        trainer: Any,
        tokenizer: Any,
        adapter_dir: Path,
    ) -> List[str]:
        """保存 LoRA adapter + tokenizer；返回 adapter 目录下的文件列表。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 5) reload + infer
    # ------------------------------------------------------------------

    def reload_and_infer(
        self,
        *,
        torch: Any,
        model_name: str,
        revision: str,
        config: Any,
        adapter_dir: Path,
        inference_prompt: str,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        """释放旧 model → 重新加载 base → 加载 adapter → 最小推理。

        返回 ``{"generated_token_count": int, "generated_text": str,
        "generated_char_count": int, "nonempty": bool}``。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def count_trainable_params(self, model: Any) -> Tuple[int, int]:
        """返回 ``(trainable_params, total_params)``。"""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return int(trainable), int(total)

    def count_lora_layers(self, model: Any) -> int:
        """返回 LoRA 层数。HF backend 通过 ``peft.tuners.lora.LoraLayer`` 识别；
        Unsloth backend 走对应包装识别。
        """
        try:
            from peft.tuners.lora import LoraLayer  # type: ignore
        except Exception:
            return 0
        return sum(1 for m in model.modules() if isinstance(m, LoraLayer))

    def is_loaded_in_4bit(self, model: Any) -> bool:
        """检查 model 是否真 4-bit 加载。"""
        if hasattr(model, "is_loaded_in_4bit"):
            try:
                return bool(model.is_loaded_in_4bit)
            except Exception:
                pass
        qc = getattr(model, "quantization_method", None) or getattr(model, "quantization_config", None)
        if qc is not None and hasattr(qc, "get"):
            try:
                return bool(qc.get("load_in_4bit")) or bool(qc.get("bits") == 4)
            except Exception:
                return False
        # Unsloth 也可能暴露 model.model.is_loaded_in_4bit（base）
        inner = getattr(model, "model", None)
        if inner is not None and inner is not model:
            return self.is_loaded_in_4bit(inner)
        return False

    def count_linear4bit(self, model: Any) -> int:
        """统计 ``Linear4bit`` 模块数量。"""
        try:
            from bitsandbytes.nn import Linear4bit  # type: ignore
        except Exception:
            inner = getattr(model, "model", None)
            if inner is not None and inner is not model:
                return self.count_linear4bit(inner)
            return 0
        return sum(1 for m in model.modules() if isinstance(m, Linear4bit))

    def resolve_revision(self, model_name: str, requested: str) -> Tuple[str, Any, bool]:
        """尝试解析 model revision 的 commit SHA。

        返回 ``(requested, resolved_sha_or_None, pinned)``。
        """
        try:
            from huggingface_hub import HfApi  # type: ignore
        except Exception:
            return requested, None, False
        try:
            api = HfApi()
            info = api.model_info(model_name, revision=requested)
            sha = getattr(info, "sha", None)
            if isinstance(sha, str) and sha:
                return requested, sha, True
        except Exception:
            pass
        return requested, None, False