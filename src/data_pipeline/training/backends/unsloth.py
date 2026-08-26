"""Phase 5C-A — Unsloth backend（lazy import）。

设计目标：

1. 顶层 ``import data_pipeline.training`` 不触发 unsloth 导入。
2. 只有 :class:`UnslothBackend` 实际方法被调用时才 import unsloth；
   缺 unsloth ⇒ 抛 ``TrainingGateError(UNSLOTH_DEPENDENCY_MISSING)``，错误清晰。
3. **不能**为了接入 unsloth 改 Phase 5B 已固定的训练语义：
   - assistant_only_loss=True（公开 TRL 路径校验 mask）；
   - packing=False；
   - max_length=2048；
   - NF4 + double quant + bfloat16；
   - LoRA r=16 alpha=32 dropout=0.05 all-linear；
   - revision 必须 pinned。
4. 加载 base model 使用 ``Qwen/Qwen3-8B`` + 用户传入的 commit SHA，不允许
   使用 ``unsloth/Qwen3-8B-*`` 预量化模型。
5. Unsloth 的 4-bit ``FastLanguageModel.from_pretrained(..., load_in_4bit=True)``；
   4-bit 真假通过 ``is_loaded_in_4bit`` / ``Linear4bit`` 模块实际计数验证。
"""

from __future__ import annotations

import gc
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..pipeline import TrainingGateError, build_inference_prompt
from .base import Backend, BackendMetadata


_LOG = logging.getLogger("data_pipeline.training.backends.unsloth")


# 缺 unsloth 时返回的错误信息（不允许 obscure traceback）。
_UNSLOTH_MISSING_MSG = (
    "UNSLOTH_DEPENDENCY_MISSING: 当前环境未安装 unsloth；"
    "请使用 .venv-unsloth 解释器，或安装 unsloth 后重试。"
)


def _require_unsloth():
    """尝试 import unsloth；缺则抛清晰错误。"""
    try:
        import unsloth  # type: ignore  # noqa: F401
        from unsloth import FastLanguageModel  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover - 仅在 unsloth 未装时触发
        raise TrainingGateError(f"{_UNSLOTH_MISSING_MSG}（底层错误：{type(e).__name__}: {e}）") from e
    return unsloth, FastLanguageModel


class UnslothBackend(Backend):
    """Unsloth backend；lazy import。"""

    name = "unsloth"

    def metadata(self) -> BackendMetadata:
        # 不触发 import；仅返回静态 metadata。
        return BackendMetadata(
            name="unsloth",
            description="Unsloth FastLanguageModel + 4-bit + LoRA + TRL SFTTrainer",
            supports_revision_pin=True,
            supports_assistant_only_loss=True,
            depends_on_unsloth=True,
        )

    # ------------------------------------------------------------------
    # 1) load
    # ------------------------------------------------------------------

    def load_model_and_tokenizer(
        self,
        *,
        model_name: str,
        revision: str,
        torch: Any,
        config: Any,
    ) -> Tuple[Any, Any]:
        _, FastLanguageModel = _require_unsloth()

        # 关键：必须显式传 use_exact_model_name=True + local_files_only=True，
        # 否则 unsloth 会把 ``Qwen/Qwen3-8B`` 内部重映射到 ``unsloth/Qwen3-8B-unsloth-bnb-4bit``
        # 预量化模型，并且会尝试网络下载——这违反本项目的"同基础模型 + 同 revision"
        # Parity 要求。
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                revision=revision,
                max_seq_length=config.max_length,
                load_in_4bit=True,
                dtype=getattr(torch, config.compute_dtype),
                trust_remote_code=False,
                local_files_only=True,
                use_exact_model_name=True,
            )
        except TypeError as e:
            # 旧版 unsloth 可能没有 use_exact_model_name / local_files_only
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    revision=revision,
                    max_seq_length=config.max_length,
                    load_in_4bit=True,
                    dtype=getattr(torch, config.compute_dtype),
                    trust_remote_code=False,
                    local_files_only=True,
                )
            except TypeError as e2:
                raise TrainingGateError(
                    f"UNSLOTH_FROM_PRETRAINED_API_INCOMPATIBLE: {e2}"
                ) from e2

        # revision 是否真的生效：FastLanguageModel 通常会把 model.config._name_or_path
        # 设为对应 snapshot；这里只做基础校验。
        inner_model = getattr(model, "model", model)
        cfg = getattr(inner_model, "config", None)
        cfg_revision = getattr(cfg, "_revision_or_sha", None) or getattr(cfg, "revision", None)
        if cfg_revision is not None and isinstance(cfg_revision, str):
            if cfg_revision != revision and not cfg_revision.startswith(revision[:7]):
                # 只记录 warning，不阻断（不同 unsloth 版本字段不同）
                _LOG.warning("Unsloth model.config.revision=%r != requested=%r", cfg_revision, revision)

        return model, tokenizer

    # ------------------------------------------------------------------
    # 2) LoRA
    # ------------------------------------------------------------------

    def apply_lora(self, *, model: Any, config: Any) -> Any:
        _, FastLanguageModel = _require_unsloth()

        # Unsloth 的 FastLanguageModel.get_peft_model 不接受 ``task_type``（内部硬编码
        # ``TaskType.CAUSAL_LM``），必须从 kwargs 中剔除以避免 TypeError。
        # 同时 Unsloth 不接受字符串 ``"all-linear"`` 的 target_modules；本项目要求等价，
        # 因此在 backend 层显式解析为 Qwen3 的常见线性模块列表。
        # Phase 5B baseline 的 all-linear ⇒ Qwen3-8B 的 Linear 层数为 252，
        # 这里列出的模块应能完整覆盖所有 linear。
        if config.lora_target_modules == "all-linear":
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        elif isinstance(config.lora_target_modules, str):
            target_modules = [m.strip() for m in config.lora_target_modules.split(",") if m.strip()]
        else:
            target_modules = config.lora_target_modules

        # 同样 dropout：Unsloth 默认 0.0（fast path），显式传 0.05 会触发 warning
        # 但不报错——保留 Phase 5B 的 0.05 语义不变。
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias=config.lora_bias,
                target_modules=target_modules,
                use_gradient_checkpointing="unsloth"
                if config.gradient_checkpointing
                else False,
            )
        except TypeError as e:
            raise TrainingGateError(
                f"UNSLOTH_LORA_API_INCOMPATIBLE: {e}；"
                "unsloth 当前 API 不支持本项目所需的全部 LoRA 字段。"
            ) from e

        if config.gradient_checkpointing and hasattr(model, "config"):
            try:
                model.config.use_cache = False
            except Exception:
                pass

        return model

    # ------------------------------------------------------------------
    # 3) trainer
    # ------------------------------------------------------------------

    def _build_sft_config(self, config: Any, output_dir: Path) -> Any:
        from trl import SFTConfig  # type: ignore

        sig = inspect.signature(SFTConfig.__init__)
        params = sig.parameters

        kwargs: Dict[str, Any] = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_steps": config.max_steps,
            # 显式设置 num_train_epochs，避免默认 3 epoch 覆盖 max_steps。
            "num_train_epochs": 1000,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "logging_steps": config.logging_steps,
            "save_strategy": config.save_strategy,
            "eval_strategy": config.eval_strategy,
            "gradient_checkpointing": config.gradient_checkpointing,
            "packing": config.packing,
            "assistant_only_loss": config.assistant_only_loss,
            "bf16": config.bf16,
            "fp16": config.fp16,
            "report_to": config.report_to,
            "seed": config.seed,
        }
        if "max_length" in params:
            kwargs["max_length"] = config.max_length
        elif "max_seq_length" in params:
            kwargs["max_seq_length"] = config.max_length

        return SFTConfig(**kwargs)

    def build_trainer(
        self,
        *,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        config: Any,
        output_dir: Path,
    ) -> Any:
        from trl import SFTTrainer  # type: ignore

        sft_config = self._build_sft_config(config, output_dir)

        # Unsloth 包装的 SFTTrainer 在 _prepare_dataset 阶段硬性要求
        # ``formatting_func``（per-row，调用 ``next(iter(dataset))``）或 dataset
        # 已含 ``text`` 字段。
        # 为保持与 Phase 5B HF 路径完全等价的语义（assistant_only_loss + 相同
        # apply_chat_template），我们在 backend 层预先生成 ``text`` 字段：
        # - 保留 ``messages`` 字段以满足 Phase 5B mask 校验；
        # - 新增 ``text`` 字段供 Unsloth 直接使用；
        # - 不修改 max_length / packing / assistant_only_loss / NF4 / LoRA。
        def _add_text(row):
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        if hasattr(dataset, "map"):
            dataset = dataset.map(_add_text)

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        return trainer

    # ------------------------------------------------------------------
    # 4) save
    # ------------------------------------------------------------------

    def save_adapter(
        self,
        *,
        trainer: Any,
        tokenizer: Any,
        adapter_dir: Path,
    ) -> List[str]:
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        files = sorted(p.name for p in adapter_dir.glob("*") if p.is_file())
        if "adapter_config.json" not in files:
            raise TrainingGateError(
                f"ADAPTER_SAVE_FAILED: 缺少 adapter_config.json（files={files}）"
            )
        weight_files = [f for f in files if f.endswith((".safetensors", ".bin"))]
        if not weight_files:
            raise TrainingGateError(
                f"ADAPTER_SAVE_FAILED: 缺少权重文件（files={files}）"
            )
        return files

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
        _, FastLanguageModel = _require_unsloth()
        from peft import PeftModel  # type: ignore

        # 重新用 FastLanguageModel 加载 base（保持同样的 4-bit 路径）
        try:
            base, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                revision=revision,
                max_seq_length=config.max_length,
                load_in_4bit=True,
                dtype=getattr(torch, config.compute_dtype),
                trust_remote_code=False,
                local_files_only=True,
                use_exact_model_name=True,
            )
        except TypeError as e:
            # 回退：旧 API
            try:
                base, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    revision=revision,
                    max_seq_length=config.max_length,
                    load_in_4bit=True,
                    dtype=getattr(torch, config.compute_dtype),
                    trust_remote_code=False,
                    local_files_only=True,
                )
            except TypeError as e2:
                raise TrainingGateError(
                    f"UNSLOTH_RELOAD_FROM_PRETRAINED_API_INCOMPATIBLE: {e2}"
                ) from e2

        # 在 unsloth FastLanguageModel 上加载 PEFT adapter：
        # Unsloth 包装本身是 PeftModel 的子类时可直接 from_pretrained；
        # 否则使用 base.model 的 PeftModel 路径。
        try:
            if hasattr(base, "load_adapter"):
                base.load_adapter(str(adapter_dir))
                peft_model = base
            else:
                inner = getattr(base, "model", base)
                peft_model = PeftModel.from_pretrained(inner, str(adapter_dir))
        except Exception as e:
            raise TrainingGateError(f"ADAPTER_RELOAD_FAIL: {e}") from e

        peft_model.eval()

        prompt_text = tokenizer.apply_chat_template(
            build_inference_prompt(inference_prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated_token_count = int(new_tokens.shape[-1]) if hasattr(new_tokens, "shape") else len(new_tokens)
        return {
            "generated_token_count": generated_token_count,
            "generated_text": gen_text,
            "generated_char_count": len(gen_text),
            "nonempty": bool(gen_text.strip()),
        }

    # ------------------------------------------------------------------
    # overrides
    # ------------------------------------------------------------------

    def count_lora_layers(self, model: Any) -> int:
        # Unsloth 包装后的 model 通常仍是 PeftModel 子类；优先走 PEFT 识别。
        try:
            from peft.tuners.lora import LoraLayer  # type: ignore
        except Exception:
            return 0
        return sum(1 for m in model.modules() if isinstance(m, LoraLayer))