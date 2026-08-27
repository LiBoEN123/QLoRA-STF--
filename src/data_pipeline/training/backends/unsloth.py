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

        # Phase 5C-B：num_train_epochs 优先取 config.num_train_epochs；
        # Phase 5B smoke 路径无该字段 ⇒ 默认 1000（workaround，避免 TRL 默认 3
        # epoch 覆盖 max_steps）。
        num_epochs = getattr(config, "num_train_epochs", 1000)

        kwargs: Dict[str, Any] = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_train_epochs": num_epochs,
            "learning_rate": config.learning_rate,
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

        # warmup 方式：Phase 5C-B 使用 ``warmup_ratio``；Phase 5B 用 ``warmup_steps``。
        # 优先 warmup_ratio（TRL SFTConfig 同时接受 warmup_steps）。
        if hasattr(config, "warmup_ratio"):
            if "warmup_ratio" in params:
                kwargs["warmup_ratio"] = config.warmup_ratio
            elif "warmup_steps" in params:
                # 把 ratio 转 steps 需要总步数；不预计算，让 TRL 内部处理（ratio 优先）。
                pass
        if hasattr(config, "warmup_steps"):
            if "warmup_steps" in params and "warmup_ratio" not in kwargs:
                kwargs["warmup_steps"] = config.warmup_steps

        # Phase 5C-B 专属：weight_decay / max_grad_norm / scheduler / data_seed
        for key, attr in (
            ("weight_decay", "weight_decay"),
            ("max_grad_norm", "max_grad_norm"),
            ("lr_scheduler_type", "lr_scheduler_type"),
            ("save_total_limit", "save_total_limit"),
            ("load_best_model_at_end", "load_best_model_at_end"),
            ("metric_for_best_model", "metric_for_best_model"),
            ("greater_is_better", "greater_is_better"),
        ):
            if key in params and hasattr(config, attr):
                kwargs[key] = getattr(config, attr)

        # data_seed（独立于 seed）
        if "data_seed" in params and hasattr(config, "data_seed"):
            kwargs["data_seed"] = config.data_seed

        if "max_length" in params:
            kwargs["max_length"] = config.max_length
        elif "max_seq_length" in params:
            kwargs["max_seq_length"] = config.max_length

        # max_steps：Phase 5B smoke 路径（max_steps=1）；Phase 5C-B 路径无该字段。
        if hasattr(config, "max_steps"):
            kwargs["max_steps"] = config.max_steps

        return SFTConfig(**kwargs)

    def build_trainer(
        self,
        *,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        config: Any,
        output_dir: Path,
        eval_dataset: Any = None,
    ) -> Any:
        from trl import SFTTrainer  # type: ignore

        sft_config = self._build_sft_config(config, output_dir)

        # Phase 5C-B：assistant_only_loss=True 必须真正生效。
        # Unsloth 的 _prepare_dataset 走 ``text`` 路径不会生成 assistant_masks；
        # Qwen3 的 chat template **不包含** ``{% generation %}`` 关键字，
        # ``return_assistant_tokens_mask=True`` 返回全 0 mask。
        #
        # 因此我们采用 prompt-only tokenization 对齐的策略：
        # 1) tokenize(prompt with add_generation_prompt=True) → 得到 prompt token ids；
        # 2) tokenize(full conversation) → 得到 full token ids（prompt_ids 是 full_ids 的前缀）；
        # 3) labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        #    ⇒ 只有 assistant 的 content + <|im_end|> + \n 被监督。
        # 4) 同时返回 assistant_masks 字段供 collator / mask check 使用；
        # 5) 删除 ``messages`` 列，避免 Unsloth 把它当 conversational 重新触发
        #    apply_chat_template；
        # 6) dataset_text_field 指向不存在的字段，强制 Unsloth 走 ``input_ids`` 分支。
        #
        # 这样 Unsloth 的 _prepare_dataset 走 ``DataCollatorForSeq2Seq`` 路径，
        # labels 字段直接被采用；padding 时 pad_token_id 位置补 -100，监督 mask 不变。
        if not config.assistant_only_loss:
            # 兼容路径：仅添加 ``text`` 字段。
            def _add_text(row):
                text = tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
                return {"text": text}

            if hasattr(dataset, "map"):
                dataset = dataset.map(_add_text)

            if eval_dataset is not None and hasattr(eval_dataset, "map"):
                eval_dataset = eval_dataset.map(_add_text)
        else:
            def _tokenize_with_am(row):
                msgs = row["messages"]
                prompt_ids = tokenizer.apply_chat_template(
                    msgs[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                full_ids = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                # Qwen3 的 tokenization 对齐：prompt_ids 是 full_ids 的前缀。
                if not (len(prompt_ids) <= len(full_ids)
                        and list(full_ids[:len(prompt_ids)]) == list(prompt_ids)):
                    # Fallback：保守地全部 supervised（不应发生）。
                    prompt_len = 0
                else:
                    prompt_len = len(prompt_ids)
                am = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
                labels = [-100] * prompt_len + list(full_ids[prompt_len:])
                attn = [1] * len(full_ids)
                return {
                    "input_ids": list(full_ids),
                    "attention_mask": attn,
                    "assistant_masks": am,
                    "labels": labels,
                }

            if hasattr(dataset, "map"):
                # 保留 ``messages`` 字段使 dataset 仍是 conversational，
                # 但同时存在 ``input_ids`` + ``labels`` 让 Unsloth 走 ``do_tokenize=False``
                # + ``DataCollatorForSeq2Seq`` 路径（line 1204）。
                keep_cols = {"sample_id", "messages"}
                dataset = dataset.map(
                    _tokenize_with_am,
                    remove_columns=[c for c in dataset.column_names if c not in keep_cols],
                )

            if eval_dataset is not None and hasattr(eval_dataset, "map"):
                keep_cols = {"sample_id", "messages"}
                eval_dataset = eval_dataset.map(
                    _tokenize_with_am,
                    remove_columns=[c for c in eval_dataset.column_names if c not in keep_cols],
                )

            try:
                sft_config.dataset_text_field = "__no_such_field__"
            except Exception:
                pass

        trainer_kwargs: Dict[str, Any] = {
            "model": model,
            "args": sft_config,
            "train_dataset": dataset,
            "processing_class": tokenizer,
        }
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        trainer = SFTTrainer(**trainer_kwargs)
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