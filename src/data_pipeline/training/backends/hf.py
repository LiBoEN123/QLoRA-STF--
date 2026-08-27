"""Phase 5C-A — HuggingFace backend（Phase 5B baseline 提取）。

只承载 Phase 5B 已验证的 4-bit + LoRA + TRL SFTTrainer 路径；不引入
新行为、新参数或新分支。
"""

from __future__ import annotations

import gc
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..pipeline import TrainingGateError, build_inference_prompt
from .base import Backend, BackendMetadata


class HfBackend(Backend):
    """Phase 5B baseline backend：Transformers + bitsandbytes + PEFT + TRL。"""

    name = "hf"

    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="hf",
            description="Transformers + bitsandbytes + PEFT + TRL SFTTrainer (Phase 5B baseline)",
            supports_revision_pin=True,
            supports_assistant_only_loss=True,
            depends_on_unsloth=False,
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

        compute_dtype = getattr(torch, config.compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.double_quant,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, use_fast=True, trust_remote_code=False
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                quantization_config=bnb_config,
                dtype=compute_dtype,
                device_map="auto",
                trust_remote_code=False,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                quantization_config=bnb_config,
                torch_dtype=compute_dtype,
                device_map="auto",
                trust_remote_code=False,
            )
        return model, tokenizer

    # ------------------------------------------------------------------
    # 2) LoRA
    # ------------------------------------------------------------------

    def apply_lora(self, *, model: Any, config: Any) -> Any:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore

        try:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        except TypeError:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type=config.lora_task_type,
            target_modules=config.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)

        if config.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
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

        if hasattr(config, "warmup_ratio"):
            if "warmup_ratio" in params:
                kwargs["warmup_ratio"] = config.warmup_ratio
        if hasattr(config, "warmup_steps"):
            if "warmup_steps" in params and "warmup_ratio" not in kwargs:
                kwargs["warmup_steps"] = config.warmup_steps

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

        if "data_seed" in params and hasattr(config, "data_seed"):
            kwargs["data_seed"] = config.data_seed

        if "max_length" in params:
            kwargs["max_length"] = config.max_length
        elif "max_seq_length" in params:
            kwargs["max_seq_length"] = config.max_length

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
        from peft import PeftModel  # type: ignore

        base, tokenizer = self.load_model_and_tokenizer(
            model_name=model_name,
            revision=revision,
            torch=torch,
            config=config,
        )
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
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
        try:
            from peft.tuners.lora import LoraLayer  # type: ignore
        except Exception:
            return 0
        return sum(1 for m in model.modules() if isinstance(m, LoraLayer))