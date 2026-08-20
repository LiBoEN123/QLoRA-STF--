"""Phase 5B — QLoRA 单步训练冒烟测试的配置与数据模型。

只定义 ``QLoRASmokeConfig``：固化当前数据集与硬件所需的训练超参。
所有字段前必须有独立中文注释（与 Phase 4 / 5A 同约定）。
"""

from __future__ import annotations

from dataclasses import dataclass


# 当前数据集真实最大 token 数；Phase 5A token_report 已锁定。
# 该值用于 gate 校验：实际超过 2048 ⇒ 立即阻断，不允许悄悄扩容。
MAX_LENGTH = 2048


# 默认基础模型仓库；与 Phase 5A 默认保持一致，避免换模型导致 token 分布失真。
DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"


# 默认 revision：无法拿到 commit SHA 时使用 main；拿到则 pin 到 SHA。
DEFAULT_REVISION = "main"


# LoRA baseline 配置：固定不变，禁止在 Phase 5B 做超参搜索。
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_BIAS = "none"
DEFAULT_LORA_TASK_TYPE = "CAUSAL_LM"
DEFAULT_LORA_TARGET_MODULES = "all-linear"


# 4-bit / NF4 量化固定配置。
DEFAULT_QUANT_TYPE = "nf4"
DEFAULT_COMPUTE_DTYPE = "bfloat16"
DEFAULT_DOUBLE_QUANT = True


# 训练参数固定配置：1 step、batch=1、accum=1。
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_MAX_STEPS = 1
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WARMUP_STEPS = 0
DEFAULT_LOGGING_STEPS = 1
DEFAULT_SAVE_STRATEGY = "no"
DEFAULT_EVAL_STRATEGY = "no"
DEFAULT_GRADIENT_CHECKPOINTING = True
DEFAULT_PACKING = False
DEFAULT_ASSISTANT_ONLY_LOSS = True
DEFAULT_BF16 = True
DEFAULT_FP16 = False
DEFAULT_REPORT_TO = "none"
DEFAULT_SEED = 42


# smoke 选样：按 sample_id 升序取前 N，再加 1 条最接近 train P95 的样本。
DEFAULT_SMOKE_SHORT_COUNT = 7
# 目标 smoke 数量：short + p95
DEFAULT_SMOKE_TARGET_COUNT = 8

# 最小推理：固定 prompt 字符串，与训练内容无关。
DEFAULT_INFERENCE_PROMPT = "什么是基层党组织？"
DEFAULT_INFERENCE_MAX_NEW_TOKENS = 64


@dataclass(frozen=True)
class QLoRASmokeConfig:
    """QLoRA 单步训练冒烟测试的冻结配置。

    字段全部冻结：相同配置 ⇒ smoke 行为确定性（不含 VRAM / loss 等运行期数据）。
    """

    # 基础模型仓库，与 Phase 5A tokenizer 审计使用模型保持一致
    model_name: str = DEFAULT_MODEL_NAME

    # Hugging Face 模型 revision；优先使用真实 commit SHA，拿不到则退回 main
    revision: str = DEFAULT_REVISION

    # 训练最大 token 长度；与数据集真实最大 token (1672) 匹配，留足 buffer
    max_length: int = MAX_LENGTH

    # 4-bit 量化类型；Phase 5B 固定 NF4，不做 FP4 / INT8 / GPTQ / AWQ 实验
    quant_type: str = DEFAULT_QUANT_TYPE

    # 量化张量的计算 dtype；本阶段固定 bfloat16，不退化为 fp16
    compute_dtype: str = DEFAULT_COMPUTE_DTYPE

    # 是否启用 bitsandbytes double quantization
    double_quant: bool = DEFAULT_DOUBLE_QUANT

    # LoRA 低秩矩阵的 rank
    lora_r: int = DEFAULT_LORA_R

    # LoRA 缩放参数 alpha
    lora_alpha: int = DEFAULT_LORA_ALPHA

    # LoRA 层训练时的 dropout
    lora_dropout: float = DEFAULT_LORA_DROPOUT

    # LoRA bias 处理方式
    lora_bias: str = DEFAULT_LORA_BIAS

    # LoRA 任务类型；固定 CAUSAL_LM
    lora_task_type: str = DEFAULT_LORA_TASK_TYPE

    # LoRA 目标模块；优先 PEFT 的 all-linear 字符串形式
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES

    # 每设备训练 batch size；smoke 阶段固定 1，避免显存失控
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE

    # 梯度累积步数；smoke 阶段固定 1
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS

    # 总训练步数；smoke 阶段固定 1
    max_steps: int = DEFAULT_MAX_STEPS

    # 学习率；仅 smoke baseline，不做超参搜索
    learning_rate: float = DEFAULT_LEARNING_RATE

    # warmup 步数
    warmup_steps: int = DEFAULT_WARMUP_STEPS

    # 日志步数
    logging_steps: int = DEFAULT_LOGGING_STEPS

    # checkpoint 保存策略；smoke 阶段固定 no，避免多余输出
    save_strategy: str = DEFAULT_SAVE_STRATEGY

    # 评估策略；smoke 阶段固定 no
    eval_strategy: str = DEFAULT_EVAL_STRATEGY

    # 是否启用梯度检查点；smoke 阶段固定 True，缓解显存
    gradient_checkpointing: bool = DEFAULT_GRADIENT_CHECKPOINTING

    # 是否开启 packing；Phase 5B 禁止，避免与 assistant_only_loss 交互复杂化
    packing: bool = DEFAULT_PACKING

    # TRL SFTTrainer 的 assistant_only_loss；本阶段必须 True
    assistant_only_loss: bool = DEFAULT_ASSISTANT_ONLY_LOSS

    # 是否启用 bf16 训练；本阶段必须 True
    bf16: bool = DEFAULT_BF16

    # 是否启用 fp16 训练；本阶段必须 False
    fp16: bool = DEFAULT_FP16

    # Trainer 上报目标；本阶段固定 none，避免 wandb / tensorboard 副作用
    report_to: str = DEFAULT_REPORT_TO

    # 随机种子；保证同一 config 跑两次行为可复现（数值部分）
    seed: int = DEFAULT_SEED

    # smoke 短样本数量（按 sample_id 升序）
    smoke_short_count: int = DEFAULT_SMOKE_SHORT_COUNT

    # smoke 目标总条数（short + 1 条 p95 样本）
    smoke_target_count: int = DEFAULT_SMOKE_TARGET_COUNT

    # smoke 最小推理使用的固定 prompt
    inference_prompt: str = DEFAULT_INFERENCE_PROMPT

    # smoke 推理允许的最大生成 token 数
    inference_max_new_tokens: int = DEFAULT_INFERENCE_MAX_NEW_TOKENS

    def __post_init__(self) -> None:
        """冻结配置的基本不变量校验。"""
        if self.max_length != MAX_LENGTH:
            raise ValueError(f"max_length 必须固定为 {MAX_LENGTH}，得到 {self.max_length}")
        if self.quant_type != "nf4":
            raise ValueError(f"quant_type 必须固定为 'nf4'，得到 {self.quant_type!r}")
        if self.compute_dtype != "bfloat16":
            raise ValueError(f"compute_dtype 必须固定为 'bfloat16'，得到 {self.compute_dtype!r}")
        if not self.double_quant:
            raise ValueError("double_quant 必须为 True（Phase 5B 固定开启 double quant）")
        if self.bf16 is not True:
            raise ValueError("bf16 必须为 True（Phase 5B 禁止退回 fp16）")
        if self.fp16 is not False:
            raise ValueError("fp16 必须为 False（Phase 5B 禁止启用 fp16）")
        if self.max_steps != 1:
            raise ValueError(f"max_steps 必须固定为 1，得到 {self.max_steps}")
        if self.packing:
            raise ValueError("packing 必须为 False（Phase 5B 禁止开启）")
        if not self.assistant_only_loss:
            raise ValueError("assistant_only_loss 必须为 True（Phase 5B 禁止退回）")
        if self.per_device_train_batch_size != 1:
            raise ValueError(
                f"per_device_train_batch_size 必须为 1，得到 {self.per_device_train_batch_size}"
            )
        if self.gradient_accumulation_steps != 1:
            raise ValueError(
                f"gradient_accumulation_steps 必须为 1，得到 {self.gradient_accumulation_steps}"
            )
        if self.lora_r != DEFAULT_LORA_R:
            raise ValueError(f"lora_r 必须固定为 {DEFAULT_LORA_R}，得到 {self.lora_r}")
        if self.lora_alpha != DEFAULT_LORA_ALPHA:
            raise ValueError(f"lora_alpha 必须固定为 {DEFAULT_LORA_ALPHA}，得到 {self.lora_alpha}")
        if self.lora_dropout != DEFAULT_LORA_DROPOUT:
            raise ValueError(f"lora_dropout 必须固定为 {DEFAULT_LORA_DROPOUT}，得到 {self.lora_dropout}")
        if self.lora_target_modules != "all-linear":
            raise ValueError(
                f"lora_target_modules 必须为 'all-linear'，得到 {self.lora_target_modules!r}"
            )
        if self.inference_max_new_tokens <= 0:
            raise ValueError(
                f"inference_max_new_tokens 必须 > 0，得到 {self.inference_max_new_tokens}"
            )