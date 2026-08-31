# Qwen3-8B + QLoRA Adapter — 基层党组织 FAQ

## 模型名称
Qwen3-8B + QLoRA Adapter（Phase 5C-B / checkpoint-288 / epoch 2）

## 模型类型
- 基座：Qwen3-8B（Causal LM）
- Adapter：PEFT / LoRA，r=16 / alpha=32 / dropout=0.05 / bias=none
- 量化：4-bit NF4 + double quant；计算 dtype = BF16
- LoRA 目标模块：all-linear（即 ``gate_proj / k_proj / o_proj /
  up_proj / q_proj / v_proj / down_proj``），共 252 层 LoRA 层，
  504 个 adapter tensor
- 训练可训练参数：43,646,976（占 base 8B 总参数约 0.9167%）

## Base Model
- model_name: `Qwen/Qwen3-8B`
- revision: `b968826d9c46dd6066d109eabc6255188de91218`（**必须 pin，
  禁止使用动态 main / latest**）
- tokenizer_signature:
  `4ac4fed6d76f2cddf5ee756990b7bd4dc40655f96e4c6be460dc3299e73651c0`

## 训练方法
- backend: Unsloth FastLanguageModel + TRL SFTTrainer
- 数据集：train 1,149 / validation 75 / **held-out test 79** 条
  基层党组织 / 党务工作 FAQ（来源《基层党务（试用版）1》）；
  test 集已与 train/validation 完全隔离（sample_id / prompt / full
  messages 均无重叠）。
- 训练：3 epoch，global_step=432，per_device_batch=8，
  gradient_accumulation=1（effective batch = 8），learning_rate=1e-4，
  linear schedule，warmup_ratio=0.03，weight_decay=0.0，max_grad_norm=1.0
- assistant_only_loss=True；packing=False；bf16=True；fp16=False
- 评估选 best：epoch 2 / checkpoint-288，best eval_loss=1.1594
  （before_training=2.1087，epoch1=1.1644，epoch2=1.1594，epoch3=1.1902）
- pre-train eval_loss=2.1087 ⇒ best eval_loss=1.1594 ⇒ validation
  改善约 45.0%

## 数据集
| split       | count | file_sha256 (前 16) |
|-------------|-------|---------------------|
| train       | 1149  | b59927afe162bc9e…   |
| validation  | 75    | 61d29e321f4c8259…   |
| test        | 79    | b77a51613ebe6b75…   |

## 训练配置
详见 ``training_summary.json``；本卡片仅列核心字段。

## Validation Results
| epoch | eval_loss |
|-------|-----------|
| 0 (pre-train) | 2.1087 |
| 1 | 1.1644 |
| **2 (best)** | **1.1594** |
| 3 | 1.1902 |

best epoch = 2，best checkpoint = checkpoint-288；
``best_adapter == checkpoint-288``（tensor-level 数值一致）。

## Held-out Test Results
held-out test 79 条；greedy / enable_thinking=False / max_new_tokens=1610

| metric          | base        | finetuned   |
|-----------------|-------------|-------------|
| test_loss       | 1.9303      | **1.2241**  |
| test_perplexity | 6.8916      | **3.4010**  |
| mean_char_f1    | 0.2702      | **0.3753**  |
| mean_rouge_l    | 0.1447      | **0.2626**  |
| empty_generation_rate | 0.0   | 0.0         |
| max_token_hit_rate    | 0.0   | 0.1139      |
| paired char F1 wins    | 26    | **53** (ties=0) |
| paired ROUGE-L wins    | 18    | **61** (ties=0) |

test_loss 改善 36.59%。Phase 5D gate：
``status=PASS / code=PASS / ready_for_phase5e=true``。

## Recommended Usage
- **不要**把 ``adapter_model.safetensors`` 单独当作当完整 8B 模型加载；
  必须搭配 pinned revision 的 ``Qwen/Qwen3-8B`` base + 4-bit NF4 量化。
- generation 推荐参数：``enable_thinking=False``、``do_sample=False``；
  ``max_new_tokens`` 推荐 256-1024。
- chat template：使用 base 模型原配的 Qwen3 chat template；
  训练时使用 ``apply_chat_template(messages, tokenize=False,
  add_generation_prompt=True, enable_thinking=False)``。

## Loading Example
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER_DIR = "outputs/phase5e/release/adapter"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    revision=BASE_REVISION,
    quantization_config=bnb,
    dtype=torch.bfloat16,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

prompt = "什么是基层党组织？"
chat = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tok(chat, return_tensors="pt").to(model.device)
with torch.no_grad():
    gen = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
print(tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Limitations
- 该 Adapter **仅在《基层党务（试用版）1》文档范围内训练**；
  对未在训练集出现的领域（其他党务文档、最新政策、其他机关党务
  知识）会回退到 base model 默认行为，**不应被理解为对任何党务问题
  的权威回答**。
- ``mean_char_f1 = 0.3753`` / ``mean_rouge_l = 0.2626`` 表明与 reference
  字面重叠仍有显著差距；本 Adapter 的实际目标是改善生成的可读性与
  字段对齐，而非与 reference 完全一致。
- 0/79 paired char F1 ties / 53 wins vs 26 losses（基座仍有约 32.9%
  paired regression）说明存在 base model 局部更优的场景，使用前请
  按实际业务抽样验证。
- 本 Adapter **不包含** base 8B 完整权重；任何 merged 8B / GGUF / GPTQ /
  AWQ 等部署形态需要独立设计部署阶段。

## Reproducibility
完整复现需要：

1. base model pinned revision: `b968826d9c46dd6066d109eabc6255188de91218`
2. dataset `data/sft/{train,validation,test}.jsonl`（SHA256 见上表）
3. Unsloth backend + 同等训练超参（详见 ``training_summary.json``）
4. Phase 5D frozen protocol：``protocol_frozen=true`` + ``eval_manifest.json``
5. 校验流程：
   - ``verify_best_adapter`` 检查 ``best_adapter == checkpoint-288``
   - ``test_predictions.jsonl`` 必须 79 行且每行字段齐全
   - ``checksums.sha256`` 必须 0 missing / 0 mismatch

## Artifact Integrity
本 release 目录的 SHA256 校验和文件见
``outputs/phase5e/release/checksums.sha256``。该文件**自身不参与
checksums 计算**（避免递归），但其内容列出 release 中所有其他稳定
文件的 SHA256。

## License / Base Model Notice
- Base Model **Qwen3-8B** 受 Qwen 团队发布条款约束；下游使用请遵守
  `Qwen/Qwen3-8B` 仓库内的 LICENSE。
- 本 Adapter 由本次 QLoRA 微调项目产生，仅用于研究 / 内部验证 /
  党务工作辅助写作，不可被理解为对任何具体党务问题提供权威法律 /
  政策建议；任何政策落地请以官方原文为准。
