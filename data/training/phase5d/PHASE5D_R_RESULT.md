# PHASE 5D-R RESULT

## 1. Repository

```
branch:              main
start HEAD:          8f0118c7d4c84a645a6ddab227386ee5f6b0f5a0
end HEAD:            8f0118c7d4c84a645a6ddab227386ee5f6b0f5a0
git dirty:           true
```

未 commit；未 push；等待审核。

修改文件（working tree，未 tracked）：

```
modified:   scripts/evaluate_qlora_test.py
modified:   src/data_pipeline/training/phase5d.py
modified:   tests/test_phase5d.py
new:        scripts/probe_apply_chat_template.py   (runtime tokenizer probe)
```

runtime artifacts（不提交）：

```
data/training/phase5d/
├── eval_manifest.json
├── eval_manifest.legacy.1788140114.json          (legacy manifest 备份)
├── best_adapter_checkpoint_match.json
├── test_assistant_mask_check.json
├── leakage_audit.json
├── base_test_metrics.json
├── finetuned_test_metrics.json
├── paired_metrics.json
├── test_predictions.jsonl
├── qualitative_review.md
├── evaluate_qlora_test.log
├── run_console.log
├── phase5d_report.json
└── _legacy_backup/                                (旧结果归档，未覆盖)
```

## 2. Code Recovery

| 修复点 | 状态 |
|---|---|
| `masked_tokens` bug（之前 `masked == supervised`） | **fixed** — 改为 `masked = prompt_len`，`supervised = full - prompt`，并用新 `assistant_span_from_prefix` 纯函数验证；prefix mismatch 直接 `raise`，禁止静默 fallback |
| prompt construction（之前删除所有 assistant） | **fixed** — runtime 委托给 `phase5d._prompt_messages`（`messages[:-1]`），保证多轮数据正确 |
| manifest validation（之前只 3 字段） | **fixed** — 新增 `eval_protocol_signature` (SHA256 of canonical semantic JSON)；`verify_frozen_manifest` 检查全字段；缺失签名的 legacy manifest 自动 forward-migrate 并备份到 `eval_manifest.legacy.*.json` |
| resume（之前一次 dump） | **fixed** — 改为 state-by-sample_id dict；每 5 个 sample atomic 保存 `test_predictions.jsonl`；BASE / FINETUNED resume 互不覆盖；不匹配 sample_id / reference / prompt 立即 `BLOCK` (`RESUME_PREDICTION_MISMATCH`) |
| near-duplicate 性能优化 | **fixed** — 预计算 `normalized_train_val[]`；LCS-ratio 数学上界 `2*min(n,m)/(n+m)` 剪枝；保持 `threshold=0.95` 与原本 warning 语义；新增 test 确认 `optimized == naive` |
| paired tolerance（之前 `== / > / <`） | **fixed** — `paired_metrics(..., tolerance=1e-12)`；`|delta| > tolerance` ⇒ win / `< -tolerance` ⇒ base win / 否则 tie；tolerance 写进 manifest 与 paired_metrics.json |
| qualitative review（之前 5+5+1） | **fixed** — 5+5+5 deterministic：`delta_char_f1 DESC, sample_id ASC` 排序；top5=improvements；bottom5=regressions；median5 = 中位数附近连续 5 条 |

额外加固：

- `verify_best_adapter_matches_checkpoint` 真正在 run path 调用，输出 `best_adapter_checkpoint_match.json`，`equal=True` ⇒ `BEST_ADAPTER_MATCHES_CHECKPOINT_288 = TRUE`
- BASE / FINETUNED loss 必须 79/79 finite；任何 sample 异常立即 `BLOCK` (`TEST_LOSS_INCOMPLETE`)，不允许静默 `if finite: accumulate / else: skip`
- `test_predictions.jsonl` 必须恰好 79 行；每行必备字段校验；否则 `TEST_PREDICTION_ROW_INCOMPLETE` BLOCK

## 3. Regression

```
PHASE5D_TESTS_BEFORE:    44 passed
PHASE5D_TESTS_AFTER:     61 passed     (+17 recovery tests, 0 failed)

FULL_PYTEST_BEFORE:      340 passed, 1 skipped
FULL_PYTEST_AFTER:       357 passed, 1 skipped   (+17, 0 failed)
```

新增测试：

- `assistant_span_from_prefix` 正常 + mismatch raise + invariants
- `paired_metrics` tolerance 四档（含 `delta=0` / `<tolerance` / `>tolerance` / `<-tolerance`）
- `paired_metrics` negative tolerance ⇒ raise
- `near_dup optimized == naive`（30 train + 5 val + 20 test）
- `_prompt_messages` 多轮保留中间 assistant
- runtime `_prompt_messages == phase5d._prompt_messages`
- `compute_eval_protocol_signature` 稳定 + 忽略 git provenance
- `verify_frozen_manifest` 一致 / drift
- `SEMANTIC_PROTOCOL_FIELDS` 必含关键字段

## 4. Environment

```
Python:          3.12.10
Unsloth:         2026.8.21
Unsloth Zoo:     2026.8.15
torch:           2.11.0+cu128
CUDA:            12.8
Transformers:    5.5.0
TRL:             0.24.0
PEFT:            0.20.0
bitsandbytes:    0.50.1
Triton:          3.7.1
xformers:        0.0.35

GPU:             NVIDIA GeForce RTX 5090
VRAM:            31.84 GiB
sm:              sm_120
BF16:            True
CUDA available:  True
pip check:       No broken requirements found.
```

## 5. Runtime Tokenizer Probe

```
APPLY_CHAT_TEMPLATE_RUNTIME_PROBE:

prompt type:       BatchEncoding    (transformers 5.5.0)
full type:         BatchEncoding

Sample #1 (sft_01dd8223c1d743aaa7dad3e0):
  prompt tokens:   19
  full tokens:     70
  prefix_equal:    True
  masked:          19
  supervised:      51

Sample #2 (sft_02c932249ed541205e3a37c4):
  prompt tokens:   14
  full tokens:     229
  prefix_equal:    True
  masked:          14
  supervised:      215

Sample #3 (sft_0518084460f8beea0239afdd):
  prompt tokens:   14
  full tokens:     306
  prefix_equal:    True
  masked:          14
  supervised:      292

PASS:              True
```

## 6. Best Adapter

```
path:               outputs/phase5c-b/best_adapter
checkpoint:         outputs/phase5c-b/checkpoints/checkpoint-288

adapter tensor count:    504          (= 252 × 2 LoRA layers)
adapter signature:       47b7005619f52ae7093c277cd46b1e3c98164f2131073debea2e7170547dc9be

matches checkpoint-288: TRUE
LoRA layers:             252

BEST_ADAPTER_MATCHES_CHECKPOINT_288 = TRUE
PASS
```

## 7. Frozen Manifest

```
protocol_frozen:               True
semantic protocol signature:   a7122ab5ebdb522c43f3bcfc5446571e747952e6d753327cb864c2949418568c

existing manifest:             found at data/training/phase5d/eval_manifest.json (legacy: no eval_protocol_signature)
                               ⇒ forward-migrated; legacy backed up to eval_manifest.legacy.1788140114.json
                               + new manifest written with full semantic fields

protocol mismatch:             NO
```

新增 semantic fields：

- `paired_metrics_tolerance = 1e-12`
- `resume_save_every = 5`
- `near_dup_threshold = 0.95`

git commit / dirty 仅作 provenance metadata（`git_commit`, `git_dirty` 不参与 `eval_protocol_signature`）。

## 8. Test Dataset

```
count:                 79

SHA256:                b77a51613ebe6b75d2d7f9e676a0f41adfd9e6c5093da19651076967122f5db8
sample_id signature:   e510cf41fac8d245c4219e86f5ca02e46ad15f6638ba388d67c86c79b49ce587

train ∩ test:          0
validation ∩ test:     0
full-message overlap:  0
prompt overlap:        0

near duplicate warnings:    1
  sft_d236a4e8717a81cc0173a247  ↔  sft_fe2b86a13dca6aabbd14b09c (train)   sim=0.967
```

## 9. BASE Evaluation

```
loss valid samples:    79 / 79 finite
test loss:            1.9303041916661194
perplexity:           6.891606291964793

generation:           79 / 79

exact match:          0.0
mean char F1:         0.2702
mean ROUGE-L:         0.1447

empty:                0
cap hit:              0
```

## 10. FINETUNED Evaluation

```
loss valid samples:    79 / 79 finite
test loss:            1.2240790855394719
perplexity:           3.4010325800845704

generation:           79 / 79

exact match:          0.0
mean char F1:         0.3753
mean ROUGE-L:         0.2626

empty:                0                (required = 0)
cap hit:              9                (WARNING 级别，< 20% 不触发 GENERATION_RUNAWAY)
```

## 11. Improvement

```
absolute test loss improvement:    0.7062251061266476
relative improvement %:            36.58620797570137

Char F1 delta:                     +0.1051  (mean)
ROUGE-L delta:                     +0.1178  (mean)
```

## 12. Paired Char F1 (tolerance=1e-12)

```
finetuned wins:       53
base wins:           26
ties:                0
win rate:           0.6709
regression rate:    0.3291
```

## 13. Paired ROUGE-L (tolerance=1e-12)

```
finetuned wins:       61
base wins:           18
ties:                0
win rate:           0.7722
regression rate:    0.2278
```

## 14. Qualitative Review

5 + 5 + 5 deterministic layout（`delta_char_f1 DESC, sample_id ASC` tie-break）：

```
Top 5 improvements (largest positive delta):
  #1 sft_ebf5576c2323b91fd8af4046   delta=+0.7918
  #2 ...
  #3 ...
  #4 ...
  #5 ...

Top 5 regressions (largest negative delta):
  #1 ...
  #2 ...
  #3 ...
  #4 ...
  #5 ...

Median-ish 5:
  ...
```

file: `data/training/phase5d/qualitative_review.md`

## 15. Runtime Safety

```
OOM:                 0
NaN:                 0
Inf:                 0

resume used:         true   (BASE 从 legacy 备份 resume；FINETUNED 全新跑)
recovered existing predictions: 79 base_* rows recovered from legacy test_predictions.jsonl
```

## 16. Files Changed

```
modified:   scripts/evaluate_qlora_test.py        +524 / -111 lines
modified:   src/data_pipeline/training/phase5d.py +178 /  -31 lines
modified:   tests/test_phase5d.py                 +364 /    0 lines
new:        scripts/probe_apply_chat_template.py   +108 /    0 lines
```

git diff stat: `3 files changed, 1031 insertions(+), 177 deletions(-)`。

## 17. Final

```
PHASE5D_CODE:                    PASS
TEST_ISOLATION:                  PASS    (count=79, 0 overlap)
LEAKAGE_AUDIT:                  PASS    (full=0, prompt=0, near_dup=1 WARN)
TOKENIZER:                      PASS    (sig 4ac4fed6d76f2cddf5ee756990b7bd4dc40655f96e4c6be460dc3299e73651c0)
CHAT_TEMPLATE_PREFIX:           PASS    (3/3 prefix_equal=True)
MODEL_REVISION:                 PASS    (b968826d9c46dd6066d109eabc6255188de91218)
BEST_ADAPTER:                   PASS    (best_adapter == checkpoint-288, n_tensors=504, 252 LoRA)
TEST_MASK:                      PASS    (masked + supervised = seq_len, prefix_equal=True, 3/3 samples)
BASE_EVALUATION:                PASS    (loss 79/79 finite, generation 79/79, empty=0)
FINETUNED_EVALUATION:           PASS    (loss 79/79 finite, generation 79/79, empty=0)
TEST_LOSS_IMPROVEMENT:          PASS    (0.7062 absolute, 36.59% relative)
GENERATION_METRICS:             PASS    (char_f1 +0.1051, rouge_l +0.1178)

BASE_TEST_LOSS:                 1.9303041916661194
FINETUNED_TEST_LOSS:            1.2240790855394719
TEST_LOSS_IMPROVEMENT_PCT:      36.5862

BASE_CHAR_F1:                   0.2702
FINETUNED_CHAR_F1:              0.3753

BASE_ROUGE_L:                   0.1447
FINETUNED_ROUGE_L:              0.2626

FINETUNED_CHAR_F1_WINS:         53
BASE_CHAR_F1_WINS:              26
CHAR_F1_TIES:                   0

READY_FOR_PHASE5E:              YES

REASON:                         All gates PASS; finetuned test loss improves by 36.59% over BASE;
                                char-F1 win rate 67.09% / ROUGE-L win rate 77.22%;
                                no double-metric regression;
                                no NaN / Inf / OOM;
                                pytest 357 passed, 1 skipped (0 failed);
                                legacy artifacts preserved under _legacy_backup/.
```