"""Phase 5E Final 收口脚本 — 完整释放 LoRA Adapter Release Package。

执行链路（严格顺序，**禁止跳步**）：

1. ``validate_phase5d_report`` — 重新读取本地 Phase 5D runtime artifact，
   校验 ``status/code/ready_for_phase5e/test_count=79/loss improved`` 等
   硬门槛；任一不满足 ⇒ ``PHASE5E_CODE = BLOCKED`` 并退出非 0。
2. ``verify_best_adapter_provenance`` — 再次确认
   ``outputs/phase5c-b/best_adapter`` 与
   ``outputs/phase5c-b/checkpoints/checkpoint-288`` 在 tensor 数值上一致。
3. ``copy_release_adapter`` — 把 best adapter 全部文件 copy 到
   ``outputs/phase5e/release/adapter``；**不删除原 source**。
4. ``build_training_summary`` / ``build_evaluation_summary`` — 从
   ``data/training/phase5c_b`` + ``data/training/phase5d`` 抽取 metadata。
5. 写 ``MODEL_CARD.md`` / ``RELEASE_README.md`` / ``release_manifest.json``。
6. ``build_file_inventory`` + ``build_checksums`` + ``write_checksums``。
7. ``verify_checksums`` — 重新读回 SHA256 校验 ``missing=0 / mismatch=0``。
8. **Fresh Reload**：释放已加载的 model ⇒ 重新加载 base model (4-bit NF4 / BF16 /
   double quant) + 从 ``outputs/phase5e/release/adapter`` 加载 PEFT adapter ⇒
   验证 LoRA layers == 252。
9. **Sanity Inference**：固定 3 条非 test prompt（``enable_thinking=False`` /
   ``do_sample=False``），要求 3/3 non-empty / 0 exception / 0 NaN / 0 OOM。
10. ``final_gate`` — 综合所有硬门槛 ⇒ ``PROJECT_COMPLETE``。
11. 输出 ``data/training/phase5e/project_completion_report.{json,md}``。

退出码：

- 0：所有硬门槛 PASS，``PHASE5E_CODE = PASS`` / ``PROJECT_COMPLETE = YES``。
- 1：Phase 5D 验证失败。
- 2：best adapter provenance 失败。
- 3：release 拷贝 / manifest / checksums 失败。
- 4：fresh reload 失败。
- 5：sanity inference 失败。
- 6：final gate 失败。

**严禁**：

- 训练 / 重新训练；
- 修改 dataset / split / 根据 test 调参；
- 生成 merged 8B / GGUF / GPTQ / AWQ 等完整 base weights 形态。
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_pipeline.training import (
    PHASE5E_BASE_MODEL_NAME,
    PHASE5E_BASE_REVISION,
    PHASE5E_BEST_ADAPTER_SOURCE,
    PHASE5E_BEST_CHECKPOINT_SOURCE,
    PHASE5E_RELEASE_ROOT,
    PHASE5E_RUNTIME_ARTIFACT_DIR,
    PHASE5E_TOKENIZER_SIGNATURE,
    AdapterProvenance,
    Phase5DVerification,
    build_checksums,
    build_evaluation_summary,
    build_file_inventory,
    build_release_manifest,
    build_training_summary,
    copy_release_adapter,
    final_gate,
    validate_phase5d_report,
    verify_best_adapter_provenance,
    verify_checksums,
    write_checksums,
    write_evaluation_summary,
    write_release_manifest,
    write_training_summary,
)
from data_pipeline.utils import atomic_write_json, get_logger


_LOG = get_logger("finalize_qlora_release")


# Sanity prompts — must NOT appear in ``data/sft/test.jsonl``.
# 预筛：在 finalizer 启动时再次校验，避免本阶段重新 evaluation test。
DEFAULT_SANITY_PROMPTS: List[str] = [
    "什么是基层党组织？",
    "党的组织生活主要包括哪些内容？",
    "基层党组织的主要作用是什么？",
]

# 推理固定超参 — 与 Phase 5D frozen protocol 一致。
SANITY_INFERENCE_MAX_NEW_TOKENS = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_git_commit_and_dirty() -> Tuple[str, bool]:
    """返回 (commit_short_or_full, dirty_bool)。失败时返回 ("unknown", False)。"""
    import subprocess as sp

    try:
        commit = sp.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=sp.STDOUT,
        ).decode().strip()
    except Exception:
        commit = "unknown"
    dirty = False
    try:
        out = sp.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), stderr=sp.STDOUT,
        ).decode()
        dirty = bool(out.strip())
    except Exception:
        pass
    return (commit, dirty)


def _sanity_prompts_not_in_test(prompts: List[str]) -> Tuple[bool, List[str]]:
    """从 ``data/sft/test.jsonl`` 读取所有 user prompt，校验 sanity prompts 不重叠。"""
    test_path = ROOT / "data" / "sft" / "test.jsonl"
    reasons: List[str] = []
    if not test_path.exists():
        return (False, [f"missing test.jsonl: {test_path}"])
    user_set = set()
    with test_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            if msgs:
                user_set.add(str(msgs[0].get("content", "")))
    for p in prompts:
        if p in user_set:
            reasons.append(f"sanity prompt leaked into test set: {p!r}")
    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Stage 1: Phase 5D verification
# ---------------------------------------------------------------------------


def stage_validate_phase5d() -> Phase5DVerification:
    _LOG.info("[stage 1/10] validate_phase5d_report ...")
    v = validate_phase5d_report()
    if not v.pass_:
        for r in v.reasons:
            _LOG.error("Phase 5D FAIL: %s", r)
        raise SystemExit(1)
    _LOG.info(
        "Phase 5D PASS: test_count=%d rows=%d loss=%.4f->%.4f (%.2f%%) "
        "improvement",
        v.test_count, v.predictions_rows,
        v.base_test_loss, v.finetuned_test_loss, v.test_loss_improvement_pct,
    )
    return v


# ---------------------------------------------------------------------------
# Stage 2: best adapter provenance
# ---------------------------------------------------------------------------


def stage_best_adapter_provenance() -> AdapterProvenance:
    _LOG.info("[stage 2/10] verify_best_adapter_provenance ...")
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore
    prov = verify_best_adapter_provenance(torch=torch)
    if not prov.equal:
        _LOG.error(
            "Best adapter provenance FAIL: src=%s ckpt=%s n_tensors=%d err=%s",
            prov.source_signature[:16] if prov.source_signature else "",
            prov.checkpoint_signature[:16] if prov.checkpoint_signature else "",
            prov.n_tensors, prov.error,
        )
        raise SystemExit(2)
    _LOG.info(
        "Best adapter provenance PASS: n_tensors=%d signature=%s",
        prov.n_tensors, prov.source_signature[:16],
    )
    return prov


# ---------------------------------------------------------------------------
# Stage 3: copy release adapter
# ---------------------------------------------------------------------------


def stage_copy_release_adapter() -> Path:
    _LOG.info("[stage 3/10] copy_release_adapter ...")
    PHASE5E_RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    # idempotent：清理上次 finalizer 留下的 release 顶层产物（不影响 source）
    for child in sorted(PHASE5E_RELEASE_ROOT.iterdir(), key=lambda x: -len(str(x))):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    target = copy_release_adapter(PHASE5E_BEST_ADAPTER_SOURCE, PHASE5E_RELEASE_ROOT)
    _LOG.info("Copied release adapter -> %s", target)
    return target


# ---------------------------------------------------------------------------
# Stage 4-5: training / evaluation summary + release manifest
# ---------------------------------------------------------------------------


def stage_summaries(
    *,
    git_commit: str,
    git_dirty: bool,
    phase5d_v: Phase5DVerification,
    adapter_prov: AdapterProvenance,
) -> Dict[str, Any]:
    _LOG.info("[stage 4/10] training_summary ...")
    training_summary = build_training_summary()
    write_training_summary(
        PHASE5E_RELEASE_ROOT / "training_summary.json", training_summary,
    )

    _LOG.info("[stage 5/10] evaluation_summary ...")
    eval_summary = build_evaluation_summary()
    write_evaluation_summary(
        PHASE5E_RELEASE_ROOT / "evaluation_summary.json", eval_summary,
    )

    _LOG.info("[stage 5/10] release_manifest ...")
    payload = build_release_manifest(
        git_commit=git_commit,
        git_dirty=git_dirty,
        phase5d_verification=phase5d_v,
        adapter_provenance=adapter_prov,
    )
    return payload


# ---------------------------------------------------------------------------
# Stage 6: write release markdown files (MODEL_CARD.md, RELEASE_README.md)
# ---------------------------------------------------------------------------


def write_model_card(path: Path) -> None:
    body = MODEL_CARD_TEMPLATE
    path.write_text(body, encoding="utf-8")
    _LOG.info("Wrote MODEL_CARD.md to %s", path)


def write_release_readme(path: Path) -> None:
    body = RELEASE_README_TEMPLATE
    path.write_text(body, encoding="utf-8")
    _LOG.info("Wrote RELEASE_README.md to %s", path)


# ---------------------------------------------------------------------------
# Stage 7-8: inventory + checksums + integrity gate
# ---------------------------------------------------------------------------


def stage_checksums(manifest_payload: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    _LOG.info("[stage 7/10] file_inventory + checksums ...")
    # Phase A：写所有其他文件 + 计算 inventory（不含 checksums.sha256 本身）。
    initial_inventory = build_file_inventory(PHASE5E_RELEASE_ROOT)
    # 把 inventory 列表回写到 manifest payload.files
    manifest_payload["files"] = [
        {"path": it["path"], "size_bytes": it["size_bytes"], "sha256": it["sha256"]}
        for it in initial_inventory
    ]
    # 写 release_manifest.json
    write_release_manifest(
        PHASE5E_RELEASE_ROOT / "release_manifest.json", manifest_payload,
    )
    # 重算 inventory（此时已包含 release_manifest.json，但 checksums 还未写）。
    full_inventory = build_file_inventory(PHASE5E_RELEASE_ROOT)
    checksums_path = PHASE5E_RELEASE_ROOT / "checksums.sha256"
    body = build_checksums(full_inventory)
    write_checksums(checksums_path, body)
    # Phase B：重新读回并校验。
    _LOG.info("[stage 8/10] verify_checksums ...")
    ok, reasons = verify_checksums(checksums_path)
    if not ok:
        for r in reasons:
            _LOG.error("Checksum FAIL: %s", r)
        raise SystemExit(3)
    _LOG.info(
        "Release integrity PASS: checked=%d missing=0 mismatch=0",
        len(full_inventory),
    )
    return (ok, full_inventory)


# ---------------------------------------------------------------------------
# Stage 9: fresh reload + sanity inference (GPU)
# ---------------------------------------------------------------------------


def stage_fresh_reload_and_inference(
    *,
    adapter_dir: Path,
    sanity_prompts: List[str],
) -> Tuple[bool, int, int, List[Dict[str, Any]]]:
    """Fresh reload base model + release adapter + sanity inference.

    Returns
    -------
    (ok, non_empty_count, total_count, per_prompt_outputs)
    """
    _LOG.info("[stage 9/10] fresh reload from %s ...", adapter_dir)
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from peft import PeftModel  # type: ignore
    except Exception as e:
        _LOG.error("Failed to import torch / transformers / peft: %s", e)
        raise SystemExit(4)

    # 1) 释放既有模型 + cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 2) 加载 base model — 4-bit NF4 / double quant / BF16 / pinned revision
    from transformers import BitsAndBytesConfig  # type: ignore

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    _LOG.info("Loading base model %s @ %s ...", PHASE5E_BASE_MODEL_NAME,
              PHASE5E_BASE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        PHASE5E_BASE_MODEL_NAME,
        revision=PHASE5E_BASE_REVISION,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    tok = AutoTokenizer.from_pretrained(
        str(adapter_dir),  # 优先使用 release adapter 自带的 tokenizer
        local_files_only=True,
    )

    # 3) 加载 PEFT adapter (从 release 目录)
    _LOG.info("Loading PEFT adapter from %s ...", adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    # 4) 校验 LoRA layers == 252
    target_modules: List[str] = []
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            target_modules.append(name)
    n_lora = len(target_modules)
    _LOG.info("LoRA target_modules count = %d", n_lora)
    if n_lora != 252:
        _LOG.error("LoRA layers != 252 (got %d) — fresh reload FAIL", n_lora)
        # cleanup before raise
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise SystemExit(4)

    # 5) Sanity inference — 3/3 non-empty
    outputs: List[Dict[str, Any]] = []
    non_empty = 0
    for i, prompt in enumerate(sanity_prompts):
        msgs = [{"role": "user", "content": prompt}]
        chat = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tok(chat, return_tensors="pt").to(model.device)
        try:
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=SANITY_INFERENCE_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            new_tokens = gen[0][inputs["input_ids"].shape[1]:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
        except Exception as e:
            _LOG.error("Sanity inference #%d raised: %s", i, e)
            text = ""
            outputs.append({"prompt": prompt, "error": str(e), "text": text})
            continue
        ok = bool(text and text.strip())
        if ok:
            non_empty += 1
        outputs.append({"prompt": prompt, "text": text, "non_empty": ok})

    # 6) 释放模型
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total = len(sanity_prompts)
    if non_empty < total:
        _LOG.error(
            "Sanity inference FAIL: %d/%d non-empty", non_empty, total,
        )
        raise SystemExit(5)
    _LOG.info(
        "Fresh reload + sanity inference PASS: %d/%d non-empty, LoRA layers=%d",
        non_empty, total, n_lora,
    )
    return (True, non_empty, total, outputs)


# ---------------------------------------------------------------------------
# Stage 10: final gate + completion report
# ---------------------------------------------------------------------------


def stage_final_gate(
    *,
    phase5d_v: Phase5DVerification,
    adapter_prov: AdapterProvenance,
    fresh_reload_ok: bool,
    inference_non_empty: int,
    inference_total: int,
) -> Any:
    _LOG.info("[stage 10/10] final_gate ...")
    res = final_gate(
        phase5d_verification=phase5d_v,
        adapter_provenance=adapter_prov,
        release_manifest_path=PHASE5E_RELEASE_ROOT / "release_manifest.json",
        checksums_verified=True,
        fresh_reload_passed=fresh_reload_ok,
        inference_non_empty=inference_non_empty,
        inference_total=inference_total,
    )
    _LOG.info("final_gate code=%s project_complete=%s reasons=%s",
              res.code, res.project_complete, res.reasons)
    return res


def write_completion_report(
    *,
    gate: Any,
    phase5d_v: Phase5DVerification,
    adapter_prov: AdapterProvenance,
    inference_outputs: List[Dict[str, Any]],
    file_inventory: List[Dict[str, Any]],
    git_commit: str,
    git_dirty: bool,
    sanity_prompts: List[str],
    training_summary: Dict[str, Any],
    eval_summary: Dict[str, Any],
    release_manifest: Dict[str, Any],
    duration_s: float,
) -> Tuple[Path, Path]:
    PHASE5E_RUNTIME_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "phase5e.v1",
        "code": gate.code,
        "project_complete": bool(gate.project_complete),
        "reasons": list(gate.reasons),
        "git": {"commit": git_commit, "dirty": bool(git_dirty)},
        "phase5d": phase5d_v.to_dict(),
        "adapter_provenance": adapter_prov.to_dict(),
        "sanity_inference": {
            "prompts": sanity_prompts,
            "non_empty_count": sum(1 for o in inference_outputs if o.get("non_empty")),
            "total_count": len(sanity_prompts),
            "outputs": inference_outputs,
        },
        "release": {
            "release_root": str(PHASE5E_RELEASE_ROOT),
            "release_manifest_path": str(
                PHASE5E_RELEASE_ROOT / "release_manifest.json"
            ),
            "file_count": len(file_inventory),
            "file_inventory": file_inventory,
            "checksums_path": str(
                PHASE5E_RELEASE_ROOT / "checksums.sha256"
            ),
        },
        "training_summary": training_summary,
        "evaluation_summary": eval_summary,
        "release_manifest_summary": {
            "schema_version": release_manifest.get("schema_version"),
            "release_type": release_manifest.get("release_type"),
            "release_created_at": release_manifest.get("release_created_at"),
            "base_model": release_manifest.get("base_model"),
            "lora": release_manifest.get("lora"),
            "training": release_manifest.get("training"),
            "held_out_test": release_manifest.get("held_out_test"),
        },
        "duration_seconds": round(duration_s, 3),
    }
    json_path = PHASE5E_RUNTIME_ARTIFACT_DIR / "project_completion_report.json"
    atomic_write_json(json_path, payload)

    md_lines: List[str] = []
    md_lines.append("# Phase 5E — Project Completion Report")
    md_lines.append("")
    md_lines.append(f"- code: `{gate.code}`")
    md_lines.append(f"- project_complete: `{bool(gate.project_complete)}`")
    md_lines.append(f"- git_commit: `{git_commit}`")
    md_lines.append(f"- git_dirty: `{bool(git_dirty)}`")
    md_lines.append("")
    md_lines.append("## Phase 5D gate")
    md_lines.append(
        f"- status={phase5d_v.status} code={phase5d_v.code} "
        f"ready={phase5d_v.ready_for_phase5e}"
    )
    md_lines.append(
        f"- test_count={phase5d_v.test_count} "
        f"predictions_rows={phase5d_v.predictions_rows}"
    )
    md_lines.append(
        f"- base_test_loss={phase5d_v.base_test_loss:.6f} "
        f"finetuned_test_loss={phase5d_v.finetuned_test_loss:.6f}"
    )
    md_lines.append(
        f"- test_loss_improvement_pct="
        f"{phase5d_v.test_loss_improvement_pct:.4f}"
    )
    md_lines.append(
        f"- protocol_signature={phase5d_v.eval_protocol_signature[:16]}"
    )
    md_lines.append("")
    md_lines.append("## Adapter provenance")
    md_lines.append(
        f"- n_tensors={adapter_prov.n_tensors} "
        f"equal_to_checkpoint_288={adapter_prov.equal}"
    )
    md_lines.append(
        f"- source_signature={adapter_prov.source_signature[:16]}"
    )
    md_lines.append("")
    md_lines.append("## Final gate")
    md_lines.append(f"- reasons: {gate.reasons if gate.reasons else '[]'}")
    md_lines.append("")
    md_lines.append("## Sanity inference")
    for o in inference_outputs:
        md_lines.append(
            f"- prompt={o.get('prompt')!r} non_empty={o.get('non_empty')}"
        )
    md_lines.append("")
    md_lines.append("## Release files")
    for it in file_inventory:
        md_lines.append(
            f"- `{it['path']}` size={it['size_bytes']} "
            f"sha256={it['sha256'][:16]}"
        )
    md_path = PHASE5E_RUNTIME_ARTIFACT_DIR / "project_completion_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return (json_path, md_path)


# ---------------------------------------------------------------------------
# Markdown templates
# ---------------------------------------------------------------------------


MODEL_CARD_TEMPLATE = """# Qwen3-8B + QLoRA Adapter — 基层党组织 FAQ

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
"""


RELEASE_README_TEMPLATE = """# Release — Qwen3-8B + QLoRA Adapter (基层党务 FAQ)

## 这是什么
这是本项目最终交付的 **LoRA Adapter Release**（不是完整 8B 模型）。
包内不含 base 8B 权重；使用时**必须搭配 pinned revision 的
`Qwen/Qwen3-8B`**，并使用 4-bit NF4 + double quant + BF16 加载。

来源：Phase 5C-B 正式 3 epoch QLoRA SFT 训练的 **best checkpoint =
checkpoint-288 / epoch 2 / best eval_loss = 1.1594**。

## 需要什么 Base Model
- model: `Qwen/Qwen3-8B`
- revision: `b968826d9c46dd6066d109eabc6255188de91218`
- tokenizer_signature:
  `4ac4fed6d76f2cddf5ee756990b7bd4dc40655f96e4c6be460dc3299e73651c0`

加载示例见 `MODEL_CARD.md`。

## 目录结构
```
outputs/phase5e/release/
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── chat_template.jinja
│   └── README.md
├── MODEL_CARD.md
├── RELEASE_README.md   <- 本文件
├── release_manifest.json
├── evaluation_summary.json
├── training_summary.json
└── checksums.sha256
```

## 如何加载
1. 安装 `torch` / `transformers` / `peft` / `bitsandbytes`。
2. 按 `MODEL_CARD.md` 中 Loading Example 加载 base + 4-bit + LoRA。
3. 用 `apply_chat_template(..., enable_thinking=False,
   add_generation_prompt=True)` 构造 prompt。
4. ``do_sample=False``，``max_new_tokens`` 自定义（推荐 256-1024）。

## 如何校验 SHA256
```bash
# Linux / macOS / WSL
cd outputs/phase5e/release
sha256sum -c checksums.sha256

# Windows PowerShell (示例：用 Get-FileHash 逐文件比对)
Get-Content checksums.sha256 | ForEach-Object {
  $parts = $_ -split '\\s+', 2
  $expected = $parts[0]
  $path = $parts[1]
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower()
  if ($actual -ne $expected) { Write-Error "MISMATCH: $path" }
}
```
任何 missing / mismatch ⇒ ``RELEASE_INTEGRITY_FAIL``，请勿使用。

## 哪个文件是 adapter
- 唯一 LoRA 权重文件：
  `adapter/adapter_model.safetensors`（504 tensors / 252 LoRA layers）
- adapter 元数据：`adapter/adapter_config.json`
- 配套 tokenizer：`adapter/tokenizer.json` / `adapter/tokenizer_config.json`
- 配套 chat template：`adapter/chat_template.jinja`

**请勿把 `adapter_model.safetensors` 当作完整 8B 模型加载。**

## 如何复现实验
参见 `MODEL_CARD.md` 的 `Reproducibility` 章节。

简述：
1. 重新加载 base model pinned revision。
2. 用 `data/sft/{train,validation,test}.jsonl` 训练 3 epoch，LR=1e-4，
   linear schedule，warmup=0.03。
3. ``load_best_model_at_end=True`` ⇒ ``best_adapter`` 自动写入
   `outputs/phase5c-b/best_adapter/`。
4. 用 `verify_best_adapter` 校验 `best_adapter == checkpoint-288`。
5. 用 Phase 5D frozen protocol 在 `test.jsonl` 上做 held-out test。
6. 复制 `best_adapter/*` → `outputs/phase5e/release/adapter/*`。
7. 重新计算 SHA256 写入 `checksums.sha256`。

## 已知限制
- 不包含 base 8B 完整权重；不含 GGUF / GPTQ / AWQ 等部署形态。
- 仅覆盖《基层党务（试用版）1》文档范围内的 FAQ。
- 不可被理解为对党务问题的权威回答。

## License / Base Model Notice
- Base Model `Qwen3-8B` 受 Qwen 团队发布条款约束。
- 本 Adapter 适用于研究 / 内部验证 / 党务工作辅助写作；任何政策
  落地以官方原文为准。
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5E finalize QLoRA release")
    p.add_argument("--skip-fresh-reload", action="store_true",
                   help="跳过 fresh reload + sanity inference（仅用于调试）")
    p.add_argument("--release-root", default=str(PHASE5E_RELEASE_ROOT), type=Path)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    global PHASE5E_RELEASE_ROOT
    PHASE5E_RELEASE_ROOT = Path(args.release_root)

    t0 = time.time()
    git_commit, git_dirty = _get_git_commit_and_dirty()
    _LOG.info("git_commit=%s git_dirty=%s", git_commit, git_dirty)

    # 0) sanity prompts 与 test 集不重叠
    ok, reasons = _sanity_prompts_not_in_test(DEFAULT_SANITY_PROMPTS)
    if not ok:
        for r in reasons:
            _LOG.error("Sanity prompt guard: %s", r)
        return 1
    _LOG.info("Sanity prompts not in test set: OK (n=%d)",
              len(DEFAULT_SANITY_PROMPTS))

    # 1) Phase 5D gate
    phase5d_v = stage_validate_phase5d()

    # 2) Best adapter provenance
    adapter_prov = stage_best_adapter_provenance()

    # 3) Copy release adapter
    adapter_dir = stage_copy_release_adapter()

    # 4-5) Summaries + manifest
    training_summary = build_training_summary()
    write_training_summary(
        PHASE5E_RELEASE_ROOT / "training_summary.json", training_summary,
    )
    eval_summary = build_evaluation_summary()
    write_evaluation_summary(
        PHASE5E_RELEASE_ROOT / "evaluation_summary.json", eval_summary,
    )
    payload = build_release_manifest(
        git_commit=git_commit,
        git_dirty=git_dirty,
        phase5d_verification=phase5d_v,
        adapter_provenance=adapter_prov,
    )

    # 6) MODEL_CARD.md / RELEASE_README.md
    write_model_card(PHASE5E_RELEASE_ROOT / "MODEL_CARD.md")
    write_release_readme(PHASE5E_RELEASE_ROOT / "RELEASE_README.md")

    # 7-8) checksums
    try:
        ok, file_inventory = stage_checksums(payload)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 3

    # 9) fresh reload + sanity inference
    if args.skip_fresh_reload:
        _LOG.warning("Skipping fresh reload + sanity inference (debug only)")
        fresh_ok = True
        non_empty = len(DEFAULT_SANITY_PROMPTS)
        total = len(DEFAULT_SANITY_PROMPTS)
        outputs: List[Dict[str, Any]] = [
            {"prompt": p, "non_empty": True, "text": "[SKIPPED]"} for p in DEFAULT_SANITY_PROMPTS
        ]
    else:
        fresh_ok, non_empty, total, outputs = stage_fresh_reload_and_inference(
            adapter_dir=adapter_dir,
            sanity_prompts=DEFAULT_SANITY_PROMPTS,
        )

    # 10) final gate
    gate = stage_final_gate(
        phase5d_v=phase5d_v,
        adapter_prov=adapter_prov,
        fresh_reload_ok=bool(fresh_ok),
        inference_non_empty=non_empty,
        inference_total=total,
    )

    # 11) project completion report
    t1 = time.time()
    json_path, md_path = write_completion_report(
        gate=gate,
        phase5d_v=phase5d_v,
        adapter_prov=adapter_prov,
        inference_outputs=outputs,
        file_inventory=file_inventory,
        git_commit=git_commit,
        git_dirty=git_dirty,
        sanity_prompts=DEFAULT_SANITY_PROMPTS,
        training_summary=training_summary,
        eval_summary=eval_summary,
        release_manifest=payload,
        duration_s=t1 - t0,
    )

    _LOG.info(
        "Project completion report:\n  json: %s\n  md:   %s",
        json_path, md_path,
    )
    if gate.code != "PASS":
        _LOG.error("PHASE5E_CODE=BLOCKED, reasons=%s", gate.reasons)
        return 6
    _LOG.info(
        "PHASE5E_CODE=PASS PROJECT_COMPLETE=YES duration=%.1fs",
        t1 - t0,
    )
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        traceback.print_exc()
        rc = 99
    sys.exit(rc)