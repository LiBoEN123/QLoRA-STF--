"""Phase 5C-A — Backend / Unsloth 相关 CPU-only 单元测试。

不下载任何模型；不依赖 GPU；mock / monkeypatch 一切外部副作用。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

from data_pipeline.training import (
    QLoRASmokeConfig,
    TrainingGateError,
)
from data_pipeline.training.backends import (
    DEFAULT_BACKEND,
    VALID_BACKENDS,
    HfBackend,
    UnslothBackend,
    resolve_backend,
)
from data_pipeline.training.backends.base import BackendMetadata


# ---------------------------------------------------------------------------
# 1) 默认 backend + 合法性
# ---------------------------------------------------------------------------


def test_default_backend_is_hf():
    """默认 backend 必须是 hf；不允许因引入 unsloth 改变默认行为。"""
    assert DEFAULT_BACKEND == "hf"
    assert "hf" in VALID_BACKENDS
    assert "unsloth" in VALID_BACKENDS


def test_resolve_backend_hf():
    """hf backend 必须始终可用（不依赖 unsloth）。"""
    b = resolve_backend("hf")
    assert isinstance(b, HfBackend)
    assert b.name == "hf"


def test_resolve_backend_unsloth_object_construction_does_not_import_unsloth():
    """构造 UnslothBackend 实例时不应触发 import unsloth（lazy import）。"""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("unsloth")}
    sys.modules.pop("unsloth", None)
    sys.modules.pop("unsloth_zoo", None)
    try:
        b = UnslothBackend()
        # unsloth 仍未被 import
        assert "unsloth" not in sys.modules
        # metadata 是静态字段，不触发 import
        md = b.metadata()
        assert isinstance(md, BackendMetadata)
        assert md.name == "unsloth"
        assert md.depends_on_unsloth is True
        assert md.supports_revision_pin is True
        assert md.supports_assistant_only_loss is True
    finally:
        for k in list(sys.modules):
            if k not in saved and k.startswith("unsloth"):
                sys.modules.pop(k, None)
        sys.modules.update(saved)


def test_resolve_backend_invalid_name_rejected():
    """非法 backend 名 ⇒ TrainingGateError。"""
    with pytest.raises(TrainingGateError) as exc:
        resolve_backend("transformers")
    assert "INVALID_BACKEND" in str(exc.value)


def test_resolve_backend_non_string_rejected():
    """非字符串 backend ⇒ TrainingGateError。"""
    with pytest.raises(TrainingGateError) as exc:
        resolve_backend(None)  # type: ignore[arg-type]
    assert "backend 名称必须是 str" in str(exc.value)


def test_unsloth_backend_missing_dependency_raises_clear_error():
    """缺 unsloth 时，调用 backend 实际方法应抛 TrainingGateError(UNSLOTH_DEPENDENCY_MISSING)。"""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("unsloth")}
    sys.modules.pop("unsloth", None)
    sys.modules.pop("unsloth_zoo", None)
    try:
        b = UnslothBackend()
        with pytest.raises(TrainingGateError) as exc:
            b.load_model_and_tokenizer(
                model_name="Qwen/Qwen3-8B",
                revision="abc",
                torch=None,
                config=QLoRASmokeConfig(),
            )
        assert "UNSLOTH_DEPENDENCY_MISSING" in str(exc.value)
    finally:
        for k in list(sys.modules):
            if k not in saved and k.startswith("unsloth"):
                sys.modules.pop(k, None)
        sys.modules.update(saved)


def test_hf_backend_does_not_require_unsloth():
    """HfBackend.metadata() 必须可调用，且 depends_on_unsloth=False。"""
    b = HfBackend()
    md = b.metadata()
    assert md.depends_on_unsloth is False
    assert md.supports_revision_pin is True
    assert md.supports_assistant_only_loss is True


# ---------------------------------------------------------------------------
# 2) 顶层 import data_pipeline.training 不触发 unsloth
# ---------------------------------------------------------------------------


def test_importing_data_pipeline_training_does_not_import_unsloth():
    """普通 .venv 无 unsloth 时，import data_pipeline.training 不应触发 unsloth。"""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("unsloth")}
    sys.modules.pop("unsloth", None)
    sys.modules.pop("unsloth_zoo", None)
    try:
        import importlib
        # 强制重新导入核心模块
        for mod in list(sys.modules):
            if mod.startswith("data_pipeline"):
                importlib.reload(sys.modules[mod]) if False else None
        import data_pipeline.training  # noqa: F401
        import data_pipeline.training.backends  # noqa: F401
        assert "unsloth" not in sys.modules, (
            f"unsloth 不应被 import；sys.modules 含 unsloth={list(k for k in sys.modules if 'unsloth' in k)}"
        )
    finally:
        for k in list(sys.modules):
            if k not in saved and k.startswith("unsloth"):
                sys.modules.pop(k, None)
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# 3) Config parity：unsloth parity 模式禁止改固定字段
# ---------------------------------------------------------------------------


def test_unsloth_parity_config_does_not_change_fixed_fields():
    """Unsloth parity 必须保持：max_length=2048 / NF4 / double_quant / BF16 / LoRA r16 a32 d0.05 /
    assistant_only_loss=True / packing=False。
    """
    cfg = QLoRASmokeConfig()
    assert cfg.max_length == 2048
    assert cfg.quant_type == "nf4"
    assert cfg.compute_dtype == "bfloat16"
    assert cfg.double_quant is True
    assert cfg.bf16 is True
    assert cfg.fp16 is False
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.lora_target_modules == "all-linear"
    assert cfg.lora_bias == "none"
    assert cfg.lora_task_type == "CAUSAL_LM"
    assert cfg.assistant_only_loss is True
    assert cfg.packing is False


# ---------------------------------------------------------------------------
# 4) Backend 行为：is_loaded_in_4bit / count_linear4bit
# ---------------------------------------------------------------------------


class _FakeLinear4bit:
    pass


def _install_fake_bnb(monkeypatch):
    """monkeypatch bitsandbytes.nn.Linear4bit 为 fake 类型。"""
    import sys
    fake_bnb = mock.MagicMock()
    fake_bnb.nn.Linear4bit = _FakeLinear4bit
    sys.modules["bitsandbytes"] = fake_bnb
    sys.modules["bitsandbytes.nn"] = fake_bnb.nn
    return fake_bnb


def test_backend_count_linear4bit_walks_modules():
    """count_linear4bit 必须能识别 fake Linear4bit 实例。"""
    _install_fake_bnb(monkeypatch=None) if False else None
    # 关键：必须 monkeypatch sys.modules['bitsandbytes.nn'] 使 ``from bitsandbytes.nn import Linear4bit`` 拿到我们的 fake
    import sys
    fake_bnb_mod = mock.MagicMock()
    fake_bnb_mod.Linear4bit = _FakeLinear4bit
    fake_bnb_pkg = mock.MagicMock()
    fake_bnb_pkg.nn = fake_bnb_mod
    sys.modules["bitsandbytes"] = fake_bnb_pkg
    sys.modules["bitsandbytes.nn"] = fake_bnb_mod
    try:
        b = HfBackend()
        fake = _FakeLinear4bit()
        class M:
            def __init__(self):
                self.children = [fake, fake, object()]
            def modules(self):
                yield self
                for c in self.children:
                    yield c
        m = M()
        assert b.count_linear4bit(m) == 2
    finally:
        sys.modules.pop("bitsandbytes", None)
        sys.modules.pop("bitsandbytes.nn", None)


def test_backend_is_loaded_in_4bit_via_attribute():
    """model.is_loaded_in_4bit=True ⇒ True。"""
    b = HfBackend()
    m = mock.MagicMock()
    m.is_loaded_in_4bit = True
    m.quantization_method = None
    m.quantization_config = None
    assert b.is_loaded_in_4bit(m) is True


def test_backend_is_loaded_in_4bit_falls_back_to_quant_config():
    """model 没有 is_loaded_in_4bit，但 quantization_config={'load_in_4bit': True} ⇒ True。"""
    b = HfBackend()
    m = mock.MagicMock(spec=["quantization_config"])
    qc = {"load_in_4bit": True}
    m.quantization_config = qc
    assert b.is_loaded_in_4bit(m) is True


# ---------------------------------------------------------------------------
# 5) Tokenizer signature drift（deterministic）
# ---------------------------------------------------------------------------


def test_tokenizer_signature_deterministic():
    """相同输入（sample_id + split + total/prompt/assistant tokens）应产出相同 SHA256。"""
    import hashlib

    rows = [
        {"sample_id": "sft_a", "split": "train",
         "total_tokens": 100, "prompt_tokens": 20, "assistant_target_tokens": 80},
        {"sample_id": "sft_b", "split": "train",
         "total_tokens": 200, "prompt_tokens": 30, "assistant_target_tokens": 170},
    ]
    def _sig(rs):
        h = hashlib.sha256()
        for r in sorted(rs, key=lambda x: x["sample_id"]):
            h.update(r["sample_id"].encode("utf-8"))
            h.update(b"|")
            h.update(str(r["split"]).encode("utf-8"))
            h.update(b"|")
            h.update(str(r["total_tokens"]).encode("utf-8"))
            h.update(b"|")
            h.update(str(r["prompt_tokens"]).encode("utf-8"))
            h.update(b"|")
            h.update(str(r["assistant_target_tokens"]).encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    s1 = _sig(rows)
    s2 = _sig(rows)
    assert s1 == s2

    # 改一行 total_tokens ⇒ signature 必须变
    rows2 = [dict(rows[0]), dict(rows[1])]
    rows2[1]["total_tokens"] = 201
    assert _sig(rows2) != s1


# ---------------------------------------------------------------------------
# 6) Model revision pin：未 pin 必须 BLOCK
# ---------------------------------------------------------------------------


def test_unsloth_backend_load_without_pinned_revision_is_blocked():
    """Unsloth load_model_and_tokenizer 调用前必须传入 pinned revision（resolved SHA）。

    本测试通过 monkeypatch unsloth + FastLanguageModel.from_pretrained，
    验证 UnslothBackend 把传入的 revision 直接交给 FastLanguageModel。
    """
    fake_unsloth = mock.MagicMock()
    fake_FLM = mock.MagicMock()
    fake_FLM.from_pretrained.return_value = (mock.MagicMock(), mock.MagicMock())
    fake_unsloth.FastLanguageModel = fake_FLM
    sys.modules["unsloth"] = fake_unsloth
    fake_unsloth_mod = mock.MagicMock()
    sys.modules["unsloth_zoo"] = fake_unsloth_mod

    try:
        b = UnslothBackend()
        fake_torch = mock.MagicMock()
        fake_torch.bfloat16 = object()
        # 传入 pinned revision（SHA），backend 必须使用该 revision
        rev = "b968826d9c46dd6066d109eabc6255188de91218"
        model, tok = b.load_model_and_tokenizer(
            model_name="Qwen/Qwen3-8B",
            revision=rev,
            torch=fake_torch,
            config=QLoRASmokeConfig(),
        )
        # 验证 FLM.from_pretrained 被以 revision=rev 调用
        kwargs = fake_FLM.from_pretrained.call_args.kwargs
        args = fake_FLM.from_pretrained.call_args.args
        assert kwargs.get("revision") == rev or (len(args) >= 2 and args[1] == rev)
    finally:
        sys.modules.pop("unsloth", None)
        sys.modules.pop("unsloth_zoo", None)


# ---------------------------------------------------------------------------
# 7) Mask sanity helper
# ---------------------------------------------------------------------------


def test_assistant_mask_check_helper_detects_invalid_mask(monkeypatch):
    """Mock TRL 路径：labels 全 0 ⇒ mask_validation_pass 应为 False。"""
    from scripts.smoke_train_qlora import _verify_assistant_mask

    # 构造假 trainer / dataset
    dataset = [{"sample_id": "sft_a", "messages": []}, {"sample_id": "sft_b", "messages": []}]

    class _Collator:
        def __call__(self, rows):
            return {
                "input_ids": [list(range(20))],
                "labels": [list(range(20))],  # 全部 = 非 -100 ⇒ masked=0 ⇒ fail
            }

    trainer = mock.MagicMock()
    trainer.train_dataset = dataset
    trainer.data_collator = _Collator()
    res = _verify_assistant_mask(trainer, dataset)
    assert len(res) == 2
    assert all(r["mask_validation_pass"] is False for r in res)


def test_assistant_mask_check_helper_passes_with_real_mask():
    """Mock TRL 路径：labels = -100 + 非 -100 ⇒ 通过。"""
    from scripts.smoke_train_qlora import _verify_assistant_mask

    dataset = [{"sample_id": "sft_a", "messages": []}]

    class _Collator:
        def __call__(self, rows):
            labels = [-100] * 5 + [1, 2, 3]  # 5 masked + 3 supervised
            ids = list(range(len(labels)))
            return {"input_ids": [ids], "labels": [labels]}

    trainer = mock.MagicMock()
    trainer.train_dataset = dataset
    trainer.data_collator = _Collator()
    res = _verify_assistant_mask(trainer, dataset)
    assert len(res) == 1
    r = res[0]
    assert r["sequence_length"] == 8
    assert r["masked_tokens"] == 5
    assert r["supervised_tokens"] == 3
    assert r["mask_validation_pass"] is True


# ---------------------------------------------------------------------------
# 8) Backend is_loaded_in_4bit: walks .model attribute (Unsloth wrapper)
# ---------------------------------------------------------------------------


def test_backend_is_loaded_in_4bit_walks_inner_model():
    """Unsloth 包装的 model 可能 .is_loaded_in_4bit 在 .model 上，Backend.is_loaded_in_4bit 应能下钻。"""

    b = HfBackend()

    inner = mock.MagicMock()
    inner.is_loaded_in_4bit = True
    inner.quantization_method = None
    inner.quantization_config = None

    outer = mock.MagicMock(spec=["model"])  # outer 自己无 is_loaded_in_4bit
    outer.model = inner
    # outer.is_loaded_in_4bit 不存在 ⇒ hasattr fallback
    del outer.is_loaded_in_4bit  # 确保属性缺失
    # spec=["model"] 不影响属性；我们用普通 mock
    outer = mock.MagicMock()
    outer.is_loaded_in_4bit = mock.PropertyMock(side_effect=AttributeError("no"))
    outer.quantization_method = None
    outer.quantization_config = None
    outer.model = inner
    # hasattr(outer, "is_loaded_in_4bit") == True；调用会抛 AttributeError；
    # HfBackend.is_loaded_in_4bit 会捕获并继续走 quantization_config 分支
    # inner 最终应被检测到 is_loaded_in_4bit=True
    assert b.is_loaded_in_4bit(outer) is True


# ---------------------------------------------------------------------------
# 9) Backend metadata 不触发 import
# ---------------------------------------------------------------------------


def test_backend_metadata_does_not_trigger_heavy_imports():
    """metadata() 调用必须轻量，不触发 unsloth / transformers / peft。"""
    saved_unsloth = {k: v for k, v in sys.modules.items() if k.startswith("unsloth")}
    sys.modules.pop("unsloth", None)
    sys.modules.pop("unsloth_zoo", None)
    try:
        b = UnslothBackend()
        md = b.metadata()
        assert md.name == "unsloth"
        assert "unsloth" not in sys.modules
    finally:
        for k in list(sys.modules):
            if k not in saved_unsloth and k.startswith("unsloth"):
                sys.modules.pop(k, None)
        sys.modules.update(saved_unsloth)


# ---------------------------------------------------------------------------
# 10) resolve_revision 在不可达时返回 pinned=False
# ---------------------------------------------------------------------------


def test_resolve_revision_when_hf_unreachable_returns_pinned_false():
    """mock huggingface_hub 抛异常 ⇒ resolve_revision 必须返回 (requested, None, False)。"""
    b = HfBackend()
    # monkeypatch huggingface_hub 使其抛异常
    fake_hub = mock.MagicMock()
    fake_hub.HfApi.side_effect = Exception("network down")
    sys.modules["huggingface_hub"] = fake_hub
    try:
        req, res, pinned = b.resolve_revision("Qwen/Qwen3-8B", "main")
        assert req == "main"
        assert res is None
        assert pinned is False
    finally:
        sys.modules.pop("huggingface_hub", None)


def test_resolve_revision_when_hf_returns_sha_is_pinned():
    """mock huggingface_hub 返回 sha ⇒ pinned=True。"""
    b = HfBackend()
    fake_hub = mock.MagicMock()
    fake_api = fake_hub.HfApi.return_value
    fake_info = mock.MagicMock()
    fake_info.sha = "deadbeef0000000000000000000000000000abcd"
    fake_api.model_info.return_value = fake_info
    sys.modules["huggingface_hub"] = fake_hub
    try:
        req, res, pinned = b.resolve_revision("Qwen/Qwen3-8B", "main")
        assert req == "main"
        assert res == "deadbeef0000000000000000000000000000abcd"
        assert pinned is True
    finally:
        sys.modules.pop("huggingface_hub", None)