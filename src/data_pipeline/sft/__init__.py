"""Phase 4 SFT 数据集构造包。

将 Phase 3 产出的 ``semantic_units.jsonl`` 中的 FAQ 单元
转成标准 chat-style SFT 样本，并完成数据质量审计与
Train / Validation / Test 确定性划分。
"""

from .pipeline import run_sft_pipeline

__all__ = ["run_sft_pipeline"]
