"""Phase 4 SFT 数据集构造的数据模型与配置。

本模块只定义：
- ``SFTSample``：单条 chat-style 训练样本（user + assistant）。
- ``SFTBuildConfig``：数据构造与划分的确定性配置。

不包含任何 IO、划分或质量审计逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


# 合法的最低置信度：仅 high 与 medium。
# low/none 拒绝进入正式训练集。
VALID_MIN_CONFIDENCE = ("high", "medium")

# 确定性划分的固定 bucket 总数（0..SPLIT_BUCKET_TOTAL-1）。
# train/validation/test 边界来自 SFTBuildConfig 的 ratio，
# 例如 train_ratio=0.90 ⇒ train_end = round(0.90 * SPLIT_BUCKET_TOTAL) = 9000。
# 不再硬编码业务边界，避免 config 形同虚设。
SPLIT_BUCKET_TOTAL = 10000

# 默认固定划分盐值；任何变更都会改变所有 split 分配。
DEFAULT_SPLIT_SALT = "qlora-sft-phase4-v1"


@dataclass(frozen=True)
class SFTBuildConfig:
    """SFT 数据构造与划分的确定性配置。

    字段全部冻结，保证 ``run_sft_pipeline`` 不会出现"运行到一半被改"
    的副作用；同输入 + 同 config ⇒ 输出完全一致。
    """

    # 最低 FAQ 置信度；只允许 high / medium，low / none 视为非法
    min_confidence: str = "medium"

    # 训练集目标比例，与 validation + test 之和为 1.0（允许浮点误差）
    train_ratio: float = 0.90

    # 验证集目标比例
    validation_ratio: float = 0.05

    # 测试集目标比例
    test_ratio: float = 0.05

    # 确定性划分使用的固定盐值，必须显式持久化在 report 中
    split_salt: str = DEFAULT_SPLIT_SALT

    def __post_init__(self) -> None:
        # 置信度合法性
        if self.min_confidence not in VALID_MIN_CONFIDENCE:
            raise ValueError(
                f"min_confidence 必须是 {VALID_MIN_CONFIDENCE} 之一，得到 {self.min_confidence!r}"
            )

        # 比例合法性
        for name in ("train_ratio", "validation_ratio", "test_ratio"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} 必须 > 0，得到 {v}")
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train/validation/test 比例之和必须为 1.0，得到 {total}"
            )

        if not self.split_salt:
            raise ValueError("split_salt 不能为空")


# ---------------------------------------------------------------------------
# 样本
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SFTSample:
    """一条确定性的 chat-style SFT 问答样本。

    messages 严格为 ``[user, assistant]`` 两轮，role 与 content 不可空，
    content 不得携带"问题："/"答案："前缀或 metadata 痕迹。
    """

    # 稳定样本 ID（SHA256(unit_id + question + answer) 截前 24 位），
    # 用于数据集去重、跨阶段追踪与 split 关联
    sample_id: str

    # Chat-style 消息列表；当前固定为 user + assistant 两轮
    messages: List[Dict[str, str]]

    # 源 SemanticUnit 与原文记录的追溯信息，不进入训练文本
    metadata: Dict[str, Any] = field(default_factory=dict)
