"""Phase 5A — tokenizer 审计的配置与数据模型。

只定义 ``TokenizerAuditConfig``。
审计过程产生的结果直接用普通 dict 承载，不额外引入 ``Result`` dataclass。
"""

from __future__ import annotations

from dataclasses import dataclass


# 候选最大序列长度默认值；只用于覆盖率统计，不决定训练最终长度。
DEFAULT_CANDIDATE_MAX_LENGTHS: tuple[int, ...] = (512, 1024, 1536, 2048, 3072, 4096)

# 默认目标模型仓库。
DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"

# 默认 revision；用户可通过 CLI 自定义。
DEFAULT_REVISION = "main"

# 每个 split 抽多少条真实样本做内部 sanity check（不写入 report）。
DEFAULT_REAL_SAMPLE_CHECK_PER_SPLIT = 5

# 强制要求的最低 transformers 版本；Qwen3 需要 >= 4.51。
MINIMUM_TRANSFORMERS_VERSION = "4.51.0"


@dataclass(frozen=True)
class TokenizerAuditConfig:
    """Phase 5A tokenizer 审计阶段的固定配置。

    字段全部冻结：同 SFT 数据 + 同 config + 同 revision ⇒ 输出 byte-for-byte 一致。
    """

    # Hugging Face 模型仓库标识，默认 Qwen3-8B
    model_name: str = DEFAULT_MODEL_NAME

    # Hugging Face 模型 revision，用于锁定 tokenizer 版本
    revision: str = DEFAULT_REVISION

    # 候选最大序列长度；用于统计 95/99/100% coverage
    candidate_max_lengths: tuple[int, ...] = DEFAULT_CANDIDATE_MAX_LENGTHS

    # 每个 split 抽几条真实样本做内部 sanity check（不写入 report）
    real_sample_check_per_split: int = DEFAULT_REAL_SAMPLE_CHECK_PER_SPLIT

    def __post_init__(self) -> None:
        # 候选长度合法性：全部为正整数
        if not self.candidate_max_lengths:
            raise ValueError("candidate_max_lengths 不能为空")
        for n in self.candidate_max_lengths:
            if not isinstance(n, int) or n <= 0:
                raise ValueError(f"candidate_max_lengths 含非法值: {n!r}")

        # 模型名 / revision 非空
        if not self.model_name:
            raise ValueError("model_name 不能为空")
        if not self.revision:
            raise ValueError("revision 不能为空")

        # 抽样数非负
        if self.real_sample_check_per_split < 0:
            raise ValueError("real_sample_check_per_split 不能为负")