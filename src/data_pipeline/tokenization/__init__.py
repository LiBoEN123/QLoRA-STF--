"""Phase 5A — Qwen3-8B tokenizer / chat template / token 长度审计包。

本包在 Phase 4 产出的 SFT 数据集之上，做训练前 tokenizer 验证：

- 加载 ``AutoTokenizer``（不加载模型权重，不初始化 GPU）。
- 强制 ``enable_thinking=False``（non-thinking SFT）。
- 使用 ``tokenizer.apply_chat_template`` 作为唯一真值。
- 统计 token 长度、coverage、prompt prefix match。
- 不修改任何上游数据，不生成训练样本。
"""

from .pipeline import run_tokenizer_audit

__all__ = ["run_tokenizer_audit"]