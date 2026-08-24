# LoRA 评论判断模型资料

## Knowledge

- [论文：LoRA——大语言模型的低秩适配](https://arxiv.org/abs/2106.09685)
  LoRA 原始论文。用于理解为什么冻结原模型、只训练少量新增参数，可以显著减少训练参数和显存需求。
- [Hugging Face PEFT：LoRA 官方指南](https://huggingface.co/docs/peft/en/developer_guides/lora)
  官方实现说明。用于配置 LoRA、选择要训练的线性层以及保存和加载训练结果。
- [Hugging Face PEFT：量化训练官方指南](https://huggingface.co/docs/peft/en/developer_guides/quantization)
  官方四位加载加 LoRA 说明。用于在消费级显卡上降低基础模型的显存占用。
- [Hugging Face bitsandbytes 官方文档](https://huggingface.co/docs/bitsandbytes/index)
  四位和八位模型加载的官方文档。用于核对当前显卡环境下的压缩加载方式。
- [Hugging Face TRL：监督微调官方文档](https://huggingface.co/docs/trl/sft_trainer)
  用成对的输入和标准答案训练语言模型的官方入口。用于训练循环、数据格式、验证和保存。

## Wisdom (Communities)

当前没有要求加入外部社区。训练出现具体显存、速度或模型兼容问题时，再根据所选基础模型寻找其官方讨论区和问题追踪记录。
