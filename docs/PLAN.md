# Qwen3-0.6B LoRA SFT 项目设计与阶段计划

日期：2026-10-28。最初选择 4B，用户随后要求在 4B 过大时换用小模型，因此改为 Qwen3-0.6B-Base，独立目录，不改动 MiniGPT。

## 当前环境（已实际检查）
- NVIDIA GeForce RTX 3060 Laptop GPU，6144 MiB；驱动 527.99。
- 现有 GPU 虚拟环境：Python 3.12.15、PyTorch 2.7.1+cu118、CUDA 可用、BF16 支持。
- 已在现有 GPU 环境安装项目运行依赖和 pytest；标准基准所需的 lm-eval 与可选量化所需的 bitsandbytes 尚未安装。安装前 E 盘剩余约 69.7 GB。
- 官方模型 Qwen/Qwen3-0.6B-Base：模型页标称 0.6B 参数、Apache-2.0；仓库约 1.2 GB。仍需通过实测确认当前设备上的训练显存与速度。

## 数据与评估边界
- 采用模型自带 Qwen Chat Template。每条 JSONL 样本是多轮 messages，要求最后一条为 assistant，且此前至少有 user。只监督最后一条 assistant 回复与结束符；历史、角色标记、系统/user 消息及 padding 为 -100。
- 测试 Qwen 模板前缀相等性。模板可能对后续对话重新排版，故不监督历史 assistant 回复；不静默采用字符偏移来映射 token 边界。
- 数据按对话前缀分组去重后固定 seed 划分；过长样本过滤并记录，不截断目标回复；提供作者编写的小样本，仅用于流程检查，不作效果研究数据。
- LoRA 目标含 q/k/v/o 和 gate/up/down 投影。支持 rank、alpha、dropout 消融与4bit量化；数据、训练配置、随机种子、训练参数量、Loss、耗时、显存峰值存本地。
- MMLU、GSM8K、IFEval 通过 lm-evaluation-harness 对同一基座和 adapter 分别运行相同任务/提示/抽样设置，原始输出存本地。只从结果 JSON 读取分数。
- 默认不启用 4bit 量化；如需量化可安装可选依赖。不能把脚本存在当作训练或基准评测已完成。

## 阶段
1. 已实现项目结构、输入校验、最后助手回复标签、padding、去重与数据拆分；6 项单元测试通过。
2. 已实现训练和生成：PEFT LoRA、可选 4bit、日志与 adapter 保存；仅用随机初始化的极小 Qwen3 配置执行了 1 步训练链路检查，未下载官方模型权重。
3. 已实现基座/adapter 的 MMLU、GSM8K、IFEval 入口，执行了 dry-run；完整基准未执行，不填写分数。
4. README、许可证、敏感文件排除和本地 Git 提交已完成；GitHub 仓库创建与推送待完成。

## 研究限制
默认示例数据过小，不能支撑能力改进结论。标准基准与训练数据须检查重叠；同一模型revision、shot数、模板、量化和解码配置都要记录。不同rank只比较同一数据/预算、需多种子。显存峰值/耗时须在真实设备运行才能填。
