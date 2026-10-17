# 简历项目经历：Qwen3-4B 数学推理后训练

整理日期：2026-10-14。面向大模型算法／后训练实习。以下正文采用用户最新报告的第三轮SFT约65%、GRPO测试正确率78%；对应评测记录仍待对齐，不将旧版GRPO结果接到新版SFT之后。

## 可粘贴到简历的正文

**Qwen3-4B 数学推理后训练｜SFT / GRPO**

技术栈：PyTorch、Transformers、TRL、PEFT/QLoRA、vLLM、Math-Verify

面向数学推理任务，完成Qwen3-4B-Base的SFT冷启动与GRPO后训练，MATH Level 2–3测试正确率由SFT起点约65%提升至**78%（约+13个百分点）**。

- **SFT数据优化：**针对微调后数学正确率下降，开展Base/SFT配对错题分析与示范质量复核；采用Base每题4次采样、答案核验及推导审查，筛选合格自身解答并统一格式，在改善输出格式的同时使正确率恢复至与Base接近。
- **GRPO数据与奖励：**基于SFT模型采样表现筛选训练题，优先保留4份回答中1–3份正确的题目；使用Math-Verify构建答案正确性奖励，结合辅助格式奖励与KL约束，完成在线采样及策略训练。
- **训练与评测工程：**基于TRL、PEFT与vLLM搭建单卡训练流程，采用NF4 QLoRA控制显存占用；建立统一评测流程，固定题目、提示、解码与判分设置，统计数学正确率、格式合规率及截断情况。

## 专业技能（可单独放入简历技能栏）

- **编程与框架：**熟悉Python，具备PyTorch及Hugging Face Transformers使用经验，能够进行训练数据处理、模型微调与评测脚本开发。
- **大模型基础：**了解Transformer架构、自注意力机制与自回归生成原理，理解SFT、LoRA/QLoRA及GRPO的基本原理。
- **后训练实践：**具备基于TRL、PEFT开展QLoRA SFT与GRPO训练的实践经验，能够配置训练参数、编写规则奖励函数，并进行训练日志与错误样本分析。
- **数据与评测：**具备数据清洗、生成解答筛选及数学推理评测经验，使用过vLLM批量生成和Math-Verify答案验证，能够开展统一评测条件下的模型效果对比。

技能表述按当前项目经历组织，不重复项目结果。LLaMA-Factory目前仅讨论过用法，不列为实操技能；Linux、Git、Docker、分布式训练及英语成绩等，待本人补充实际掌握情况后再决定是否加入。“熟悉Python”和“能够开发脚本”也应以本人实际编写、阅读和修改能力为准，不将AI协助产出的代码自动等同于独立开发能力。

本节参考[郑辉公开简历的专业技能分组](https://www.huii.top/zhenghui_CV_20260313.pdf)、[Tathagata Debnath的Technical Skills](https://tathagatadebnath.com/)及[Pranjul Rathour的技能栏](https://pranjulrathour.scult.in/resume)，借鉴分类、关键词和简洁表达，不借用其资历与技能范围。

## 数字与经历口径（不放进简历正文）

- 505条来自已有训练记录；OpenMathReasoning-mini来源由用户补充确认，原始抽样方法仍需追溯。
- 第三轮“Base每题采样4次筛选自身轨迹”及约65%的MATH正确率来自用户最新陈述，尚未与本地某一份训练清单、checkpoint和评测文件对齐。简历提交前应能展示对应记录。
- 第三轮最终训练条数尚未确认。600道MATH候选、200道补充，以及后来讨论的1200道GSM8K均不写成已经使用的数据。
- 旧归档的SFT v3→GRPO为另一条实验记录，包含GSM8K与MATH混合数据；不将其3072题、1536步或历史评测值写成新版第三轮SFT之后的结果。
- 12条漏判、11个有效题组属于历史训练后的奖励审计。离线修复不等于重新训练，不能称为因此提升了模型正确率。
- GRPO测试正确率78%来自用户本轮补充，正文按此前MATH Level 2–3主测试语境写入。与SFT约65%相比的约13个百分点是算术差值；提交前需对齐同一测试清单、checkpoint、提示、解码与评分版本的评测记录，当前未在本地独立核验该结果。不采用早期讨论的“65%→80%”或“SFT提高1个百分点”作为本版结论。
- 60%→96%的格式指标暂不写入正文，待确认它属于上述第三轮checkpoint及同一评测口径。
- 不写“解决奖励欺骗”“消融证明最佳超参数”“加入难题突破平台期”：目前对话中的这部分包含面试假设和排查方案，不能当作已完成实验。
- 本文按项目工作表述；面试介绍应区分本人设计与核验、AI工具协助、上游框架实现，不将GRPO算法或框架实现归为个人原创。

## 参考与本地证据

写作采用具体动作、方法与可核对结果，避免堆叠算法名或未确认提升。参考[Harvard简历指南](https://careerservices.fas.harvard.edu/resources/create-a-strong-resume/)；训练流程术语参考[Open-R1](https://github.com/huggingface/open-r1)。引用支持写作方式与技术背景，不证明个人项目成果。

2026-10-14进一步参考以下公开材料，按“一句项目目标与结果＋三条具体贡献”压缩正文：

- [公开个人简历：SFT、GRPO与vLLM实习经历](https://logcreative.tech/about/parts/resume/)：观察如何将工具与具体交付关联，不能据公开简历判定其录用效果。
- [公开个人简历：后训练与Agent算法经历](https://zeng-yirong.github.io/cv/resume-A-general.html)：参考问题、方法、结果的组织方式，不借用其成果或资历。
- [上海人工智能实验室：垂域大模型训练算法实习生](https://www.shlab.org.cn/joinus/detail/7678626977301760294?mode=campus&subject=7538649065812363566)：岗位列出训练数据构建、训练流程、过程诊断、实验分析和模型评测，作为本简历选取重点的依据。

本次压缩移除了初始505条数量、每次SFT迭代顺序、EOS监督细节和标准GRPO公式要素的罗列。这些仍适合面试展开，但不比“为何改数据、如何选RL题、如何验证结果”更值得占用正文篇幅。保留78%作为用户提供的结果；它不是本次网页检索或本地运行核验出的新结论。

- [历史训练归档](../deliverables/SFTv3_GRPO_1536_training_handoff/TRAINING_REPORT.md)
- [实验索引](TRAINING_EXPERIMENT_INDEX.md)
- [项目面试介绍与历史诊断](INTERVIEW.md)
- [当前MATH数据方案](MATH_L2_L3_DATA_CONFIGURATION.md)
- [初始SFT数据来源](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)
