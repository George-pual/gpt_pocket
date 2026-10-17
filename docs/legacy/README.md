# Qwen3-4B Reasoning Lab

面向单卡资源约束的数学推理后训练项目：**QLoRA SFT → GRPO → 评分器审计 → 配对评测**。

项目目标是改善规定格式下的数学解答、正常结束与生成效率，并检验是否保留最终答案正确率。这里保留失败实验及其边界；没有将格式提升直接解释为推理能力提升。

## 快速体验：不需要 GPU

Python 3.10+，在项目根目录运行：

```powershell
python -m venv .venv-portfolio
.venv-portfolio/Scripts/python -m pip install -e .
.venv-portfolio/Scripts/python -m unittest discover -s tests -v
.venv-portfolio/Scripts/r1-lab report --root . --output reports/local-review
```

已有可写 Python 环境时可直接 `python -m pip install -e .`。Linux 的虚拟环境可执行文件位于 `.venv-portfolio/bin/`。

打开 `reports/local-review/index.html`。输出还包括可供程序读取的 `report.json`。目录已存在时拒绝覆盖，请更换输出目录。历史实验原始 JSONL 只在本地工作区中；公开代码不附带权重和数据。没有原始文件时仍可运行单元测试，真实数据集成测试会明确跳过，报告命令会报缺文件。

也可以直接打开本仓库已生成的 [离线评测页](../../reports/portfolio/index.html) 和 [汇总数据](../../reports/portfolio/report.json)。这两份文件只含汇总指标、来源相对路径及哈希，不含完整题面、模型回答或权重，无外部网页依赖。

## 已有实验与真实结论

下表为保存回答按 `audited-final-answer-v3.2` 的自动判对率，未确认保留在分母中；不是人工逐题认证的真实正确率。

| 模型 | MATH L2–3 /2024 | GSM8K /1319 | MATH 平均 tokens |
|---|---:|---:|---:|
| Base | 74.70% | 85.44% | 427.7 |
| SFT v3 | 65.71% | 80.14% | 365.5 |
| GRPO-1024 | 72.28% | 85.90% | 279.4 |
| GRPO-1280 | 71.84% | 85.75% | 291.1 |
| GRPO-1536 | 72.58% | 85.75% | 275.3 |
| SFT v4 | 68.48% | 未评测 | 233.9 |

相对 SFT v3，GRPO-1536 的 MATH 自动判对率提高 **6.87 个百分点**，截断率从 **5.63% 降至 2.57%**，格式且 EOS 从 **91.55% 升至 96.49%**。但其 MATH 判对率仍低于 Base **2.12 个百分点**。GSM8K 相对 Base 仅多判对 4 题，不能据此宣称稳定超越 Base。

MATH 2024 已包含旧 200 题。旧题已用于实验反馈，报告将剩余 1824 题独立展示。已排除 1 道已知训练精确重叠，未做语义去重，无法排除预训练污染。

## 技术工作

- **单卡后训练**：Qwen3-4B-Base，NF4 QLoRA，LoRA r=32/alpha=64；v3 用 505 条样本冷启动。GRPO 两阶段共 1536 optimizer steps、3072 个不同题组、12288 条回答。
- **采样与训练一致性**：NF4 策略训练与 BF16 vLLM 生成交替执行，流式同步解量化权重与 LoRA delta；检查冻结参数未被修改，记录采样/训练 log-prob 差异并启用重要性修正。
- **奖励与有效信号诊断**：每题 4 条回答，正确性奖励加最高 0.1 格式奖励；分析组内标准化、全同奖励组和截断组屏蔽。明确区分策略奖励梯度与 KL 梯度。
- **评分器审计**：历史训练奖励与事后修订评分分开；覆盖多答案、单位和答案提取边界，不把重评分称作训练效果提升。
- **本次新增工程层**：可安装 CLI、严格 schema/类型/题序检查、SHA256 输入冻结、数据集规范化精确重叠审计、按 ID 配对、Wilson 区间、离线结果页和 CPU CI。

固定工作量生成 benchmark 中，vLLM 508.14 tokens/s，Transformers SDPA 177.66 tokens/s，约 **2.86× 生成吞吐**。此数不含评分、反向传播、同步与初始化，不能写成端到端训练加速。

## 使用与目录

```powershell
# 训练/验证/测试去重检查；发现重复或重叠退出码为 1
r1-lab audit --split train=path/to/train.jsonl --split validation=path/to/validation.jsonl --split test=path/to/test.jsonl
```

输入题面支持 `question`、`problem`、`messages`/`prompt` 内的 user 内容。该检查只验证规范化精确重叠，不验证数学正确性或语义泄漏。工具错误退出码为 2。

```text
src/r1_lab/          新增：独立、无 GPU 依赖的检查与报告工具
configs/            新增：固定原始评测文件与 SHA256
tests/              新增：负例、配对、数值及真实归档一致性检查
docs/               新增：实验论证、面试讲稿、复现边界与下一轮实验
reports/portfolio/  新增：从现有回答生成的离线评测页
vllm_speed_tools/   已有：后训练、奖励、吞吐及审计脚本
closure_grpo_experiment/  已有：训练公共模块和历史实验
datasets/           本地：数据及逐题评测，公开仓库默认忽略
deliverables/       本地：冻结训练交付包，公开仓库默认忽略
```

原始脚本和实验归档保持原样。新工程层消费已保存评分，**不会重新推理、重新判分或启动训练**。完整流程见 [项目 Pipeline](PROJECT_PIPELINE.md)，SFT与GRPO的现象、分析、处理及效果见 [训练问题排查复盘](TRAINING_RECORD.md)，运行配置清单见 [实验索引](TRAINING_EXPERIMENT_INDEX.md)，原因判断与消融设计见 [训练诊断](TRAINING_DIAGNOSIS.md)。训练复现的环境及限制见 [复现说明](REPRODUCIBILITY.md)，论证过程见 [实验故事与面试讲稿](INTERVIEW.md)，后续可验证改进见 [实验计划](NEXT_EXPERIMENTS.md)。

参考开源项目重新设计的数据内容分层、难度分层及SFT/GRPO选数方案见 [数据设计方案](DATA_DESIGN_PROPOSAL.md)。其中规模、比例与筛选规则是待验证的起始配置，尚未构建或训练，不替代历史记录。

从“Base答对而SFT答错”分析、定向补数据，到GRPO难度混合、日志诊断和关键超参数消融，集中见 [项目推进主线](PROJECT_DEVELOPMENT_STORY.md)。包含本地2024题配对复算及8个代表退步题复核，区分已有证据与待验证实验。

当前以MATH L2–3与尽量保持Base正确率为目标的优先路线见 [任务划分与SFT能力保持](SFT_ACCURACY_PRESERVATION.md)：使用同等级训练题、筛选Base适配的合格示范，再验证小幅更新；不默认继续堆数据或沿退化adapter追加训练。

与这条路线衔接的 [当前GRPO数据方案](GRPO_DATA_PLAN.md) 统一了来源划分、G4难度筛选、3894生成预算、512题试运行及最多1024题合格池，区分待执行方案与历史实际用量。

## 参考项目

| 项目 | 参考内容 | 在本项目中的取舍 |
|---|---|---|
| [Hugging Face Open-R1](https://github.com/huggingface/open-r1) | 配置、训练、评测组织 | 补齐结果入口与复现文档，不声称复现完整 DeepSeek-R1 |
| [TRL v0.23.1](https://github.com/huggingface/trl/tree/v0.23.1) | SFT/GRPO trainer 与配置 | 保留历史版本，避免私有方法改写与新 API 不兼容 |
| [TinyZero](https://github.com/Jiayi-Pan/TinyZero) | 最小实验和过程展示 | 仅参考展示方式；上游已声明不再积极维护 |

参考日期：2026-10-12。新增 CPU 工具独立实现；历史代码中的上游实现及其许可证需单独核对后再公开，不对整个工作区擅自添加统一许可证。
