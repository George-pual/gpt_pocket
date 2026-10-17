# 复现范围

## CPU 工程层

`python -m pip install -e .` 后可使用 `r1-lab`。无运行时第三方依赖。CI 在 Linux Python 3.10/3.12 上运行 CPU 单元测试；本地有完整数据时另外验证新汇总与冻结的历史汇总一致。CI workflow 已添加，是否在 GitHub 成功运行须以上传后的 Actions 记录为准。

报告只重算保存 verdict 的统计，不运行 Math-Verify，也不宣称重新验证了最终答案。SHA256 能发现输入在基准冻结后变化，不能证明输入本来正确或实验真实；首次冻结的依据仍是原实验日志和 manifest。

`configs/evidence_registry.json` 固定 11 份模型回答和 1 份 MATH 子集成员文件。`tools/freeze_evidence.py` 是显式初始化工具，已有 registry 时拒绝覆盖。新结果应建立单独版本的 registry，不能为了消除 hash 报错悄悄更新基准。

`r1-lab report` 验证 scorer schema、布尔值类型、EOS/截断冲突、非负整数 token、唯一 ID、相同题目及参考答案。配对按 ID 完成，文件顺序改变不影响比较。MATH 分为 all/previous200/remaining1824，不重复累计。公开汇总不导出题面和模型回答。

## 历史 GPU 训练

历史代码需要 Linux CUDA 环境及模型权重；当前 Windows CPU 验证不等同于 GPU 训练复现。不要直接把全部库升级到最新版，历史代码覆写 TRL 私有方法，存在版本兼容约束。

| 内容 | 历史已知配置 |
|---|---|
| 模型 | Qwen3-4B-Base / 冻结 NF4 Base＋LoRA |
| TRL | 0.23.1 |
| vLLM | 0.10.2 |
| 数学验证 | Math-Verify 0.8.0，历史奖励与事后 scorer 分离 |
| LoRA | r=32，alpha=64，dropout=0 |
| SFT v3 | 505 条，lr=2e-5，有效 batch=16，32 optimizer steps |
| GRPO | 每题 4 条回答，beta=0.04，max_completion_length=3894 |
| 阶段 1 | 1024 steps，lr=5e-6 线性衰减 |
| 阶段 2 | 512 additional steps，lr=2e-6，重新初始化 optimizer/scheduler |

其他依赖的精确版本应从原始 `run_manifest.json` 及对应归档中读取；没有核对的版本不补写成“已验证 requirements”。完整训练入口与公共模块在本地 `deliverables/SFTv3_GRPO_1536_training_handoff/START_COMMANDS.md` 及 `code/historical`，其中路径为当时机器路径。恢复训练前需要真实模型、adapter、数据、GPU 和重新映射的路径。

v3 的第二阶段是加载 adapter 后重置优化器与调度器，不是无缝恢复 optimizer state。完整 GRPO 权重未包含在本地训练交付包中，CPU 报告可以复现，端到端训练目前仍依赖额外资产。

## 发布与来源

根目录原先不是 Git 仓库。本次未自动发布到 GitHub。`.gitignore` 默认排除原始数据、模型、归档、实验结果目录与 SSH 文件。发布前仍需检查准备上传的文件；ignore 不会移除已经被 Git 跟踪的文件。

参考了 [Open-R1](https://github.com/huggingface/open-r1) 的项目组织、[TRL 历史版本](https://github.com/huggingface/trl/tree/v0.23.1) 的训练接口和 [TinyZero](https://github.com/Jiayi-Pan/TinyZero) 的实验展示。没有复制上游结果作为本项目指标。历史快照可能包含上游代码，应在具体分发前保留其许可证与来源。
