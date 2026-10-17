# Qwen3 Math Reasoning Lab

Qwen3-4B-Base 数学推理后训练项目：**审核数据 → QLoRA SFT → vLLM GRPO → 评分器审计 → 统一评测**。

本仓库整理实际运行的训练、答案核验、数据筛选和实验记录。主结果来自同一官方 MATH Level 2–3 测试集的 2024 题，原始模型回答按固定 v4 评分器统一核验。

## 最新结果

| 模型 | 判对/总数 | 自动最终答案正确率 |
|---|---:|---:|
| Qwen3-4B-Base | 1417/2024 | 70.01% |
| SFT（367条直接审核示范） | 1570/2024 | 77.57% |
| 第一轮 GRPO（699题） | 1628/2024 | 80.43% |
| 第二轮 GRPO（549题 replay） | 1724/2024 | **85.18%** |

第二轮相对第一轮：176 题从未通过变为通过，80 题从通过变为未通过，**净增96题、4.74个百分点**。原 v2 分数独立保留；评分器修复带来的涨分没有算入这一差值。

统一标准：固定2024题、同一提示词与Base权重，n=1、temperature=0、max_tokens=3894、seed=3407；Math-Verify 0.8.0、ANTLR 4.13.2和固定v4源码。测试不调用API，无法提取/解析/核验的回答仍留在分母中。历史复评在Windows，本轮原评分在Linux；历史复评无记录超时，Base有1份核验异常按未通过保留。这里衡量最终答案得分，不是逐步推导认证或多次独立训练的因果结论。

详细结果见 [实验记录](docs/EXPERIMENTS.md) 和 [机器可读汇总](reports/math_l2_l3_v4_summary.json)。

## 本项目完成了什么

- 从600道MATH题的Base自生成回答出发，直接审核得到367条SFT示范，进行冷启动。
- 用模型每题生成4答的结果筛选GRPO训练题；第一轮699题，后续1200题重筛后采用全部549道有效题。
- 第二轮使用3题12答的vLLM生成批次，训练microbatch=2、累积2、缓存6个微批次，共549次更新。
- 数学正确奖励1；数学正确且原格式合格加0.1；错误回答奖励0。API仅在核验不明确时摘取原文答案，再由本地Math-Verify核验。
- 诊断实际数学对错信号、格式信号、屏蔽组、KL和梯度；取回模型/数据/代码/日志共2631文件并逐文件SHA256校验。

## CPU检查

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[verify]"
python -m unittest discover -s tests -v
```

不需要GPU或API密钥。`verify`可选依赖用于数学核验回归；纯CPU数据去重/证据工具本身只用标准库。本次本地运行46项测试，44项通过；1项需要PyTorch、1项需要本地原始归档，明确跳过。155个Python文件语法检查通过。

## GPU复现

训练和推理入口、数据schema、批次配置及限制见 [复现说明](docs/REPRODUCING.md)。需要自备原始Qwen3-4B-Base、合格训练/验证/测试数据及兼容CUDA环境。仓库不附模型权重或完整题库，不能从本仓库单独重建人工审核的367条示范；这是代码和汇总结果的可运行归档，不是完整可下载实验数据集。

## 目录

```text
src/r1_lab/                 CPU证据检查、按ID配对、精确去重及旧报告CLI
tools/                      SFT、GRPO、vLLM评测、评分器及审计入口
tests/                      数学核验、奖励门控、API摘取和批次回归
configs/                    固定评分合同、历史运行配置及无密钥API模板
reports/                    最新统一评分与训练信号汇总；portfolio为较早实验
docs/                       最新实验、复现边界、已知问题
docs/legacy/                2026-10-15之前的实验说明，独立口径
closure_grpo_experiment/     历史训练公共模块与核验回归来源
vllm_speed_tools/            历史吞吐、训练和审计工具
```

较早实验采用不同数据、模型阶段和评分器，详见 [旧项目说明](docs/legacy/README.md)。旧portfolio页不代表上表的367/699/549主线，不能混合比较。

## 奖励与已知限制

[评分机制与已知问题](docs/SCORING_AND_LIMITATIONS.md)保留此次已发现的问题：部分API摘取结果与Math-Verify衔接不佳，5份空回答被错误升级为未确认并导致整组屏蔽。历史实现保留以解释已完成训练；没有用事后修复覆盖历史奖励或伪造重训结果。

本次整理不改变冻结评分器SHA。完整回答、权重、API缓存及服务器状态留在本地，上传内容只含源码、模板与汇总证据。第三方依赖与来源见 [依赖说明](docs/THIRD_PARTY.md)。
