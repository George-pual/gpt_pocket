# 运行与复现边界

## CPU

安装`python -m pip install -e ".[verify]"`后，运行`python -m unittest discover -s tests -v`。不需要API/GPU。
数学核验入口`tools/math_verification.py`，加载器`tools/scorer_runtime_guard.py`验证实际导入文件、函数来源、SHA256、schema和依赖版本。固定源码SHA为`26565fd5639523aaad27b7211d32623d7a5a95605d605e1ee23dec69c412bec8`。

`r1-lab audit --split train=path/to/train.jsonl --split test=path/to/test.jsonl`进行规范化精确去重；不证明语义无泄漏。`r1-lab report`是旧实验报告工具，依赖configs/evidence_registry.json中未随仓库分发的本地文件；它不会生成本轮549题报告。

## 训练环境

GPU训练历史锚点：Linux、TRL 0.23.1、vLLM 0.10.2、NF4 QLoRA、BF16 vLLM。requirements-gpu.txt不是完整环境锁；CUDA/PyTorch、驱动、Transformers及其他依赖需配置为相互兼容的版本。历史GPU训练已完成，本次整理只验证CPU回归，不声称重新验证完整GPU环境。

## 数据结构

训练/验证JSONL由用户自行准备。SFT入口需字段`row_id, prompt, completion, answer`，prompt为system/user消息数组，completion包含`<end_working_out><SOLUTION>答案</SOLUTION><|endoftext|>`；示范需人工审核且不能截断。SFT会检查训练/验证ID无重叠并验证token监督mask。

GRPO入口需字段`row_id, problem, prompt, answer`，每题在线生成4答。官方测试入口需`row_id, problem, answer, level, type`。使用同一固定测试题集与参考答案，开发选择和测试评估应分开。

## SFT

```bash
python tools/train_math_self600.py --model /path/to/base --data /path/to/sft_train.jsonl --validation /path/to/sft_validation.jsonl --output /path/to/fresh_sft --learning-rate 5e-6 --steps 64
```

## 第二轮GRPO

```bash
python tools/train_direct_grpo.py --model /path/to/base --adapter /path/to/previous_grpo_adapter --data /path/to/train549.jsonl --output /path/to/fresh_grpo --questions-per-generation 3 --train-microbatch 2 --gradient-accumulation 2 --vllm-memory 0.60 --expected-grader-sha256 26565fd5639523aaad27b7211d32623d7a5a95605d605e1ee23dec69c412bec8 --expected-reward-sha256 0cecbfaaa21dac778b507c4dc054c991065acac5111784983d6de89ab593d9a5
```

可选API摘取：从configs/api_training.example.json复制本地配置，设置HTTPS端点，密钥只放`MATH_REVIEW_API_KEY`环境变量，追加`--api-review-config /path/to/local_config.json`。原训练使用DeepSeek只摘取原文，禁止API解题或替换答案，标准答案不发给API；是否发送原题/回答由实际运行者授权。默认上述命令不调用API；带API的历史549题运行与不带API的新训练不等价。

## 官方评测

```bash
python tools/run_math_stage.py --model /path/to/base --data /path/to/official_test.jsonl --output /path/to/fresh_eval --arms '{"grpo":"/path/to/adapter"}' --expected-grader-sha256 26565fd5639523aaad27b7211d32623d7a5a95605d605e1ee23dec69c412bec8
```

该入口不调用API。不同模型按row_id配对，所有无法核验回答保留在固定分母中。历史重评入口和finalize工具保留原本地归档布局要求；它们不是开箱即用的数据下载器。

## 内容范围

本仓库不分发Base/adapter权重、完整MATH题库、原始逐题回答、完整训练日志、API请求缓存或服务器凭据。汇总结果可检查，原始逐题证据需对应本地归档；不能仅凭本仓库独立审核全部推导。数据和模型使用条件应分别遵循其来源要求。
