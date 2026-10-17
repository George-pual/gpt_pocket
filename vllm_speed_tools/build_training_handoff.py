"""Build a credential-free, source-backed SFT/GRPO training handoff."""
import hashlib,json,pickletools,shutil,tarfile,zipfile
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'datasets/grpo_candidates_gsm8k_math_l2_l3'
T=D/'training_audit_1536'
OUT=ROOT/'deliverables/SFTv3_GRPO_1536_training_handoff'
PHASES=['E1_sft_v3_half_math1024_fast_v2','E1_sft_v3_expanded_continue1536_v1']
def read(p):return [json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rs):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs),encoding='utf-8')
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def archive_select(src,dst,names=None):
    # Selected regular files only; never extract archive-provided paths directly.
    with tarfile.open(src) as t:
        for m in t.getmembers():
            if not m.isfile():continue
            name=Path(m.name).name
            if names is not None and name not in names:continue
            if name in {'deployment_status.json','screening_server_run.json'}:continue
            target=dst/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(t.extractfile(m).read())

def main():
    if OUT.exists():raise RuntimeError('Refuse overwrite of handoff directory')
    OUT.mkdir(parents=True)
    # Actual SFT args, read as pickle opcodes without executing pickle payloads.
    argsfile=ROOT/'sft_closure_results/models/E1_sft_v3/adapter/training_args.bin'
    with zipfile.ZipFile(argsfile) as z:
        ops=list(pickletools.genops(z.read(next(n for n in z.namelist() if n.endswith('data.pkl')))))
    wanted={'per_device_train_batch_size','gradient_accumulation_steps','learning_rate','num_train_epochs',
            'max_length','warmup_ratio','weight_decay','seed','data_seed','max_grad_norm',
            'completion_only_loss','packing','bf16','gradient_checkpointing'}
    extracted={}
    for i,(op,arg,pos) in enumerate(ops):
        if isinstance(arg,str) and arg in wanted:
            for nextop,value,_ in ops[i+1:i+5]:
                if nextop.name in ['BININT','BININT1','BININT2','BINFLOAT','NEWTRUE','NEWFALSE','NONE']:
                    extracted[arg]=True if nextop.name=='NEWTRUE' else False if nextop.name=='NEWFALSE' else value
                    break
    extracted.update(source='Static pickle opcode inspection; no unpickling executed',source_sha256=digest(argsfile))
    save(OUT/'configs/sft_actual_args.json',extracted)
    assert extracted['per_device_train_batch_size']==4 and extracted['gradient_accumulation_steps']==4
    for name in ['adapter_config.json','chat_template.jinja','tokenizer_config.json','special_tokens_map.json']:
        copy(argsfile.parent/name,OUT/'configs'/('sft_'+name))
    copy(argsfile,OUT/'configs/sft_training_args.bin')
    copy(ROOT/'sft_closure_results/models/E1_sft_v3/download_manifest.json',OUT/'configs/sft_adapter_manifest.json')
    merged=ROOT/'sft_quality_audit_v2/merged_supplied_answers'
    for name in ['train_messages_505.jsonl','train_merged_505.jsonl','questions_answers_505.txt','base_sft_readiness.json','SPECIAL_ANSWERS_REVIEW.md','FORMAT_CHECK.md']:
        copy(merged/name,OUT/'data/sft505'/name)
    copy(ROOT/'closure_grpo_experiment/inputs/validation.jsonl',OUT/'data/sft505/validation_57.jsonl')
    for folder,src in [('initial_pool_3816',D/'half_math_v3'),('expanded_pool_7527',D/'expanded_math_v3'),('continue_selected_1024',D/'expanded_math_v3/continue1536')]:
        for name in ['train_candidates.jsonl','train_candidate_scores.jsonl','validation.jsonl','manifest.json']:
            copy(src/name,OUT/'data/grpo'/folder/name)
    for ds in ['gsm8k','math']:
        for name in ['train.jsonl','manifest.json']:
            copy(ROOT/'datasets'/ds/name,OUT/'data/source_datasets'/ds/name)
    copy(ROOT/'datasets/gsm8k/test.jsonl',OUT/'data/evaluation/gsm8k_test_1319.jsonl')
    for name in ['questions.jsonl','manifest.json']:
        copy(ROOT/'datasets/math/test_l2_l3_200'/name,OUT/'data/evaluation/math_fixed200'/name)
    for name in ['questions.jsonl','questions_all_official.jsonl','overlaps_training.jsonl','manifest.json','gold_scoring_preflight.json','README.md']:
        copy(ROOT/'datasets/math/test_l2_l3_full'/name,OUT/'data/evaluation/math_full_l2_l3'/name)
    pool={x['row_id']:x for x in read(D/'expanded_math_v3/train_candidates.jsonl')}
    ledger=[];consumed=[];phaseinfo=[];seen=set()
    for j,phase in enumerate(PHASES):
        src=T/phase/'B_components';rollouts=read(src/'rollouts.jsonl');phaseids=[]
        for x in rollouts:
            if x['row_id'] in seen:continue
            seen.add(x['row_id']);phaseids.append(x['row_id']);consumed.append(pool[x['row_id']])
            ledger.append(dict(row_id=x['row_id'],phase=phase,cumulative_generation_step=x['step']+(1024 if j else 0),
                               dataset='GSM8K' if x['source']=='gsm8k' else 'MATH '+pool[x['row_id']]['level']))
        write(OUT/'data/grpo/actually_consumed'/f'phase{j+1}_questions.jsonl',[pool[k] for k in phaseids])
        phaseinfo.append(dict(phase=phase,unique_questions=len(phaseids),answers=len(rollouts),counts=dict(Counter(x['dataset'] for x in ledger if x['phase']==phase))))
        for name in ['grpo_config.json','run_manifest.json','experiment.json','format_reward_config.json','completion_summary.json','selected_train_ids.json']:
            copy(src/name,OUT/'configs'/phase/name)
        for name in ['rollouts.jsonl','trainer_log.json','metrics.csv','vllm_sync.jsonl','padding_audit.jsonl']:
            if (src/name).exists():copy(src/name,OUT/'logs'/phase/name)
        copy(T/phase/'B_components.log',OUT/'logs'/phase/'stdout.log')
        for cp in src.glob('checkpoint-*/trainer_state.json'):
            copy(cp,OUT/'logs'/phase/cp.parent.name/'trainer_state.json')
    assert len(consumed)==3072 and [x['unique_questions'] for x in phaseinfo]==[2048,1024]
    write(OUT/'data/grpo/actually_consumed/all_3072_questions.jsonl',consumed)
    write(OUT/'data/grpo/actually_consumed/exposure_ledger.jsonl',ledger)
    save(OUT/'data/grpo/actually_consumed/summary.json',phaseinfo)
    for name in ['combined_metrics_0_1536.csv','FULL_GRPO_TRAINING_0_1536.log','audit_summary.json','ANALYSIS.md']:
        copy(T/name,OUT/'logs'/name)
    for folder in ['all_rollout_advantage_audit_v2','correctness_groups_five_bins','learning_rate_decay_audit']:
        for p in (T/folder).iterdir():
            if p.is_file():copy(p,OUT/'audits'/folder/p.name)
    # Historical training code: frozen deployment archives, not today's scorer.
    hist=OUT/'code/historical'
    for name in ['common.py','train_closure_grpo.py','closure_rewards.py','rewards_article.py','dense_rollout.py','prepare_bc.py']:
        copy(ROOT/'closure_grpo_experiment'/name,hist/'shared'/name)
    archive_select(ROOT/'sft_quality_audit_v2/sft_v3_launch_bundle.tar.gz',hist/'sft',
                   {'train_reviewed_sft.py','common.py','prepare_bc.py','math_verification.py','SFT_FROM_BASE.md'})
    # Use archived v1 verifier even though the working source is now v2.
    copy(hist/'sft/math_verification.py',hist/'shared/math_verification.py')
    needed={'train_reward_scale128_v3.py','train_reward_pilot_v3.py','trimmed_pilot_trainer.py',
            'batched_pilot_trainer.py','safe_vllm_trainer.py','pilot_rewards_v3.py','dense_format_reward_v3.py',
            'eval_reward_pilot_v3.py','prepare_v3_half_math.py','prepare_v3_expanded_pool.py','prepare_v3_continue1536.py',
            'screen_sft_difficulty.py','screen_grpo_candidates_v3.py','run_continue1536_v3.sh','choose_generation_batch.py'}
    archive_select(D/'expanded_math_v3/continue1536/deployment.tar.gz',hist/'phase2',needed)
    for p in (hist/'phase2').glob('*.py'):copy(p,hist/'phase1'/p.name)
    archive_select(D/'half_math_v3/deployment.tar.gz',hist/'phase1',{'train_reward_scale128_v3.py'})
    archive_select(D/'half_math_v3/fast_tools.tar.gz',hist/'phase1',
                   {'train_reward_scale128_v3.py','batched_pilot_trainer.py','choose_generation_batch.py','run_half_math1024_fast_v3.sh'})
    # Archived dependencies sourced from phase2 are disclosed; top-level run
    # manifests, not file names, determine the historical run settings.
    for name in ['math_verification.py','pilot_rewards_v3.py','rescore_math_v2.py','audit_all_rollout_advantages_v2.py',
                 'split_correctness_groups_0_to_4.py','analyze_lr_decay_1536.py','eval_gsm8k_base_sftv3.py','eval_math_full_l2_l3.py']:
        copy(ROOT/'vllm_speed_tools'/name,OUT/'code/post_training_v2_and_evaluation'/name)
    copy(T/'trl_grpo_trainer_v0.23.1_reference.py',OUT/'code/reference/trl_grpo_trainer_v0.23.1.py')
    configs=[json.loads((T/p/'B_components/grpo_config.json').read_text(encoding='utf-8')) for p in PHASES]
    keys=['per_device_train_batch_size','gradient_accumulation_steps','generation_batch_size','steps_per_generation','num_generations',
          'num_iterations','learning_rate','lr_scheduler_type','warmup_steps','max_steps','max_prompt_length','max_completion_length',
          'temperature','top_p','top_k','min_p','beta','epsilon','epsilon_high','max_grad_norm','optim','weight_decay',
          'scale_rewards','loss_type','importance_sampling_level','vllm_mode','vllm_gpu_memory_utilization','vllm_enable_sleep_mode',
          'vllm_importance_sampling_correction','vllm_importance_sampling_cap','seed','data_seed','save_only_model']
    paramtable='| 参数 | 0–1024 步 | 1024→1536 步 |\n|---|---|---|\n'
    for k in keys:paramtable+=f'| `{k}` | `{configs[0].get(k)}` | `{configs[1].get(k)}` |\n'
    versions=json.loads((T/PHASES[0]/'B_components/run_manifest.json').read_text(encoding='utf-8'))['versions']
    save(OUT/'configs/versions.json',dict(versions,vllm='0.10.2'))
    commands='''# 历史训练命令（参考路径，运行前映射到本机）

这是历史 v1 奖励的复现命令，使用 code/historical；如使用 post_training_v2 中评分器，属于新的训练实验。不要覆盖已有训练目录。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HANDOFF=/path/to/SFTv3_GRPO_1536_training_handoff
export PY=/root/autodl-tmp/vllm_speed_env/bin/python
export MODEL=/root/autodl-tmp/qwen3_hf/model
export RUNS=/path/to/new_runs
export PYTHONPATH="$HANDOFF/code/historical/shared:$HANDOFF/code/historical/sft"

$PY "$HANDOFF/code/historical/sft/train_reviewed_sft.py" --model "$MODEL" \
  --data "$HANDOFF/data/sft505/train_messages_505.jsonl" \
  --validation "$HANDOFF/data/sft505/validation_57.jsonl" \
  --microbatch 4 --gradient-accumulation 4 --max-length 2048 --output "$RUNS/E1_sft_v3"

# 以下以历史 SFT adapter 为输入，保留历史哈希校验。
# 重新 SFT 训练得到的 adapter 未必位级相同；若更换输入，需更新校验并标为新实验。
export SFT=/root/autodl-tmp/qwen3_hf/runs/E1_sft_v3/adapter
export PYTHONPATH="$HANDOFF/code/historical/phase1:$HANDOFF/code/historical/shared"
$PY "$HANDOFF/code/historical/phase1/train_reward_scale128_v3.py" --model "$MODEL" --adapter "$SFT" \
  --data-dir "$HANDOFF/data/grpo/initial_pool_3816" --format-weight .1 --format-mode components \
  --prepared-data --microbatch 2 --generation-batch 32 --vllm-memory .5 --steps 1024 \
  --checkpoint-steps 256,512,1024 --output "$RUNS/GRPO_1024"

export PYTHONPATH="$HANDOFF/code/historical/phase2:$HANDOFF/code/historical/shared"
$PY "$HANDOFF/code/historical/phase2/train_reward_scale128_v3.py" --model "$MODEL" \
  --adapter "$RUNS/GRPO_1024/B_components/checkpoint-1024" --reference-adapter "$SFT" \
  --expected-adapter-sha256 3a159f8045efc10bc1aae34901abd200751a16936d2798cfcce258101874112c \
  --base-step 1024 --learning-rate 2e-6 --save-training-state \
  --data-dir "$HANDOFF/data/grpo/continue_selected_1024" --format-weight .1 --format-mode components \
  --prepared-data --microbatch 2 --generation-batch 32 --vllm-memory .5 --steps 512 \
  --checkpoint-steps 256,512 --output "$RUNS/GRPO_continue1536"
```

注意：上述第一阶段输出目录为 GRPO_1024，本训练脚本不会自动创建 B_components 子目录；因此续训 adapter 路径应为 `$RUNS/GRPO_1024/checkpoint-1024`。历史原始启动脚本目录布局保存在各阶段代码中，以其 run_manifest.json 的 output 字段为准。
'''
    # Correct illustrative output path directly; retain a simple layout.
    commands=commands.replace('$RUNS/GRPO_1024/B_components/checkpoint-1024','$RUNS/GRPO_1024/checkpoint-1024')
    commands=commands[:commands.index('\n注意：上述第一阶段')]+'''\n这些命令重现配置与流程，不保证随机采样、GPU 算子和软件环境差异下位级一致。两阶段均是重新构建 trainer；第二阶段加载 adapter 权重，不传 resume_from_checkpoint，故优化器和调度器重新初始化。\n'''
    (OUT/'START_COMMANDS.md').write_text(commands,encoding='utf-8')
    report='''# SFT v3 → GRPO 1536：训练交付说明

这份资料描述实际已经完成的训练，依据运行配置、启动归档、逐题 rollout 与逐步指标。它不是下一轮训练计划。v3 是实验版本：Qwen3-4B-Base + 从零初始化的 QLoRA adapter + 用户提供的 505 条解答。

## 1. 模型、硬件与模板

- Base：Qwen3-4B-Base；训练时 Base 权重冻结，NF4 4bit、double quant、BF16 计算，仅更新 LoRA。
- LoRA：r=32、alpha=64、dropout=0、bias=none，作用于 q/k/v/o_proj 和 gate/up/down_proj。
- GRPO 硬件运行记录：单张 NVIDIA GeForce RTX 4080 SUPER，平台显存上限约 32 GB。训练和 vLLM 在同一卡交替执行。
- vLLM 0.10.2，colocate、sleep 模式，显存利用配置 0.5。训练同步代码把 NF4 解量化权重与 LoRA delta 临时相加后转 BF16 流式加载进 vLLM，不修改冻结的训练权重；每次同步重置 prefix cache。
- 自定义单轮模板，不使用 Instruct 的 enable_thinking 开关。Prompt 已包含 `<start_working_out>`，回答应生成推导、`<end_working_out><SOLUTION>最终答案</SOLUTION>`，以 `<|endoftext|>`（151643）结束。完整模板见 configs/sft_chat_template.jinja 与两个 run_manifest.json。

## 2. SFT 冷启动

数据：用户提供的 505 条题目/解答；特殊题 322、430、483 的处理说明及文本清理记录一并提供。外部既有 SFT 验证集 57 条监控 loss，不是数学 benchmark。

SFT 从 Base 新建 LoRA，不加载先前 adapter。microbatch=4、梯度累积=4、有效批量=16，lr=2e-5，1 epoch、实际 32 optimizer steps，linear scheduler、warmup_ratio=0.1、weight_decay=0.01、AdamW 8bit。max_length=2048，实际序列最长 1777，不截断；只监督 completion 和真实 EOS，prompt 与 padding 屏蔽，packing=False、gradient checkpointing=True。参数已对保存的 training_args.bin 做静态 opcode 核查，详见 configs/sft_actual_args.json。

记录：train_loss=0.8959483；57 条验证集 loss 从 0.2172062 变为 0.2123663。它们不能代替数学准确率。历史 SFT adapter SHA256 为 5461e88ae8dba6a3fb536e2bd93bdd8e6f3098a3234943eaef28bc0196017b9b。

## 3. GRPO 数据筛选和实际训练覆盖

原始训练来源：GSM8K train 7473；MATH train 7500 中 Level 2 为 1348、Level 3 为 1592。候选构建去重后 10412 题（GSM8K 7473、MATH 2939）。先由 SFT v3 用 vLLM 对每题一次请求生成 4 条，temperature=1.0，按正确数筛出 1–3/4 的混合组；归档原混合组 7631，再做验证/SFT/已有评测重叠与截断排除。

第一次选出的候选池 3816 题：GSM8K 1952、MATH L2 915、L3 949，保留全部当时合格的 MATH，并抽取 GSM8K。正式前 1024 步只实际采样其中 2048 道不同题：GSM8K 1040、L2 491、L3 517，共 8192 条回答。不是把 3816 题训练完。

之后扩大合格池至 7527 题：GSM8K 5663、L2 915、L3 949。从未被前段实际采样的题中选 1024 题：GSM8K 524、L2 250、L3 250，再训练 512 步，共 4096 条回答。两阶段累计 3072 道不同题：GSM8K 1564、L2 741、L3 767，共 12288 条回答。并未覆盖全部 MATH Level 2/3 原始训练集。

data/grpo 提供准备池及筛选回答；actually_consumed 提供实际出现的题、首次生成步数与数据类别。准备清单中的 prepared_not_launched 等字段是当时状态，最终完成状态以 configs 下 completion_summary 和日志为准。所有 3072 个题组在此两阶段只采样一次；同一批生成数据随后被拆分供多个 optimizer step 使用。

## 4. GRPO 参数与批次含义

'''+paramtable+'''

每批生成 32 条回答=8 道题×每题4条；每个 optimizer step 使用 8 条回答=2 道题，microbatch2×累积4。因此一个生成批次分配给连续4次更新，后面的更新使用该批生成时的旧策略回答。num_iterations=1 不等于每次 optimizer step 都重新生成。训练实际 batched_forward=False、score_batch=1；microbatch2仍逐回答执行前向，仅删除 masked right-padding，不应宣称已启用真正的批量前向加速。

两阶段分别使用 5e-6→0（1024步）与 2e-6→0（512步）的线性衰减，warmup_steps=2。第二阶段加载1024步adapter，优化器/调度器重置，参考策略仍固定为原 SFT v3。这不是完整 optimizer-state 无缝恢复。前段实际耗时14343.7秒（约3.98小时），续段7602.5秒（约2.11小时），不含筛选和评测。

## 5. 训练时的奖励、advantage 与屏蔽

历史奖励版本 sft-v3-pilot-numeric-units-v1：正确性0/1；GSM8K首先完整数值及白名单单位解析，MATH使用Math-Verify0.8.0。只从合法最终答案提取，不在推导中寻找碰巧匹配标准的数字，boxed fallback要求唯一。

格式分量：合法推导结束边界0.05、合法非空SOLUTION0.03、完整格式且正常结束0.02，总上限0.1。总reward=正确性+格式贡献，范围0–1.1；没有额外长度或重复惩罚。KL beta=0.04另外进入loss，不包含在总reward。

每题4回答组内：A_i=(r_i−组均值)/(样本标准差+1e-4)。GRPO token级概率比clip区间[0.8,1.2]；max_grad_norm=1；vLLM重要性修正比值=exp(旧训练侧logp−vLLM采样logp)，仅上截断到2。标准GRPO回答级advantage作用于有效token，无逐步推理正确性奖励。

只要一个回答没有EOS，就把同题四回答整个组的completion_mask和advantage置零，排除策略与KL损失；53组被屏蔽，剩3019组。全批无有效token则报错。全组总reward相同则正确性/格式策略advantage为零，但未被屏蔽的回答仍可能有KL梯度。没有在线动态过滤全部全错或全对组。

## 6. Checkpoint 与评测

前段保存累计256、512、1024；续段local checkpoint-256对应累计1280，local checkpoint-512对应累计1536。各自原始服务端路径、adapter哈希、版本见configs与START_COMMANDS.md。续段保存了optimizer状态，但这些权重/状态未下载进本交付包；这里只提供训练资料，不包含Base或GRPO权重。原SFT adapter在本地 sft_closure_results/models/E1_sft_v3/adapter。

训练后早期内部评测为独立100题×4条采样；随后官方评测为GSM8K test1319和固定MATH test200（L2/L3各100），每题一次确定性回答：temperature0、top_p1、top_k−1、max_tokens3894、EOS151643、max_model_len8192、seed3407、BF16 Base+native LoRA、相同模板。训练验证的100题来自train split但与所选GRPO训练题分离，不能称官方test。

新准备的MATH全量test：官方L2 894、L3 1131；精确规范化排除1道与训练重叠后，独立主评测2024题（L2 894、L3 1130）。所有原固定200题包含其中。该全量评测尚未启动，不要把数据预检通过写成模型已测完。GSM8K官方test仍1319题。

## 7. 历史评分与事后修复必须分开

实际训练使用历史v1，后来本地修复为v2，重评历史rollout但未回改参数或历史奖励。code/historical 保留归档v1，code/post_training_v2_and_evaluation 单列此交付整理时的v2快照；工作区还可能有其他候选评分器实验，不应混作本次历史训练配置。

v2全量审计：12条正确性奖励由0改1，11个有效题组、44条advantage改变（有效回答0.364%）。它是本次评分修复识别出的范围，不是所有潜在评分错误上限。实际训练数据、原始rewards、correct字段保持原样，事后v2字段单独保存。原训练分组0/4=313、1/4=584、2/4=833、3/4=882、4/4=460。详细阶段及数据类别分组、学习率尾段衰减审计在audits。

独立测试原结果与v2数学重评分表：

| 模型 | GSM8K /1319 | MATH原评分 /200 | MATH v2 /200 |
|---|---:|---:|---:|
| Base |1102|142|149|
| SFT v3 |1052|131|133|
| GRPO1024 |1128|132|135|
| GRPO1280 |1127|138|140|
| GRPO1536 |1128|135|137|

## 8. 怎么阅读与复现

先看本说明、configs 中原始参数和 START_COMMANDS.md；再看data/grpo/actually_consumed的3072题和logs里的12288条rollout。configs保存完整参数，本文表格只摘重要项。数据说明中的池条数、有效题组数、回答条数、optimizer步数分别计量，不可混用。

file_manifest.json列出全部交付文件大小与SHA256，verify_package.py可无GPU校验。数据包不含SSH账号密码。Base权重、GRPO checkpoint及其optimizer状态没有假装包含；如需运行，须使用同一Base与对应adapter，并映射服务器路径和准备同版本环境。phase1部分未改动公共依赖取自后续部署归档，而非逐文件证明的原始运行快照；环境和采样非确定性也使位级复现不能保证。SFT初始32步完整逐步日志本地未归档，提供其完成记录、原参数、adapter清单与训练代码；不拿其他SFT实验日志冒充。
'''
    (OUT/'TRAINING_REPORT.md').write_text(report,encoding='utf-8')
    verifier='''import hashlib,json\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nm=json.loads((root/'file_manifest.json').read_text(encoding='utf-8'))\nerrors=[]\nfor name,info in m['files'].items():\n p=root/name\n if not p.is_file() or p.stat().st_size!=info['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:errors.append(name)\nprint('FILES',len(m['files']),'ERRORS',len(errors))\nif errors:print(errors)\nraise SystemExit(bool(errors))\n'''
    (OUT/'verify_package.py').write_text(verifier,encoding='utf-8')
    manifest={str(p.relative_to(OUT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(OUT.rglob('*')) if p.is_file()}
    save(OUT/'file_manifest.json',dict(files=manifest,file_count=len(manifest),weights_included=False,credentials_included=False))
    zp=OUT.with_suffix('.zip')
    with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT.parent))
    print(json.dumps(dict(folder=str(OUT),zip=str(zp),files=len(manifest)+1,zip_bytes=zp.stat().st_size,actual_training_questions=len(consumed)),ensure_ascii=False))

if __name__=='__main__':main()
