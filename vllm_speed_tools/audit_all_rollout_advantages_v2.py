"""Reconstruct TRL 0.23.1 advantages from both archived training phases."""
import concurrent.futures
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from rescore_math_v2 import worker

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'datasets/grpo_candidates_gsm8k_math_l2_l3/training_audit_1536'
OUT=T/'all_rollout_advantage_audit_v2'
PHASES=['E1_sft_v3_half_math1024_fast_v2','E1_sft_v3_expanded_continue1536_v1']
def read(p):return [json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sign(x):return 1 if x>1e-7 else -1 if x < -1e-7 else 0

def regrade(x):
    if x['source']=='gsm8k':
        from pilot_rewards_v3 import numeric_answer
        from math_verification import extract_final_answer,completion_text
        import re
        pred,source,error=extract_final_answer(x['completion'])
        if source=='boxed' and len(re.findall(r'\\boxed\s*\{',completion_text(x['completion'])))!=1:
            error='multiple_boxed_answers'
        if error:return dict(correct=False,status='extraction_failed',prediction=pred,error=error)
        a,b=numeric_answer(pred),numeric_answer(x['gold'])
        if a is not None and b is not None:
            return dict(correct=a==b,status='correct' if a==b else 'incorrect',prediction=pred,error=None)
    return worker((x['completion'],x['gold']))

def advantage(rewards):
    mean=sum(rewards)/4
    std=math.sqrt(sum((v-mean)**2 for v in rewards)/3)  # torch.std correction=1
    return [(v-mean)/(std+1e-4) for v in rewards],mean,std

def main():
    OUT.mkdir(exist_ok=True)
    poolrows={x['row_id']:x for x in read(T.parent/'expanded_math_v3/train_candidates.jsonl')}
    rows=[];hashes={}
    for j,phase in enumerate(PHASES):
        p=T/phase/'B_components/rollouts.jsonl';hashes[str(p)]=digest(p)
        cfg=json.loads((p.parent/'grpo_config.json').read_text(encoding='utf-8'))
        assert cfg['scale_rewards']=='group' and cfg['num_generations']==4 and cfg['reward_weights']==[1.0,0.1]
        for x in read(p):
            x.update(phase=phase,cumulative_generation_step=x['step']+(1024 if j else 0))
            x['dataset_group']='GSM8K' if x['source']=='gsm8k' else 'MATH '+poolrows[x['row_id']]['level']
            rows.append(x)
    pool=concurrent.futures.ProcessPoolExecutor(max_workers=4)
    try:
        fs=[pool.submit(regrade,x) for x in rows]
        for i,(x,f) in enumerate(zip(rows,fs),1):
            x['v2_verdict']=f.result(timeout=30)
            if i%3000==0:print('regraded',i,flush=True)
    finally:
        for p in list((getattr(pool,'_processes',None) or {}).values()):
            if p.is_alive():p.terminate()
        pool.shutdown(wait=True,cancel_futures=True)
    groups=defaultdict(list)
    for x in rows:groups[(x['phase'],x['step'],x['row_id'])].append(x)
    aggregate=defaultdict(Counter);details=[]
    correctness_transitions=Counter();zero_std_transitions=Counter()
    for (phase,step,rid),g in groups.items():
        assert len(g)==4
        oldr=[x['rewards'][0]+.1*x['rewards'][1] for x in g]
        newr=[float(x['v2_verdict']['correct'])+.1*x['rewards'][1] for x in g]
        olda,om,ostd=advantage(oldr);newa,nm,nstd=advantage(newr)
        assert abs(sum(olda))<1e-6 and abs(sum(newa))<1e-6
        if oldr!=newr:
            category=lambda n:'0/4' if n==0 else '4/4' if n==4 else '1-3/4'
            correctness_transitions[category(sum(v>=1 for v in oldr))+' -> '+category(sum(v>=1 for v in newr))]+=1
            zero_std_transitions[str(ostd<1e-8)+' -> '+str(nstd<1e-8)]+=1
        dropped=any(x['truncated'] for x in g)
        effective_old=[0.]*4 if dropped else olda
        effective_new=[0.]*4 if dropped else newa
        changed=[abs(a-b)>1e-7 for a,b in zip(effective_old,effective_new)]
        sample_stats=Counter(answers=4,groups=1,dropped_groups=int(dropped),effective_groups=int(not dropped),
            effective_answers=4*int(not dropped),reward_changes=sum(a!=b for a,b in zip(oldr,newr)),
            reward_0_to_1=sum(not x['rewards'][0] and x['v2_verdict']['correct'] for x in g),
            reward_1_to_0=sum(bool(x['rewards'][0]) and not x['v2_verdict']['correct'] for x in g),
            reward_affected_groups=int(oldr!=newr),effective_affected_groups=int(any(changed)),
            changed_effective_advantages=sum(changed),old_zero_std_groups=int(ostd<1e-8),new_zero_std_groups=int(nstd<1e-8))
        for a,b in zip(effective_old,effective_new):
            if dropped:continue
            sa,sb=sign(a),sign(b)
            sample_stats['negative_to_positive']+=int(sa==-1 and sb==1)
            sample_stats['positive_to_negative']+=int(sa==1 and sb==-1)
            sample_stats['zero_to_nonzero']+=int(sa==0 and sb!=0)
            sample_stats['nonzero_to_zero']+=int(sa!=0 and sb==0)
            sample_stats['same_sign_changed_magnitude']+=int(sa==sb and abs(a-b)>1e-7)
            sample_stats['advantage_absolute_difference_sum']+=abs(a-b)
            sample_stats['old_advantage_absolute_sum']+=abs(a)
        for key in ['ALL',g[0]['dataset_group'],phase]:aggregate[key].update(sample_stats)
        item=dict(phase=phase,row_id=rid,dataset_group=g[0]['dataset_group'],step=step,
                  cumulative_generation_step=g[0]['cumulative_generation_step'],dropped=dropped,
                  old_rewards=oldr,new_rewards=newr,old_mean=om,new_mean=nm,old_std=ostd,new_std=nstd,
                  old_raw_advantages=olda,new_raw_advantages=newa,
                  old_effective_advantages=effective_old,new_effective_advantages=effective_new)
        details.append(item)
        for i,x in enumerate(g):
            x.update(sample_index_in_group=i,dropped_group=dropped,
                     old_total_reward=oldr[i],new_total_reward=newr[i],
                     old_reconstructed_advantage=effective_old[i],new_reconstructed_advantage=effective_new[i])
    assert len(rows)==12288 and len(groups)==3072
    assert all(digest(Path(p))==h for p,h in hashes.items())
    for filename,rs in [('rescored_rollouts.jsonl',rows),('all_groups.jsonl',details),
                        ('affected_groups.jsonl',[x for x in details if x['old_rewards']!=x['new_rewards']])]:
        (OUT/filename).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs),encoding='utf-8')
    summary=dict(schema='archived-rollout-advantage-counterfactual-v2',formula='A_i=(r_i-mean_group)/(sample_std_group+1e-4); r=correctness+0.1*format; drop entire group if any answer lacks EOS',
                 input_sha256=hashes,scorer_sha256=digest(ROOT/'closure_grpo_experiment/math_verification.py'),
                 trl_reference_sha256=digest(T/'trl_grpo_trainer_v0.23.1_reference.py'),statistics=dict(aggregate),
                 affected_group_correctness_transitions=dict(correctness_transitions),
                 affected_group_zero_std_transitions=dict(zero_std_transitions),
                 scope='Complete 0-1536 rollout coverage, changes detected by current v2 only. Not an exhaustive semantic adjudication or measurement of actual token gradients, optimizer changes, or causal final accuracy.')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report='# 全历史 rollout：优势信号污染核查\n\n已重评 0–1536 步的全部 12,288 条回答（3,072 个四回答题组），GSM8K 采用原 numeric_answer 路径，MATH 采用 v2 修复层及 Math-Verify 0.8.0。保持格式奖励、唯一 boxed 规则和整组截断屏蔽不变，原始文件 SHA256 验证未变。\n\n公式依据官方 TRL v0.23.1 源码：总奖励=正确性+0.1×格式分量；advantage=(奖励−组均值)/(样本标准差+0.0001)。有一条截断就屏蔽整组的策略与 KL 损失，故有效 advantage 记零。浮点数使用 Python 双精度重建，变化阈值 1e-7；不承诺与 GPU float32 位级相同。\n\n| 数据 | 回答数 | 正确性奖励改变 | 有效受影响题组 | 有效 advantage 改变 | 负→正 | 正→负 | 零→非零 | 非零→零 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for key in ['GSM8K','MATH Level 2','MATH Level 3','ALL']:
        c=aggregate[key]
        vals=[c[x] for x in ['answers','reward_changes','effective_affected_groups','changed_effective_advantages','negative_to_positive','positive_to_negative','zero_to_nonzero','nonzero_to_zero']]
        report+='| '+key+' | '+' | '.join(map(str,vals))+' |\n'
    report+='\n“污染”指相对于 v2 重评分重建出的奖励/advantage 差异，不是全部真实评分错误的上界。全量覆盖回答不等于逐条人工核实了数学正确性；v2 仍可能漏判复杂文字表达。回答级 advantage 作用于 token 损失，但梯度还受概率比、裁剪、KL、长度归一化、优化器等影响，不能把受影响回答比例等同于错误参数更新比例。\n\n完整重评分（原 correct/rewards 保留）见 rescored_rollouts.jsonl；逐题组新旧 advantage 见 all_groups.jsonl；变化题组见 affected_groups.jsonl。summary.json 含数据集、阶段分组统计、输入哈希和评分器哈希。\n'
    (OUT/'ANALYSIS.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary['statistics'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
