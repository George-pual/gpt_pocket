"""Audit actual logged LR and learning signals without retraining."""
import csv,json,statistics
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'datasets/grpo_candidates_gsm8k_math_l2_l3/training_audit_1536'
OUT=T/'learning_rate_decay_audit';OUT.mkdir(exist_ok=True)

def read(p):return [json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]
def values(rows,k):return [float(x[k]) for x in rows if x.get(k) not in ('',None)]
def mean(v):return statistics.mean(v) if v else None

def main():
    with (T/'combined_metrics_0_1536.csv').open(encoding='utf-8-sig',newline='') as f:metrics=list(csv.DictReader(f))
    assert len(metrics)==1536
    groups=read(T/'per_question_training_correctness.jsonl')
    configs={}
    for phase in ['E1_sft_v3_half_math1024_fast_v2','E1_sft_v3_expanded_continue1536_v1']:
        configs[phase]=json.loads((T/phase/'B_components/grpo_config.json').read_text(encoding='utf-8'))
    windows=[]
    for lo,hi in [(1,256),(257,512),(513,768),(769,1024),(1025,1280),(1281,1536),(961,1024),(1473,1536)]:
        rs=[x for x in metrics if lo<=int(x['cumulative_step'])<=hi]
        gs=[x for x in groups if lo<=x['generation_step']+1<=hi]
        lr=values(rs,'learning_rate');gn=values(rs,'grad_norm')
        windows.append(dict(start=lo,end=hi,steps=len(rs),lr_first=lr[0],lr_last=lr[-1],lr_mean=mean(lr),lr_sum=sum(lr),
            grad_norm_mean=mean(gn),grad_norm_median=statistics.median(gn),
            clip_ratio_mean=mean(values(rs,'clip_ratio/region_mean')),kl_mean=mean(values(rs,'kl')),
            sampled_groups=len(gs),mixed_groups=sum(0<x['correct']<4 for x in gs),
            dropped_groups=sum(x['dropped'] for x in gs),
            pilot_correct_mean=mean(values(rs,'pilot/correct_rate'))))
    phase_stats={}
    for phase,c in configs.items():
        rs=[x for x in metrics if x['phase']==phase]
        lr=values(rs,'learning_rate');peak=c['learning_rate']
        phase_stats[phase]=dict(steps=len(rs),configured_peak_lr=peak,logged_lr_sum=sum(lr),logged_lr_mean=mean(lr),
            steps_at_or_below_10pct_peak=sum(v<=.1*peak for v in lr),
            steps_at_or_below_25pct_peak=sum(v<=.25*peak for v in lr),
            schedule=c['lr_scheduler_type'],warmup_steps=c['warmup_steps'],last_logged_lr=lr[-1])
    points={str(s):float(next(x['learning_rate'] for x in metrics if int(x['cumulative_step'])==s)) for s in [1,3,256,512,768,1024,1025,1027,1280,1408,1472,1536]}
    summary=dict(phases=phase_stats,windows=windows,lr_at_logged_step=points,
                 scope='Logged scheduler LR, not parameter-update norms. Different training windows use different questions. No causal attribution without controlled rerun.')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report='# 0–1536 步学习率衰减核查\n\n实际配置为分段线性衰减：前 1024 步峰值 5e-6，warmup=2，线性降到 0；从 1024 checkpoint 续训的 512 步重建优化器与调度器，峰值改为 2e-6，再次 warmup=2、线性降到 0。以下使用逐步日志 learning_rate，末条日志与调度器推进后的最终 0 可能存在一步偏移，不混用。\n\n| 累计更新阶段 | 平均学习率 | 首/末日志学习率 | 混合题组 / 总题组 | 梯度范数中位数 | 比例裁剪均值 |\n|---|---:|---|---:|---:|---:|\n'
    for w in windows:
        report+=f"| {w['start']}–{w['end']} | {w['lr_mean']:.3e} | {w['lr_first']:.3e} / {w['lr_last']:.3e} | {w['mixed_groups']}/{w['sampled_groups']} | {w['grad_norm_median']:.4f} | {w['clip_ratio_mean']:.3%} |\n"
    report+='\n后段仍持续采到混合题组，梯度范数未归零，同时日志学习率逼近零。这支持“后期仍有学习信号但更新步长被明显压低”的机制解释。grad_norm 是梯度指标，不是 Adam 参数更新范数；不能仅凭它量化实际更新量，也不能把所有梯度都归为正确性奖励。\n\n两个阶段各自前半段消耗约 75% 的线性学习率总和，后半段约 25%；最后四分之一约占该阶段学习率总和的 6.3%。学习率总和仅用于描述调度器分配，不能视作学习量或准确率提升的线性度量。\n\n续训阶段多做 512 步（前阶段步数的 50%），但日志学习率总和约为前阶段的 20%；它没有按前阶段平均步长做 512 步。第一次及续训各自末端的新采样训练题都以很小步长更新，可能降低训练预算的利用率。数据不同、梯度方向不同、优化器重置且初始权重不同，不能仅凭此证明衰减导致准确率平台。\n\n已有独立测试：1024/1280/1536 步 GSM8K 正确数分别 1128/1127/1128；MATH v2 重评分分别 135/140/137（每次 200 题）。这证明指标未持续增长，不能识别 LR 与数据、奖励、泛化等因素各自的因果影响。\n\n建议验证：从同一 1024 checkpoint 出发，统一使用修复后的评分器、相同题序/采样参数/512 步预算/优化器初始化。A 组沿用 2e-6→0 线性衰减；B 组维持相同峰值和 warmup，但线性降至 1e-6 并保持下限。两组在相同步数使用同一独立测试集比较正确率、格式率、截断与 KL。该方案是待验证的实验，不是已经证明更优的配置；此次仅分析，未启动训练。\n'
    (OUT/'ANALYSIS.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
