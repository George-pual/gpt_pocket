"""Five-bin rollout correctness histograms, before/after v2 regrading."""
import json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'datasets/grpo_candidates_gsm8k_math_l2_l3/training_audit_1536'
OUT=T/'correctness_groups_five_bins';OUT.mkdir(exist_ok=True)
def read(p):return [json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]

def table(records):
    lines=['| 数据 / 阶段 | 题组数 | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 |','|---|---:|---:|---:|---:|---:|---:|']
    for label,counts in records:
        total=sum(counts.values())
        lines.append('| '+label+' | '+str(total)+' | '+' | '.join(f'{counts.get(str(k),0)} ({counts.get(str(k),0)/total:.2%})' for k in range(5))+' |')
    return '\n'.join(lines)

def main():
    groups=read(T/'all_rollout_advantage_audit_v2/all_groups.jsonl')
    old_lookup={x['row_id']:x for x in read(T/'per_question_training_correctness.jsonl')}
    hist=defaultdict(Counter);stagehist=defaultdict(Counter);transitions=Counter();records=[]
    for g in groups:
        old=sum(v>=1 for v in g['old_rewards']);new=sum(v>=1 for v in g['new_rewards'])
        assert old==old_lookup[g['row_id']]['correct']
        lo=(g['cumulative_generation_step']//256)*256+1
        stage=f'{lo}-{lo+255}'
        transitions[f'{old}/4 -> {new}/4']+=1
        for version,k in [('original',old),('v2',new)]:
            for scope in ['all_sampled','dropped' if g['dropped'] else 'effective']:
                for ds in ['ALL',g['dataset_group']]:
                    hist[(version,scope,ds)][str(k)]+=1
                    stagehist[(version,scope,stage,ds)][str(k)]+=1
        records.append(dict(row_id=g['row_id'],dataset_group=g['dataset_group'],stage=stage,
                            original_correct=old,v2_correct=new,total=4,dropped=g['dropped']))
    summary=dict(denominator='Each bin counts questions with exactly k correct answers among 4 samples; format reward excluded. Percentages use all groups in the corresponding table row.',
                 histograms=[dict(version=v,scope=s,dataset=d,total=sum(c.values()),counts=dict(c)) for (v,s,d),c in hist.items()],
                 by_stage=[dict(version=v,scope=s,stage=st,dataset=d,total=sum(c.values()),counts=dict(c)) for (v,s,st,d),c in stagehist.items()],
                 original_to_v2_group_transitions=dict(transitions))
    assert sum(hist[('v2','all_sampled','ALL')].values())==3072
    assert sum(hist[('v2','effective','ALL')].values())==3019
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'per_question.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf-8')
    dsorder=['GSM8K','MATH Level 2','MATH Level 3','ALL']
    report='# 每题四采样正确性：0/4、1/4、2/4、3/4、4/4\n\n范围：0–1536 步全部 3,072 个不同训练题组。每个格子的百分比以所在行全部题组为分母，不是以混合组为分母。分组仅看正确性，不加入格式奖励。阶段间题目不同，不能直接当成同题进步。v2 重评分仍可能漏判复杂表达。\n\n'
    for version,title in [('original','原训练评分'),('v2','v2 修复后重评分')]:
        report+='## '+title+'：全部采样组\n\n'+table([(ds,hist[(version,'all_sampled',ds)]) for ds in dsorder])+'\n\n'
        report+='### 排除整组截断屏蔽\n\n'+table([(ds,hist[(version,'effective',ds)]) for ds in dsorder])+'\n\n'
    report+='## v2：按累计更新阶段\n\n'
    for ds in dsorder:
        report+='### '+ds+'\n\n'+table([(f'{lo}-{lo+255}',stagehist[('v2','all_sampled',f'{lo}-{lo+255}',ds)]) for lo in [1,257,513,769,1025,1281]])+'\n\n'
    report+='原评分说明历史训练实际收到的奖励；v2 表是事后修正，不代表训练当时已经使用这些奖励。有效题组仍可能总奖励标准差为零或仅有格式差异，不能把“未被屏蔽”直接等同于存在正确性策略梯度。完整阶段×数据×评分版本×屏蔽状态计数见 summary.json，逐题计数见 per_question.jsonl。\n'
    (OUT/'ANALYSIS.md').write_text(report,encoding='utf-8')
    print(json.dumps({v:{s:{ds:dict(hist[(v,s,ds)]) for ds in dsorder} for s in ['all_sampled','effective']} for v in ['original','v2']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
