"""Compare component-reward checkpoints with archived binary-reward run."""
import json,tarfile,random
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]/'datasets/grpo_candidates_gsm8k_math_l2_l3/reward_scale128_v3'
OUT=ROOT/'format_components_results';OUT.mkdir(exist_ok=True)
with tarfile.open(ROOT/'format_components_results.tar.gz') as t:
    for m in t.getmembers():
        if m.isfile():
            p=OUT/m.name
            assert p.resolve().is_relative_to(OUT.resolve())
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
paths={'binary128':ROOT/'results/eval_step_128',**{f'components{s}':OUT/f'eval_step_{s}' for s in (32,64,128)}}
sets={k:[json.loads(x) for x in (p/'answers.jsonl').read_bytes().splitlines()] for k,p in paths.items()}
summaries={k:json.loads((p/'summary.json').read_text()) for k,p in paths.items()}
groups={};fail={}
for k,rows in sets.items():
    d=defaultdict(list);counts=Counter()
    for r in rows:
        d[r['row_id']].append(r)
        if r['format_eos']:continue
        c=r['completion'];c=c if isinstance(c,str) else '\n'.join(x['content'] for x in c)
        counts['wrong_end_tag']+='</start_working_out>' in c
        counts['missing_or_multiple_end_tag']+=c.count('<end_working_out>')!=1
        counts['missing_or_multiple_solution_tags']+=c.count('<SOLUTION>')!=1 or c.count('</SOLUTION>')!=1
        counts['trailing_text']+='</SOLUTION>' in c and not c.strip().endswith('</SOLUTION>')
    assert len(d)==100 and all(len(v)==4 and {x['sample_index'] for x in v}=={0,1,2,3} for v in d.values())
    groups[k]=d;fail[k]=dict(counts)
assert all(set(d)==set(groups['binary128']) for d in groups.values())
assert all(s['data_sha256']==summaries['binary128']['data_sha256'] and s['sampling']==summaries['binary128']['sampling'] for s in summaries.values())
ci={}
for metric in ('correct','format_eos','normal_correct'):
    def value(rs):return sum(bool(r['correct'] and r['format_eos']) if metric=='normal_correct' else bool(r[metric]) for r in rs)/4
    delta=[value(groups['components128'][i])-value(groups['binary128'][i]) for i in sorted(groups['binary128'])]
    rng=random.Random(3407);boot=sorted(sum(rng.choices(delta,k=100))/100 for _ in range(10000))
    ci[metric]={'change_pp':sum(delta),'paired_question_bootstrap_95ci_pp':[100*boot[249],100*boot[9749]]}
result={'summaries':summaries,'format_failures_nonexclusive':fail,'components128_vs_binary128':ci,'scope':'Same internal100 questions,4 samples/question; single seed historical comparison, intervals not adjusted for checkpoint selection'}
(OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 分项格式奖励结果','','同一内部验证100题，每题4次。单seed历史对照，不能替代独立测试。','','| 模型 | 正确/400 | 格式完整/400 | 正确且合规/400 | 至少一次答对/100 | 错误结束标签 |','|---|---:|---:|---:|---:|---:|']
for name,s in summaries.items():
    o=s['overall'];lines.append(f"| {name} | {o['correct']} | {o['format_eos']} | {o['normal_correct']} | {o['observed_any_correct']} | {fail[name]['wrong_end_tag']} |")
lines+=['','128步分项奖励比上一轮128步：'+json.dumps(ci,ensure_ascii=False),'','建议将分项奖励作为半量训练候选配置；格式仍未解决，继续跟踪错误结束标签和数学正确率。历史单seed比较的区间跨零时，不应宣称已证明分项奖励更好。']
(OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'overall':{k:s['overall'] for k,s in summaries.items()},'format_failures':fail,'ci':ci},ensure_ascii=False,indent=2))
