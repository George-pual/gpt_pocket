"""Paired question-level bootstrap for one-arm before/after pilot."""
import argparse,json,random
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    def load(name):
        path=root/name/'answers.jsonl'
        rows=[json.loads(x) for x in path.read_bytes().splitlines() if x.strip()]
        d={}
        for r in rows:d.setdefault(r['row_id'],[]).append(r)
        assert len(d)==100 and all(len(v)==4 and {s['sample_index'] for s in v}=={0,1,2,3} for v in d.values())
        return d
    before=load('baseline_eval');after=load('B_correctness_format_eval');assert before.keys()==after.keys()
    ids=sorted(before);results={}
    for metric in ['correct','format_eos','normal_correct','truncated']:
        def value(ss):return sum((s['correct'] and s['format_eos']) if metric=='normal_correct' else s[metric] for s in ss)/4
        b=[value(before[i]) for i in ids];e=[value(after[i]) for i in ids];delta=[y-x for x,y in zip(b,e)]
        rng=random.Random(3407);boot=sorted(sum(rng.choices(delta,k=100))/100 for _ in range(5000))
        results[metric]=dict(before=sum(b)/100,after=sum(e)/100,change=sum(delta)/100,paired_question_bootstrap_95ci=[boot[124],boot[4874]])
    output=dict(scope='Internal heldout candidate questions; one arm, one seed, 32 updates; no causal comparison against pure-correctness training',questions=100,answers_per_model=400,metrics=results)
    (root/'comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# SFT v3 短程GRPO前后比较','','内部留出100题，每题4条回答。仅B组训练32步，没有A组。','','| 指标 | 训练前 | 训练后 | 变化（百分点） |','|---|---:|---:|---:|']
    for name,m in results.items():text.append(f"| {name} | {m['before']:.2%} | {m['after']:.2%} | {100*m['change']:+.2f} |")
    text+=['','本结果只能检查短程更新的可行性和前后变化，不能证明格式奖励优于纯正确性。按题目聚类的配对bootstrap区间见comparison.json；同题4条不能视作4道独立题。判定应结合独立数学匹配率、合规正确率、格式率、截断率及梯度/KL稳定性。']
    (root/'COMPARISON.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
