"""Local, read-only analysis of paired official-test completions."""
import json,re
from pathlib import Path
from collections import Counter,defaultdict

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'datasets/gsm8k/base_sftv3_test/results'
OUT=R/'grpo1024_error_audit'
OUT.mkdir(exist_ok=True)
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
mathmeta={r['row_id']:r for r in read(ROOT/'datasets/math/test_l2_l3_200/questions.jsonl')}
allstats={}
for kind,old in [('gsm8k','results_vllm'),('math','results_math_l2_l3')]:
    arms={a:read(R/old/a/'answers.jsonl') for a in ('base','sft_v3')}
    arms['grpo1024']=read(R/'grpo1024_test'/kind/'grpo1024/answers.jsonl')
    maps={a:{x['row_id']:x for x in rows} for a,rows in arms.items()}
    assert len({tuple(m) for m in maps.values()})==1
    stats={a:dict(total=len(rs),wrong=sum(not x['correct'] for x in rs),truncated=sum(x['truncated'] for x in rs),wrong_truncated=sum(not x['correct'] and x['truncated'] for x in rs),wrong_format_complete=sum(not x['correct'] and x['format_eos'] for x in rs)) for a,rs in arms.items()}
    stats['pairs']={a:dict(regressions=sum(maps[a][x['row_id']]['correct'] and not x['correct'] for x in arms['grpo1024']),rescues=sum(not maps[a][x['row_id']]['correct'] and x['correct'] for x in arms['grpo1024'])) for a in ('base','sft_v3')}
    if kind=='math':
        stats['by_type']={t:{a:dict(total=len(xs),correct=sum(x['correct'] for x in xs),truncated=sum(x['truncated'] for x in xs)) for a,rs in arms.items() if (xs:=[x for x in rs if mathmeta[x['row_id']]['type']==t])} for t in sorted({m['type'] for m in mathmeta.values()})}
    wrong=[]
    for x in arms['grpo1024']:
        if x['correct']:continue
        w=dict(x,base_correct=maps['base'][x['row_id']]['correct'],sft_correct=maps['sft_v3'][x['row_id']]['correct'])
        if kind=='math':w.update(type=mathmeta[x['row_id']]['type'],reference_solution=mathmeta[x['row_id']]['solution'])
        wrong.append(w)
    (OUT/f'{kind}_wrong.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in wrong),encoding='utf-8')
    lines=[]
    for i,x in enumerate(wrong):
        lines.extend([f"\n[{i}] {x['row_id']} type={x.get('type')} base={x['base_correct']} sft={x['sft_correct']} trunc={x['truncated']} format={x['format_eos']}",f"Q: {x['question']}",f"GOLD: {x['gold']} PRED: {x['prediction']}",f"OUTPUT: {x['completion'][:200]} ... {x['completion'][-1100:]}"])
    (OUT/f'{kind}_review.txt').write_text('\n'.join(lines),encoding='utf-8')
    allstats[kind]=stats
(OUT/'statistics.json').write_text(json.dumps(allstats,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(allstats,ensure_ascii=False,indent=2))
