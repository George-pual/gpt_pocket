"""Compare saved evaluations using paired question-cluster bootstrap."""
import json, random, tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/grpo_candidates_gsm8k_math_l2_l3/reward_scale128_v3/results'
OUT.mkdir(exist_ok=True)
old = ROOT / 'datasets/grpo_candidates_gsm8k_math_l2_l3/reward_pilot_v3/results/baseline_eval'
sets = {'SFT_v3': [json.loads(x) for x in (old/'answers.jsonl').read_bytes().splitlines()]}
summaries = {'SFT_v3': json.loads((old/'summary.json').read_text())}
with tarfile.open(OUT.parent/'evaluation_results.tar.gz') as archive:
    for step in (32, 64, 128):
        name = f'step_{step}'
        folder = OUT / f'eval_step_{step}'
        folder.mkdir(exist_ok=True)
        for filename in ('answers.jsonl', 'summary.json'):
            data = archive.extractfile(f'eval_step_{step}/{filename}').read()
            (folder/filename).write_bytes(data)
        sets[name] = [json.loads(x) for x in (folder/'answers.jsonl').read_bytes().splitlines()]
        summaries[name] = json.loads((folder/'summary.json').read_text())

def grouped(rows):
    result = defaultdict(list)
    for row in rows:
        result[row['row_id']].append(row)
    assert len(result) == 100
    assert all(len(v) == 4 and {x['sample_index'] for x in v} == {0,1,2,3} for v in result.values())
    return result

groups = {k: grouped(v) for k,v in sets.items()}
ids = sorted(groups['SFT_v3'])
assert all(sorted(g) == ids for g in groups.values())
assert all(s['data_sha256'] == summaries['SFT_v3']['data_sha256'] and s['sampling'] == summaries['SFT_v3']['sampling'] for s in summaries.values())

def metric(rows, key):
    return sum(bool(r['correct'] and r['format_eos']) if key == 'normal_correct' else bool(r[key]) for r in rows)/4

comparisons = {}
for before, after in [('SFT_v3','step_32'), ('SFT_v3','step_64'), ('SFT_v3','step_128'), ('step_64','step_128')]:
    stats = {}
    for key in ('correct', 'format_eos', 'normal_correct', 'truncated'):
        delta = [metric(groups[after][i],key)-metric(groups[before][i],key) for i in ids]
        rng = random.Random(3407)
        boot = sorted(sum(rng.choices(delta,k=100))/100 for _ in range(10000))
        stats[key] = {'change_pp': sum(delta), 'ci95_pp': [100*boot[249],100*boot[9749]], 'questions_improved':sum(d>0 for d in delta), 'questions_worsened':sum(d<0 for d in delta)}
    comparisons[f'{before}_to_{after}'] = stats

failures = {}
for name, rows in sets.items():
    counts = Counter()
    for row in rows:
        if row['format_eos']: continue
        c = row['completion']
        if not isinstance(c,str): c = '\n'.join(r['content'] for r in c)
        if '</start_working_out>' in c: counts['wrong_end_tag'] += 1
        if c.count('<end_working_out>') != 1: counts['missing_or_multiple_end_tag'] += 1
        if c.count('<SOLUTION>') != 1 or c.count('</SOLUTION>') != 1: counts['missing_or_multiple_solution_tags'] += 1
        if '</SOLUTION>' in c and not c.strip().endswith('</SOLUTION>'): counts['content_after_solution'] += 1
        if '<start_working_out>' in c: counts['repeated_start_tag'] += 1
    failures[name] = dict(counts)

output = {'scope':'Internal heldout 100 candidate questions, 4 samples per question, single training run; same dataset hash and sampling settings; intervals are paired question-cluster bootstrap, not corrected for checkpoint selection.', 'summaries':summaries, 'comparisons':comparisons, 'format_failure_counts_nonexclusive':failures}
(OUT/'comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
lines = ['# SFT v3 GRPO 128 步评测分析', '', '同一内部留出集100题，每题4次采样，T=1.0。此集用于checkpoint选择，不是最终独立测试集。', '', '| 模型 | 正确 | 格式完整且EOS | 正确且格式完整 | 正常结束 | 截断 | 至少一次正确的题目 |', '|---|---:|---:|---:|---:|---:|---:|']
for name,s in summaries.items():
    o=s['overall']
    lines.append(f"| {name} | {o['correct']}/400 | {o['format_eos']}/400 | {o['normal_correct']}/400 | {o['eos']}/400 | {o['truncated']}/400 | {o['observed_any_correct']}/100 |")
lines += ['', '128步相对原始SFT v3的变化（百分点；95%按题目配对bootstrap区间）：']
for key,s in comparisons['SFT_v3_to_step_128'].items(): lines.append(f"- {key}: {s['change_pp']:+.2f}，区间 {s['ci95_pp']}")
lines += ['', '优先保留128步：单次采样正确率和正确且合规率最高，未截断。64步四次至少一次答对覆盖93题，高于128步90题，因此探索覆盖没有单调改善。格式提升较小，区间跨零，尚不能认定格式问题已解决。', '', 'GSM8K：132/240→162/240；MATH：54/160→73/160。均有提升。原始SFT错误结束标签69条，128步具体统计见comparison.json。', '', '下一步先保留原始SFT及三个checkpoint，在新的独立题目上确认128步收益；若重点改善结束标签，另开格式奖励实验，每次只修改一个因素。当前单臂结果不能证明格式奖励优于纯正确性奖励，也不能证明继续加步数一定受益。']
(OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'comparisons':comparisons,'format_failures':failures},ensure_ascii=False,indent=2))
