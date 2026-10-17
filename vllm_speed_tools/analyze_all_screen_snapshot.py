"""Read-only aggregate of unique scored records in screening snapshots."""
import json, tarfile, statistics, csv, re
from collections import Counter
from pathlib import Path
from decimal import Decimal, InvalidOperation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/grpo_candidates_gsm8k_math_l2_l3'
OUT = DATA / 'combined_analysis'

def read_lines(data):
    return [json.loads(x) for x in data.splitlines() if x.strip()]

def metrics(rows):
    ss = [s for r in rows for s in r['samples']]
    mixed = [r for r in rows if r['group'] == 'mixed']
    return dict(questions=len(rows), answers=len(ss), groups=dict(Counter(r['group'] for r in rows)),
        histogram=dict(Counter(r['correct_count'] for r in rows)),
        correct=sum(s['correct'] for s in ss), any_correct=sum(r['correct_count']>0 for r in rows),
        normal=sum(s['normal_end'] for s in ss), eos=sum(s['eos_finished'] for s in ss),
        truncated=sum(s['truncated'] for s in ss), normal_correct=sum(s['normal_end'] and s['correct'] for s in ss),
        tokens_mean=round(statistics.mean(s['tokens'] for s in ss),2) if ss else None,
        verification=dict(Counter(s['math_verification']['status'] for s in ss)),
        mixed_with_normal_correct=sum(r['normal_correct_count']>0 for r in mixed),
        mixed_without_parsed_wrong=sum(not any(s['math_verification']['status']=='incorrect' for s in r['samples']) for r in mixed))

def main():
    pool = read_lines((DATA/'grpo_candidates.jsonl').read_bytes())
    questions = {r['row_id']:r for r in pool}
    rows = read_lines((DATA/'E1_sft_v3_grpo_screen_shard0_n2500/scored_questions.jsonl').read_bytes())
    manifests = {}
    for filename in ['first_snapshot.tar.gz','three_snapshot.tar.gz']:
        with tarfile.open(OUT/filename) as tar:
            for member in tar.getmembers():
                b=tar.extractfile(member).read()
                if member.name.endswith('scored_questions.jsonl'):
                    part=read_lines(b)
                    manifests[member.name.split('/')[0]]={'scored_snapshot':len(part)}
                    rows.extend(part)
    ids=[r['row_id'] for r in rows]
    assert len(ids)==len(set(ids)), 'Overlapping shards'
    assert set(ids)<=questions.keys()
    for r in rows:
        assert r['answer']==questions[r['row_id']]['answer']
        assert len(r['samples'])==4
        assert {s['sample_index'] for s in r['samples']}=={0,1,2,3}
        assert sum(s['correct'] for s in r['samples'])==r['correct_count']
    stats={'scope':'Scored snapshot, not a held-out evaluation; original grades unchanged',
        'target_questions':len(pool),'scored_questions':len(rows),'missing_questions':len(pool)-len(rows),
        'parts':manifests,'overall':metrics(rows),'by_source':{},'by_math_level':{},'by_math_type':{}}
    for source in ['gsm8k','math']:
        stats['by_source'][source]=metrics([r for r in rows if r['source']==source])
    for key,field in [('by_math_level','level'),('by_math_type','type')]:
        for value in sorted({str(questions[r['row_id']].get(field)) for r in rows if r['source']=='math'}):
            part=[r for r in rows if r['source']=='math' and str(questions[r['row_id']].get(field))==value]
            stats[key][value]=metrics(part)
    flags=[]
    for r in rows:
        if r['source']!='gsm8k': continue
        for s in r['samples']:
            if s['correct']: continue
            pred=s['math_verification'].get('prediction') or ''
            match=re.fullmatch(r'([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*([A-Za-z][A-Za-z ]{0,45})?\.?', pred.strip().strip('$').strip())
            if not match: continue
            try: equal=Decimal(match[1].replace(',',''))==Decimal(r['answer'])
            except InvalidOperation: equal=False
            if equal:
                flags.append(dict(row_id=r['row_id'],sample_index=s['sample_index'],prediction=pred,gold=r['answer'],category='unit_suffix_review' if match[2] else 'numeric_punctuation',completion=s['completion']))
    stats['gsm_grading_review_flags']=dict(Counter(f['category'] for f in flags))
    (OUT/'statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,records in [('grading_review_flags.jsonl',flags),('ungradable_gold_questions.jsonl',[dict(question=questions[r['row_id']],scored=r) for r in rows if r['group']=='ungradable_gold'])]:
        with (OUT/name).open('w',encoding='utf-8') as f:
            for r in records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (OUT/'question_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['row_id','source','level','type','group','correct_of_4','normal_correct_of_4'])
        for r in rows:
            q=questions[r['row_id']];w.writerow([r['row_id'],r['source'],q.get('level'),q.get('type'),r['group'],r['correct_count'],r['normal_correct_count']])
    text=['# SFT v3 全部分片筛选快照分析','',f'已判分 {len(rows)}/{len(pool)} 题；每题4条回答。未完成题不计作全错。分片ID无重复，与候选题ID、标准答案逐条一致。原评分未修改。','', '| 来源 | 已判题数 | 全错 | 对1–3次 | 全对 | gold无法判分 | 单条匹配率 | 四次至少一次匹配 | 格式完整结束 | 截断 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in [('总体',stats['overall']),*stats['by_source'].items(),*stats['by_math_level'].items()]:
        g=m['groups'];n=m['answers'];q=m['questions']
        text.append(f"| {name} | {q} | {g.get('all_wrong',0)} | {g.get('mixed',0)} | {g.get('all_correct',0)} | {g.get('ungradable_gold',0)} | {m['correct']/n:.2%} | {m['any_correct']/q:.2%} | {m['normal']/n:.2%} | {m['truncated']} |")
    text+=['','这些是训练候选采样表现，不是独立测试准确率。匹配率基于现有自动判分规则，存在单位、标点、答案提取造成的误差。上表匹配率分母包含隔离的gold解析失败题；这些题不应当解释为数学全错。', '',f'GSM8K规则扫描待复核：{json.dumps(stats["gsm_grading_review_flags"],ensure_ascii=False)}。这些是候选误判，不能全部自动加分。应重判现有回答，无需重新推理。', '',f'当前mixed组中，至少一条合规正确回答的题数为 {stats["overall"]["mixed_with_normal_correct"]}；全部不匹配回答均为提取/解析失败的mixed题数为 {stats["overall"]["mixed_without_parsed_wrong"]}，其奖励差异不能全部归因于数学对错。', '', '无法判分gold记录单独保存，排除出训练候选，先检查解析适配。数学题型明细见statistics.json；逐题明细见question_inventory.csv。', '', '## MATH题型（当前已判分范围）', '', '| 题型 | 题数 | 单条匹配率 | mixed题数 |', '|---|---:|---:|---:|']
    for name,m in stats['by_math_type'].items():
        text.append(f"| {name} | {m['questions']} | {m['correct']/m['answers']:.2%} | {m['groups'].get('mixed',0)} |")
    text+=['', '## 使用建议', '', '1. GSM8K提供较多组内奖励差异，可作为主训练候选；MATH二级可提供更难的补充，三级仍有可探索样本，但四次全错比例更高。', '2. 先修正数字标点、单位与MATH表达式解析规则，对保存的回答重判；保留原始评分版本，避免混用不同奖励定义。', '3. 纯正确性奖励下，四次全对和全错组当前没有组内优势差异；这是有限四次采样，不能认定题目永久过易或不可解。若另有格式奖励，仍可能产生组内差异。', '4. 模型EOS停止率高，但标签合格率仍低。格式奖励与数学奖励应分别记录，避免把解析差异全部当作数学学习信号。', '5. 筛选数据是训练候选，不能替代独立验证集，也不能据此保证后续GRPO有效。']
    (OUT/'ANALYSIS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
