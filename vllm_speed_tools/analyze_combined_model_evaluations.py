"""Combine disjoint MATH subsets and prior GSM8K using saved responses only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import re
import logging

from audited_answer_scoring_v32 import SCHEMA, score_answer
from report_math_remaining_eval import ARMS, read_rows, metrics
import rescore_fixed_answers
from rescore_fixed_answers import bounded_map


def worker_v32(payload):
    logging.getLogger('math_verify').setLevel(logging.ERROR)
    return score_answer(*payload)


# Use the existing process timeouts with an explicit versioned scorer worker.
rescore_fixed_answers.worker = worker_v32
rescore_fixed_answers.SCHEMA = SCHEMA


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_rows(p, rows):
    p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def repeated(text):
    words=text.split()
    return max(Counter(tuple(words[i:i+20]) for i in range(max(0,len(words)-19))).values(),default=0)>=3


def extra_metrics(rows):
    m=metrics(rows)
    m.update(correct_and_format=sum(r['audited_score_v3']['correct'] and r['format_eos'] for r in rows),
        at_least_1024_tokens=sum(r['tokens']>=1024 for r in rows),
        at_least_2048_tokens=sum(r['tokens']>=2048 for r in rows),
        mean_nontruncated_tokens=statistics.mean(r['tokens'] for r in rows if not r['truncated']))
    return m


def paired(a,b):
    assert [(r['row_id'],r['question'],r['gold']) for r in a]==[(r['row_id'],r['question'],r['gold']) for r in b]
    def correct(r):return bool(r['audited_score_v3']['correct'])
    gain=[(x,y) for x,y in zip(a,b) if not correct(x) and correct(y)]
    loss=[(x,y) for x,y in zip(a,b) if correct(x) and not correct(y)]
    return dict(gain=len(gain),loss=len(loss),net=len(gain)-len(loss),
        both_correct=sum(correct(x) and correct(y) for x,y in zip(a,b)),
        losses_with_second_truncated=sum(y['truncated'] for x,y in loss),
        losses_with_second_unconfirmed=sum(y['audited_score_v3']['status'] not in ('correct','incorrect') for x,y in loss))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output;out.mkdir(exist_ok=False)
    workspace=Path(__file__).resolve().parents[1]
    math_root=workspace/'datasets/math/test_l2_l3_remaining_v1/scored_reference_review_v1'
    previous=workspace/'datasets/gsm8k/base_sftv3_test/results/audited_scorer_v3_final_r2'
    math_manifest=json.loads((math_root/'manifest.json').read_text('utf-8'))
    gsm_manifest=json.loads((previous/'manifest.json').read_text('utf-8'))
    scorer=Path(__file__).with_name('audited_answer_scoring_v32.py')
    assert SCHEMA=='audited-final-answer-v3.2'
    prior_scorer=Path(__file__).with_name('audited_answer_scoring.py')
    assert sha(prior_scorer)==math_manifest['scorer_sha256']
    old_source=prior_scorer.read_text('utf-8')
    expected=old_source.replace("SCHEMA = 'audited-final-answer-v3.1'", "SCHEMA = 'audited-final-answer-v3.2'").replace(
        '    aliases = [a for a, b in UNIT_NAMES.items() if b == base]\n',
        "    speed_request = re.search(r'(?:^|[.?!]\\s+)(?:How fast\\b|What (?:is|was|will be) (?:the )?(?:average )?speed\\b)[^?]*\\?\\s*$', question, re.I)\n    if speed_request and base != 'miles_per_hour':\n        return False\n    aliases = [a for a, b in UNIT_NAMES.items() if b == base]\n    if base == 'miles_per_hour' and re.search(r'\\bmiles?\\s+per\\s+hour\\b', question, re.I):\n        return True\n")
    expected=expected.replace('def normalize_pair(prediction, gold, question):\n',
        "def normalize_pair(prediction, gold, question):\n    prediction = re.sub(r'\\bmiles?\\s+per\\s+hour\\b', 'mph', prediction, flags=re.I)\n    gold = re.sub(r'\\bmiles?\\s+per\\s+hour\\b', 'mph', str(gold), flags=re.I)\n")
    alias_rule="    if base == 'miles_per_hour' and re.search(r'\\bmiles?\\s+per\\s+hour\\b', question, re.I):\n        return True\n"
    expected=expected.replace(alias_rule,alias_rule+"    if base == 'miles_per_hour' and re.search(r'\\bmiles?\\b', question, re.I) and re.search(r'\\bhours?\\b', question, re.I) and re.search(r'\\bhow fast\\b|\\bspeed\\b', question, re.I):\n        return True\n")
    assert scorer.read_text('utf-8')==expected
    controls=[('750 mph','750','How fast? Average speed is 120 Miles per hour.',True,True),
        ('750 mph','750','How fast? Average speed is 120 miles per hour.',True,True),
        ('750 mph','750','How many dollars are needed?',True,False),
        ('751 mph','750','How fast in miles per hour?',True,False),
        ('750 hours','750','What is the speed in miles per hour?',True,False),
        ('John runs at a speed of 10 miles per hour.','10','How fast does John run, in miles per hour?',True,True),
        ('10 mph','10','John runs 60 miles in 6 hours. How fast does he run?',True,True),
        ('10 mph','10','John runs 60 miles in 6 minutes. How fast does he run?',True,False),
        ('1%','1','What is the probability?',True,False)]
    control_verdicts=bounded_map([('<SOLUTION>'+p+'</SOLUTION>',g,q,e) for p,g,q,e,want in controls],4,20)
    for c,v in zip(controls,control_verdicts):assert v['correct']==c[-1],(c,v)
    save(out/'unit_alias_controls.json',[dict(prediction=c[0],gold=c[1],question=c[2],expected=c[-1],verdict=v) for c,v in zip(controls,control_verdicts)])
    # Verify old source hashes/configs before reusing the fixed GSM8K answers.
    gsm_rows={};tasks=[];lookup={};before_hashes={}
    for arm in ARMS:
        source=previous/'gsm8k'/f'{arm}.jsonl'
        rows=read_rows(source);assert len(rows)==1319
        before_hashes[str(source)]=sha(source)
        gsm_rows[arm]=rows
        config=gsm_manifest['source_configs'][arm+'/gsm8k']
        assert config['sampling']==dict(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=3894,eos_token_id=151643,ignore_eos=False)
        for field in ('data_sha256','prompt_tokens_sha256','sampling','chat_template'):
            assert config[field]==gsm_manifest['source_configs']['base/gsm8k'][field]
        for row in rows:
            task=(row['completion'],row['gold'],row['question'],row['eos'] and not row['truncated'])
            if task not in lookup:
                lookup[task]=len(tasks);tasks.append(task)
    print('GSM8K_SAVED_ANSWERS',6595,'UNIQUE_CPU_TASKS',len(tasks),flush=True)
    verdicts=bounded_map(tasks,4,20)
    grouped={'gsm8k':{},'math_all':{},'math_old200':{},'math_new1824':{}}
    details={'gsm8k':{},'math_all':{},'math_new1824':{}}
    changes=[]
    (out/'gsm8k_v31').mkdir()
    (out/'math_v32').mkdir()
    math_changes=[]
    for arm in ARMS:
        rows=gsm_rows[arm]
        for r in rows:
            task=(r['completion'],r['gold'],r['question'],r['eos'] and not r['truncated'])
            old=r['audited_score_v3'];new=verdicts[lookup[task]]
            if old['correct']!=new['correct'] or old['status']!=new['status']:
                changes.append(dict(arm=arm,row_id=r['row_id'],previous=old,revised=new))
            r['previous_score_v3']=old;r['audited_score_v3']=new
            r['repeated_20_word_span_3_times']=repeated(r['completion'])
        write_rows(out/'gsm8k_v31'/f'{arm}.jsonl',rows)
        grouped['gsm8k'][arm]=extra_metrics(rows);details['gsm8k'][arm]=rows
        m=read_rows(math_root/f'{arm}.jsonl')
        # Only this question-context predicate can reach the new v3.2 branch.
        # Reuse the exact v3.1 result elsewhere, preserving its old schema as
        # provenance. Recompute every affected question for every model.
        affected=[r for r in m if re.search(r'\bmiles?\s+per\s+hour\b|\bhow fast\b|\bWhat (?:is|was|will be) (?:the )?(?:average )?speed\b',r['question'],re.I)
            or re.search(r'\bmiles?\s+per\s+hour\b',r['completion']+' '+r['gold'],re.I)
            or (re.search(r'\bmiles?\b',r['question'],re.I) and re.search(r'\bhours?\b',r['question'],re.I) and re.search(r'\bspeed\b',r['question'],re.I))]
        assert all('scoring_reference_policy' not in r for r in affected)
        fixed=bounded_map([(r['completion'],r['gold'],r['question'],r['eos'] and not r['truncated']) for r in affected],4,20)
        by_id={r['row_id']:v for r,v in zip(affected,fixed)}
        for r in m:
            old=r['audited_score_v3']
            new=by_id.get(r['row_id'],dict(old,schema=SCHEMA))
            r['prior_v31_reference_review_score']=old
            r['audited_score_v3']=new
            if old['correct']!=new['correct'] or old['status']!=new['status']:
                math_changes.append(dict(arm=arm,row_id=r['row_id'],previous=old,revised=new))
        write_rows(out/'math_v32'/f'{arm}.jsonl',m)
        for name,selected in [('math_all',m),('math_old200',[r for r in m if r['reused']]),('math_new1824',[r for r in m if not r['reused']])]:
            grouped[name][arm]=extra_metrics(selected)
            if name in details:details[name][arm]=selected
        for k in ('total','correct','truncated','repeated','format_eos'):
            assert grouped['math_all'][arm][k]==grouped['math_old200'][arm][k]+grouped['math_new1824'][arm][k]
    comparisons={name:{arm:paired(rows['base'],rows[arm]) for arm in ARMS if arm!='base'} for name,rows in details.items()}
    comparisons['grpo1536_vs_1024']={name:paired(rows['grpo1024'],rows['grpo1536']) for name,rows in details.items()}
    save(out/'summary.json',grouped);save(out/'paired_comparisons.json',comparisons)
    write_rows(out/'gsm8k_scoring_version_changes.jsonl',changes)
    write_rows(out/'math_unit_alias_changes.jsonl',math_changes)
    report=['# 两轮 MATH 与此前 GSM8K 的联合分析','',
        '仅分析已有回答，没有模型训练或新生成。MATH将不重叠的旧200题与新增1824题按题数合并为2024题；GSM8K1319题单列。所有模型使用相同原prompt与greedy解码，输出预算3894。',
        '统一评分口径为audited-final-answer-v3.2：保留v3.1的提取与等价逻辑，补充题面Miles per hour与答案mph的单位别名，并防止明确问速度时把hours等单位误当速度；正反例检查通过。MATH沿用完整性修订的参考答案策略，并对所有触及速度上下文规则的题逐模型重评分；其余记录与v3.1等价，复用既有判定。GSM8K本次全量CPU复评分。',
        f'GSM8K相对旧v3共改变{len(changes)}条记录的正确性或状态，MATH相对修订版v3.1改变{len(math_changes)}条。此前MATH200题也使用本次参考答案修订后的分数，不能直接照抄旧报告数字。','',
        '|模型|上次MATH200题|本次MATH1824题|合并MATH2024题|GSM8K1319题|','|---|---:|---:|---:|---:|']
    for arm,label in ARMS.items():
        xs=[grouped[n][arm] for n in ('math_old200','math_new1824','math_all','gsm8k')]
        report.append('|'+label+'|'+ '|'.join(f"{x['correct']}/{x['total']}（{x['correct']/x['total']:.2%}）" for x in xs)+'|')
    for name,title in [('math_all','合并MATH2024题的收尾表现'),('gsm8k','GSM8K1319题的收尾表现')]:
        report+=['',f'## {title}','','|模型|格式且EOS|EOS|截断|重复|平均tokens|中位tokens|至少2048tokens|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm,label in ARMS.items():
            x=grouped[name][arm];n=x['total']
            report.append(f"|{label}|{x['format_eos']/n:.2%}|{x['eos']/n:.2%}|{x['truncated']}/{n}（{x['truncated']/n:.2%}）|{x['repeated']/n:.2%}|{x['mean_tokens']:.1f}|{x['median_tokens']:.1f}|{x['at_least_2048_tokens']}/{n}|")
    report+=['','## 配对比较','','每个数都比较同一道题的已有回答；改善表示前者未判对、后者判对，退步反之。未确认仍按未判对统计。',
        '|题集|从Base到|改善题数|退步题数|净变化|退步中后者截断|','|---|---|---:|---:|---:|---:|']
    for name in ('math_all','gsm8k'):
        for arm in ARMS:
            if arm=='base':continue
            x=comparisons[name][arm]
            report.append(f"|{name}|{ARMS[arm]}|{x['gain']}|{x['loss']}|{x['net']:+d}|{x['losses_with_second_truncated']}|")
    report+=['','## 解释','',
        '1. SFT v3确实更常输出规定标签格式，但没有达到“改善收尾且不损害答题正确率”的目标。MATH旧200题、新1824题和GSM8K的自动判对率均低于Base；两种题集截断率也都高于Base。',
        '2. MATH平均长度下降不能单独证明不过度推理：SFT v3中位数明显下降，同时截断、重复更高，说明多数答案较短与少数失败长回答并存。GSM8K的平均长度反而增加。具体内部原因尚不能由这些指标确定，不能把全部差距归因于近期抽查发现的数据错误。',
        '3. GRPO在SFT v3基础上恢复了较多数学正确率，并改善格式、长度及收尾。MATH上三个GRPO仍低于Base；GSM8K上则与Base接近、自动分数略高。',
        '4. GRPO训练步数与效果不呈单调关系。上次MATH200题是1280最高，而新增1824题和合并2024题是1536略高。1024与1536的合并MATH正确数仅差6，GSM8K仅差2，不能认定哪个版本有稳定的普遍优势。',
        '5. 若以本次MATH最终答案正确率为首要标准，Base最强；若必须同时输出规定格式并希望回答较短，GRPO1536是现有候选，GRPO1024也应保留。现有SFT v3不能仅因格式提升就视为成功。这里只是依据已有测试结果选择候选，没有启动新训练。','',
        '## 边界','',
        '- 自动判对率不是人工审核的真实正确率，无法提取、无法解析或不支持表达的回答仍留在分母中，未确认数量在summary.json单列。',
        '- 旧200题已被用于反馈，新增1824题应作为主要确认依据。没有把旧200题重复加到2024题上，也没有把MATH与GSM8K混成一个总分。',
        '- 已做已知训练题的规范化精确去重，未做语义去重，无法排除Base预训练已见题目。6题或32题收尾诊断不混入此benchmark汇总。',
        '- 仅检查最终答案等价和生成行为，不认证每一步推导。重复指标为相同连续20个空白分词片段至少出现3次，不能捕捉所有语义重复。',
        '- 今天新增1390条补充数据尚未训练，本报告不是对新补充数据训练效果的评测。','']
    (out/'ANALYSIS.md').write_text('\n'.join(report),encoding='utf-8')
    for path,h in before_hashes.items():assert sha(Path(path))==h
    for name in ('audited_answer_scoring_v32.py','audited_answer_scoring.py','rescore_fixed_answers.py','report_math_remaining_eval.py',Path(__file__).name):
        shutil.copy2(Path(__file__).with_name(name),out/name)
    save(out/'manifest.json',dict(status='complete_no_training_or_inference',schema=SCHEMA,scorer_sha256=sha(scorer),
        gsm8k_inputs_sha256=before_hashes,math_scores_manifest_sha256=sha(math_root/'manifest.json'),
        source_generation_configs=gsm_manifest['source_configs'],packages=math_manifest['packages'],
        unique_cpu_tasks=len(tasks),workers=4,timeout_seconds=20,gsm8k_score_changes=len(changes),
        math_unit_alias_score_changes=len(math_changes),math_unit_context_questions_recomputed_per_model=len(affected),
        unchanged_math_reuse_proof='Byte comparison restricts v3.2 diff to schema and speed unit context branches. Every question matching either new branch is rescored in every arm; all other mathematical grading operations are identical.',
        outputs_sha256={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps({'gsm8k_correct':{arm:grouped['gsm8k'][arm]['correct'] for arm in ARMS},'version_changes':len(changes),'paired':comparisons},indent=2),flush=True)


if __name__=='__main__':main()
