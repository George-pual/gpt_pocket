"""Review reference completeness and uniformly rescore the seven affected items.

Uses the frozen v3.1 extraction/equivalence code, saved answers only, no inference.
The manually derived policy is constructed without consulting model answers.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from report_math_remaining_eval import ARMS, metrics, read_rows
from rescore_fixed_answers import bounded_map


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def reference_policy():
    reviewed = [
        ('algebra', 414, 'complete_set', ['-2,1'], ['-2', '1', '-2,1,3'],
         'Domain x>=-2. Squaring gives (x+2)(x-1)=0; both -2 and 1 satisfy the original equation.'),
        ('intermediate_algebra', 409, 'complete_set', [r'-\frac32,-\frac34'], [r'-\frac32', r'-\frac34', '3'],
         'x!=-1; y=(x/(x+1))^2 gives y=9. The two signs give -3/2 and -3/4; both are valid.'),
        ('intermediate_algebra', 563, 'complete_set', [r'\frac25,3'], [r'\frac25', '3', '0'],
         'Domain excludes -1,0,1. Clearing denominators gives (x-3)(5x-2)=0; 2/5 and 3 both satisfy the original rational equation.'),
        ('intermediate_algebra', 576, 'one_of', ['(10,0)', '(0,0)'], ['(5,0)', '(10,1)'],
         '(x-5)^2/20-y^2/5=1 has center (5,0), c=sqrt(20+5)=5; either focus is allowed by the question.'),
        ('intermediate_algebra', 772, 'one_of', [r'(\frac32,-\frac52)', r'(-\frac52,-\frac52)'], ['(0,0)', r'(-\frac12,-\frac52)'],
         '(x+1/2)^2/4-(y+5/2)^2/16=1; center (-1/2,-5/2), horizontal vertex offset 2. Either vertex is requested.'),
        ('intermediate_algebra', 796, 'one_of', ['(3,-5)', '(0,0)'], ['(-3,-5)', '(3,5)'],
         'Oddness guarantees (3,-5). Original official solution also accepts (0,0), explicitly assuming zero is in the domain; retain that official convention and record the assumption.'),
        ('precalculus', 108, 'complete_set', [r'30^\circ,45^\circ,105^\circ'], [r'105^\circ', r'30^\circ,45^\circ', '30,45,100'],
         'Cosine law gives sqrt(3)/2 and sqrt(2)/2, hence 30 and 45 degrees. Angle sum gives 105 degrees; all three are required.'),
    ]
    return {'version': 'math-test-reference-completeness-v1',
        'scope': 'Manual review of all eleven distinct-multibox test references; four incomplete sets repaired, three questions allow alternative answers. Other four entries require no change.',
        'unchanged_reviewed_ids': [f'math:{s}/test-00000-of-00001.parquet:{i}' for s, i in
            [('algebra',405),('geometry',62),('number_theory',298),('prealgebra',254)]],
        'policy_frozen_before_reading_affected_model_completions': True,
        'extraction_and_equivalence': 'Unchanged audited-final-answer-v3.1. For complete_set compare against the complete set, never against each member. For one_of accept equivalence to any listed complete option.',
        'entries': [dict(row_id=f'math:{s}/test-00000-of-00001.parquet:{i}', kind=k,
            accepted_complete_references=g, negative_controls=n, independent_justification=proof)
            for s,i,k,g,n,proof in reviewed]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--completed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output
    out.mkdir(exist_ok=False)
    policy = reference_policy()
    save(out/'reviewed_reference_policy.json', policy)
    questions = {r['row_id']:r for r in read_rows(a.bundle/'questions_all.jsonl')}
    entries = {r['row_id']:r for r in policy['entries']}
    queue = read_rows(a.bundle/'reference_multibox_review_queue.jsonl')
    assert {r['row_id'] for r in queue} == set(entries) | set(policy['unchanged_reviewed_ids'])
    tasks, tests = [], []
    for e in entries.values():
        q = questions[e['row_id']]['problem']
        for prediction, expected in [(g,True) for g in e['accepted_complete_references']] + [(n,False) for n in e['negative_controls']]:
            indices=[]
            for gold in e['accepted_complete_references']:
                indices.append(len(tasks))
                tasks.append(('<SOLUTION>'+prediction+'</SOLUTION>',gold,q,True))
            tests.append(dict(row_id=e['row_id'], prediction=prediction, expected_correct=expected, indices=indices))
    verdicts = bounded_map(tasks,4,20)
    for t in tests:
        t['observed_correct'] = any(verdicts[i]['correct'] for i in t.pop('indices'))
        assert t['observed_correct'] == t['expected_correct'], t
    save(out/'reference_policy_controls.json', tests)
    # Existing scores were independently validated against all raw chunks by
    # report_math_remaining_eval.py before this reference-only repair.
    validation=json.loads((a.completed/'comparison_validation.json').read_text('utf-8'))
    assert validation['checks']=='passed' and len(validation['arms'])==5
    old_manifest = json.loads((a.completed/'scored_v31/manifest.json').read_text('utf-8'))
    scorer = Path(__file__).with_name('audited_answer_scoring.py')
    assert sha(scorer)==old_manifest['scorer_sha256']=='e239ae9419b37fb7183b0f5ff562cda3934e87534d7d34072baa6c8229251951'
    summary, changes, review, input_hashes = {}, [], [], {}
    for arm in ARMS:
        source=a.completed/'scored_v31'/f'{arm}.jsonl'
        rows=read_rows(source)
        assert len(rows)==2024 and [r['row_id'] for r in rows]==list(questions)
        input_hashes[arm]=sha(source)
        tasks=[]; spans=[]
        for idx,r in enumerate(rows):
            if r['row_id'] not in entries:
                continue
            e=entries[r['row_id']]; start=len(tasks)
            for gold in e['accepted_complete_references']:
                tasks.append((r['completion'],gold,r['question'],r['eos'] and not r['truncated']))
            spans.append((idx,start,len(tasks)))
        verdicts=bounded_map(tasks,4,20)
        for idx,start,end in spans:
            r=rows[idx]; e=entries[r['row_id']]
            alternatives=verdicts[start:end]
            chosen=next((v for v in alternatives if v['correct']), alternatives[0])
            old=r['audited_score_v3']
            r['previous_last_box_score_v31']=old
            r['scoring_reference_policy']=dict(version=policy['version'], **e)
            r['reference_alternative_verdicts_v31']=alternatives
            r['audited_score_v3']=chosen
            if old['correct']!=chosen['correct'] or old['status']!=chosen['status']:
                changes.append(dict(model=arm,row_id=r['row_id'],previous=old,repaired=chosen))
        # Raw gold, completion and generation metrics are retained byte-for-byte
        # as strings/values; the explicitly separate policy controls scoring.
        for r,original in zip(rows,read_rows(source)):
            for field in ('row_id','question','gold','completion','tokens','eos','truncated','reused','format_eos','repeated_20_word_span_3_times'):
                assert r[field]==original[field],(arm,r['row_id'],field)
            if r['audited_score_v3']['status'] not in ('correct','incorrect'):
                review.append(dict(model=arm,**r))
        summary[arm] = {'all':metrics(rows), 'remaining':metrics([r for r in rows if not r['reused']]),
            'reused':metrics([r for r in rows if r['reused']]),
            'remaining_by_level':{lv:metrics([r for r in rows if not r['reused'] and r['level']==lv]) for lv in ('Level 2','Level 3')}}
        write_rows(out/f'{arm}.jsonl',rows)
    save(out/'summary.json',summary)
    write_rows(out/'score_changes.jsonl',changes)
    write_rows(out/'review_queue.jsonl',review)
    shutil.copy2(scorer,out/scorer.name)
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('rescore_fixed_answers.py'),out/'rescore_fixed_answers.py')
    shutil.copy2(Path(__file__).with_name('report_math_remaining_eval.py'),out/'report_math_remaining_eval.py')
    write_report(out,summary,changes)
    save(out/'manifest.json',dict(status='complete_saved_answers_only_no_training_or_inference',
        reference_policy_version=policy['version'], reference_policy_sha256=sha(out/'reviewed_reference_policy.json'),
        scorer_schema='audited-final-answer-v3.1', scorer_sha256=sha(scorer),
        packages=old_manifest['packages'], workers=4, task_timeout_seconds=20,
        runner_sha256=sha(Path(__file__)), source_scored_inputs_sha256=input_hashes,
        source_validation_sha256=sha(a.completed/'comparison_validation.json'),
        original_archive_sha256=sha(a.bundle/'results_and_scores.tar.gz'),
        repaired_reference_items=7, reference_controls_passed=len(tests),
        rows_per_arm=2024, new_rows_per_arm=1824, reused_rows_per_arm=200,
        score_changes=len(changes), raw_answers_and_metrics_unchanged=True,
        unchanged_items='Reuse exact frozen v3.1 scores; same inputs and scorer, only seven reviewed references changed.',
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({'changes':len(changes),'remaining':{arm:summary[arm]['remaining']['correct'] for arm in ARMS}},indent=2))


def write_report(out, summary, changes):
    lines=['# MATH Level 2/3 五模型结果（参考答案完整性修订版）','',
        '五模型生成和结果下载已完成。统一使用冻结的 audited-final-answer-v3.1；本次只修订参考答案策略并复评分，没有重新生成回答或训练模型。',
        '抽查补充训练数据时发现“最后一个boxed”可能漏掉多答案，于是审核了测试集中全部11条存在不同boxed文本的参考解答。4题需要完整答案集，3题允许任选一个有效答案，其余4题无需改动。',
        '原题面、原gold、全部模型回答及旧评分保留；修订策略另存reviewed_reference_policy.json，并统一应用于五模型。完整集题不会只答一个解就判对。',
        '主要结果为此次新增1824题（L2=794，L3=1030）；此前200题曾用于反馈，合并2024题只作补充。官方2025题中有1题与已知训练题精确重复，已排除。','',
        '|模型|新增题自动判对|判对率|未确认|EOS率|截断率|重复率|平均tokens|格式且EOS|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,label in ARMS.items():
        x=summary[arm]['remaining'];n=x['total']
        lines.append(f"|{label}|{x['correct']}/{n}|{x['correct']/n:.2%}|{x['unconfirmed']}|{x['eos']/n:.2%}|{x['truncated']/n:.2%}|{x['repeated']/n:.2%}|{x['mean_tokens']:.1f}|{x['format_eos']/n:.2%}|")
    lines += ['', '|模型|新增L2|新增L3|全部2024题|', '|---|---:|---:|---:|']
    for arm,label in ARMS.items():
        a,b=[summary[arm]['remaining_by_level'][lv] for lv in ('Level 2','Level 3')]
        x=summary[arm]['all']
        lines.append(f"|{label}|{a['correct']}/{a['total']}（{a['correct']/a['total']:.2%}）|{b['correct']}/{b['total']}（{b['correct']/b['total']:.2%}）|{x['correct']}/2024（{x['correct']/2024:.2%}）|")
    lines += ['',f'参考答案修订共改变{len(changes)}条回答的判定，详见score_changes.jsonl。',
        '与背景任务自动生成的scored_v31历史报告相比，应采用本修订报告。参考答案能对自己判对，不等于参考答案完整；原自检没有发现这种遗漏。',
        '固定解码：temperature=0、top_p=1、top_k=-1、3894输出tokens、EOS=151643，沿用原自定义prompt；没有Qwen thinking模式开关。',
        '未确认回答按未判对计入自动判对率，不能一律视为数学错误。这里衡量最终答案，不代表推理每一步都正确。',
        'EOS与标签格式分别统计。重复率为同一连续20个空白分词片段出现至少3次的启发式指标，不能覆盖所有过度推理。',
        'SFT v3的格式率高于Base，但本子集的自动判对率、截断率和重复率更差；平均回答更短不能单独证明收尾改善。此结论针对已有SFT v3权重，今天补充的1390条数据尚未训练。',
        '仅做规范化精确去重，没有语义去重，无法排除预训练污染。与此前6题长度诊断分开解释。',
        '奇函数题796按官方参考接受(3,-5)和(0,0)；后者假设定义域包含0，已在策略中明确保留官方约定。','',
        '复现：先运行原report_math_remaining_eval.py验证145个chunk及五模型题序，再运行repair_math_test_references.py --bundle <原bundle> --completed <下载结果目录> --output <新的修订评分目录>。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
