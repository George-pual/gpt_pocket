"""Versioned manual adjudications of known MATH false negatives; never overwrite scores.

Reviews extracted answers for all four arms using the same semantic criteria.
This is an additive confirmed-error audit, not a replacement benchmark grader.
"""
import json,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'datasets/gsm8k/base_sftv3_test/results'
OUT=R/'math_grader_audit_v1';OUT.mkdir(exist_ok=True)
PATHS={'base':R/'results_math_l2_l3/base/answers.jsonl','sft_v3':R/'results_math_l2_l3/sft_v3/answers.jsonl','grpo1024':R/'grpo1024_test/math/grpo1024/answers.jsonl','grpo1280':R/'grpo1280_test/math/grpo1280/answers.jsonl'}
def rid(topic,index):return f'math:{topic}/test-00000-of-00001.parquet:{index}'
# A priori semantic criteria applied across arms: unique stated final value;
# exact symbolic equivalence; time-string equivalence; units already requested
# by the question. Percent-context and malformed tags are reported separately.
RULES={
 rid('algebra',634):('clear_answer_in_sentence',{'base'},'Final sentence explicitly states degree 5.'),
 rid('algebra',977):('explicit_unit',{'base'},'368 kilowatts is the requested rounded power, gold 368.'),
 rid('counting_and_probability',355):('clear_answer_in_sentence',{'base'},'Final sentence explicitly states 13/204.'),
 rid('geometry',234):('explicit_unit',{'base'},'80 square centimeters is gold area 80.'),
 rid('geometry',354):('clear_answer_in_sentence',{'base'},'Final sentence explicitly states 3 new containers.'),
 rid('prealgebra',460):('clear_answer_in_sentence',{'base'},'Final sentence explicitly states count 3; format score remains false.'),
 rid('prealgebra',298):('time_string',set(PATHS),'4:30 p.m. equals the reference time.'),
 rid('precalculus',184):('plain_sqrt',{'base','sft_v3','grpo1024'},'The two roots 2+sqrt(6),2-sqrt(6) exactly match reference; 1280 already scored correct.'),
 rid('precalculus',395):('imaginary_unit_parser_assumptions',{'base','grpo1024','grpo1280'},'Negative imaginary term matches gold; SFT has positive imaginary term and remains wrong.'),
}
CONTEXT={rid('prealgebra',635):({'base'},'Question asks what percent; 90% matches bare reference 90.'),rid('counting_and_probability',269):({'sft_v3','grpo1024','grpo1280'},'Question asks percent; reasoning states 100%, final answer 100 omits percent symbol. Contextually correct, unit omission retained.')}
MALFORMED={rid('intermediate_algebra',805):({'base'},'Correct boxed -5/7 but duplicate opening SOLUTION and missing close. Do not count in strict-extraction correction.')}
records=[];summary={};input_hashes={}
for arm,p in PATHS.items():
    rows=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==200 and len({x['row_id'] for x in rows})==200
    input_hashes[arm]=hashlib.sha256(p.read_bytes()).hexdigest()
    changes=[];context=[];malformed=[]
    for x in rows:
        key=x['row_id'];cat=None;note=None;scope=None
        if key in RULES and arm in RULES[key][1]:cat,_,note=RULES[key];scope='confirmed_answer_expression_false_negative';changes.append(key)
        elif key in CONTEXT and arm in CONTEXT[key][0]:cat='percent_question_context';note=CONTEXT[key][1];scope='contextual_unit_interpretation';context.append(key)
        elif key in MALFORMED and arm in MALFORMED[key][0]:cat='correct_math_malformed_tags';note=MALFORMED[key][1];scope='mathematics_only_invalid_format';malformed.append(key)
        if scope:
            assert not x['correct'],(arm,key,'unexpected original score')
            records.append(dict(arm=arm,row_id=key,category=cat,scope=scope,note=note,question=x['question'],gold=x['gold'],prediction=x['prediction'],completion=x['completion'],original_verdict=x['strict_verdict'],original_format_complete=x['format_eos'],original_correct=x['correct']))
    raw=sum(x['correct'] for x in rows)
    summary[arm]=dict(questions=200,original_correct=raw,original_accuracy=raw/200,confirmed_expression_false_negatives=len(changes),confirmed_expression_adjusted_correct=raw+len(changes),confirmed_expression_adjusted_accuracy=(raw+len(changes))/200,additional_percent_context_cases=len(context),context_inclusive_adjusted_correct=raw+len(changes)+len(context),context_inclusive_adjusted_accuracy=(raw+len(changes)+len(context))/200,correct_math_invalid_tags_not_in_adjusted_scores=len(malformed),original_verdict_statuses=dict(Counter(x['strict_verdict']['status'] for x in rows)))
assert [summary[a]['confirmed_expression_false_negatives'] for a in PATHS]==[9,2,3,2]
assert len(records)==21
payload=dict(schema='manual-confirmed-math-false-negative-audit-v1',scope='Confirmed additive corrections only; original positives were not exhaustively audited, not a full replacement benchmark. No inference, training reward or original outputs changed.',input_sha256=input_hashes,models=summary)
(OUT/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'adjudications.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
