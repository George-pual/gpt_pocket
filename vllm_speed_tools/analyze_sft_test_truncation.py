"""Audit all saved SFT test truncations; preserve original benchmark grading."""
import json,re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'datasets/gsm8k/base_sftv3_test/results'
OUT=ROOT/'sft_truncation_audit';OUT.mkdir(exist_ok=True)
norm=lambda s:re.sub(r'\s+',' ',s).strip()
# Individually inspected exceptions to the repeated-question text detector.
PARAPHRASED_COPY={'gsm8k:main:test:777','gsm8k:main:test:1193','gsm8k:main:test:1295'}
DECIMAL={'gsm8k:main:test:491','math:number_theory/test-00000-of-00001.parquet:199','math:prealgebra/test-00000-of-00001.parquet:710'}
FORMULA={'math:counting_and_probability/test-00000-of-00001.parquet:305','math:intermediate_algebra/test-00000-of-00001.parquet:510','math:intermediate_algebra/test-00000-of-00001.parquet:626'}
ENUMERATION={'gsm8k:main:test:236','gsm8k:main:test:301','math:algebra/test-00000-of-00001.parquet:391','math:intermediate_algebra/test-00000-of-00001.parquet:92','math:number_theory/test-00000-of-00001.parquet:429','math:prealgebra/test-00000-of-00001.parquet:371','math:precalculus/test-00000-of-00001.parquet:281'}
INEQUALITY={'math:intermediate_algebra/test-00000-of-00001.parquet:628'}
records=[];summaries={}
for dataset,folder in [('gsm8k','results_vllm'),('math','results_math_l2_l3')]:
    rows=[json.loads(s) for s in (ROOT/folder/'sft_v3/answers.jsonl').read_bytes().splitlines()]
    bad=[q for q in rows if q['truncated']]
    for q in bad:
        c=q['completion'];lines=Counter(norm(s) for s in c.splitlines() if s.strip())
        copied=norm(c).count(norm(q['question']))>=3 or any(n>=5 and len(line)>=40 and (line in norm(q['question']) or norm(q['question']) in line) for line,n in lines.items())
        if copied or q['row_id'] in PARAPHRASED_COPY:category='repeated_question'
        elif q['row_id'] in DECIMAL:category='unbounded_decimal_expansion'
        elif q['row_id'] in FORMULA:category='repeated_formula'
        elif q['row_id'] in ENUMERATION:category='unproductive_enumeration_or_recursion'
        elif q['row_id'] in INEQUALITY:category='unproductive_inequality_recursion'
        else:raise AssertionError(('Unreviewed case',q['row_id']))
        assert q['tokens']==3894 and not q['eos']
        records.append(dict(q,dataset=dataset,review_category=category,
                            end_tag_count=c.count('<end_working_out>'),solution_open_count=c.count('<SOLUTION>'),
                            solution_close_count=c.count('</SOLUTION>')))
    part=[q for q in records if q['dataset']==dataset]
    summaries[dataset]=dict(total_questions=len(rows),truncated=len(part),categories=dict(Counter(q['review_category'] for q in part)),
                           with_end_tag=sum(q['end_tag_count']>0 for q in part),with_solution_open=sum(q['solution_open_count']>0 for q in part),with_solution_close=sum(q['solution_close_count']>0 for q in part))
false_positives=[q['row_id'] for q in records if q['correct'] and q['review_category']=='repeated_question']
assert len(records)==81 and len(false_positives)==2
(OUT/'truncated_answers_reviewed.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in records),encoding='utf-8')
report=dict(status='reviewed_saved_outputs',summaries=summaries,false_positive_correct_ids=false_positives,
            original_metrics_preserved=True,diagnostic_gsm8k_correct_excluding_identified_false_positives=1050,
            diagnostic_gsm8k_accuracy=1050/1319,
            limitation='Output-pattern diagnosis only; does not establish training-data causality or rule out inference-engine effects. Enumeration includes finite but inefficient expansions.')
(OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# SFT v3截断回答检查','','共检查81条：GSM8K 68条、MATH 13条，全部达到3894新token上限。','','| 数据集 | 复述题目 | 小数展开 | 公式重复 | 枚举/递推 | 不等式反复递推 |','|---|---:|---:|---:|---:|---:|']
for name,s in summaries.items():
    c=s['categories'];lines.append(f"| {name} | {c.get('repeated_question',0)} | {c.get('unbounded_decimal_expansion',0)} | {c.get('repeated_formula',0)} | {c.get('unproductive_enumeration_or_recursion',0)} | {c.get('unproductive_inequality_recursion',0)} |")
lines+=['','GSM8K的68条全部没有推理结束标签及答案标签；MATH有12条没有推理结束标签，1条打开答案标签后不断枚举整数，全部没有关闭答案标签。',
'GSM8K两条数字回退判对是误判：'+', '.join(false_positives)+'。原评测不覆盖，排除这两例后的诊断正确率为1050/1319=79.61%。Base其余数字回退样本尚未完整人工审计，此项不是两模型统一重评分。',
'增加生成上限预计主要延长这些重复/枚举过程，不能作为主要修复；逐项枚举不一定是无限循环。',
'现有输出说明生成退化和不会收束；未证明是SFT样本、训练实现或推理引擎中的哪一项造成。应比较GRPO1024同题表现，并用少量相同Prompt在Transformers复现开头重复，排查引擎差异。']
(OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))

