"""Audit completed screening outputs without changing original scores or jobs."""
import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/grpo_candidates_gsm8k_math_l2_l3'
RUN = DATA / 'E1_sft_v3_grpo_screen_shard0_n2500'
OUT = RUN / 'analysis'


def read(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    rows = read(RUN / 'scored_questions.jsonl')
    raw = {r['row_id']: r for r in read(RUN / 'raw_generations.jsonl')}
    pool = read(DATA / 'grpo_candidates.jsonl')[:2500]
    questions = {r['row_id']: r for r in pool}
    assert len(rows) == len(raw) == len(questions) == 2500
    assert len({r['row_id'] for r in rows}) == 2500
    assert {r['row_id'] for r in rows} == raw.keys() == questions.keys()
    for row in rows:
        assert row['source'] == 'gsm8k' and row['answer'] == questions[row['row_id']]['answer']
        assert len(row['samples']) == 4
        assert {s['sample_index'] for s in row['samples']} == {0, 1, 2, 3}
        assert row['correct_count'] == sum(s['correct'] for s in row['samples'])
        assert row['normal_correct_count'] == sum(s['correct'] and s['normal_end'] for s in row['samples'])
        for a, b in zip(row['samples'], raw[row['row_id']]['samples']):
            for key in ['completion', 'tokens', 'finish_reason', 'stop_reason', 'sample_index']:
                assert a[key] == b[key]
    samples = [s for r in rows for s in r['samples']]
    mixed = [r for r in rows if r['group'] == 'mixed']
    flags = []
    for row in rows:
        for sample in row['samples']:
            if sample['correct']:
                continue
            prediction = sample['math_verification'].get('prediction')
            if not prediction:
                continue
            match = re.fullmatch(r'([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*([A-Za-z][A-Za-z ]{0,45})?\.?',
                                 prediction.strip().strip('$').strip())
            if not match:
                continue
            try:
                equal = Decimal(match[1].replace(',', '')) == Decimal(row['answer'])
            except InvalidOperation:
                equal = False
            if equal:
                flags.append({'row_id': row['row_id'], 'sample_index': sample['sample_index'],
                              'category': 'unit_suffix_review' if match[2] else 'numeric_punctuation',
                              'prediction': prediction, 'gold': row['answer'],
                              'original_status': sample['math_verification']['status'],
                              'problem': questions[row['row_id']]['problem'],
                              'completion': sample['completion']})
    write(OUT / 'grading_review_flags.jsonl', flags)
    # This is a sensitivity calculation only; do not overwrite any original grades.
    pure_flags = {(f['row_id'], f['sample_index']) for f in flags if f['category'] == 'numeric_punctuation'}
    adjusted_groups, changed = Counter(), []
    for row in rows:
        k = sum(s['correct'] or (row['row_id'], s['sample_index']) in pure_flags for s in row['samples'])
        group = 'mixed' if 0 < k < 4 else 'all_correct' if k == 4 else 'all_wrong'
        adjusted_groups[group] += 1
        if group != row['group']:
            changed.append({'row_id': row['row_id'], 'old_group': row['group'], 'sensitivity_group': group})
    lengths = sorted(s['tokens'] for s in samples)
    stats = {'questions': len(rows), 'samples': len(samples), 'source': 'First 2500 GSM8K training candidates, not a held-out benchmark',
             'groups': dict(Counter(r['group'] for r in rows)),
             'correct_count_histogram': dict(Counter(r['correct_count'] for r in rows)),
             'correct_samples': sum(s['correct'] for s in samples),
             'observed_any_correct_questions': sum(r['correct_count'] > 0 for r in rows),
             'format_valid': sum(s['format_valid'] for s in samples),
             'eos_finished': sum(s['eos_finished'] for s in samples),
             'normal_end': sum(s['normal_end'] for s in samples),
             'correct_and_normal_end': sum(s['normal_end'] and s['correct'] for s in samples),
             'truncated': sum(s['truncated'] for s in samples),
             'invalid_gold_questions': sum(not r['gold_verification']['correct'] for r in rows),
             'verification_status': dict(Counter(s['math_verification']['status'] for s in samples)),
             'tokens': {'mean': statistics.mean(lengths), 'median': statistics.median(lengths),
                        'p90': lengths[int(.90*(len(lengths)-1))], 'p95': lengths[int(.95*(len(lengths)-1))],
                        'max': max(lengths)},
             'mixed_with_at_least_one_normal_correct': sum(r['normal_correct_count'] > 0 for r in mixed),
             'mixed_without_normal_correct': sum(r['normal_correct_count'] == 0 for r in mixed),
             'mixed_with_parsed_wrong': sum(any(s['math_verification']['status'] == 'incorrect' for s in r['samples']) for r in mixed),
             'mixed_without_parsed_wrong': sum(not any(s['math_verification']['status'] == 'incorrect' for s in r['samples']) for r in mixed),
             'grading_review_flags': dict(Counter(f['category'] for f in flags)),
             'numeric_punctuation_only_sensitivity': {'groups': dict(adjusted_groups), 'changed_questions': changed,
                                                     'not_applied_to_original_outputs': True},
             'source_hashes': {name: hashlib.sha256((RUN/name).read_bytes()).hexdigest() for name in
                               ['scored_questions.jsonl', 'raw_generations.jsonl', 'run_manifest.json']}}
    (OUT / 'analysis_statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT / 'question_inventory.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row_id','group','correct_of_4','normal_correct_of_4','normal_end_of_4','mean_tokens'])
        for row in rows:
            writer.writerow([row['row_id'],row['group'],row['correct_count'],row['normal_correct_count'],row['normal_end_count'],
                             statistics.mean(s['tokens'] for s in row['samples'])])
    # Original screening membership is unchanged. Provide optional diagnostic subsets.
    mixed_records = {r['row_id']: r for r in read(RUN / 'mixed.jsonl')}
    assert mixed_records.keys() == {r['row_id'] for r in mixed}
    write(OUT / 'mixed_with_normal_correct_1735.jsonl',
          [mixed_records[r['row_id']] for r in mixed if r['normal_correct_count'] > 0])
    write(OUT / 'mixed_without_normal_correct_177.jsonl',
          [mixed_records[r['row_id']] for r in mixed if r['normal_correct_count'] == 0])
    by_id = {r['row_id']: r for r in rows}
    case_notes = {
        2: '标准5：100/2+15+30=95，差5。四条里只有一条匹配；其他回答错读关系或把50+15+30算成80。体现可学习的数学错误。',
        3: '标准42：一条输出42 pages，推导和数值都正确，但原评分incorrect。单位后缀误判已确认；另一条12是真实错误。',
        11: '标准5：(95+15-3*5-4*15)/7=5。四条均未匹配；可见基础算术和收支建模错误。全错四次不证明永远无法解出。',
        13: '标准35：10+5+20=35。唯一匹配样本缺正确结束标签，其他样本将两磅总价10误认为每磅10。原mixed组没有合规正确样本。',
        15: '标准448000：(15-6)*500*5*20-2000。四条均错；有回答把一次性制作费2000每周扣除，得到410000。',
        17: '标准43：65-4-6-12=43。一条完整算出43却用错标签；另一条先算对43，后续写其他题目并覆盖答案。',
        27: '标准400：800*(1-0.7)+400*(1-3/5)。四条均错，混淆第二批文件与全部文件的删除比例。',
        29: '标准1000：0.4x=0.25(x+600)。一条推导及首个答案正确，但继续生成其他题目，触发多答案提取失败。停止问题不能等同于不会数学。',
        40: '标准476：10*40+2*38。唯一匹配样本使用错误结束标签，另外回答有错用折扣和完全跑题。',
    }
    cases = []
    for index, note in case_notes.items():
        key = f'gsm8k:main:train:{index}'
        cases.append({'row_id': key, 'problem': questions[key]['problem'], 'gold': questions[key]['answer'],
                      'review_note': note, 'scored_record': by_id[key]})
    (OUT / 'reviewed_examples.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    report = '''# SFT v3：首2500道GSM8K候选题筛选分析

## 数据和结论

对象为GSM8K训练候选的前2500题，每题vLLM独立采样4条，共10000条，T=1.0、BF16 Base + v3 LoRA、最大新token3894。不是独立测试集、不是MATH、不是对此前100道混合题的复测，也没有对应Base结果。因此不能以55%对比此前1%来声称SFT提升54个百分点。

当前规则下，1912题（76.48%）出现组内正确性差异，适合作为GRPO候选；这证明在该批题及该采样配置下存在较多可利用的奖励差异，不证明后续GRPO训练一定有效或收敛。判分误差需先处理。

## 原始筛选结果（未改分）

| 四次匹配次数 | 题数 | 比例 |
|---|---:|---:|
| 0 | 239 | 9.56% |
| 1 | 449 | 17.96% |
| 2 | 734 | 29.36% |
| 3 | 729 | 29.16% |
| 4 | 349 | 13.96% |

1–3次匹配合计1912题。总匹配5500/10000=55.00%。四次至少一次匹配2261/2500=90.44%（观察到的pass@4）。四次全错/全对只是有限采样结果，不能当成真实成功概率等于0/1。

## 输出行为

| 指标 | 结果 |
|---|---:|
| EOS结束 | 9992/10000，99.92% |
| 格式完整且EOS结束 | 6535/10000，65.35% |
| 答案匹配且格式完整结束 | 4059/10000，40.59% |
| 长度上限截断 | 8/10000，0.08% |
| 平均/中位新token | 134.15 / 101 |

EOS结束不代表标签正确。3465条格式不完整，不能主要归咎于8条截断。典型问题是</start_working_out>替代规定结束标记、重复答案块，以及答完后续写其他题目。

在原1912题中，1735题至少有一条“答案匹配且格式完整结束”的样本；177题虽然有匹配数字，却没有合规正确样本。另存了这两组用于诊断，没有替换原mixed.jsonl。

## 判分问题：需要在训练奖励中修正

原状态：correct 5500、incorrect 3085、extraction_failed 1137、prediction_parse_failed 278。原始记录2500题的标准答案均能解析。

抽查确认：train:3的最终答案42 pages对应标准42，推导正确，却被判incorrect。扫描已提取的答案，找到252条“显式单一数字等于gold、但原评分失败”的复核候选：47条纯数字/句末标点形式，205条带英文后缀。205条必须核查单位和数量级，不能一律去掉后缀判对，例如4 million dollars中的million不能随便忽略。

这只是保守的规则扫描，未搜索推理里的最后数字，也没有从不闭合/多答案块中强行救答案；可能还有遗漏。正确最终数值也不证明推导每一步正确。

405个原mixed组的所有不匹配样本都属于提取失败或解析失败；不能确认这些组的差异来自数学能力。其余1507组至少含一条已解析却判不等价的样本，但这里也可能混入单位写法的误判。

建议：保留原始评分，先制定GSM8K专用的数字、标点和单位处理规则，再对现有10000条回答重判，无需重新推理。MATH表达式评分应保持独立规则；不要通过任意抽取最后一个数字修复。

## 代表性案例

'''
    for case in cases:
        report += f"- **{case['row_id']}**：{case['review_note']}\n\n"
    report += '''## 建议使用方式

- 当前1912题是按原规则得到的候选量，尚不应当成最终无误的GRPO训练量。
- 1735题可作为更严格的优先候选参考，但同样需要重判，不能自动视作人工确认的优质数学题。
- 349题四次全对可留作容易题/格式检查组，239题四次全错可留作难题和错误诊断组；不要把这批训练候选当作独立评测数据。
- 当前任务继续筛选其他分片；本次分析未修改正在运行的奖励函数、筛选阈值、原始回答或分组。

## 文件

analysis_statistics.json保存可复算统计；question_inventory.csv逐题汇总；grading_review_flags.jsonl包含252条复核候选的题面和原文；reviewed_examples.json包含9道代表题的全部4条回答。下载副本与原始raw/scored回答逐样本核对一致。
'''
    (OUT / 'ANALYSIS_2500.md').write_text(report, encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
