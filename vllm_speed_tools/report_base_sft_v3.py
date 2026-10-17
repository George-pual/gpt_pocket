"""Write a comparison report once the four evaluation cells have finished."""
import argparse
import json
import random
import time
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def paired_interval(deltas):
    rng = random.Random(3407)
    n = len(deltas)
    estimates = sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return [estimates[250], estimates[9749]]


p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--wait', action='store_true')
p.add_argument('--exitcode', type=Path)
a = p.parse_args()
while not (a.output/'comparison.json').exists():
    if not a.wait:
        raise SystemExit('Comparison is not complete')
    if a.exitcode and a.exitcode.exists() and a.exitcode.read_text().strip() != '0':
        raise SystemExit('Evaluation failed; inspect the evaluation log')
    time.sleep(30)
report = json.loads((a.output/'comparison.json').read_text())
arms = report['arms']
lines = ['# Base 与 SFT v3 对照评测', '',
         '100 道同源独立留出题；两模型各 100 次 Greedy 和 400 次 Temperature=1.0、G=4 生成。', '',
         '| 模型 / 解码 | 正确率 | 正常结束率 | EOS 结束率 | 格式合格率 | 重复率 | 截断率 | 平均生成 tokens |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for arm in ('base', 'sft_v3'):
    for mode in ('greedy', 'sample_t1_g4'):
        s = arms[arm+'/'+mode]
        values = [s[k+'_rate'] for k in ('correct', 'normal_end', 'eos_finished', 'format_valid', 'repetitive', 'truncated')]
        lines.append('| '+arm+' / '+mode+' | '+' | '.join(f'{v:.1%}' for v in values)+f' | {s["mean_tokens"]:.1f} |')
lines += ['', '| 模型 / T=1、G=4 | 至少一次答对（观测 pass@4） | 同组有对有错 | 四次全错 | 唯一回答比例 |',
          '|---|---:|---:|---:|---:|']
for arm in ('base', 'sft_v3'):
    s = arms[arm+'/sample_t1_g4']
    lines.append('| '+arm+' | '+' | '.join(f'{s[k]:.1%}' for k in
        ('observed_pass_at_4', 'mixed_correctness_group_rate', 'all_wrong_group_rate', 'mean_unique_completion_fraction'))+' |')
paired = {}
for mode in ('greedy', 'sample_t1_g4'):
    by_arm = {}
    for arm in ('base', 'sft_v3'):
        groups = {}
        for r in records(a.output/arm/mode/'rollouts.jsonl'):
            groups.setdefault(r['row_id'], []).append(r)
        by_arm[arm] = groups
    assert by_arm['base'].keys() == by_arm['sft_v3'].keys()
    deltas = [sum(r['correct'] for r in by_arm['sft_v3'][qid])/len(by_arm['sft_v3'][qid])
              -sum(r['correct'] for r in by_arm['base'][qid])/len(by_arm['base'][qid]) for qid in by_arm['base']]
    paired[mode] = {'correctness_difference': sum(deltas)/len(deltas),
                    'paired_question_bootstrap_95_interval': paired_interval(deltas),
                    'bootstrap_resamples': 10000, 'seed': 3407}
    ci = paired[mode]['paired_question_bootstrap_95_interval']
    lines += ['', f'{mode} 正确率差（SFT−Base）：{sum(deltas)/len(deltas)*100:+.1f} 个百分点；'
              f'按题配对 bootstrap 95% 区间 [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] 个百分点。']
lines += ['', '若正确率变化不大，而正常结束率、格式和重复情况改善，应视为符合冷启动目标；不要求 SFT 单独显著提高解题正确率。', '',
          '范围：这是同源留出诊断，不是外部独立 benchmark。已排除本轮及既有 SFT 训练/验证题与此前已生成题号；不保证语义或预训练去污染。',
          '数学评分使用 Math-Verify 最终答案等价判断，不证明推导正确。采样以题为单位汇总，四次采样不作为四道独立试题计算区间。',
          '两边使用同一原始 BF16 Base 权重、模板、EOS、3894 tokens 上限与解码配置；SFT 增加冻结的 v3 adapter。训练使用 NF4 QLoRA，当前评测是 BF16 部署表示。',
          '完整配置、原始回答和各项验证状态见 manifest.json、四个 rollouts.jsonl、comparison.json。', '']
(a.output/'paired_statistics.json').write_text(json.dumps(paired, indent=2)+'\n', encoding='utf-8')
(a.output/'COMPARISON.md').write_text('\n'.join(lines), encoding='utf-8')
print('REPORT_SAVED', a.output/'COMPARISON.md')
