"""Audit archived overnight results and compare the same internal validation set."""
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'datasets/grpo_candidates_gsm8k_math_l2_l3'
OUT = ROOT / 'half_math_v3/results'
OUT.mkdir(exist_ok=True)
with tarfile.open(OUT.parent / 'half_math1024_results.tar.gz') as archive:
    for member in archive.getmembers():
        if member.isfile():
            target = OUT / member.name
            assert target.resolve().is_relative_to(OUT.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())

summaries, failures = {}, {}
ids = None
for step in (256, 512, 1024):
    folder = OUT / f'eval_step_{step}'
    summary = json.loads((folder / 'summary.json').read_text())
    rows = [json.loads(line) for line in (folder / 'answers.jsonl').read_bytes().splitlines()]
    groups, counts = defaultdict(list), Counter()
    for row in rows:
        groups[row['row_id']].append(row)
        if not row['format_eos']:
            c = row['completion']
            if not isinstance(c, str):
                c = '\n'.join(x['content'] for x in c)
            counts['wrong_end_tag'] += '</start_working_out>' in c
            counts['missing_or_multiple_end_tag'] += c.count('<end_working_out>') != 1
            counts['solution_tags'] += c.count('<SOLUTION>') != 1 or c.count('</SOLUTION>') != 1
            counts['trailing_text'] += '</SOLUTION>' in c and not c.strip().endswith('</SOLUTION>')
    assert len(rows) == 400 and len(groups) == 100
    assert all(len(v) == 4 and {r['sample_index'] for r in v} == {0, 1, 2, 3} for v in groups.values())
    assert ids is None or set(groups) == ids
    ids = set(groups)
    for metric in ('correct', 'format_eos', 'truncated', 'eos'):
        assert sum(bool(r[metric]) for r in rows) == summary['overall'][metric]
    assert sum(bool(r['correct'] and r['format_eos']) for r in rows) == summary['overall']['normal_correct']
    summaries[str(step)] = summary
    failures[str(step)] = dict(counts)

shutdown = json.loads((OUT / 'auto_shutdown_status.json').read_text())
for rel, digest in shutdown['artifact_hashes'].items():
    p = OUT / rel
    if p.is_file():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, rel
assert shutdown['status'] == 'validated_shutdown_requested'
assert (OUT / 'stage.txt').read_text().strip() == 'complete'
assert (OUT / 'pipeline.exitcode').read_text().strip() == '0'
completion = json.loads((OUT / 'B_components/completion_summary.json').read_text())
assert completion['status'] == 'complete' and completion['steps'] == 1024
assert len({s['data_sha256'] for s in summaries.values()}) == 1
assert all(s['sampling'] == summaries['256']['sampling'] for s in summaries.values())
report = {'summaries': summaries, 'format_failures_nonexclusive': failures,
          'training': completion, 'archive_checks_passed': True}
(OUT / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# 1024步训练结果', '', '同一内部验证集100题，每题4次，Temperature=1.0。', '',
         '| 步数 | 正确 | 格式完整 | 正确且格式完整 | 截断 |', '|---|---:|---:|---:|---:|']
for step, summary in summaries.items():
    o = summary['overall']
    lines.append(f"| {step} | {o['correct']}/400 | {o['format_eos']}/400 | {o['normal_correct']}/400 | {o['truncated']} |")
lines += ['', '1024步为本轮三个checkpoint中综合最佳。训练约3小时59分；三个评测合计约4分41秒，不含模型加载。',
          '格式问题仍有77/400条；该内部验证集已用于选checkpoint，不能直接作为独立泛化成绩。',
          '1024步使用2048个题组，未遍历3816题的整个候选池。', '',
          '格式错误计数（可重叠）：', json.dumps(failures, ensure_ascii=False, indent=2)]
(OUT / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'overall': {k: v['overall'] for k, v in summaries.items()}, 'failures': failures, 'checks': 'passed'}, ensure_ascii=False, indent=2))
