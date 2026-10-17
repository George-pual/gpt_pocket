"""Prepare a directly reviewable, training-only structural-diagnosis seed set."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'datasets/math/base_self600_20261015'
OUT = ROOT / 'reasoning_structure_audit'


def read(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def write(p, rows):
    p.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def main():
    source = {r['row_id']: r for r in read(ROOT / 'remote_results/grpo_reserved.jsonl')}
    scores = read(ROOT / 'grpo_supplement_v3/v3_scores.jsonl')
    verdicts = collections.defaultdict(dict)
    for r in read(ROOT / 'grpo_supplement_v3/sample_verdicts.jsonl'):
        verdicts[r['row_id']][r['sample_index']] = r['verdict']
    zero = [s for s in scores if s['k'] == 0 and s['u'] == 0 and s['bad_end'] == 0
            and s['gold_parser_supported'] and not s['excluded_reason']]
    strict = [s for s in zero if all(v['status'] == 'incorrect' for v in verdicts[s['row_id']].values())]
    assert len(zero) == 344 and len(strict) == 22
    ids = {s['row_id'] for s in strict}
    packets = []
    for p in sorted((ROOT / 'remote_results/grpo_screen_direct367_v1/sft/groups').glob('*.json')):
        group = json.loads(p.read_text(encoding='utf-8'))
        if group['row_id'] not in ids:
            continue
        r = source[group['row_id']]
        packets.append(dict(audit_index=len(packets), row_id=r['row_id'], problem=r['problem'],
                            answer=r['answer'], reference_solution=r['reference_solution'],
                            type=r['type'], level=r['level'],
                            samples=[dict(sample_index=s['sample_index'], completion=s['completion'],
                                          verdict=verdicts[r['row_id']][s['sample_index']]) for s in group['samples']]))
    OUT.mkdir(exist_ok=True)
    write(OUT / 'strict_zero_review_packets.jsonl', packets)
    summary = dict(task_zero_count=len(zero), four_incorrect_count=len(strict),
                   mixed_incorrect_and_format_failure_count=sum(0 < s['protocol_failures'] < 4 for s in zero),
                   four_format_failure_count=sum(s['protocol_failures'] == 4 for s in zero),
                   screening_model='SFT step64, frozen cached four-answer screening',
                   scope='All four automatic final-answer verdicts incorrect is a review trigger, not proof of a reasoning capability deficit.',
                   inputs_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                       ROOT / 'remote_results/grpo_reserved.jsonl', ROOT / 'grpo_supplement_v3/v3_scores.jsonl',
                       ROOT / 'grpo_supplement_v3/sample_verdicts.jsonl']})
    (OUT / 'screening_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
