"""Analyze the frozen 1200-question proposal; never changes data or trains."""
from collections import Counter, defaultdict
import hashlib
import json
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'datasets/math/base_self600_20261015'
OUT = ROOT / 'grpo_supplement_v3'


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(k, n=4):
    z = 1.959963984540054
    d = 1+z*z/n
    center = (k/n+z*z/(2*n))/d
    half = z*sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/d
    return [center-half, center+half]


def main():
    data = read(OUT/'train_expanded.jsonl')
    scores = {r['row_id']: r for r in read(OUT/'v3_scores.jsonl')}
    verdicts = defaultdict(list)
    for r in read(OUT/'sample_verdicts.jsonl'):
        verdicts[r['row_id']].append(r['verdict'])
    history = {r['row_id']: r for r in read(ROOT/'grpo_log_analysis/failure_audit/group_counterfactual.jsonl')}
    previous = read(ROOT/'remote_results/grpo_selected_direct367_v1/train.jsonl')
    manifest = json.loads((OUT/'selection_report.json').read_text(encoding='utf-8'))
    assert digest(OUT/'train_expanded.jsonl') == manifest['train_sha256']
    assert len(data) == len({r['row_id'] for r in data}) == 1200

    def summarize(rows):
        kinds = Counter()
        statuses = Counter()
        for r in rows:
            vs = verdicts[r['row_id']]
            assert len(vs) == 4
            ss = Counter(v['status'] for v in vs)
            assert set(ss) <= {'correct','incorrect','extraction_failed'}
            assert ss['correct'] == r.get('screen_correct_count', scores[r['row_id']]['k'])
            assert 0 < ss['correct'] < 4
            statuses.update(ss)
            kinds['mathematical_incorrect_plus_protocol_failure' if ss['incorrect'] and ss['extraction_failed']
                  else 'mathematical_incorrect_only_negatives' if ss['incorrect']
                  else 'protocol_failure_only_negatives'] += 1
        return dict(questions=len(rows), sample_status_counts=dict(statuses), group_types=dict(kinds),
                    any_mathematical_incorrect_groups=kinds['mathematical_incorrect_plus_protocol_failure']+kinds['mathematical_incorrect_only_negatives'])

    overall = summarize(data)
    assert overall['sample_status_counts'] == dict(correct=2400, extraction_failed=1666, incorrect=734)
    assert sum(overall['group_types'].values()) == 1200
    per_difficulty = {}
    for d,k in [('easy',3),('medium',2),('hard',1)]:
        rows = [r for r in data if r['difficulty']==d]
        per_difficulty[d] = dict(**summarize(rows), observed_k=k,
            observed_task_success=k/4, wilson_95_interval_iid_illustration=interval(k),
            plug_in_next_g4_mixed_probability_iid_illustration=1-(k/4)**4-(1-k/4)**4)
    all_eligible = [s for s in scores.values() if s['eligible']]
    strict_cap = Counter(s['difficulty'] for s in all_eligible
                         if all(v['status'] in ('correct','incorrect') for v in verdicts[s['row_id']]))
    # These are scorer verdict counts, not a mathematical proof of answer validity.
    strict_max = min(strict_cap['easy']//3, strict_cap['medium']//4, strict_cap['hard']//3)*10
    output = dict(status='analyzed', proposal_sha256=manifest['train_sha256'], overall=overall,
        by_difficulty=per_difficulty,
        by_origin={o:summarize([r for r in data if r['selection_role']==o]) for o in ['retained_previous','supplement']},
        extraction_error_counts=dict(Counter(v.get('error') for r in data for v in verdicts[r['row_id']] if v['status']=='extraction_failed')),
        old_new_question_counts=dict(Counter(r['selection_role'] for r in data)),
        old_retained_previous_online_contrast=sum(history[r['row_id']]['original_correctness_contrast'] for r in data if r['row_id'] in history),
        old_retained_previous_online_participating=sum(not history[r['row_id']]['originally_dropped'] for r in data if r['row_id'] in history),
        subjects={s:dict(selected=sum(r['type']==s for r in data),
                         previous=sum(r['type']==s for r in previous),
                         pool=sum(r['type']==s for r in all_eligible),
                         signal=summarize([r for r in data if r['type']==s])) for s in sorted({r['type'] for r in data})},
        difficulty_pool_utilization={d:per_difficulty[d]['questions']/sum(s['difficulty']==d for s in all_eligible) for d in per_difficulty},
        strict_mathematical_verdict_pool_by_difficulty=dict(strict_cap),
        maximum_strict_3_4_3_pool_total=strict_max,
        iid_illustration_expected_mixed_groups=sum(v['questions']*v['plug_in_next_g4_mixed_probability_iid_illustration'] for v in per_difficulty.values()),
        scopes=['Difficulty is SFT cached task success, not intrinsic MATH level or current GRPO competence.',
                'IID mixed-group estimates and intervals are teaching illustrations, not forecasts or calibrated difficulty intervals.',
                'Protocol failure negatives are task failures, not proven mathematically wrong responses.',
                'Historical online contrast counts were observed during an evolving old policy, not fixed final-model rescreening.',
                'No semantic structure coverage audit or new training has been completed.'],
        training_started=False, data_changed=False)
    (OUT/'rationality_analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
