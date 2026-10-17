"""Task-success eligibility; preserve the frozen mathematical verifier's verdicts."""
POLICY = 'task_success_v1'
CONFIRMED_TASK_STATUSES = {'correct', 'incorrect', 'extraction_failed'}


def unconfirmed(verdict):
    return verdict['status'] not in CONFIRMED_TASK_STATUSES


def summarize(group, gold):
    samples = group['samples']
    k = sum(bool(s['correct']) for s in samples) if len(samples) == 4 else None
    u = sum(unconfirmed(s['verdict']) for s in samples)
    bad_end = sum(not s['eos'] or s['truncated'] for s in samples)
    supported = bool(gold['correct'])
    eligible = bool(len(samples) == 4 and not group.get('excluded_reason')
                    and supported and not u and not bad_end and 0 < k < 4)
    return dict(row_id=group['row_id'], k=k, u=u, bad_end=bad_end,
                protocol_failures=sum(s['verdict']['status'] == 'extraction_failed' for s in samples),
                eligible=eligible, gold_parser_supported=supported,
                difficulty={3: 'easy', 2: 'medium', 1: 'hard'}.get(k),
                level=group['level'], type=group['type'], excluded_reason=group.get('excluded_reason'),
                outcome_policy=POLICY)
