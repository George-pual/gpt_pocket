"""Dependency-free selection rules; rates are estimates, not true difficulty."""
import math


def difficulty_decision(correct, trials=16, low=.2, high=.7):
    if trials < 2 or not 0 <= correct <= trials or not 0 <= low < high <= 1:
        raise ValueError('Invalid counts or thresholds')
    rate = correct / trials
    # Wilson interval is diagnostic only; the requested selection uses point estimates.
    z = 1.96
    denominator = 1 + z*z/trials
    center = (rate + z*z/(2*trials)) / denominator
    radius = z * math.sqrt(rate*(1-rate)/trials + z*z/(4*trials*trials)) / denominator
    return {'correct': correct, 'trials': trials, 'estimated_accuracy': rate,
            'keep': low <= rate <= high,
            'bucket': 'hard' if rate < low else 'easy' if rate > high else 'eligible',
            'wilson95': [center-radius, center+radius]}


def informative_indices(correctness, group_size=4):
    """Keep entire groups containing both correct and incorrect completions."""
    if group_size < 2 or len(correctness) % group_size:
        raise ValueError('Incomplete groups')
    if any(value not in (False, True, 0, 1) for value in correctness):
        raise ValueError('Correctness must be binary')
    indices, counts = [], {'all_wrong': 0, 'all_correct': 0, 'mixed': 0}
    for start in range(0, len(correctness), group_size):
        total = sum(correctness[start:start+group_size])
        bucket = 'all_wrong' if total == 0 else 'all_correct' if total == group_size else 'mixed'
        counts[bucket] += 1
        if bucket == 'mixed':
            indices.extend(range(start, start+group_size))
    return indices, counts


def apply_selected_update(items, update):
    """An empty selection must not call backward, optimizer or scheduler."""
    if not items:
        return False
    update(items)
    return True
