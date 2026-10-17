"""Behavior shaping; not a reasoning-step correctness verifier."""
from math_verification import completion_text
from closure_rewards import complete_format


def repetition_ratio(text, n=12):
    words = completion_text(text).lower().split()
    count = len(words) - n + 1
    if count <= 0:
        return 0.0
    unique = {tuple(words[i:i+n]) for i in range(count)}
    return 1.0 - len(unique) / count


def repetition_penalty(completions, **kwargs):
    return [-.1 * repetition_ratio(c) for c in completions]


def make_complete_end_reward(eos_id):
    def complete_end_reward(completions, completion_ids, **kwargs):
        if len(completions) != len(completion_ids):
            raise ValueError('Reward batch mismatch')
        return [.05 if eos_id in ids and complete_format(c) else 0.0
                for c, ids in zip(completions, completion_ids)]
    return complete_end_reward
