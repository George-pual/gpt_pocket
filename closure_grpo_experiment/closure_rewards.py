"""Correct final answers dominate; EOS alone never earns a bonus."""
import re
from math_verification import verify_final_answer, completion_text


def complete_format(text):
    text = completion_text(text).strip()
    text = re.sub(r'(?:<\|endoftext\|>|<\|im_end\|>)\s*$', '', text).strip()
    return (text.count('<end_working_out>') == 1
            and text.count('<SOLUTION>') == 1 and text.count('</SOLUTION>') == 1
            and bool(re.fullmatch(r'.+<end_working_out>\s*<SOLUTION>[^<>]+</SOLUTION>', text, re.S)))


def make_end_reward(eos_id, weight):
    def correct_end_bonus(completions, answer, completion_ids, **kwargs):
        if not len(completions) == len(answer) == len(completion_ids):
            raise ValueError('Reward batch mismatch')
        return [weight if eos_id in ids and complete_format(c)
                and verify_final_answer(c, gold)['correct'] else 0.0
                for c,gold,ids in zip(completions,answer,completion_ids)]
    return correct_end_bonus


def soft_penalty(length, cap=3894, buffer=512, weight=.5):
    return -weight * min(1.0, max(0.0, (length-(cap-buffer))/buffer))
