"""Partial format reward, capped at 0.1; complete-format metric stays unchanged."""
import re
from math_verification import completion_text

VERSION = 'sft-v3-format-components-v1'

def components(completion, ended):
    c = completion_text(completion).strip()
    c = re.sub(r'(?:<\|endoftext\|>|<\|im_end\|>)\s*$', '', c).strip()
    end = '<end_working_out>'
    start, close = '<SOLUTION>', '</SOLUTION>'
    # Reward a genuine reasoning boundary, not an isolated or misplaced tag.
    boundary = bool(c.count(end) == 1 and '</start_working_out>' not in c
                    and '<start_working_out>' not in c and c.split(end)[0].strip()
                    and start not in c.split(end)[0] and close not in c.split(end)[0])
    answer = bool(c.count(start) == 1 and c.count(close) == 1
                  and re.search(r'<SOLUTION>\s*[^<>\s][^<>]*</SOLUTION>', c, re.S))
    whole = bool(ended and boundary and answer
                 and re.fullmatch(r'.+<end_working_out>\s*<SOLUTION>[^<>]+</SOLUTION>', c, re.S))
    # No format reward from an unfinished generation; matches dropped-group policy.
    return {'boundary': 0.05 * boundary * bool(ended),
            'answer': 0.03 * answer * bool(ended),
            'whole': 0.02 * whole}

def normalized_reward(completion, ended):
    return sum(components(completion, ended).values()) / 0.1
