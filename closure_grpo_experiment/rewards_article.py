"""Four rewards adapted from the Volcengine GRPO tutorial.

Ratio-based partial credit is preserved for reproduction; it is not accuracy.
"""
import re
from common import END, SOL_START, SOL_END

EXACT = re.compile(
    re.escape(END) + r".*?" + re.escape(SOL_START) + r"(.+?)" +
    re.escape(SOL_END) + r"\s*(?:<\|endoftext\|>)?\s*$", re.S | re.M)
NUMBER = re.compile(r"<SOLUTION>.*?\s*([-]?[\d\.,]+)", re.S | re.M)

def texts(completions):
    return [c if isinstance(c, str) else c[0]["content"] for c in completions]

def match_format_exactly(completions, **kwargs):
    return [3.0 if EXACT.search(c) else 0.0 for c in texts(completions)]

def match_format_approximately(completions, **kwargs):
    return [sum(0.5 if c.count(tag) == 1 else -1.0
                for tag in [END, SOL_START, SOL_END]) for c in texts(completions)]

def check_answer(completions, answer, **kwargs):
    rewards = []
    for c, gold in zip(texts(completions), answer):
        m = EXACT.search(c)
        if m is None:
            rewards.append(-2.0)
            continue
        guess, gold = m.group(1), str(gold)
        if guess == gold:
            rewards.append(5.0)
        elif guess.strip() == gold.strip():
            rewards.append(3.5)
        else:
            try:
                ratio = float(guess) / float(gold)
                rewards.append(2.0 if 0.9 <= ratio <= 1.1 else
                               1.5 if 0.8 <= ratio <= 1.2 else -2.5)
            except (ValueError, ZeroDivisionError, OverflowError):
                rewards.append(-4.5)
    return rewards

def check_numbers(completions, answer, **kwargs):
    rewards = []
    for c, gold in zip(texts(completions), answer):
        m = NUMBER.search(c)
        if m is None:
            rewards.append(-2.5)
        else:
            try:
                rewards.append(3.5 if float(m.group(1).strip().replace(',', '')) ==
                               float(str(gold).strip()) else -1.5)
            except (ValueError, OverflowError):
                rewards.append(0.0)
    return rewards

REWARDS = [match_format_exactly, match_format_approximately, check_answer, check_numbers]
