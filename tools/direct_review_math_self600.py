"""Display source-grounded review packets; decisions are written by the primary agent."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'datasets/math/base_self600_20261015'
SOURCE=ROOT/'direct_review_source'


def read(p):
    with p.open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def packets():
    rows=read(SOURCE/'questions.jsonl')
    answers=read(SOURCE/'results/format_recovered_candidates.jsonl')
    by_id={r['row_id']:[] for r in rows}
    for r in answers:
        if r['math_verification']['correct'] and r['eos_finished'] and not r['truncated']:
            by_id[r['row_id']].append(r)
    return [(r,sorted(by_id[r['row_id']],key=lambda c:(c['tokens'],c['sample_index']))) for r in rows if by_id[r['row_id']]]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser()
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--count',type=int,default=10)
    p.add_argument('--alternative',type=int,default=0)
    a=p.parse_args()
    items=packets()
    for index in range(a.start,min(a.start+a.count,len(items))):
        q,candidates=items[index]
        if a.alternative>=len(candidates):
            print('NO_ALTERNATIVE',index);continue
        c=candidates[a.alternative]
        print('\nITEM',index,'SAMPLE',c['sample_index'],'ID',q['row_id'],'STRATUM',q['level'],q['type'],'GOLD',q['answer'])
        print('QUESTION:',q['problem'])
        print('REFERENCE:',q['reference_solution'])
        print('CANDIDATE:',c['completion'])
    print('\nTOTAL_REVIEW_QUESTIONS',len(items))


if __name__=='__main__':main()
