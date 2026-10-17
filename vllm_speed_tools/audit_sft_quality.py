"""Structural SFT audit plus review queue; never certifies proof correctness."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def problem(row):
    if 'problem' in row:
        return str(row['problem'])
    prompt = row.get('prompt', [])
    if isinstance(prompt,list):
        return '\n'.join(str(m['content']) for m in prompt if m['role']=='user')
    raise ValueError('Cannot canonicalize question; do not use formatted prompt as problem')


def key(row):
    return re.sub(r'\s+',' ',problem(row)).strip().lower()


def flags(row, excluded, seen):
    completion = row.get('completion','')
    reasons = []
    question_key = key(row)
    if question_key in excluded:
        reasons.append('split_overlap')
    if question_key in seen:
        reasons.append('duplicate_question')
    seen.add(question_key)
    if not completion.endswith('</SOLUTION><|endoftext|>'):
        reasons.append('missing_normal_end')
    if (completion.count('<end_working_out>') != 1 or completion.count('<SOLUTION>') != 1
            or completion.count('</SOLUTION>') != 1):
        reasons.append('malformed_format')
    if '<think>' in completion or '</think>' in completion:
        reasons.append('unexpected_think_tags')
    words = completion.lower().split()
    grams = [tuple(words[i:i+12]) for i in range(max(0,len(words)-11))]
    repetition = 1-len(set(grams))/len(grams) if grams else 0.
    if repetition > .05:
        reasons.append('repetitive_12grams')
    boxes = len(re.findall(r'\\boxed\s*\{',completion))
    if boxes > 1:
        reasons.append('multiple_boxed_review')
    if row.get('tokens',0)>2048:
        reasons.append('recorded_overlong')
    return reasons,repetition


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train',required=True)
    p.add_argument('--exclude',nargs='+',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--verify-final',action='store_true',help='Requires installed Math-Verify; no fallback')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError('Fresh audit directory required')
    out.mkdir(parents=True)
    excluded = {key(row) for path in a.exclude for row in read(path)}
    if a.verify_final:
        from math_verification import require_math_verify, verify_final_answer
        require_math_verify()
    seen,counts,records,reviews = set(),Counter(),[],[]
    rows = read(a.train)
    for index,row in enumerate(rows):
        reasons,repetition = flags(row,excluded,seen)
        grade = None
        if a.verify_final:
            grade = verify_final_answer(row.get('completion',''),row.get('answer'))
            if not grade['correct']:
                reasons.append('final_answer_not_verified')
        counts.update(reasons)
        records.append({'index':index,'id':row.get('id'),'flags':reasons,
                        'repetition_ratio':repetition,'math_verification':grade})
        # All examples remain unapproved for reasoning quality until reviewed.
        reviews.append({'index':index,'id':row.get('id'),'problem':problem(row),
            'answer':row.get('answer'),'completion':row.get('completion'),
            'automatic_flags':reasons,'review_status':'pending',
            'conditions_preserved':None,'derivation_valid':None,'branches_complete':None,
            'arithmetic_checked':None,'answer_checked':None,'concise_normal_end':None,
            'reviewer':None,'evidence':None})
    for name,data in [('audit.jsonl',records),('reasoning_review_queue.jsonl',reviews)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
    summary = {'examples':len(rows),'flagged_examples':sum(bool(r['flags']) for r in records),
        'flags':dict(counts),'final_verifier_run':a.verify_final,
        'source_sha256':hashlib.sha256(Path(a.train).read_bytes()).hexdigest(),
        'proof_verification':'NOT performed. Passing this audit does not certify reasoning quality.',
        'approved_for_new_sft':0,
        'exclusions_sha256':{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in a.exclude}}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
