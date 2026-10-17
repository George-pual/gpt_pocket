"""Recover only explicit final-answer regions; preserve strict raw scores separately.

No search for a gold-matching number in reasoning, no gold substitution, and no
multiple-answer or substantive-post-answer-tail salvage. Reasoning still needs review.
"""
import argparse
import collections
import json
from pathlib import Path
import re


def region(text):
    openings=list(re.finditer(r'<SOLUTION>',text,re.I))
    if not openings:
        return None
    start=openings[-1].end()
    previous=text[:openings[-1].start()]
    if any(s.strip() for s in re.findall(r'<SOLUTION>(.*?)</SOLUTION>',previous,re.I|re.S)):
        return None
    previous=re.sub(r'<SOLUTION>\s*</SOLUTION>','',previous,flags=re.I)
    if re.search(r'</?SOLUTION',previous,re.I):
        return None
    tail=text[start:]
    close=re.search(r'</SOLUTION>',tail,re.I)
    if close:
        after=tail[close.end():]
        for marker in ('<|endoftext|>','<end_working_out>','</end_working_out>',r'\end{working_out}'):
            after=after.replace(marker,'')
        if not re.fullmatch(r'[\s.\\\[\]{}]*',after):
            return None
        tail=tail[:close.start()]
    tail=re.sub(r'</?(?:start_working_out|end_working_out)>','',tail,flags=re.I)
    tail=tail.replace('<|endoftext|>','').strip()
    if not tail or len(tail)>1000 or re.search(r'</?SOLUTION',tail,re.I):
        return None
    return previous,tail


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',required=True)
    a=p.parse_args()
    from math_verification import verify_final_answer,extract_final_answer
    from review_math_self600 import formatted
    root=Path(a.results)
    dest=root/'format_recovered_candidates.jsonl'
    assert not dest.exists(),'Never overwrite recovery evidence'
    rows=[];counts=collections.Counter()
    with (root/'scored_generations.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            counts['raw_answers']+=1
            original=r['math_verification']
            if not original['correct'] and original['status'] in ('extraction_failed','prediction_parse_failed') and r['eos_finished'] and not r['truncated']:
                isolated=region(r['completion'])
                if isolated:
                    body,final=isolated
                    candidate,_,error=extract_final_answer(final)
                    if error:
                        math=list(re.finditer(r'\\\((.+?)\\\)|\$(.+?)\$|\\\[(.+?)\\\]',final,re.S))
                        if len(math)==1:
                            outside=final[:math[0].start()]+final[math[0].end():]
                            if not re.search(r'[\d=+*/<>]',outside):
                                candidate=next(g for g in math[0].groups() if g is not None)
                            else:candidate=final
                        else:candidate=final
                    candidate=candidate.strip()
                    recovered=verify_final_answer('<SOLUTION>'+candidate+'</SOLUTION>',r['gold'])
                    if recovered['correct']:
                        try:
                            completion=formatted(body,candidate)
                        except ValueError:
                            pass
                        else:
                            r=dict(r,original_completion=r['completion'],raw_math_verification=original,
                                   completion=completion,math_verification=recovered,
                                   token_ids_for_original_completion=True,
                                   format_recovery={'method':'Explicit final SOLUTION region, optionally its sole math span; no gold substitution',
                                                    'original_final_region':final,'isolated_final_answer':candidate})
                            counts['additional_answer_correct_after_boundary_repair']+=1
            counts['candidate_answer_correct']+=bool(r['math_verification']['correct'])
            rows.append(r)
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    tmp.replace(dest)
    (root/'format_recovery_summary.json').write_text(json.dumps(dict(counts),indent=2),encoding='utf-8')
    print(json.dumps(dict(counts)),flush=True)


if __name__=='__main__':main()
