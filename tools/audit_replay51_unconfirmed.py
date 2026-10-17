"""Explain all 54 unconfirmed responses within 51 normally ended G4 groups.

Offline audit only: preserve original scores; no API calls or training changes.
Candidate spans were reviewed in the original question/response context.
"""
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
from scorer_runtime_guard import load_verified_scorer

ROOT=Path(__file__).resolve().parents[1]/'datasets/math/base_self600_20261015/grpo_replay549_extract_v4_20261016_batch12'
OUT=ROOT/'normal_end_uncertain_audit'
CATEGORIES={
    'sentence_not_math_result':[2,3,10,14,15,16,19,29,35,39,40,41,42,43,45,47,48,49,52,53],
    'equivalent_or_approximate_forms_rejected':[33,51],
    'damaged_or_unsupported_math_notation':[0,4,6,11,30,31,34,36],
    'verbal_no_solution_or_undefined':[20,21,44,46,50],
    'api_returned_null':[1,5,7,8,9,13,18,22,24,25,26,27,28,37,38],
    'api_output_not_contiguous_verbatim':[12,17,23,32],
}
# These candidates encode the stated answer, never the independently solved answer.
# Notation repair and word-number conversion are disclosed audit transformations.
CANDIDATES={2:'15',3:'15',10:'6',14:'10',15:'44.84',16:'70.75',19:'40 feet',29:'15',
    35:'12',39:'1',40:'101',41:r'\text{two lines}',42:'84',43:'83',45:'2 kPa',47:'300',48:'36',49:'12',52:'3',53:'1',
    33:'-3/2',51:'4/13',4:r'5\sqrt{15}',6:r'\begin{pmatrix}6\\3\end{pmatrix}',
    11:r'6\frac{1}{2}\text{ feet}',30:r'(-4\sqrt{3},-4)',31:r'137^\circ',36:r'10\sqrt{15}',
    12:'250 square centimeters',17:r'\frac{13}{5}',32:r'\frac{29}{3},\frac{5}{12}'}
NULL_REASONS={1:'program_without_output',5:'off_topic_or_corrupted',7:'off_topic_or_corrupted',8:'unfinished_verbal_calculation',
    9:'empty_response',13:'empty_response',18:'empty_response',22:'unresolved_conflicting_answers',24:'corrupted_math_without_final_result',
    25:'program_without_output',26:'empty_response',27:'answer_to_another_problem',28:'repetition_without_final_conclusion',
    37:'empty_response',38:'drifting_and_unresolved_final_claims'}

def main():
    rows=[json.loads(s) for s in (OUT/'packets.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==54 and len({r['row_id'] for r in rows})==51
    labels={i:k for k,indices in CATEGORIES.items() for i in indices};assert len(labels)==54 and set(labels)==set(range(54))
    scorer,runtime=load_verified_scorer()
    results=[]
    for r in rows:
        i=r['audit_index'];api=r['api_review'];answer=json.loads(api['content'])['answer']
        result=dict(audit_index=i,row_id=r['row_id'],sample_index=r['sample_index'],
            category=labels[i],problem=r['problem'],gold=r['gold'],original_response=r['completion'],
            original_api_output=api['content'],original_extracted_answer=api.get('extracted_answer'),
            original_math_verdict=r['verdict'],original_math_recheck=(r.get('api_extraction_recheck') or {}).get('verdict'),
            offline_candidate=CANDIDATES.get(i),null_reason=NULL_REASONS.get(i))
        assert r['eos'] and not r['truncated'] and r['grading_unconfirmed']
        if api['status']=='extracted':assert isinstance(answer,str) and answer in r['completion']
        elif answer is not None:assert answer not in r['completion']
        if i in CANDIDATES:
            result['offline_verdict']=scorer.verify_final_answer('<SOLUTION>'+CANDIDATES[i]+'</SOLUTION>',r['gold'],question=r['problem'])
        else:result['offline_verdict']=None
        results.append(result)
    confirmed=[r for r in results if (r['offline_verdict'] or {}).get('status') in {'correct','incorrect'}]
    by_question=defaultdict(list)
    for r in results:by_question[r['row_id']].append(r)
    summary=dict(questions=51,answers_in_these_questions=204,unconfirmed_answers=54,other_confirmed_answers=150,
        categories={k:len(v) for k,v in CATEGORIES.items()},api_extracted=35,api_null=15,api_nonverbatim=4,
        null_subcategories=dict(Counter(NULL_REASONS.values())),candidate_rechecks=len(CANDIDATES),
        offline_confirmed=len(confirmed),offline_correct=sum(r['offline_verdict']['status']=='correct' for r in confirmed),
        offline_incorrect=sum(r['offline_verdict']['status']=='incorrect' for r in confirmed),
        fully_recoverable_questions_with_reviewed_candidates=sum(all((r.get('offline_verdict') or {}).get('status') in {'correct','incorrect'} for r in rr) for rr in by_question.values()),
        offline_still_unconfirmed=[r['audit_index'] for r in results if r['offline_verdict'] and r['offline_verdict']['status'] not in {'correct','incorrect'}],
        scorer_runtime=runtime,api_calls=0,production_scoring_changed=False,training_restarted=False,
        input_sha256=hashlib.sha256((OUT/'packets.jsonl').read_bytes()).hexdigest(),
        scope='Diagnosis and disclosed manual answer-span/notation rechecks; not a tested generic automatic normalization rule.')
    (OUT/'decisions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
