"""Shadow-check v4 without overwriting the frozen v3 validation artifacts."""
import json
from pathlib import Path
import validate_scorer_v3 as audit
from scorer_runtime_guard import load_verified_scorer


def main():
    scorer,runtime=load_verified_scorer()
    out=audit.ROOT/'scorer_v4_validation'
    audit.OUT=out
    audit.verify_final_answer=scorer.verify_final_answer
    audit.require_math_verify=scorer.require_math_verify
    audit.SCHEMA=scorer.SCHEMA
    audit.SCORER_SOURCE=Path(scorer.__file__)
    audit.main()
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    summary['scorer_sha256']=runtime['sha256']
    summary['grader_runtime']=runtime
    packets=audit.read(audit.ROOT/'reasoning_structure_audit/strict_zero_review_packets.jsonl')
    text=next(r for r in packets if r['audit_index']==15)
    additional=[]
    for s in text['samples']:
        verdict=scorer.verify_final_answer(s['completion'],text['answer'],question=text['problem'])
        expected='correct' if s['sample_index'] in (0,1,2) else 'incorrect'
        assert verdict['status']==expected,(s['sample_index'],verdict)
        additional.append(dict(row_id=text['row_id'],sample_index=s['sample_index'],expected=expected,verdict=verdict))
    summary['actual_ordered_name_samples']=4
    summary['actual_ordered_name_expected_statuses_matched']=True
    summary['known_incomplete_gold_action']='Quarantine source row; never silently correct the verifier reference'
    (out/'additional_samples.json').write_text(json.dumps(additional,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('V4_VALIDATED',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
