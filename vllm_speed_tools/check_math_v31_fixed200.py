"""Check the symbolic-answer repair on the five archived MATH-200 files."""
import json
import argparse
from pathlib import Path
from audited_answer_scoring import SCHEMA
from rescore_fixed_answers import bounded_map
from math_remaining_eval import digest, save, write_rows, ARMS


def main():
    root=Path(__file__).resolve().parents[1]
    bundle=root/'datasets/math/test_l2_l3_remaining_v1'
    prior=root/'datasets/gsm8k/base_sftv3_test/results/audited_scorer_v3_final_r2/math'
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', default=str(bundle/'fixed200_v31_check'))
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(exist_ok=False)
    changes=[]; summary={}
    for arm in ARMS:
        path=prior/f'{arm}.jsonl'
        rows=[json.loads(x) for x in path.read_bytes().splitlines() if x.strip()]
        tasks=[(r['completion'],r['gold'],r['question'],r['eos'] and not r['truncated']) for r in rows]
        verdicts=bounded_map(tasks,4,20)
        for r,v in zip(rows,verdicts):
            if v['correct']!=r['audited_score_v3']['correct']:
                changes.append(dict(model=arm, row_id=r['row_id'], prior=r['audited_score_v3'], revised=v,
                                    question=r['question'],gold=r['gold'],completion=r['completion']))
        summary[arm]=dict(total=len(rows), prior_correct=sum(r['audited_score_v3']['correct'] for r in rows),
                          revised_correct=sum(v['correct'] for v in verdicts), source_sha256=digest(path))
        print(arm,summary[arm],flush=True)
    save(out/'summary.json',dict(schema=SCHEMA,scorer_sha256=digest(Path(__file__).with_name('audited_answer_scoring.py')),
                               models=summary, changed=len(changes)))
    write_rows(out/'changes.jsonl',changes)


if __name__=='__main__':
    main()
