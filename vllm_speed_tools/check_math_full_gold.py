"""Preflight full MATH references using the versioned evaluation scorer."""
import concurrent.futures,json
from pathlib import Path
from rescore_math_v2 import worker

def main():
    root=Path(__file__).resolve().parents[1]/'datasets/math/test_l2_l3_full'
    rows=[json.loads(x) for x in (root/'questions.jsonl').read_bytes().splitlines() if x.strip()]
    pool=concurrent.futures.ProcessPoolExecutor(max_workers=4)
    try:
        futures=[pool.submit(worker,('<SOLUTION>'+x['answer']+'</SOLUTION>',x['answer'])) for x in rows]
        failures=[]
        for row,f in zip(rows,futures):
            verdict=f.result(timeout=30)
            if not verdict['correct']:failures.append(dict(row_id=row['row_id'],gold=row['answer'],verdict=verdict))
    finally:
        for p in list((getattr(pool,'_processes',None) or {}).values()):
            if p.is_alive():p.terminate()
        pool.shutdown(wait=True,cancel_futures=True)
    summary=dict(checked=len(rows),failed=len(failures),failures=failures)
    (root/'gold_scoring_preflight.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
