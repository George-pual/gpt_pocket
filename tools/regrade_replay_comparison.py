"""Regrade archived baseline responses under the exact current test grader."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import argparse,hashlib,json,sys,zipfile
from pathlib import Path

SCORER=None
def initialize(expected):
    global SCORER
    from scorer_runtime_guard import load_verified_scorer
    SCORER,_=load_verified_scorer(expected,run_preflight=False)
def grade(item):
    arm,row,problem=item
    verdict=SCORER.verify_final_answer(row['completion'],row['gold'],question=problem)
    return arm,{**row,'original_correct':row['correct'],'original_verdict':row['verdict'],'correct':bool(verdict['correct']),'verdict':verdict}
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--historical-only',action='store_true');args=ap.parse_args();root=Path(args.root).resolve()
    assert root.name=='grpo_replay549_extract_v4_20261016_batch12'
    plan=json.loads((root/'plan.json').read_text());inputs=json.loads((root/'baseline_manifest.json').read_text())
    from scorer_runtime_guard import load_verified_scorer
    _,runtime=load_verified_scorer(plan['grader_sha256'])
    questions=read(root/'data/official_test.jsonl');by_id={r['row_id']:r for r in questions};assert len(by_id)==2024
    out=root/'comparison_regraded_v4';out.mkdir(exist_ok=False)
    tasks=[]
    for arm,meta in inputs.items():
        p=root/meta['file'];assert sha(p)==meta['sha256'];rows=read(p)
        assert len(rows)==len({r['row_id'] for r in rows})==2024 and {r['row_id'] for r in rows}==set(by_id)
        for r in rows:assert r['gold']==by_id[r['row_id']]['answer'];tasks.append((arm,r,by_id[r['row_id']]['problem']))
    results={arm:[] for arm in inputs};completed=0
    with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(plan['grader_sha256'],)) as pool:
        for arm,row in pool.map(grade,tasks,chunksize=8):
            results[arm].append(row);completed+=1
            if completed%250==0:print('REGRADE_PROGRESS',completed,len(tasks),flush=True)
    table={}
    for arm,rows in results.items():
        rows.sort(key=lambda r:r['row_id'])
        (out/f'{arm}_answers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        correct=sum(r['correct'] for r in rows)
        table[arm]=dict(questions=len(rows),original_correct=sum(r['original_correct'] for r in rows),
            original_accuracy=sum(r['original_correct'] for r in rows)/len(rows),correct=correct,accuracy=correct/len(rows),
            statuses=dict(Counter(r['verdict']['status'] for r in rows)),
            changed_by_grader=sum(r['correct']!=r['original_correct'] for r in rows),
            by_level={lv:dict(questions=sum(r['level']==lv for r in rows),correct=sum(r['correct'] for r in rows if r['level']==lv)) for lv in ['Level 2','Level 3']})
    if args.historical_only:
        summary=dict(status='historical_regrading_complete_current_retrieval_pending',grader_runtime=runtime,table=table,
            source_inputs=inputs,test_sha256=plan['test_sha256'],caveat='Archived baseline responses regraded only; no generation, training or API. Current test result not yet retrieved.')
        save(out/'summary.json',summary)
        print('HISTORICAL_REGRADE_COMPLETE',json.dumps(summary,ensure_ascii=True),flush=True)
        return
    current=read(root/'test_after_grpo/grpo_replay549/answers.jsonl');assert len(current)==2024
    manifest=json.loads((root/'test_after_grpo/manifest.json').read_text())
    assert manifest['grader_sha256']==runtime['sha256'] and manifest['data_sha256']==plan['test_sha256']
    assert {r['row_id'] for r in current}==set(by_id)
    table['grpo2']=dict(questions=2024,correct=sum(r['correct'] for r in current),accuracy=sum(r['correct'] for r in current)/2024,
        statuses=dict(Counter(r['verdict']['status'] for r in current)))
    old={r['row_id']:r for r in results['grpo1']};new={r['row_id']:r for r in current}
    gained=[rid for rid in old if not old[rid]['correct'] and new[rid]['correct']]
    regressed=[rid for rid in old if old[rid]['correct'] and not new[rid]['correct']]
    assert len(gained)-len(regressed)==table['grpo2']['correct']-table['grpo1']['correct']
    summary=dict(status='complete',grader_runtime=runtime,table=table,gained=len(gained),regressed=len(regressed),
        net_questions=len(gained)-len(regressed),percentage_point_change=100*(len(gained)-len(regressed))/2024,
        caveat='Historical responses regraded only, no generation or training; original results retained; same fixed 2024 test. No API judging.',
        source_inputs=inputs,test_sha256=plan['test_sha256'])
    save(out/'summary.json',summary)
    save(out/'transitions.json',dict(gained_row_ids=gained,regressed_row_ids=regressed))
    files=list(out.iterdir())+[root/'baseline_manifest.json',root/'tools/regrade_replay_comparison.py']
    archive=root/'comparison_regraded_v4.zip';manifest_files={}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            name=p.relative_to(root).as_posix();content=p.read_bytes();z.writestr(name,content);manifest_files[name]=hashlib.sha256(content).hexdigest()
        z.writestr('manifest.json',json.dumps(manifest_files))
    save(root/'comparison_receipt.json',dict(status='complete',archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(manifest_files),summary=summary))
    print('REGRADE_COMPLETE',json.dumps(summary,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
