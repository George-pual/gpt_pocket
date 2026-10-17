"""Pinned current-GRPO vLLM screening only; never starts GRPO training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'rescreen_plan.json').read_text(encoding='utf-8'))
    from scorer_runtime_guard import load_verified_scorer
    scorer,runtime=load_verified_scorer(plan['grader_sha256'])
    rows=[json.loads(s) for s in (root/'data/candidates1200.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    assert len(rows)==len({r['row_id'] for r in rows})==1200
    assert digest(root/'data/candidates1200.jsonl')==plan['data_sha256']
    assert not {r['row_id'] for r in rows}&set(plan['quarantined_source_rows'])
    for row in rows:
        assert [m['role'] for m in row['prompt']]==['system','user']
        assert row['prompt'][1]['content']==row['problem']
    project=Path('/root/autodl-tmp/math_self600_20261015')
    model=Path('/root/autodl-tmp/qwen3_hf/model')
    adapter=project/'grpo_direct367_v1/adapter'
    assert digest(adapter/'adapter_model.safetensors')==plan['adapter_sha256']
    for name,expected in plan['base_weight_sha256'].items():assert digest(model/name)==expected
    status=dict(stage='grader_and_model_verified',grader_runtime=runtime,model=str(model),adapter=str(adapter),
                adapter_sha256=plan['adapter_sha256'],questions=len(rows),training_started=False)
    (root/'preflight.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False),flush=True)
    if args.preflight_only:return
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True)
        gpu_error=None
    except OSError as exc:
        gpu=None;gpu_error=str(exc)
    if gpu is None or gpu.returncode or not gpu.stdout.strip():
        status.update(stage='waiting_for_gpu',reason='No usable GPU detected; no vLLM generation started',gpu_probe_error=gpu_error)
        (root/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(status,ensure_ascii=False),flush=True)
        return
    output=root/'screen_current_grpo'
    status.update(stage='screen_running',gpu_name=gpu.stdout.strip())
    (root/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        subprocess.run([sys.executable,str(root/'tools/run_math_stage.py'),'--model',str(model),
                        '--data',str(root/'data/candidates1200.jsonl'),'--output',str(output),
                        '--arms',json.dumps({'grpo_final':str(adapter)}),'--screen','--batch','16',
                        '--expected-grader-sha256',plan['grader_sha256']],check=True)
    except subprocess.CalledProcessError as exc:
        status.update(stage='screen_failed',returncode=exc.returncode)
        (root/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    from grpo_outcome_policy import summarize
    gold=json.loads((output/'grpo_final/gold_checks.json').read_text(encoding='utf-8'))
    scores=[]
    for p in sorted((output/'grpo_final/groups').glob('*.json')):
        g=json.loads(p.read_text(encoding='utf-8'))
        score=summarize(g,gold[g['row_id']])
        statuses=Counter(s['verdict']['status'] for s in g['samples'])
        score.update(mathematical_contrast=bool(score['eligible'] and statuses['correct'] and statuses['incorrect']),
                     protocol_only_contrast=bool(score['eligible'] and not statuses['incorrect']),
                     verdict_counts=dict(statuses),calibration_model='Previous GRPO final adapter, freshly generated G4')
        scores.append(score)
    assert len(scores)==1200
    (root/'fresh_scores.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in scores),encoding='utf-8')
    status.update(stage='screen_complete',groups=len(scores),eligible=sum(s['eligible'] for s in scores),
                  mathematical_contrast=sum(s['mathematical_contrast'] for s in scores),
                  protocol_only_contrast=sum(s['protocol_only_contrast'] for s in scores))
    (root/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
