"""Freeze the user-selected full 549-question replay pool and launch bundle."""
from collections import Counter
import hashlib,json,shutil,zipfile
from pathlib import Path
from prepare_math_self600 import canonical

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'datasets/math/base_self600_20261015'
AUDIT=ROOT/'grpo_rescreen_v4_results'
OUT=ROOT/'grpo_replay549_extract_v4_20261016_batch12'

def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write(p,x):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=False)
    for name in ['data','tools','configs','tests']: (OUT/name).mkdir()
    rows=read(AUDIT/'remote/data/candidates1200.jsonl');by_id={r['row_id']:r for r in rows}
    annotations={r['row_id']:r for r in read(AUDIT/'annotated_scores.jsonl')}
    overrides={(r['row_id'],r['sample_index']):r for r in read(AUDIT/'answer_correctness_reaudit_final.jsonl')}
    assert len(rows)==len(by_id)==1200 and len(overrides)==724
    chosen=[];dispositions=[];used=set()
    for p in sorted((AUDIT/'remote/screen_current_grpo/grpo_final/groups').glob('*.json')):
        g=json.loads(p.read_text(encoding='utf-8'));rid=g['row_id'];a=annotations[rid]
        if a['category'] not in {'strict_math','mixed_math_protocol','protocol_only'}:continue
        assert len(g['samples'])==4 and a['bad_end']==a['u']==0 and a['gold_parser_supported']
        decisions=[]
        for s in g['samples']:
            key=(rid,s['sample_index']);original=s['verdict']['status']
            verdict=original;source='original_math_verify_v4'
            if original not in {'correct','incorrect'}:
                o=overrides[key];verdict=o['final_verdict'];source=o['verdict_source'];used.add(key)
                assert o['completion']==s['completion'] and o['gold']==by_id[rid]['answer']
            assert verdict in {'correct','incorrect'} and s['eos'] and not s['truncated']
            decisions.append(dict(sample_index=s['sample_index'],original_status=original,status=verdict,source=source))
        k=sum(d['status']=='correct' for d in decisions)
        dispositions.append(dict(row_id=rid,correct_count=k,eligible=0<k<4,samples=decisions))
        if not 0<k<4:continue
        r=by_id[rid]
        assert r['row_id']!='math:precalculus/train-00000-of-00001.parquet:26'
        chosen.append(dict(r,previous_sft_difficulty=r['difficulty'],previous_sft_correct_count=r['screen_correct_count'],
            difficulty={1:'hard',2:'medium',3:'easy'}[k],screen_correct_count=k,
            screen_grading_schema='math_verify_v4_plus_direct_answer_reaudit',
            calibration_adapter_sha256='8bd99e8a8a75abbe5208a4ee1bf8b5ade805dffcd91dab589e84a754a9cc6186',
            replay_signal_tier=a['category']))
    assert len(used)==724 and len(dispositions)==780
    chosen.sort(key=lambda r:hashlib.sha256(('20261016|replay549|'+r['row_id']).encode()).hexdigest())
    assert len(chosen)==len({r['row_id'] for r in chosen})==len({canonical(r['problem']) for r in chosen})==549
    assert Counter(r['difficulty'] for r in chosen)==Counter(easy=352,medium=145,hard=52)
    assert len({r['type'] for r in chosen})==7
    for r in chosen:
        assert [m['role'] for m in r['prompt']]==['system','user'] and r['prompt'][1]['content']==r['problem']
        assert not {'completion','reference_solution'}&r.keys()
    keys={canonical(r['problem']) for r in chosen};overlap={}
    for label,p in {'sft':ROOT/'reviewed_direct/train.jsonl','development':ROOT/'development.jsonl',
        'official_test':ROOT/'remote_results/official_test/questions.jsonl'}.items():
        overlap[label]=len(keys&{canonical(r['problem']) for r in read(p)});assert overlap[label]==0
    write(OUT/'data/train.jsonl',chosen);write(OUT/'data/screen_dispositions.jsonl',dispositions)
    test=ROOT/'remote_results/official_test/questions.jsonl';assert len(read(test))==2024
    shutil.copyfile(test,OUT/'data/official_test.jsonl')
    pilot=read(ROOT/'api_answer_review_pilot/binary8_v2/records.jsonl');assert len(pilot)==8
    write(OUT/'data/api_extraction_pilot8.jsonl',[{k:r[k] for k in ['row_id','sample_index','problem','gold','completion','eos','regraded_status']} for r in pilot])
    code=['train_direct_grpo.py','online_grpo_reward.py','online_reward_contract.json','api_answer_review.py',
        'math_verification.py','scorer_runtime_guard.py','scorer_v4_contract.json','run_math_stage.py',
        'run_grpo_replay549.py','grpo_batching.py']
    for name in code:shutil.copyfile(REPO/'tools'/name,OUT/'tools'/name)
    shutil.copyfile(ROOT/'remote_results/tools/common.py',OUT/'tools/common.py')
    shutil.copyfile(REPO/'tests/test_grpo_batching.py',OUT/'tests/test_grpo_batching.py')
    cfg=json.loads((REPO/'configs/math_answer_review_api.json').read_text(encoding='utf-8'))
    cfg.update(max_calls=4*len(chosen),thinking='disabled');cfg.pop('reasoning_effort',None)
    save(OUT/'configs/api_training.json',cfg)
    preplan=json.loads((ROOT/'rescreen_grpo1200_v4/rescreen_plan.json').read_text(encoding='utf-8'))
    plan=dict(questions=549,difficulty=dict(Counter(r['difficulty'] for r in chosen)),
        subject=dict(Counter(r['type'] for r in chosen)),level=dict(Counter(r['level'] for r in chosen)),
        origin=dict(Counter(r['selection_role'] for r in chosen)),exact_overlap=overlap,
        selection='User explicitly chose all 549 reviewed contrast questions; no half or 3:4:3 constraint this round',
        adapter_sha256=preplan['adapter_sha256'],base_weight_sha256=preplan['base_weight_sha256'],
        grader_sha256=preplan['grader_sha256'],reward_sha256=sha(OUT/'tools/online_grpo_reward.py'),
        train_sha256=sha(OUT/'data/train.jsonl'),test_sha256=sha(OUT/'data/official_test.jsonl'),
        source_sha256={p.name:sha(p) for p in [AUDIT/'annotated_scores.jsonl',AUDIT/'answer_correctness_reaudit_final.jsonl',AUDIT/'answer_correctness_reaudit_final_summary.json']},
        training=dict(epochs=1,updates=549,generations=4,questions_per_generation=3,training_microbatch=2,
            gradient_accumulation=2,generation_batch_size=12,steps_per_generation=6,vllm_memory=.60,
            learning_rate=5e-6,warmup_steps=10,
            scheduler='linear',beta=.04,format_weight=.1,initial_model='Previous final GRPO adapter; new optimizer/scheduler'),
        api=dict(role='Verbatim original-answer extraction only; correctness from pinned local Math-Verify',
            max_training_calls=2196,pilot_calls=8,key_persisted=False),
        evaluation='Same official MATH L2-L3 2024 questions with original strict Math-Verify metric; API-free for comparison',
        shutdown_authorized=False)
    save(OUT/'plan.json',plan)
    manifest={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    save(OUT/'bundle_manifest.json',manifest)
    archive=ROOT/'deployment'/f'{OUT.name}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
    save(ROOT/'deployment'/f'{OUT.name}_bundle_receipt.json',dict(bundle=str(archive),sha256=sha(archive),bytes=archive.stat().st_size))
    print(json.dumps(dict(path=str(OUT),bundle_sha256=sha(archive),**plan),ensure_ascii=True),flush=True)

if __name__=='__main__':main()
