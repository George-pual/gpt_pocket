"""Package the pinned grader and screening-only continuation, without secrets."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'datasets/math/base_self600_20261015'


def main():
    stage=ROOT/'rescreen_grpo1200_v4'
    (stage/'tools').mkdir(parents=True,exist_ok=True)
    (stage/'data').mkdir(exist_ok=True)
    (stage/'backup_evidence').mkdir(exist_ok=True)
    code=['math_verification.py','scorer_runtime_guard.py','scorer_v4_contract.json',
          'run_math_stage.py','grpo_outcome_policy.py','rescreen_grpo1200_v4.py','cleanup_math_remote_project.py']
    for name in code:shutil.copyfile(REPO/'tools'/name,stage/'tools'/name)
    shutil.copyfile(ROOT/'remote_results/tools/common.py',stage/'tools/common.py')
    data=ROOT/'grpo_supplement_v3/train_expanded.jsonl'
    shutil.copyfile(data,stage/'data/candidates1200.jsonl')
    for name in ['all_results_manifest.json','final_retrieval_verification.json']:
        shutil.copyfile(ROOT/'remote_results'/name,stage/'backup_evidence'/name)
    validation=json.loads((ROOT/'scorer_v4_validation/summary.json').read_text(encoding='utf-8'))
    assert not validation['false_accept_indices'] and not validation['wrong_disagreement_indices']
    assert validation['actual_ordered_name_expected_statuses_matched']
    contract=json.loads((stage/'tools/scorer_v4_contract.json').read_text())
    assert hashlib.sha256((stage/'tools/math_verification.py').read_bytes()).hexdigest()==contract['sha256']==validation['scorer_sha256']
    original=json.loads((ROOT/'remote_results/grpo_screen_direct367_v1/manifest.json').read_text(encoding='utf-8'))
    plan=dict(purpose='Fresh G4 screening of 1200 candidates using previous GRPO final adapter; no training',
        grader_sha256=contract['sha256'],grading_schema=contract['schema'],
        data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),questions=1200,
        adapter_sha256='8bd99e8a8a75abbe5208a4ee1bf8b5ade805dffcd91dab589e84a754a9cc6186',
        base_weight_sha256=original['model_weights'],
        quarantined_source_rows=['math:precalculus/train-00000-of-00001.parquet:26'],
        source_gold_policy='Known incomplete original gold is quarantined, not silently corrected',
        sampling=dict(n=4,temperature=1.0,max_tokens=3894),
        classification='Report mathematical contrast separately from protocol-only task contrast; do not force quotas',
        training_authorized_by_this_bundle=False)
    (stage/'rescreen_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    (stage/'START.txt').write_text('/root/autodl-tmp/vllm_speed_env/bin/python /root/autodl-tmp/math_self600_20261015/rescreen_grpo1200_v4/tools/rescreen_grpo1200_v4.py\n',encoding='utf-8')
    manifest={p.relative_to(stage).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.rglob('*') if p.is_file() and p.name!='bundle_manifest.json'}
    (stage/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=ROOT/'deployment/grpo1200_rescreen_v4.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(stage).as_posix())
    result=dict(bundle=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(manifest)+1)
    (ROOT/'deployment/grpo1200_rescreen_v4_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
