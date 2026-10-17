"""Verify all retrieved files, reproduce eligibility, and export separate pools."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

from grpo_outcome_policy import summarize
from prepare_math_self600 import canonical

ROOT=Path(__file__).resolve().parents[1]/'datasets/math/base_self600_20261015'
OUT=ROOT/'grpo_rescreen_v4_results'
REMOTE=OUT/'remote'
EXPECTED_ARCHIVE='4b5b553f6af2ec0cb29936a3eb0943dc7bf2d2000ca6d5369b7f27c147dcaad5'
GRADER='26565fd5639523aaad27b7211d32623d7a5a95605d605e1ee23dec69c412bec8'
ADAPTER='8bd99e8a8a75abbe5208a4ee1bf8b5ade805dffcd91dab589e84a754a9cc6186'


def read(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def write(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def main():
    archive=OUT/'rescreen_results.zip';assert sha(archive)==EXPECTED_ARCHIVE
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('rescreen_results_manifest.json'))
        assert len(manifest)==1224 and set(z.namelist())==set(manifest)|{'rescreen_results_manifest.json'}
        for name,expected in manifest.items():
            path=Path(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
        REMOTE.mkdir(exist_ok=True);z.extractall(REMOTE)
    assert all(sha(REMOTE/name)==expected for name,expected in manifest.items())
    cfg=json.loads((REMOTE/'screen_current_grpo/manifest.json').read_text(encoding='utf-8'))
    assert cfg['grader_sha256']==sha(REMOTE/'tools/math_verification.py')==GRADER
    assert cfg['grading_schema']=='math-verify-final-answer-v4'
    assert cfg['adapter_sha256']['grpo_final']==ADAPTER
    assert cfg['grader_runtime']['preflight_passed']
    assert cfg['math_verify_version']=='0.8.0'
    assert cfg['grader_runtime']['dependencies']['antlr4-python3-runtime']=='4.13.2'
    assert sha(REMOTE/'tools/grpo_outcome_policy.py')==sha(Path(__file__).with_name('grpo_outcome_policy.py'))
    data=read(REMOTE/'data/candidates1200.jsonl'); source={r['row_id']:r for r in data}
    assert len(data)==len(source)==1200
    assert cfg['data_sha256']==sha(REMOTE/'data/candidates1200.jsonl')==sha(ROOT/'grpo_supplement_v3/train_expanded.jsonl')
    gold=json.loads((REMOTE/'screen_current_grpo/grpo_final/gold_checks.json').read_text(encoding='utf-8'))
    fresh={r['row_id']:r for r in read(REMOTE/'fresh_scores.jsonl')};assert len(fresh)==1200
    groups=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((REMOTE/'screen_current_grpo/grpo_final/groups').glob('*.json'))]
    assert len(groups)==1200
    verdicts=Counter();errors=Counter();categories=Counter();by_group={};scored=[];sample_count=0;tokens=0
    for i,(g,r) in enumerate(zip(groups,data)):
        assert g['row_id']==r['row_id'] and g['question_index']==i and g['gold']==r['answer']
        assert len(g['samples'])==4 and {s['sample_index'] for s in g['samples']}==set(range(4))
        ss=Counter()
        for s in g['samples']:
            assert s['verdict']['schema']==cfg['grading_schema'] and s['correct']==s['verdict']['correct']
            assert s['tokens']==len(s['token_ids'])
            ss[s['verdict']['status']]+=1;verdicts[s['verdict']['status']]+=1;sample_count+=1;tokens+=s['tokens']
            if s['verdict'].get('error'):errors[s['verdict']['error']]+=1
        result=summarize(g,gold[r['row_id']])
        for key,value in result.items():assert fresh[r['row_id']][key]==value,(i,key)
        mathematical=bool(result['eligible'] and ss['correct'] and ss['incorrect'])
        assert fresh[r['row_id']]['mathematical_contrast']==mathematical
        assert fresh[r['row_id']]['protocol_only_contrast']==bool(result['eligible'] and not ss['incorrect'])
        if result['eligible']:
            category='strict_math' if not ss['extraction_failed'] else 'mixed_math_protocol' if ss['incorrect'] else 'protocol_only'
        elif not result['gold_parser_supported']:category='unsupported_gold'
        elif result['u']:category='unconfirmed_math_verdict'
        elif result['bad_end']:category='abnormal_termination'
        elif result['k']==4:category='all_task_correct'
        elif result['k']==0:category='all_task_unsuccessful'
        else:category='other_excluded'
        categories[category]+=1;by_group[r['row_id']]=category
        scored.append(dict(**fresh[r['row_id']],category=category))
    assert sample_count==4800 and categories['strict_math']==224 and categories['mixed_math_protocol']==190 and categories['protocol_only']==366
    assert sum(categories.values())==1200
    exports=defaultdict(list)
    for s in scored:
        if not s['eligible']:continue
        r=source[s['row_id']]
        row=dict(r,previous_sft_difficulty=r['difficulty'],previous_sft_correct_count=r['screen_correct_count'],
                 difficulty=s['difficulty'],screen_correct_count=s['k'],screen_grading_schema=cfg['grading_schema'],
                 calibration_adapter_sha256=ADAPTER,calibration_model='Previous GRPO final, fresh G4',
                 replay_signal_tier=s['category'])
        assert [m['role'] for m in row['prompt']]==['system','user'] and row['prompt'][1]['content']==row['problem']
        assert not {'completion','reference_solution'}&row.keys()
        exports[s['category']].append(row)
    combined=exports['strict_math']+exports['mixed_math_protocol']
    pools=dict(strict_math=exports['strict_math'],mixed_math_protocol=exports['mixed_math_protocol'],
               math_priority_all=combined,protocol_only=exports['protocol_only'])
    ids={r['row_id'] for r in combined};assert len(ids)==len(combined)==414
    for pool in pools.values():
        assert len(pool)==len({r['row_id'] for r in pool})==len({canonical(r['problem']) for r in pool})
    keys={canonical(r['problem']) for r in data}
    for name in ['reviewed_direct/train.jsonl','development.jsonl','official_test/questions.jsonl']:
        assert not keys&{canonical(r['problem']) for r in read(ROOT/'remote_results'/name)}
    write(OUT/'annotated_scores.jsonl',scored)
    def count_pool(rows):
        dc=Counter(r['difficulty'] for r in rows)
        return dict(questions=len(rows),difficulty=dict(dc),origin=dict(Counter(r['selection_role'] for r in rows)),
                    subject=dict(Counter(r['type'] for r in rows)),level=dict(Counter(r['level'] for r in rows)),
                    maximum_exact_3_4_3_without_replacement=min(dc['easy']//3,dc['medium']//4,dc['hard']//3)*10)
    pool_summary={}
    for name,rows in pools.items():
        p=OUT/f'train_{name}.jsonl';write(p,rows);pool_summary[name]=dict(**count_pool(rows),sha256=sha(p))
    completion=json.loads((REMOTE/'rescreen_completion_receipt.json').read_text(encoding='utf-8'))
    original_strict=json.loads((REMOTE/'screen_current_grpo/grpo_final/summary.json').read_text(encoding='utf-8'))
    assert original_strict['eligible']==categories['strict_math']
    report=dict(status='complete_verified',questions=1200,samples=4800,completion=completion,
                category_counts=dict(categories),verdict_status_counts=dict(verdicts),error_counts=dict(errors),
                overlapping_flags=dict(unconfirmed_groups=sum(s['u']>0 for s in scored),
                                       abnormal_end_groups=sum(s['bad_end']>0 for s in scored),
                                       unsupported_gold_groups=sum(not s['gold_parser_supported'] for s in scored)),
                raw_task_success_k_distribution=dict(Counter(s['k'] for s in scored)),
                task_eligible_total=sum(s['eligible'] for s in scored),
                task_eligible_difficulty=dict(Counter(s['difficulty'] for s in scored if s['eligible'])),
                pool_summaries=pool_summary,total_generated_tokens=tokens,
                scope='Training-candidate recalibration only, not an official held-out MATH test or proof/step verification.',
                old_sft_buckets_preserved_in_exports=True,force_balancing_applied=False,
                new_training_started=False,server_shutdown=False)
    save(OUT/'summary.json',report)
    save(OUT/'retrieval_verification.json',dict(status='verified',archive_sha256=EXPECTED_ARCHIVE,
         files_verified=len(manifest),questions=1200,samples=4800,grader_sha256=GRADER,adapter_sha256=ADAPTER,
         eligibility_recomputed_matches_remote=True,exact_sft_dev_test_dedup_passed=True,
         scope='All newly archived generation, scoring, configuration, and log files verified; no new model training.'))
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8');main()
