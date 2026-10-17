"""Prepare, resume inference, and score the remaining official MATH L2/L3 test.

No training or shutdown actions. The original 200 responses are reused verbatim.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import statistics
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path

ARMS = ['base', 'sft_v3', 'grpo1024', 'grpo1280', 'grpo1536']
RELATIVE_ADAPTERS = {
    'sft_v3': 'E1_sft_v3/adapter',
    'grpo1024': 'E1_sft_v3_half_math1024_fast_v2/B_components/checkpoint-1024',
    'grpo1280': 'E1_sft_v3_expanded_continue1536_v1/B_components/checkpoint-256',
    'grpo1536': 'E1_sft_v3_expanded_continue1536_v1/B_components/checkpoint-512',
}


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return [json.loads(x) for x in Path(p).read_bytes().splitlines() if x.strip()]


def save(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)


def write_rows(p, rows):
    Path(p).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def prepare(a):
    root = Path(__file__).resolve().parents[1]
    out = Path(a.bundle)
    if out.exists():
        raise RuntimeError('Refuse overwrite: '+str(out))
    source = root/'datasets/math/test_l2_l3_full'
    prior = root/'datasets/gsm8k/base_sftv3_test/results/audited_scorer_v3_final_r2/manifest.json'
    old = json.loads(prior.read_text(encoding='utf-8'))
    data_manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert digest(source/'questions.jsonl') == data_manifest['data_sha256']
    rows = read(source/'questions.jsonl')
    by_id = {r['row_id']: r for r in rows}
    assert len(rows) == len(by_id) == 2024
    reused, common_ids = {}, None
    for arm in ARMS:
        info = old['inputs'][arm+'/math']
        path = Path(info['path'])
        assert digest(path) == info['sha256']
        previous = read(path)
        ids = {r['row_id'] for r in previous}
        assert len(previous) == len(ids) == 200
        assert common_ids is None or common_ids == ids
        common_ids = ids
        for r in previous:
            q = by_id[r['row_id']]
            assert (r['question'], r['gold'], r['level']) == (q['problem'], q['answer'], q['level'])
        reused[arm] = previous
    remaining = [r for r in rows if r['row_id'] not in common_ids]
    assert len(remaining) == 1824
    norm = lambda s: re.sub(r'\s+', '', unicodedata.normalize('NFKC', s)).lower()
    training_paths = [root/'datasets/math/train.jsonl', root/'sft_quality_audit_v2/merged_supplied_answers/train_merged_505.jsonl']
    training_paths += list((root/'datasets/grpo_candidates_gsm8k_math_l2_l3').glob('**/train_candidates.jsonl'))
    training_paths += list((root/'datasets/grpo_candidates_gsm8k_math_l2_l3').glob('**/validation.jsonl'))
    fingerprints = {norm(r['problem']) for p in training_paths for r in read(p)}
    assert not [r['row_id'] for r in rows if norm(r['problem']) in fingerprints], 'Training overlap requires review'
    out.mkdir(parents=True)
    (out/'reused').mkdir()
    shutil.copy2(source/'questions.jsonl', out/'questions_all.jsonl')
    write_rows(out/'questions_remaining.jsonl', remaining)
    for arm in ARMS:
        write_rows(out/'reused'/f'{arm}.jsonl', reused[arm])
    for name in ['math_remaining_eval.py', 'rescore_fixed_answers.py']:
        shutil.copy2(Path(__file__).with_name(name), out/name)
    for name in ['audited_answer_scoring.py', 'test_audited_answer_scoring.py']:
        shutil.copy2(prior.parent/name, out/name)
    shared = root/'qwen3-4b-qlora-sft-grpo-deepseek/outputs/qwen3_hf_autodl/common.py'
    shutil.copy2(shared, out/'common.py')
    refconfig = old['source_configs']['base/math']
    for arm in ARMS:
        for field in ['sampling', 'chat_template', 'prompt_tokens_sha256']:
            assert old['source_configs'][arm+'/math'][field] == refconfig[field]
    assert digest(out/'audited_answer_scoring.py') == old['scorer_sha256']
    manifest = dict(status='prepared_not_launched', arms=ARMS, full_count=2024, reused_per_arm=200,
        generate_per_arm=1824, remaining_levels=dict(Counter(r['level'] for r in remaining)),
        full_levels=dict(Counter(r['level'] for r in rows)), prior_manifest_sha256=digest(prior),
        sampling=refconfig['sampling'], chat_template=refconfig['chat_template'],
        prior_prompt_tokens_sha256=refconfig['prompt_tokens_sha256'],
        expected_adapter_sha256={arm:old['source_configs'][arm+'/math']['adapter_sha256'] for arm in RELATIVE_ADAPTERS},
        relative_adapters=RELATIVE_ADAPTERS,
        overlap_checks={str(p.relative_to(root)):dict(sha256=digest(p), rows=len(read(p))) for p in training_paths},
        packages_for_scoring=old['packages'],
        files={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()},
        scope='Official test, exact normalized training overlap excluded; no semantic dedup or pretraining contamination claim.')
    save(out/'manifest.json', manifest)
    print(json.dumps({k:manifest[k] for k in ['status','full_count','generate_per_arm','remaining_levels']}, indent=2))


def load_bundle(bundle):
    b = Path(bundle)
    m = json.loads((b/'manifest.json').read_text(encoding='utf-8'))
    for rel, sha in m['files'].items():
        assert digest(b/rel) == sha, ('Bundle changed', rel)
    return b, m


def generate(a):
    import fcntl
    b, m = load_bundle(a.bundle)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'generation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **kw):
        save(out/'status.json', dict(stage=stage, pid=os.getpid(), updated=time.time(), **kw))
    # Do not compete with an existing GPU job, or silently wait behind it.
    gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True)
    assert not gpu.strip(), 'GPU occupied: '+gpu
    model = Path(a.model)
    reference = json.loads((model/'hf_reference.json').read_text())
    assert reference['repo'] == 'Qwen/Qwen3-4B-Base'
    status('verifying_weights')
    for filename, info in reference['weights'].items():
        assert digest(model/filename) == info['sha256'], ('Base weight mismatch', filename)
    paths = {arm:Path(a.runs_root)/rel for arm,rel in RELATIVE_ADAPTERS.items()}
    for arm, path in paths.items():
        assert digest(path/'adapter_model.safetensors') == m['expected_adapter_sha256'][arm], arm
    from common import tokenizer, messages
    t = tokenizer(paths['sft_v3'])
    assert t.chat_template == m['chat_template'] and t.eos_token_id == 151643
    old = read(b/'reused/base.jsonl')
    old_prompts = [t.apply_chat_template(messages(r['question']), tokenize=True, add_generation_prompt=True) for r in old]
    assert hashlib.sha256(json.dumps(old_prompts).encode()).hexdigest() == m['prior_prompt_tokens_sha256']
    rows = read(b/'questions_remaining.jsonl')
    prompts = [t.apply_chat_template(messages(r['problem']), tokenize=True, add_generation_prompt=True) for r in rows]
    assert max(map(len, prompts))+3894 <= 8192
    provenance = dict(bundle_manifest_sha256=digest(b/'manifest.json'), sampling=m['sampling'],
        prompt_tokens_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        adapter_sha256=m['expected_adapter_sha256'], base_reference=reference, chat_template=t.chat_template,
        engine=dict(dtype='bfloat16', max_model_len=8192, max_num_seqs=128, max_num_batched_tokens=8192,
                    gpu_memory_utilization=.88, seed=3407, chunk_size=64),
        packages={p:importlib.metadata.version(p) for p in ['torch','vllm','transformers']})
    config = out/'evaluation_config.json'
    if config.exists():
        assert json.loads(config.read_text()) == provenance, 'Resume configuration mismatch'
    else:
        save(config, provenance)
    if a.dry_run:
        status('preflight_passed'); print('PREFLIGHT_PASSED', flush=True); return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    status('loading_model')
    llm = LLM(model=str(model), tokenizer=str(paths['sft_v3']), dtype='bfloat16',
        enable_lora=True, max_lora_rank=32, max_model_len=8192, gpu_memory_utilization=.88,
        max_num_seqs=128, max_num_batched_tokens=8192, enable_prefix_caching=True, seed=3407)
    params = SamplingParams(n=1, temperature=0., top_p=1., top_k=-1, max_tokens=3894,
                            stop_token_ids=[t.eos_token_id], seed=3407)
    try:
        for arm in ARMS:
            folder = out/arm
            folder.mkdir(exist_ok=True)
            for abandoned in folder.glob('chunk_*.tmp'):
                abandoned.unlink()  # Only uncommitted chunks inside this evaluation folder.
            request = None if arm == 'base' else LoRARequest(arm, ARMS.index(arm), str(paths[arm]))
            completed = 0
            for start in range(0, len(rows), 64):
                part = rows[start:start+64]
                chunk = folder/f'chunk_{start:05d}.jsonl'
                if chunk.exists():
                    saved = read(chunk)
                    assert [(r['row_id'],r['question'],r['gold']) for r in saved] == [(r['row_id'],r['problem'],r['answer']) for r in part]
                else:
                    outputs = llm.generate([{'prompt_token_ids':p} for p in prompts[start:start+64]],
                                           params, lora_request=request, use_tqdm=False)
                    assert len(outputs) == len(part)
                    saved = []
                    for r, result in zip(part, outputs):
                        assert len(result.outputs) == 1
                        s = result.outputs[0]
                        ended = s.finish_reason == 'stop' and s.stop_reason in (None,t.eos_token_id)
                        saved.append(dict(row_id=r['row_id'], question=r['problem'], gold=r['answer'],
                            level=r['level'], subject=r['type'], completion=s.text, tokens=len(s.token_ids),
                            eos=ended, truncated=s.finish_reason=='length', finish_reason=s.finish_reason,
                            stop_reason=s.stop_reason, reused=False))
                    tmp = chunk.with_suffix('.tmp')
                    write_rows(tmp, saved)
                    tmp.replace(chunk)
                completed += len(part)
                status('generating', arm=arm, completed=completed, total=len(rows),
                       completed_arms=ARMS[:ARMS.index(arm)])
                print('EVAL_PROGRESS', arm, completed, '/', len(rows), flush=True)
            if arm != 'base':
                assert digest(paths[arm]/'adapter_model.safetensors') == m['expected_adapter_sha256'][arm]
        status('generation_complete', questions_per_arm=len(rows), completed_arms=ARMS)
    except BaseException as exc:
        status('failed', error=repr(exc)); raise
    finally:
        llm.llm_engine.engine_core.shutdown()
    os.sync()
    print('GENERATION_COMPLETE', flush=True)
    # Avoid the known vLLM 0.10.2 interpreter-finalization crash after saved results.
    import sys
    sys.stdout.flush()
    os._exit(0)


def score(a):
    from audited_answer_scoring import SCHEMA
    from rescore_fixed_answers import bounded_map
    b, m = load_bundle(a.bundle)
    # The immutable generation bundle retains v3. The versioned v3.1 CPU
    # repair protects whole symbolic answers; all five arms use this same file.
    assert SCHEMA == 'audited-final-answer-v3.1'
    source = Path(a.output or a.bundle)
    out = Path(a.score_output)
    if out.exists():
        raise RuntimeError('Refuse overwrite: '+str(out))
    out.mkdir(parents=True)
    assert all(importlib.metadata.version(p)==v for p,v in m['packages_for_scoring'].items())
    full = read(b/'questions_all.jsonl')
    golds = bounded_map([('<SOLUTION>'+r['answer']+'</SOLUTION>',r['answer'],r['problem'],True) for r in full],4,20)
    failures = [dict(row_id=r['row_id'],gold=r['answer'],verdict=v) for r,v in zip(full,golds) if not v['correct']]
    save(out/'reference_self_checks.json', dict(total=len(full), failed=len(failures), failures=failures))
    if a.gold_only:
        print('GOLD_CHECK', len(full), 'failures', len(failures), flush=True); return
    # Keep the audited scorer byte-for-byte unchanged. Unsupported reference
    # types remain in the test and require explicit review for every model.
    reference_review_ids = {r['row_id'] for r in failures}
    summary, review = {}, []
    def repeated(text):
        words = text.split()
        counts = Counter(tuple(words[i:i+20]) for i in range(max(0,len(words)-19)))
        return max(counts.values(),default=0) >= 3
    def stats(rs):
        n=len(rs)
        return dict(total=n, correct=sum(r['audited_score_v3']['correct'] for r in rs),
            accuracy=sum(r['audited_score_v3']['correct'] for r in rs)/n,
            eos=sum(r['eos'] for r in rs), eos_rate=sum(r['eos'] for r in rs)/n,
            truncated=sum(r['truncated'] for r in rs), truncation_rate=sum(r['truncated'] for r in rs)/n,
            mean_tokens=statistics.mean(r['tokens'] for r in rs), median_tokens=statistics.median(r['tokens'] for r in rs),
            repeated=sum(r['repeated_20_word_span_3_times'] for r in rs),
            repetition_rate=sum(r['repeated_20_word_span_3_times'] for r in rs)/n,
            format_eos=sum(r['format_eos'] for r in rs),
            format_rate=sum(r['format_eos'] for r in rs)/n,
            reference_requires_review=sum(r['reference_requires_review'] for r in rs),
            status_counts=dict(Counter(r['audited_score_v3']['status'] for r in rs)))
    input_hashes = {}
    arms = a.arms.split(',') if a.arms else ARMS
    assert len(arms) == len(set(arms)) and set(arms) <= set(ARMS)
    for arm in arms:
        previous = [dict(r,reused=True) for r in read(b/'reused'/f'{arm}.jsonl')]
        paths = sorted((source/arm).glob('chunk_*.jsonl'))
        new = [r for p in paths for r in read(p)]
        expected = read(b/'questions_remaining.jsonl')
        assert [(r['row_id'],r['question'],r['gold']) for r in new] == [(r['row_id'],r['problem'],r['answer']) for r in expected]
        by_id = {r['row_id']:r for r in previous+new}
        assert len(previous)+len(new) == len(by_id) == 2024
        rows = [by_id[r['row_id']] for r in full]
        verdicts = bounded_map([(r['completion'],r['gold'],r['question'],r['eos'] and not r['truncated']) for r in rows],4,20)
        for r,v in zip(rows,verdicts):
            r['audited_score_v3']=v
            r['repeated_20_word_span_3_times']=repeated(r['completion'])
            r['reference_requires_review'] = r['row_id'] in reference_review_ids
            text = re.sub(r'(?:<\|endoftext\|>|<\|im_end\|>)\s*$', '', r['completion'].strip()).strip()
            fmt = bool(r['eos'] and '<start_working_out>' not in text
                and text.count('<end_working_out>') == 1 and text.count('<SOLUTION>') == 1
                and text.count('</SOLUTION>') == 1
                and re.fullmatch(r'.+<end_working_out>\s*<SOLUTION>[^<>]+</SOLUTION>', text, re.S))
            if r['reused']:
                assert r['format_eos'] == fmt, 'Historical format definition mismatch'
            r['format_eos'] = fmt
            if r['reference_requires_review'] or v['status'] not in ['correct','incorrect']:
                review.append(dict(model=arm, **r))
        write_rows(out/f'{arm}.jsonl',rows)
        summary[arm] = dict(all=stats(rows),remaining=stats([r for r in rows if not r['reused']]),
            reused=stats([r for r in rows if r['reused']]),
            by_level={level:stats([r for r in rows if r['level']==level]) for level in ['Level 2','Level 3']})
        input_hashes.update({str(p):digest(p) for p in paths})
        save(out/'summary.json',summary)
        print('SCORED',arm,summary[arm]['all'],flush=True)
    write_rows(out/'review_queue.jsonl',review)
    save(out/'manifest.json',dict(schema=SCHEMA, bundle_sha256=digest(b/'manifest.json'), inputs=input_hashes,
        scorer_sha256=digest(Path(__file__).with_name('audited_answer_scoring.py')),
        runner_sha256=digest(Path(__file__)), packages=m['packages_for_scoring'],
        final_accuracy_status=('requires_reference_review' if reference_review_ids else 'automatic_scoring_complete'),
        repetition_definition='At least one whitespace-tokenized 20-word contiguous span occurs 3 or more times; heuristic only.',
        provenance='200 existing responses plus 1824 new responses per arm; same prompt and decoding; no training.'))
    for filename in ['audited_answer_scoring.py','math_remaining_eval.py','rescore_fixed_answers.py','test_audited_answer_scoring.py']:
        shutil.copy2(Path(__file__).with_name(filename),out/filename)
    lines=['# MATH Level 2/3 补测结果','',
        '每个模型复用原 200 题回答，补测 1824 题，合计 2024 题。官方 2025 题中排除 1 题与训练集精确重叠。',
        '统一使用 audited-final-answer-v3.1 评分器、原 prompt、temperature=0、max_tokens=3894。',
        '未做语义去重，无法排除预训练污染。重复率是同一 20 词片段出现至少三次的启发式指标。','',
        '|模型|Level 2 自动判对|Level 3 自动判对|自动判对率|EOS 率|截断率|重复率|平均 tokens|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,s in summary.items():
        x=s['all']; l2=s['by_level']['Level 2']; l3=s['by_level']['Level 3']
        lines.append(f"|{arm}|{l2['correct']}/{l2['total']}|{l3['correct']}/{l3['total']}|{x['accuracy']:.2%}|{x['eos_rate']:.2%}|{x['truncation_rate']:.2%}|{x['repetition_rate']:.2%}|{x['mean_tokens']:.1f}|")
    lines += ['',f'参考答案预检未通过 {len(reference_review_ids)} 题。',
              f'共 {len(review)} 条评分器未确认的回答另存 review_queue.jsonl，按未判对计入自动评分正确率，不能一概解释为数学错误。',
              '这是官方 MATH test Level 2/3 子集的评测；与此前 6 题长度诊断及固定 200 题结果分开解释。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','generate','score'])
    p.add_argument('--bundle',required=True)
    p.add_argument('--output')
    p.add_argument('--score-output')
    p.add_argument('--model',default='/root/autodl-tmp/qwen3_hf/model')
    p.add_argument('--runs-root',default='/root/autodl-tmp/qwen3_hf/runs')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--gold-only',action='store_true')
    p.add_argument('--arms',help='Optional subset to score; generation always uses all five arms.')
    args=p.parse_args()
    {'prepare':prepare,'generate':generate,'score':score}[args.action](args)
