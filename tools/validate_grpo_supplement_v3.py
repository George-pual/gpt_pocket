"""Validate the local supplement without generating answers or starting training."""
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from math_verification_v3_frozen import verify_final_answer
from prepare_math_self600 import canonical

ROOT = Path(__file__).resolve().parents[1] / 'datasets/math/base_self600_20261015'
REMOTE = ROOT / 'remote_results'
OUT = ROOT / 'grpo_supplement_v3'


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((OUT / 'selection_report.json').read_text(encoding='utf-8'))
    data, new = read(OUT / 'train_expanded.jsonl'), read(OUT / 'supplement.jsonl')
    old = read(REMOTE / 'grpo_selected_direct367_v1/train.jsonl')
    scores = {row['row_id']: row for row in read(OUT / 'v3_scores.jsonl')}
    verdicts = read(OUT / 'sample_verdicts.jsonl')
    cache = read(OUT / 'verification_cache.jsonl')
    assert len(cache) == len({r['key'] for r in cache}) == 10686
    assert len(data) == len({r['row_id'] for r in data}) == 1200
    assert len(new) == len({r['row_id'] for r in new}) == 526
    old_ids = {row['row_id'] for row in old}
    assert len(old_ids) == 699
    assert {r['row_id'] for r in new} == {r['row_id'] for r in data} - old_ids
    assert all(scores[r['row_id']]['eligible'] for r in data)
    assert collections.Counter(r['difficulty'] for r in data) == dict(easy=360, medium=480, hard=360)
    source = {r['row_id']: r for r in read(REMOTE / 'grpo_reserved.jsonl')}
    for row in data:
        assert row['prompt'] == source[row['row_id']]['prompt']
        assert [message['role'] for message in row['prompt']] == ['system', 'user']
        assert row['prompt'][1]['content'] == row['problem']
        assert not ({'completion', 'reference_solution', 'solution', 'messages'} & row.keys())
        assert row['answer'] == source[row['row_id']]['answer']
        assert row['screen_correct_count'] == scores[row['row_id']]['k']
    keys = {canonical(row['problem']) for row in data}
    assert len(keys) == 1200
    for name in ['reviewed_direct/train.jsonl', 'development.jsonl', 'official_test/questions.jsonl']:
        assert not keys & {canonical(r['problem']) for r in read(REMOTE / name)}
    assert not {canonical(r['problem']) for r in new} & {canonical(r['problem']) for r in old}
    for name, expected in report['input_sha256'].items():
        assert digest(ROOT / name) == expected
    manifest = json.loads((REMOTE / 'all_results_manifest.json').read_text(encoding='utf-8'))
    for name in report['input_sha256']:
        relative = name.removeprefix('remote_results/')
        assert digest(REMOTE / relative) == manifest[relative]
    assert digest(OUT / 'train_expanded.jsonl') == report['train_sha256']
    assert digest(OUT / 'supplement.jsonl') == report['supplement_sha256']
    assert not any('timeout' in r['verdict']['status'] or 'error' in r['verdict']['status'] for r in cache)
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    screen_manifest = json.loads((REMOTE / 'grpo_screen_direct367_v1/manifest.json').read_text(encoding='utf-8'))
    tokenizer = Tokenizer.from_file(str(REMOTE / 'sft_direct367_v1/adapter/tokenizer.json'))
    template = ImmutableSandboxedEnvironment().from_string(screen_manifest['template'])
    token_ids = [tokenizer.encode(template.render(messages=row['prompt'], eos_token='<|endoftext|>',
                 add_generation_prompt=True), add_special_tokens=False).ids for row in source.values()]
    assert hashlib.sha256(json.dumps(token_ids).encode()).hexdigest() == screen_manifest['prompt_ids_sha256']
    lengths = {row_id: len(ids) for row_id, ids in zip(source, token_ids)}
    max_prompt_tokens = max(lengths[row['row_id']] for row in data)
    assert max_prompt_tokens <= 1024

    # A consistency check of the public API against the persistent-worker path;
    # this is not an independent estimate of the scorer's mathematical accuracy.
    buckets = collections.defaultdict(list)
    for row in verdicts:
        v = row['verdict']
        buckets[(v['status'], bool(v.get('normalization_rule')))].append(row)
    checks = [item for key in sorted(buckets) for item in buckets[key][:2]]
    groups = {}
    for path in sorted((REMOTE / 'grpo_screen_direct367_v1/sft/groups').glob('*.json')):
        g = json.loads(path.read_text(encoding='utf-8'))
        groups[g['row_id']] = g
    def check(row):
        g = groups[row['row_id']]
        s = next(s for s in g['samples'] if s['sample_index'] == row['sample_index'])
        r = source[row['row_id']]
        result = verify_final_answer(s['completion'], r['answer'], question=r['problem'])
        assert result == row['verdict'], (row['row_id'], row['sample_index'], result, row['verdict'])
        return dict(row_id=row['row_id'], sample_index=row['sample_index'], status=result['status'], matched=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(check, checks))
    removed = read(OUT / 'previous_not_retained.jsonl')
    reasons = collections.Counter()
    for row in removed:
        s = row['v3_score']
        if s['u']:
            reasons['unconfirmed_verdict'] += 1
        elif not s['gold_parser_supported']:
            reasons['gold_parser_unsupported'] += 1
        elif s['bad_end']:
            reasons['abnormal_termination'] += 1
        elif s['k'] == 0:
            reasons['all_task_answers_unsuccessful'] += 1
        elif s['k'] == 4:
            reasons['all_correct'] += 1
        else:
            reasons['other'] += 1
    result = dict(status='passed', expanded_count=len(data), supplement_count=len(new),
                  retained_count=len(data)-len(new), not_retained_reasons=dict(reasons),
                  cached_sample_status_counts=dict(collections.Counter(r['verdict']['status'] for r in verdicts)),
                  public_api_consistency_checks=checked,
                  exact_normalized_dedup_passed=True, prompt_reference_separation_passed=True,
                  maximum_prompt_tokens=max_prompt_tokens, prompt_limit=1024,
                  all_candidate_prompt_fingerprint_matches_original_screening=True,
                  source_hashes_match_original_retrieval_manifest=True,
                  scope='Local data validation only. No GPU generation or training. Exact dedup does not prove semantic independence.')
    (OUT / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
