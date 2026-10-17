"""Paired frozen BF16 Base/SFT-v3 evaluation, greedy and T=1 G=4."""
import argparse
import collections
import fcntl
import gc
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from common import tokenizer, messages, read_jsonl, save_json, file_hash
from prepare_bc import canonical
from math_verification import verify_final_answer, require_math_verify
from closure_rewards import complete_format
from reward_v2 import repetition_ratio


def problem(row):
    return row.get('problem') or next(m['content'] for m in row.get('messages', row.get('prompt', [])) if m['role'] == 'user')


def select(root, data, out):
    excluded = set()
    paths = [data, root/'prepared/sft.jsonl']
    paths += [root/f'runs/E1_closure_sft_v1/{folder}/{name}.jsonl'
              for folder in ('data', 'data_original') for name in ('train', 'validation')]
    for path in paths:
        excluded.update(canonical(problem(r)) for r in read_jsonl(path))
    prior_ids = set()
    prior_paths = list((root/'runs').rglob('rollouts.jsonl'))
    for path in prior_paths:
        for r in read_jsonl(path):
            if r.get('row_id') is not None:
                prior_ids.add(r['row_id'])
    source = root/'prepared/grpo_article.jsonl'
    pool = read_jsonl(source)
    random.Random(20261010).shuffle(pool)
    chosen, seen = [], set(excluded)
    for row in pool:
        q = problem(row)
        key = canonical(q)
        if key in seen or row['row_id'] in prior_ids:
            continue
        gold_check = verify_final_answer('<SOLUTION>'+str(row['answer'])+'</SOLUTION>', row['answer'])
        if not gold_check['correct']:
            continue
        seen.add(key)
        chosen.append({'row_id': row['row_id'], 'problem': q, 'answer': row['answer'], 'prompt': messages(q)})
        if len(chosen) == 100:
            break
    assert len(chosen) == 100
    qpath = out/'questions.jsonl'
    qpath.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in chosen), encoding='utf-8')
    save_json(out/'selection.json', {
        'questions': 100, 'seed': 20261010, 'questions_sha256': file_hash(qpath),
        'source_sha256': file_hash(source), 'row_ids': [r['row_id'] for r in chosen],
        'exclusions_sha256': {str(p): file_hash(p) for p in paths},
        'prior_rollout_files': len(prior_paths), 'prior_rollout_ids_excluded': len(prior_ids),
        'no_canonical_training_validation_overlap': True,
        'scope': 'Fresh within-source held-out diagnostic, not external benchmark; semantic/pretraining contamination unknown.'})
    return chosen


def summarize(records, group_size):
    groups = collections.defaultdict(list)
    for r in records:
        groups[r['row_id']].append(r)
    assert all(len(g) == group_size for g in groups.values())
    keys = ('correct', 'eos_finished', 'format_valid', 'normal_end', 'truncated', 'repetitive')
    result = {'questions': len(groups), 'generations': len(records),
              **{k+'_rate': statistics.mean(r[k] for r in records) for k in keys},
              'mean_tokens': statistics.mean(r['tokens'] for r in records),
              'mean_repeated_12gram_ratio': statistics.mean(r['repeated_12gram_ratio'] for r in records),
              'verification_status_counts': dict(collections.Counter(r['math_verification']['status'] for r in records))}
    result['observed_pass_at_'+str(group_size)] = statistics.mean(any(r['correct'] for r in g) for g in groups.values())
    result['mixed_correctness_group_rate'] = statistics.mean(0 < sum(r['correct'] for r in g) < len(g) for g in groups.values())
    result['all_wrong_group_rate'] = statistics.mean(not any(r['correct'] for r in g) for g in groups.values())
    result['mean_unique_completion_fraction'] = statistics.mean(len({r['completion'] for r in g})/len(g) for g in groups.values())
    return result


def main():
    p = argparse.ArgumentParser()
    for k in ('storage', 'data', 'output'):
        p.add_argument('--'+k, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--sft-only', action='store_true')
    p.add_argument('--base-only', action='store_true')
    p.add_argument('--sample-only', action='store_true')
    p.add_argument('--generations', type=int, choices=(1, 4), default=4)
    a = p.parse_args()
    if a.sft_only and a.base_only:
        p.error('--sft-only and --base-only are mutually exclusive')
    root, out = Path(a.storage), Path(a.output)
    if out.exists() and not a.resume:
        raise RuntimeError('Use a fresh evaluation output directory')
    out.mkdir(parents=True, exist_ok=a.resume)
    lock = (out/'gpu_eval.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require_math_verify()
    rows = read_jsonl(out/'questions.jsonl') if a.resume else select(root, Path(a.data), out)
    assert len(rows) == 100
    assert file_hash(out/'questions.jsonl') == json.loads((out/'selection.json').read_text())['questions_sha256']
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig, StoppingCriteria, StoppingCriteriaList, set_seed
    from peft import PeftModel
    t = tokenizer(root/'model')
    t.padding_side = 'left'
    prompts = [t.apply_chat_template(r['prompt'], tokenize=False, add_generation_prompt=True) for r in rows]
    prompt_ids = t(prompts, add_special_tokens=False)['input_ids']
    adapter = root/'runs/E1_sft_v3/adapter'
    adapter_hash = file_hash(adapter/'adapter_model.safetensors')
    shared = dict(max_new_tokens=3894, min_new_tokens=0, num_beams=1,
                  eos_token_id=t.eos_token_id, pad_token_id=t.pad_token_id, bos_token_id=None,
                  repetition_penalty=1.0, use_cache=True)
    configs = {'greedy': GenerationConfig(**shared, do_sample=False),
               'sample_t1_g4': GenerationConfig(**shared, do_sample=True, temperature=1.0, top_p=1.0, top_k=0)}
    manifest = {
        'experiment': 'Base vs SFT v3', 'questions': 100, 'total_generations': 1000,
        'model': str(root/'model'), 'adapter': str(adapter), 'adapter_sha256': adapter_hash,
        'seed': 3407, 'batch_size': a.batch_size, 'chat_template': t.chat_template,
        'prompt_tokens_sha256': hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
        'generation_configs': {k: v.to_dict() for k,v in configs.items()},
        'representation': 'Both use original dense BF16 Base weights; SFT adds frozen v3 LoRA. Training used NF4 QLoRA.',
        'normal_end_definition': 'EOS emitted and exactly one closed SOLUTION at end after end_working_out',
        'correctness_definition': 'Math-Verify final-answer equivalence, independent of normal_end metric; no reasoning proof verification',
        'repetition_definition': 'Repeated lowercased whitespace 12-gram ratio > 0.05',
        'pass_at_4_definition': 'Observed fraction of questions with at least one correct answer among four samples',
        'versions': {k: __import__('importlib.metadata', fromlist=['version']).version(k) for k in ('torch','transformers','peft','math-verify')},
    }
    if a.resume:
        assert json.loads((out/'manifest.json').read_text()) == manifest, 'Evaluation configuration changed'
    else:
        save_json(out/'manifest.json', manifest)
    if a.sample_only:
        configs = {'sample_t1_g'+str(a.generations): configs['sample_t1_g4']}
    arms = ['base'] if a.base_only else (['sft_v3'] if a.sft_only else ['sft_v3', 'base'])
    count_per_arm = (0 if a.sample_only else 100) + 100*a.generations
    save_json(out/'execution_order.json', {'arms': arms, 'modes': list(configs), 'resumed': a.resume})
    save_json(out/'active_scope.json', {'arms': arms, 'questions': 100,
        'total_generations': count_per_arm*len(arms), 'greedy_cancelled': a.sample_only,
        'sample_generations_per_question': a.generations, 'status': 'running',
        'base_cancelled': a.sft_only, 'existing_base_outputs': 'Retained but excluded from SFT-only results'})
    set_seed(3407)
    model = AutoModelForCausalLM.from_pretrained(root/'model', local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
    model.requires_grad_(False)
    model.eval()
    summaries = {}
    started = time.time()
    class Progress(StoppingCriteria):
        def __init__(self, width, label):
            self.width, self.label, self.last = width, label, time.time()
        def __call__(self, input_ids, scores, **kwargs):
            if time.time()-self.last >= 45:
                self.last = time.time()
                print('GENERATING', self.label, input_ids.shape[1]-self.width, flush=True)
            return False
    for arm in arms:
        if arm == 'sft_v3':
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
            model.requires_grad_(False)
            model.eval()
        elif isinstance(model, PeftModel):
            # Unload without merging, restoring the unchanged original Base.
            model = model.unload()
            model.requires_grad_(False)
            model.eval()
        for mode, config in configs.items():
            n = 1 if mode == 'greedy' else a.generations
            target = out/arm/mode
            target.mkdir(parents=True, exist_ok=a.resume)
            records = read_jsonl(target/'rollouts.jsonl') if (target/'rollouts.jsonl').exists() else []
            expanded = [(i,j) for i in range(len(rows)) for j in range(n)]
            assert [(r['row_id'], r['sample_index']) for r in records] == [(rows[i]['row_id'],j) for i,j in expanded[:len(records)]]
            assert len(records) == len(expanded) or len(records) % a.batch_size == 0, 'Resume requires complete saved batches'
            print('EVAL_CELL_START', arm, mode, 'saved', len(records), flush=True)
            with (target/'rollouts.jsonl').open('a', encoding='utf-8') as f:
                for start in range(len(records), len(expanded), a.batch_size):
                    batch = expanded[start:start+a.batch_size]
                    set_seed(3407+start)
                    x = t([prompts[i] for i,j in batch], padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
                    width = x.input_ids.shape[1]
                    with torch.inference_mode():
                        outputs = model.generate(**x, generation_config=config,
                            stopping_criteria=StoppingCriteriaList([Progress(width, arm+'/'+mode)]))
                    for (i,j), ids in zip(batch, outputs[:,width:].cpu().tolist()):
                        eos = t.eos_token_id in ids
                        if eos:
                            ids = ids[:ids.index(t.eos_token_id)+1]
                        text = t.decode(ids, skip_special_tokens=True)
                        ratio = repetition_ratio(text)
                        grade = verify_final_answer(text, rows[i]['answer'])
                        valid = complete_format(text)
                        r = {'row_id': rows[i]['row_id'], 'sample_index': j, 'gold': rows[i]['answer'],
                             'completion': text, 'tokens': len(ids), 'eos_finished': eos,
                             'format_valid': valid, 'normal_end': eos and valid,
                             'truncated': not eos and len(ids)>=3894, 'repetitive': ratio>.05,
                             'repeated_12gram_ratio': ratio, 'correct': grade['correct'], 'math_verification': grade}
                        records.append(r)
                        f.write(json.dumps(r, ensure_ascii=False)+'\n')
                    f.flush()
                    save_json(out/'progress.json', {'arm': arm, 'mode': mode, 'completed': len(records),
                        'total_this_mode': len(expanded), 'runtime_seconds': time.time()-started})
                    print('EVAL_PROGRESS', arm, mode, len(records), '/', len(expanded), flush=True)
                    del outputs, x
            summary = summarize(records, n)
            save_json(target/'summary.json', summary)
            summaries[arm+'/'+mode] = summary
            save_json(out/'comparison_partial.json', summaries)
    assert file_hash(adapter/'adapter_model.safetensors') == adapter_hash
    save_json(out/('sft_summary.json' if a.sft_only else ('base_summary.json' if a.base_only else 'comparison.json')),
              {'status': 'complete', 'runtime_seconds': time.time()-started, 'arms': summaries})
    if a.sft_only:
        lines = ['# SFT v3 生成评测', '', f'100 道固定留出题，T=1.0、G={a.generations} 采样 {100*a.generations} 条。Base 评测已取消。'+('Greedy 已取消。' if a.sample_only else '另含 Greedy 100 条。'), '',
                 '| 解码 | 正确率 | 正常结束率 | 格式合格率 | 重复率 | 截断率 | 平均 tokens |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for mode in configs:
            s = summaries['sft_v3/'+mode]
            lines.append('| '+mode+' | '+' | '.join(f'{s[k+"_rate"]:.1%}' for k in
                ('correct','normal_end','format_valid','repetitive','truncated'))+f' | {s["mean_tokens"]:.1f} |')
        s = summaries['sft_v3/sample_t1_g'+str(a.generations)]
        lines += ['', f'采样至少一次答对（观测 pass@{a.generations}）：{s["observed_pass_at_"+str(a.generations)]:.1%}。',
                  f'同组有对有错：{s["mixed_correctness_group_rate"]:.1%}；四次全错：{s["all_wrong_group_rate"]:.1%}。', '',
                  '范围：同源留出诊断；最终答案评分使用 Math-Verify，不证明推导正确。没有 Base 对照，不能据此判断 SFT 相对 Base 的提升。',
                  '共享生成配置保留于 manifest.json；本轮实际执行范围以 active_scope.json 为准。', '']
        (out/'SFT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    if a.base_only and a.sample_only:
        mode = 'sample_t1_g'+str(a.generations)
        sft_path = out/'sft_v3'/mode/'rollouts.jsonl'
        if sft_path.exists():
            sft_rows = read_jsonl(sft_path)
            base_rows = read_jsonl(out/'base'/mode/'rollouts.jsonl')
            assert [(r['row_id'],r['sample_index'],r['gold']) for r in base_rows] == [(r['row_id'],r['sample_index'],r['gold']) for r in sft_rows]
            paired = {'base': summarize(base_rows,a.generations), 'sft_v3': summarize(sft_rows,a.generations),
                      'base_right_sft_wrong': sum(b['correct'] and not s['correct'] for b,s in zip(base_rows,sft_rows)),
                      'base_wrong_sft_right': sum(not b['correct'] and s['correct'] for b,s in zip(base_rows,sft_rows)),
                      'questions_sha256':file_hash(out/'questions.jsonl'), 'mode':mode}
            save_json(out/'sample_comparison.json',paired)
            lines = ['# Base 与 SFT v3：同题单次采样对照', '',
                     '两边使用相同 100 道题、Prompt、模板、EOS、T=1.0、top_p=1、top_k=0、3894 tokens 上限、seed 与批次划分。', '',
                     '| 模型 | 严格金标匹配率 | 正常结束率 | EOS 结束率 | 重复率 | 截断率 | 平均 tokens |',
                     '|---|---:|---:|---:|---:|---:|---:|']
            for arm in ('base','sft_v3'):
                s = paired[arm]
                lines.append('| '+arm+' | '+' | '.join(f'{s[k+"_rate"]:.1%}' for k in
                    ('correct','normal_end','eos_finished','repetitive','truncated'))+f' | {s["mean_tokens"]:.1f} |')
            lines += ['', f'Base 匹配但 SFT 不匹配：{paired["base_right_sft_wrong"]} 条；SFT 匹配但 Base 不匹配：{paired["base_wrong_sft_right"]} 条。', '',
                      '这是一组固定 seed 的同源留出诊断，不是外部 benchmark。严格金标匹配受题干歧义和答案解析影响，不等同于全部真实数学正确率。未验证推导正确性。',
                      '两边均为 BF16 推理；SFT 训练采用 NF4 QLoRA。Base 未加载 adapter。每题一次不能推断 pass@4 或组内奖励多样性。', '']
            (out/'BASE_SFT_SAMPLE_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('COMPARISON_COMPLETE', json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
