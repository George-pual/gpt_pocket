"""Build SFT targets solely from recorded primary-agent decisions and frozen outputs.

Never chooses answers by matching gold. Gold is used only for post-build grading.
All edits are explicit and asserted against the reviewed candidate.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys

from direct_review_math_self600 import ROOT, SOURCE, packets, read

sys.path.insert(0, str(SOURCE / 'tools'))
from common import messages
from math_verification import extract_final_answer, verify_final_answer
from prepare_math_self600 import canonical


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def tidy_answer(value):
    value = value.strip()
    # Only unwrap whole final regions. Never search the reasoning for an answer.
    for _ in range(5):
        old = value
        for left, right in [('$$', '$$'), ('$', '$'), (r'\[', r'\]'), (r'\(', r'\)')]:
            if value.startswith(left) and value.endswith(right) and len(value) > len(left)+len(right):
                value = value[len(left):-len(right)].strip()
                break
        if value.startswith(r'\boxed{') and value.endswith('}'):
            depth = 0
            whole = True
            for k, ch in enumerate(value[len(r'\boxed'):]):
                if ch == '{': depth += 1
                elif ch == '}': depth -= 1
                if depth == 0 and k != len(value[len(r'\boxed'):])-1:
                    whole = False
                    break
            if whole: value = value[len(r'\boxed{'):-1].strip()
        if value == old: break
    assert value and len(value) < 200 and not re.search(r'<|>|\n', value), repr(value)
    return value


def curate(candidate, decision):
    text = candidate['completion']
    edit_log = []
    # Replacements run on the full reviewed text before selecting a complete method.
    for old, new in decision.get('replacements', []):
        assert old in text, (decision['review_index'], 'missing replacement', repr(old))
        text = text.replace(old, new)
        edit_log.append(dict(operation='replace', old=old, new=new))
    if decision.get('keep_from'):
        marker = decision['keep_from']
        assert marker in text, (decision['review_index'], 'missing keep_from')
        text = text[text.index(marker):]
        edit_log.append(dict(operation='keep_from', marker=marker))
    for start, end in decision.get('remove_between', []):
        assert start in text and end in text[text.index(start):]
        a = text.index(start)
        b = text.index(end, a)
        text = text[:a]+text[b:]
        edit_log.append(dict(operation='remove_between', start=start, end=end))
    if decision.get('trim_after'):
        marker = decision['trim_after']
        assert marker in text, (decision['review_index'], 'missing trim_after', repr(marker))
        text = text[:text.index(marker)+len(marker)]
        edit_log.append(dict(operation='trim_after_complete_solution', marker=marker))
    answer = tidy_answer(decision.get('final_answer', candidate['math_verification']['prediction']))
    solution = re.search(r'<SOLUTION>(.*?)</SOLUTION>', text, re.I | re.S)
    if decision.get('copy_final_to_reasoning'):
        assert solution, (decision['review_index'], 'missing solution to move')
        body = text[:solution.start()]+'\n'+solution[1]
        edit_log.append(dict(operation='move_solution_region_to_reasoning'))
    else:
        opening = re.search(r'<SOLUTION>', text, re.I)
        body = text[:opening.start()] if opening else text
    # Remove known control boundaries only; no answer-line heuristic cuts proof text.
    body = re.sub(r'\\?<\\?/?(?:start|end)_working_out\\?>', '', body, flags=re.I)
    body = re.sub(r'\[(?:start|end)_working_out\]', '', body, flags=re.I)
    body = re.sub(r'\\(?:begin|end)\{working_out\}', '', body, flags=re.I)
    body = body.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
    # Discard alternate answer boundary tags while retaining their reviewed content.
    body = re.sub(r'</?(?:start_solution|end_solution)>', '', body, flags=re.I).strip()
    assert body and not re.search(r'</?SOLUTION\b|working_out|<\|', body, re.I), (decision['review_index'], 'remaining boundary')
    completion = body+'\n<end_working_out><SOLUTION>'+answer+'</SOLUTION><|endoftext|>'
    edit_log.append(dict(operation='normalize_boundaries_and_final_region', final_answer=answer))
    assert extract_final_answer(completion) == (answer, 'solution', None)
    assert completion.count('<end_working_out>') == 1 and completion.count('<|endoftext|>') == 1
    return completion, answer, edit_log


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', default='sft_closure_results/models/E1_sft_v3/adapter/tokenizer.json')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--format-only', action='store_true')
    args = ap.parse_args()
    out = ROOT/'reviewed_direct'
    decisions = read(ROOT/'direct_audit_decisions.jsonl')
    by_index = collections.defaultdict(list)
    for row in decisions: by_index[row['review_index']].append(row)
    all_questions = read(SOURCE/'questions.jsonl')
    items = packets()
    accepted, provenance, dispositions = [], [], []
    for i, (question, candidates) in enumerate(items):
        ds = by_index[i]
        passes = [d for d in ds if d['verdict'].startswith('pass')]
        exclusions = [d for d in ds if d['verdict'].startswith('exclude_question')]
        assert passes or exclusions, ('pending question', i)
        if not passes:
            dispositions.append(dict(row_id=question['row_id'], review_index=i, status='excluded_reasoning_or_source', decisions=ds))
            continue
        d = passes[-1]
        c = next(c for c in candidates if c['sample_index']==d['sample_index'])
        completion, final, edits = curate(c, d)
        row = dict(row_id=question['row_id'], problem=question['problem'], answer=question['answer'],
                   level=question['level'], type=question['type'], prompt=messages(question['problem']),
                   completion=completion, sample_index=c['sample_index'], review_index=i,
                   reasoning_review_status='source_grounded_direct_agent_review_pass', review_approved=True,
                   review_reason=d['reason'], final_answer=final)
        accepted.append(row)
        provenance.append(dict(row_id=question['row_id'], review_index=i, sample_index=c['sample_index'],
             reference_solution=question['reference_solution'], original_completion=c.get('original_completion', c['completion']),
             review_input_completion=c['completion'], original_token_ids=c['token_ids'],
             original_token_ids_scope='Original generated output only; edited completion has different token IDs.',
             original_math_verification=c.get('raw_math_verification', c['math_verification']),
             recovered_math_verification=c['math_verification'], decision=d, edits=edits,
             completion_sha256=hashlib.sha256(completion.encode()).hexdigest()))
        dispositions.append(dict(row_id=question['row_id'], review_index=i, status='accepted_direct_review', sample_index=c['sample_index']))
    eligible = {q['row_id'] for q, _ in items}
    dispositions += [dict(row_id=q['row_id'], status='excluded_no_correct_eos_candidate', reasoning_reviewed=False)
                     for q in all_questions if q['row_id'] not in eligible]
    assert len(accepted)==367 and len(dispositions)==600
    problems = {canonical(q['problem']) for q in accepted}
    assert len(problems)==len(accepted)
    for other in ['development.jsonl','grpo_reserved.jsonl']:
        assert not problems & {canonical(q['problem']) for q in read(ROOT/other)}
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    tok = Tokenizer.from_file(args.tokenizer)
    manifest = json.loads((SOURCE/'results/run_manifest.json').read_text(encoding='utf-8'))
    template = ImmutableSandboxedEnvironment().from_string(manifest['chat_template'])
    def prefix(q):
        rendered = template.render(messages=messages(q['problem']), eos_token='<|endoftext|>', add_generation_prompt=True)
        return tok.encode(rendered, add_special_tokens=False).ids
    frozen_prompts = [prefix(q) for q in all_questions]
    fingerprint = hashlib.sha256(json.dumps(frozen_prompts).encode()).hexdigest()
    assert fingerprint == manifest['prompt_token_ids_sha256'], 'Local tokenizer must reproduce all 600 inference prompts exactly'
    assert tok.token_to_id('<|endoftext|>') == 151643
    lengths = []
    for row in accepted:
        prompt_ids = prefix(row)
        completion_ids = tok.encode(row['completion'], add_special_tokens=False).ids
        assert completion_ids[-1]==151643 and completion_ids.count(151643)==1
        lengths.append(len(prompt_ids)+len(completion_ids))
        row['formatted_tokens'] = lengths[-1]
        assert lengths[-1] <= 2048, ('too_long', row['review_index'], lengths[-1])
    jsonl(out/'format_preview.jsonl', accepted)
    jsonl(out/'provenance.jsonl', provenance)
    jsonl(out/'question_dispositions.jsonl', dispositions)
    save(out/'format_check.json', dict(count=len(accepted), max_tokens=max(lengths),
         prompt_fingerprint=fingerprint, tokenizer_sha256=sha(args.tokenizer), max_length=2048,
         no_truncation=True, no_exact_dev_or_grpo_overlap=True, answer_equivalence_recheck='pending'))
    if args.format_only:
        print('FORMAT_PASS', len(accepted), 'MAX_TOKENS', max(lengths), flush=True)
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(verify_final_answer, r['completion'], r['answer']):i for i,r in enumerate(accepted)}
        done = 0
        for future in as_completed(pending):
            i = pending[future]
            accepted[i]['final_answer_verification'] = future.result()
            done += 1
            if done%25==0 or done==len(accepted):
                print('VERIFY', done, '/', len(accepted), flush=True)
    failures = [r for r in accepted if not r['final_answer_verification']['correct']]
    jsonl(out/'verification_failures.jsonl', failures)
    assert not failures, ('final answer recheck failed', [(r['review_index'],r['final_answer_verification']) for r in failures])
    check=json.loads((out/'format_check.json').read_text(encoding='utf-8'))
    check['answer_equivalence_recheck']='pass'
    save(out/'format_check.json',check)
    jsonl(out/'train.jsonl', accepted)
    jsonl(out/'messages.jsonl', [dict(messages=r['prompt']+[dict(role='assistant', content='<start_working_out>'+r['completion'].removesuffix('<|endoftext|>'))]) for r in accepted])
    train_sha = sha(out/'train.jsonl')
    save(out/'direct_agent_spotcheck.json', dict(status='pass', method='Primary agent direct source-grounded review of every selected solution',
         selected_questions_reviewed=len(accepted), eligible_questions_reviewed=406, train_sha256=train_sha,
         decisions_sha256=sha(ROOT/'direct_audit_decisions.jsonl'), auxiliary_model_used=False))
    save(out/'summary.json', dict(status='sft_data_ready', train_questions=len(accepted), inference_questions=600,
         eligible_review_questions=406, excluded_reasoning_or_source_questions=39, excluded_no_correct_eos_questions=194,
         train_sha256=train_sha, provenance_sha256=sha(out/'provenance.jsonl'),
         decisions_sha256=sha(ROOT/'direct_audit_decisions.jsonl'), max_sequence_tokens=max(lengths),
         mean_sequence_tokens=sum(lengths)/len(lengths), tokenizer_prompt_fingerprint=fingerprint,
         no_truncation=True, all_final_answers_reverified=True, auxiliary_model_used=False,
         source_gold_errors=[d for d in decisions if d.get('corrected_reference_answer')],
         strata=dict(collections.Counter(r['level']+' / '+r['type'] for r in accepted)),
         sft_training_started=False, grpo_started=False,
         limitations='Direct mathematical review is not formal proof verification. Development set may have historical exposure.'))
    print('SFT_DATA_READY', train_sha, flush=True)


if __name__=='__main__':
    main()
