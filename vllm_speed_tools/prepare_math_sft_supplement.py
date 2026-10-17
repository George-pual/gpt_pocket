"""Collect every MATH L2/3 train question unused by the 0--1536 GRPO lineage.

No sampling, model generation, training, tokenizer truncation or scoring changes.
Requires pyarrow only to read all 5000 official test references for exclusion.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text).replace('$','')).casefold()


def problem(row):
    if row.get('problem'): return row['problem']
    if row.get('question'): return row['question']
    for msg in row.get('prompt',row.get('messages',[])):
        if isinstance(msg,dict) and msg.get('role')=='user': return msg['content']
    raise ValueError('Reference has no question text')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reviewed-overrides',type=Path,
                        help='Explicit, manually reviewed complete-answer and solution repairs')
    args=parser.parse_args()
    w=args.workspace.resolve(); out=args.output.resolve()
    if out.exists(): raise RuntimeError('Refuse to overwrite: '+str(out))
    import pyarrow.parquet as pq
    sys.path.insert(0,str(w/'vllm_speed_tools'))
    from math_verification import extract_final_answer
    from audited_answer_scoring import _boxes
    sys.path.insert(0,str(w/'datasets/math/test_l2_l3_remaining_v1'))
    from train_reviewed_sft import convert_messages
    from common import messages,END,SOL_START,SOL_END
    consumed_dir=w/'deliverables/SFTv3_GRPO_1536_training_handoff/data/grpo/actually_consumed'
    paths={
        'math_train':w/'datasets/math/train.jsonl',
        'candidate_answers':w/'datasets/grpo_candidates_gsm8k_math_l2_l3/math_level2_3_candidates.jsonl',
        'consumed':consumed_dir/'all_3072_questions.jsonl',
        'ledger':consumed_dir/'exposure_ledger.jsonl',
        'sft505':w/'sft_quality_audit_v2/revision_20261011_user15/train_merged_505.jsonl',
    }
    source=read(paths['math_train']); consumed=read(paths['consumed']); ledger=read(paths['ledger'])
    policy=json.loads(args.reviewed_overrides.read_text(encoding='utf-8')) if args.reviewed_overrides else {}
    if policy:
        assert policy['source_math_train_sha256']==digest(paths['math_train'])
        source_snapshot=Path(policy['source_supplement'])
        assert digest(source_snapshot)==policy['source_supplement_sha256']
        paths['reviewed_overrides']=args.reviewed_overrides.resolve()
    overrides=policy.get('overrides',{})
    reviewed=policy.get('reviewed_rows',{})
    assert len(source)==7500 and len(consumed)==len(ledger)==3072
    assert len({r['row_id'] for r in consumed})==3072
    assert {r['row_id'] for r in consumed}=={r['row_id'] for r in ledger}
    assert min(r['cumulative_generation_step'] for r in ledger)==0
    assert max(r['cumulative_generation_step'] for r in ledger)==1532
    assert Counter(r['phase'] for r in ledger)=={
        'E1_sft_v3_half_math1024_fast_v2':2048,
        'E1_sft_v3_expanded_continue1536_v1':1024}
    # Verify the compact ledger independently against both training rollout logs.
    rollout_paths=[
        w/'deliverables/SFTv3_GRPO_1536_training_handoff/logs/E1_sft_v3_half_math1024_fast_v2/rollouts.jsonl',
        w/'deliverables/SFTv3_GRPO_1536_training_handoff/logs/E1_sft_v3_expanded_continue1536_v1/rollouts.jsonl']
    rollout_ids=[]
    for path in rollout_paths:
        counts=Counter()
        with path.open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line); counts[r['row_id']]+=1
        assert set(counts.values())=={4}
        rollout_ids.extend(counts)
        paths['rollouts_'+path.parent.name]=path
    assert len(rollout_ids)==3072 and set(rollout_ids)=={r['row_id'] for r in consumed}
    trained_ids={r['row_id'] for r in consumed}
    trained_keys={canonical(problem(r)) for r in consumed}
    old_sft=read(paths['sft505'])
    assert len(old_sft)==505
    old_sft_keys={canonical(problem(r)) for r in old_sft}
    heldout=defaultdict(set)
    official_test=[]
    parquet_paths=sorted((w/'datasets/math/test_l2_l3_200/raw').glob('*/test-*.parquet'))
    assert len(parquet_paths)==7
    for path in parquet_paths:
        paths['official_test_'+path.parent.name]=path
        for i,r in enumerate(pq.read_table(path).to_pylist()):
            rid=f'math:{path.parent.name}/{path.name}:{i}'
            heldout[canonical(r['problem'])].add('official_math_test')
            official_test.append(dict(row_id=rid,problem=r['problem'],level=r['level'],type=r['type']))
    assert len(official_test)==5000
    validation_paths=sorted((w/'datasets/grpo_candidates_gsm8k_math_l2_l3').glob('**/validation.jsonl'))
    validation_paths += [w/'sft_quality_audit_v2/eval_questions_100/questions.jsonl',
                         w/'closure_grpo_experiment/inputs/validation.jsonl',
                         w/'closure_grpo_experiment/inputs/prior_eval_questions.jsonl',
                         w/'sft_closure_data/validation.jsonl']
    for path in validation_paths:
        paths['heldout_'+str(path.relative_to(w)).replace('\\','/')]=path
        for r in read(path): heldout[canonical(problem(r))].add(str(path.relative_to(w)).replace('\\','/'))
    candidates=read(paths['candidate_answers'])
    cached_answers={r['row_id']:r['answer'] for r in candidates}
    cached_by_key={canonical(r['problem']):r['answer'] for r in candidates}
    all_l23=[r for r in source if r['level'] in ('Level 2','Level 3')]
    assert len(all_l23)==2940
    exclusions=[]; supplements=[]; seen={}; quality_flags=[]
    for r in all_l23:
        key=canonical(r['problem']); reasons=[]
        if r['id'] in trained_ids or key in trained_keys: reasons.append('consumed_by_grpo_0_1536')
        if key in old_sft_keys: reasons.append('existing_sft505')
        if key in heldout: reasons.append('heldout_validation_or_test')
        if key in seen: reasons.append('duplicate_supplement_question')
        if reasons:
            exclusions.append(dict(row_id=r['id'],level=r['level'],type=r['type'],problem=r['problem'],
                                   reasons=reasons,heldout_sources=sorted(heldout.get(key,[])),
                                   duplicate_of=seen.get(key)))
            continue
        seen[key]=r['id']
        answer,answer_source,error=extract_final_answer(r['solution'])
        cached=cached_answers.get(r['id'],cached_by_key.get(key))
        if error:
            # The source preparation supports a final single-digit unbraced box.
            single=re.search(r'\\boxed\s*([0-9])\s*[.$\s]*$',r['solution'])
            if single:
                answer=single[1]; answer_source='final_single_digit_unbraced_box'; error=None
        if error or not answer: raise ValueError((r['id'],'Unrecoverable source answer',error))
        assert cached is not None and answer==cached, (r['id'],answer,cached)
        boxes=_boxes(r['solution'])
        distinct_box_values={re.sub(r'\s+','',box[2]) for box in boxes}
        if len(distinct_box_values)>1 and r['id'] not in overrides:
            raise ValueError((r['id'],'Different boxed answers require explicit completeness review; refusing last-box-only target'))
        reasoning=re.sub(r'\\boxed\s*','',r['solution']).strip()
        reasoning=re.sub(r'(?:Final\s+answer|Answer)\s*:', 'Conclusion:',reasoning,flags=re.I)
        source_reasoning=reasoning
        solution=reasoning+'\nFinal answer: \\boxed{'+answer+'}'
        if r['id'] in overrides:
            override=overrides[r['id']]
            assert override['previous_answer']==answer
            answer=override['answer']; solution=override['solution']
            assert solution.count('\nFinal answer:')==1
            reasoning=solution.split('\nFinal answer:',1)[0].strip()
            final,_,final_error=extract_final_answer(solution)
            assert not final_error and final==answer
        msgs=[dict(role='user',content=r['problem']),dict(role='assistant',content=solution)]
        prepared=convert_messages(dict(messages=msgs))
        assert prepared['answer']==answer
        assert prepared['completion']==reasoning+'\n'+END+SOL_START+answer+SOL_END+'<|endoftext|>'
        flags=[]
        if len(boxes)>1: flags.append('multiple_source_boxes')
        if '[asy]' in r['problem'] or '[asy]' in r['solution']: flags.append('asymptote_markup')
        if len(source_reasoning.split())<20: flags.append('short_source_solution')
        if re.search(r'\b(?:clearly|obviously|by symmetry)\b',source_reasoning,re.I): flags.append('compressed_justification_wording')
        rec=dict(supplement_index=len(supplements)+1,row_id=r['id'],source='math',split='train',
                 level=r['level'],type=r['type'],problem=r['problem'],answer=answer,
                 reference_solution=r['solution'],solution=solution,messages=msgs,
                 answer_extraction=('manual_reviewed_override' if r['id'] in overrides else answer_source),
                 original_last_box_answer=cached,quality_flags=flags,
                 grpo_0_1536_exposure=False,mathematical_steps_independently_reviewed=r['id'] in reviewed,
                 manual_review=reviewed.get(r['id']),revision=overrides.get(r['id']))
        supplements.append(rec)
        if flags: quality_flags.append(dict(row_id=r['id'],flags=flags,problem=r['problem'],reference_solution=r['solution'],manual_review=reviewed.get(r['id'])))
    assert supplements and len(supplements)+len(exclusions)==2940
    selected_keys={canonical(r['problem']) for r in supplements}
    assert len(selected_keys)==len(supplements)
    assert not (selected_keys & trained_keys or selected_keys & old_sft_keys or selected_keys & set(heldout))
    # All admissible questions are included; no length/answer-shape/difficulty subsampling.
    assert {r['id'] for r in all_l23 if not (
        r['id'] in trained_ids or canonical(r['problem']) in trained_keys
        or canonical(r['problem']) in old_sft_keys or canonical(r['problem']) in heldout
    )} == {r['row_id'] for r in supplements}
    out.mkdir(parents=True)
    write(out/'supplement_records.jsonl',supplements)
    write(out/'supplement_train_messages.jsonl',[dict(messages=r['messages']) for r in supplements])
    write(out/'supplement_prepared.jsonl',[convert_messages(dict(messages=r['messages'])) for r in supplements])
    write(out/'excluded_questions.jsonl',exclusions)
    write(out/'quality_review_queue.jsonl',quality_flags)
    write(out/'official_test_exclusion_reference.jsonl',official_test)
    (out/'questions_and_solutions.txt').write_text('\n\n'.join(
        f"Supplement {r['supplement_index']} | {r['row_id']} | {r['level']} | {r['type']}\n{r['problem']}\n\n{r['solution']}"
        for r in supplements)+'\n',encoding='utf-8')
    reason_counts=Counter(reason for r in exclusions for reason in r['reasons'])
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        status='data_prepared_no_training',source_repository='EleutherAI/hendrycks_math',
        source_revision='21a5633873b6a120296cce3e2df9d5550074f4a3',source_split='train',
        selection='All Level 2/3 train questions unused by the actual 0--1536 GRPO lineage, excluding existing SFT questions and heldout validation/test exact matches.',
        requested_step_note='User referred to 1534; audited saved lineage is 1024+512=1536 updates. Generation ledger is batched through step1532, covering its last four updates; all consumed questions from the saved1536 model are excluded.',
        grpo_total_unique_consumed=3072,grpo_math_unique_consumed=1508,
        source_level2_3_rows=2940,source_level_counts=dict(Counter(r['level'] for r in all_l23)),
        selected=len(supplements),selected_level_counts=dict(Counter(r['level'] for r in supplements)),
        selected_subject_counts=dict(Counter(r['type'] for r in supplements)),
        excluded=len(exclusions),exclusion_reason_counts=dict(reason_counts),
        exclusion_counts_note='Reason counts can overlap; excluded counts unique source rows.',
        grpo_logs_independently_match_ledger=True,official_test_references=5000,
        duplicate_policy='NFKC, remove dollar delimiters, remove whitespace, casefold; IDs also checked. No semantic/LaTeX-equivalence deduplication.',
        source_solution_policy='Preserve official reference_solution byte-for-byte as a string. Use explicit reviewed overrides where provided; otherwise remove boxed styling only in reasoning, rename preexisting final-answer labels, append one final boxed answer. Different boxed values require explicit review, never blind last-box selection.',
        validation_policy='No new split: include every admissible unused question. Previously heldout questions remain excluded.',
        quality_review_flagged=len(quality_flags),quality_flag_counts=dict(Counter(f for r in quality_flags for f in r['flags'])),
        quality_note='Official worked solutions, not newly generated answers. Extraction/format checks do not independently verify every mathematical step. Flags are review prompts, not error verdicts.',
        training_started=False,intended_starting_adapter='runs/E1_sft_v3/adapter',
        manually_reviewed_selected=sum(r['row_id'] in reviewed for r in supplements),
        explicit_solution_rewrites=sum(r['row_id'] in overrides for r in supplements),
        mathematically_unreviewed_selected=sum(r['row_id'] not in reviewed for r in supplements),
        tokenizer_length_checked=False,no_truncation_or_tail_removal=True,
        original_converter_passed=len(supplements),
        input_sha256={str(p.relative_to(w)).replace('\\','/'):digest(p) for p in paths.values()},
        output_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
        preparation_script_sha256=digest(Path(__file__)))
    save(out/'manifest.json',manifest)
    lines=['# LoRA v3 MATH 补充数据集','',
        f"从官方 MATH train 的 2940 条 Level 2/3 题中，保留全部 {len(supplements)} 条符合条件的题，不做固定数量抽样。",'',
        '依据截至1536步（1024+512）的实际GRPO训练账本及两段rollout日志排除已消耗题。不是把进入候选池或做过筛选的题一概视为已训练。用户所说1534步按已保存1536模型的完整暴露范围处理，最后一批生成在1532步记录。',
        '已排除已有SFT505题以及已知验证集和全部5000条官方MATH test中的规范化精确重复题。验证集不挪作训练，也不再从本补充集抽取新的验证集。','',
        '## 数量','',f"- GRPO实际消耗的MATH题：1508（L2=741，L3=767）。",
        f"- 补充集：{len(supplements)}；难度分布：{manifest['selected_level_counts']}。",
        f"- 排除{len(exclusions)}条原始记录；各原因计数（可重叠）：{dict(reason_counts)}。",'',
        '## 文件','',
        '- supplement_train_messages.jsonl：新增题的SFT输入，每行只有messages，user/assistant两条。',
        '- supplement_records.jsonl：保留题号、来源ID、官方原解答、整理后解答及质量提示。',
        '- supplement_prepared.jsonl：原预处理函数转换的prompt/completion，结尾为end_working_out、SOLUTION及EOS文本。',
        '- questions_and_solutions.txt：完整题面与解答，方便人工复核。',
        '- excluded_questions.jsonl：所有排除题及原因。',
        '- quality_review_queue.jsonl：多boxed、图形标记、短解答或压缩论证用语的复核提示；这些提示不是错误判定。',
        '- manifest.json：来源、去重策略、训练暴露核对、统计及哈希。','',
        '## 使用范围','',
        '仅准备数据，没有启动训练或改变已有LoRA v3权重。新增题均附官方worked solution，未为全部新题独立认证每一步推导。',
        f"本版本含{manifest['manually_reviewed_selected']}条人工数学审查记录、{manifest['explicit_solution_rewrites']}条显式解答修订；其余{manifest['mathematically_unreviewed_selected']}条没有获得本轮数学审查。多boxed值不同的记录须经显式审核，不能机械取最后一个。",
        '保留完整解答，没有因长度、答案类型或质量提示丢题，也没有截断示范结尾。正式训练前仍需核对实际tokenizer长度、上下文预算、prompt masking以及最终EOS参与监督。',
        '未做语义去重，不能排除Base预训练已见过这些题。此数据只能作为训练补充，不再用于独立benchmark。','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('selected','selected_level_counts','selected_subject_counts','excluded','exclusion_reason_counts','quality_review_flagged')},ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
