"""Export reviewed supplement, quarantine unresolved examples, validate provenance."""
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'datasets/math/sft_v3_supplement_unseen_grpo1536_v2'
OUT=ROOT/'datasets/math/sft_v3_supplement_unseen_grpo1536_v3_review'
sys.path.insert(0,str(ROOT/'datasets/math/test_l2_l3_remaining_v1'))
from common import END,SOL_START,SOL_END
from prepare_math_sft_supplement import canonical,problem
from train_reviewed_sft import convert_messages

def read(p):return [json.loads(x) for x in p.read_text('utf8').splitlines() if x.strip()]
def write(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    source=read(SRC/'supplement_records.jsonl')
    draft=read(OUT/'supplement_records_review_draft.jsonl')
    progress=json.loads((OUT/'review_progress.json').read_text('utf8'))
    assert sha(SRC/'supplement_records.jsonl')==progress['source_sha256']
    assert progress['remaining']==0 and not progress['unapplied_repairs']
    checks=json.loads((OUT/'supporting_math_checks.json').read_text('utf8'))
    assert checks['status']=='passed' and all(x['passed'] for x in checks['checks'])
    reviews=read(SRC/'remaining_reasoning_audit/reviews.jsonl')
    previous=read(SRC/'manual_audit/manual_reviews.jsonl')
    assert len(source)==len(draft)==1390 and len(reviews)==1310 and len(previous)==80
    assert {r['row_id'] for r in reviews}.isdisjoint(r['row_id'] for r in previous)
    assert {r['row_id'] for r in reviews+previous}=={r['row_id'] for r in source}
    accepted=[];quarantine=[];prepared=[];all_reviews=[]
    for before,after in zip(source,draft):
        assert before['row_id']==after['row_id'] and before['problem']==after['problem']
        assert before['reference_solution']==after['reference_solution']
        note=after.get('remaining_manual_review')
        if note:
            assert note['source_solution_sha256']==hashlib.sha256(before['solution'].encode()).hexdigest()
        all_reviews.append(dict(supplement_index=after['supplement_index'],row_id=after['row_id'],
            disposition=('accepted' if after['mathematical_steps_independently_reviewed'] else 'quarantined'),
            review=note or after['manual_review'],repaired_in_this_round='remaining_revision' in after))
        if not after['mathematical_steps_independently_reviewed']:
            assert note['verdict']=='uncertain'
            quarantine.append(after);continue
        assert after['messages']==[dict(role='user',content=after['problem']),dict(role='assistant',content=after['solution'])]
        assert after['solution'].count('\\boxed')==1
        converted=convert_messages(dict(messages=after['messages']))
        assert converted['answer']==after['answer'],after['supplement_index']
        reasoning=after['solution'].split('\nFinal answer:',1)[0].strip()
        assert converted['completion']==reasoning+'\n'+END+SOL_START+after['answer']+SOL_END+'<|endoftext|>'
        assert converted['completion'].count('<|endoftext|>')==1
        accepted.append(after);prepared.append(converted)
    assert len(accepted)==1379 and len(quarantine)==11
    old_manifest=json.loads((SRC/'manifest.json').read_text('utf8'))
    consumed=ROOT/'deliverables/SFTv3_GRPO_1536_training_handoff/data/grpo/actually_consumed/all_3072_questions.jsonl'
    old_sft=ROOT/'sft_quality_audit_v2/revision_20261011_user15/train_merged_505.jsonl'
    reference_paths=[consumed,old_sft,SRC/'official_test_exclusion_reference.jsonl']
    reference_paths.extend(ROOT/p for p in old_manifest['input_sha256'] if p.startswith(('datasets/grpo_candidates_', 'sft_quality_audit_v2/eval_questions_', 'closure_grpo_experiment/inputs/', 'sft_closure_data/validation')) and p.endswith(('validation.jsonl','questions.jsonl')))
    reference_paths=list(dict.fromkeys(reference_paths))
    keys={canonical(r['problem']) for r in accepted}
    assert len(keys)==len(accepted)
    overlap_report=[]
    for path in reference_paths:
        ref=read(path)
        overlaps=keys & {canonical(problem(r)) for r in ref}
        assert not overlaps,path
        overlap_report.append(dict(path=str(path.relative_to(ROOT)),rows=len(ref),overlaps=0,sha256=sha(path)))
    assert not keys & {canonical(r['problem']) for r in quarantine}
    # Persist only validated, accepted rows under filenames used for future training.
    write(OUT/'supplement_records.jsonl',accepted)
    write(OUT/'supplement_prepared.jsonl',prepared)
    write(OUT/'supplement_train_messages.jsonl',[dict(messages=r['messages']) for r in accepted])
    write(OUT/'quarantined_records.jsonl',quarantine)
    write(OUT/'reviewed_records_all.jsonl',draft)
    write(OUT/'all_review_dispositions.jsonl',all_reviews)
    save(OUT/'structure_and_leakage_audit.json',dict(status='passed',accepted=len(accepted),quarantined=len(quarantine),
        reviewed_total=len(all_reviews),original_references_preserved=True,one_final_answer_and_text_eos=True,
        existing_converter_passed=len(accepted),duplicate_policy=old_manifest['duplicate_policy'],
        no_semantic_deduplication=True,references=overlap_report))
    (OUT/'questions_and_solutions.txt').write_text('\n\n'.join(
        f"Supplement {r['supplement_index']} | {r['row_id']} | {r['level']} | {r['type']}\n{r['problem']}\n\n{r['solution']}" for r in accepted)+'\n',encoding='utf8')
    # Preserve the prior and current review evidence alongside the final data.
    for name in ('manual_audit','remaining_reasoning_audit'):
        shutil.copytree(SRC/name,OUT/name,dirs_exist_ok=True)
    code=OUT/'preparation_code';code.mkdir(exist_ok=True)
    for name in ('review_remaining_math_reasoning.py','repair_remaining_math_reasoning.py','remaining_reasoning_repairs_extra.py','verify_remaining_math_repairs.py','finalize_remaining_math_review.py','train_reviewed_sft.py','math_verification.py','audited_answer_scoring.py'):
        shutil.copy2(ROOT/'vllm_speed_tools'/name,code/name)
    shutil.copy2(ROOT/'datasets/math/test_l2_l3_remaining_v1/common.py',code/'common.py')
    repairs=read(OUT/'reasoning_repairs.jsonl')
    answer_changes=[dict(supplement_index=r['supplement_index'],before=r['previous_answer'],after=r['answer']) for r in repairs if r['previous_answer']!=r['answer']]
    manifest=dict(status='all_remaining_reasoning_reviewed_repaired_and_exported_no_training',
        source_directory=str(SRC),source_sha256=sha(SRC/'supplement_records.jsonl'),
        source_repository=old_manifest['source_repository'],source_revision=old_manifest['source_revision'],
        source_split='train',reviewed_original_records=1390,previously_reviewed=80,new_reviewed=1310,
        new_verdicts=progress['verdicts'],new_repairs=len(repairs),previous_solution_rewrites=30,
        accepted=len(accepted),quarantined=len(quarantine),quarantined_indices=[r['supplement_index'] for r in quarantine],
        answer_changes_this_round=answer_changes,level_counts=dict(Counter(r['level'] for r in accepted)),
        subject_counts=dict(Counter(r['type'] for r in accepted)),
        prior_data_and_models_unchanged=True,training_started=False,
        validation_policy='No new split. Existing heldout validation and all 5000 official MATH test questions remain excluded.',
        review_scope='Per-record model-assisted mathematical reasoning review, with explicit derivations, domain checks, missing-case checks, and selected exact computational checks. Not formal theorem proving or an independent human certification.',
        supplemental_checks=checks['total_checks'],tokenizer_collator_audit='pending; old v2 tokenizer audit is not reused as proof for changed examples',
        output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    save(OUT/'manifest.json',manifest)
    report=f'''# MATH SFT 补充集逐题审查结果

已审完上次剩余的 **1310 条**，合计覆盖原补充集 **1390 条**。本轮通过原解审查 1185 条，修订 114 条，暂隔离 11 条；可用补充集 **1379 条**。没有启动训练。

114 条修订包含错误公式、漏项、错误符号/文字，以及补充定义域、回代、唯一性、极值可达性和完整分类。不能把 114 全部解释为最终答案错误。

两条最终数学答案改动：
- Supplement 737：两图仅一交点的参数，原答案仅 `2` 漏掉线性情形 `a=0`，修为 `0,2`。
- Supplement 1287：由 `sin(2α)=1/2` 求最小夹角，应是 `15°`，原答案 `30°` 错误；补了具体单位向量验证可达。

Supplement 800 仅调整输出格式：按题面 HH:MM:SS 要求改为 `04:51:06`，去掉 p.m.。

修订示例：Supplement 750 给出 x=618 的全部十四项并证明其他整数不可能达到十二项；Supplement 548 改掉不成立的不等式，以两正项 AM-GM 证明全局最小值；Supplement 1224 用有向坐标覆盖垂足在内外的所有情况；Supplement 1326 枚举八边形顶点的五种间隔组合证明最小面积；Supplement 1371、1388 用严格凹性补齐交点计数证明。

11 条存疑记录已单独保存到 `quarantined_records.jsonl`，不会出现在三个可用训练数据文件里：
'''+''.join(f"- Supplement {r['supplement_index']}：{r['remaining_manual_review']['reasoning_check']}\n" for r in quarantine)+'''
这些存疑不等于最终答案全部错误。例如年龄题 896 的 26 岁满足条件，但原解仅试少数立方数，尚缺全范围唯一性证明；坐标题 1302、1376 涉及径向坐标是否允许负值的约定。原题条件没有被偷偷修改。

文件：
- `supplement_train_messages.jsonl`：1379 条 user/assistant 消息，回答为推导加唯一最终 boxed 答案。
- `supplement_prepared.jsonl`：使用项目原转换函数生成，推导后为 `<end_working_out><SOLUTION>答案</SOLUTION><|endoftext|>`。
- `supplement_records.jsonl`：可用记录，保留原题、原始官方解答、修订解答与审查依据；原 supplement_index 保持不变，隔离造成编号空缺属于正常。
- `reviewed_records_all.jsonl`：全部1390条审查后快照，包含11条隔离记录，不能直接当作训练入口。
- `reasoning_repairs.jsonl`：本轮114条修改前后对照。此前30条修改保存在 `manual_audit/reviewed_patches.json`。
- `all_review_dispositions.jsonl`：全部1390条的审查处置；`remaining_reasoning_audit/reviews.jsonl` 为本轮逐题记录。
- `supporting_math_checks.json`：47项符号、精确有理数或有限穷举辅助检查通过；`structure_and_leakage_audit.json` 为格式及精确去重复核。

与实际 GRPO 0–1536 已用题、原 SFT505、现有验证题及全部5000道官方 MATH test 再核对，规范化精确交集为0。仍未做语义去重。没有新划分训练/验证集，原留出集保持排除。

“逐条审查”表示逐条阅读并核算推导、给出具体审查记录；它是模型辅助数学审查，不是形式化证明器或另一位数学审稿人的独立认证。通过格式和答案检查也不自动证明每一步正确。此前 v2 文件、模型权重和已完成评测均保留，新文件只是数据修订，不代表模型效果已改善。
'''
    (OUT/'README.md').write_text(report,encoding='utf8')
    print(json.dumps(dict(accepted=len(accepted),quarantined=len(quarantine),repairs=len(repairs),answer_changes=answer_changes),ensure_ascii=False))

if __name__=='__main__':main()
