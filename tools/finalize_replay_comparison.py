"""Validate archived answer lineage and summarize the common v4 metric."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    base = Path(__file__).resolve().parents[1] / 'datasets/math/base_self600_20261015'
    root = base / 'grpo_replay549_extract_v4_20261016_batch12'
    remote = root / 'remote_results'
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    retrieval = json.loads((remote / 'final_retrieval_verification.json').read_text(encoding='utf-8'))
    checksums = json.loads((remote / 'all_results_manifest.json').read_text(encoding='utf-8'))
    inputs = json.loads((root / 'baseline_manifest.json').read_text(encoding='utf-8'))
    history = json.loads((root / 'comparison_regraded_v4/summary.json').read_text(encoding='utf-8'))
    assert retrieval['status'] == 'verified' and retrieval['files_verified'] == 2631
    assert history['grader_runtime']['sha256'] == plan['grader_sha256']
    assert history['grader_runtime']['preflight_passed']
    assert history['test_sha256'] == plan['test_sha256']
    assert sha(remote / 'tools/math_verification.py') == plan['grader_sha256']
    assert sha(base.parents[2] / 'tools/math_verification.py') == plan['grader_sha256']
    assert sha(remote / 'data/official_test.jsonl') == plan['test_sha256']
    questions = {r['row_id']: r for r in read(remote / 'data/official_test.jsonl')}
    assert len(questions) == 2024
    current_manifest = json.loads((remote / 'test_after_grpo/manifest.json').read_text(encoding='utf-8'))
    current_path = remote / 'test_after_grpo/grpo_replay549/answers.jsonl'
    assert sha(current_path) == checksums[current_path.relative_to(remote).as_posix()]
    assert current_manifest['grader_sha256'] == plan['grader_sha256']
    assert sha(remote / 'training/adapter/adapter_model.safetensors') == retrieval['adapter_sha256']
    assert current_manifest['adapter_sha256']['grpo_replay549'] == retrieval['adapter_sha256']
    common_keys = ['model_weights', 'data_sha256', 'prompt_ids_sha256', 'sampling', 'seed_policy', 'template']
    prior_manifests = [base / 'remote_results/test_direct367_v1/manifest.json',
                       base / 'remote_results/test_after_grpo_direct367_v1/manifest.json']
    for path in prior_manifests:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        for key in common_keys:
            assert manifest[key] == current_manifest[key], (str(path), key)
    old_manifest = json.loads(prior_manifests[1].read_text(encoding='utf-8'))
    assert old_manifest['adapter_sha256']['grpo'] == plan['adapter_sha256']
    arms, evidence = {}, {}
    for arm, meta in inputs.items():
        source = base / 'remote_results' / meta['original_source']
        frozen = root / meta['file']
        graded = root / f'comparison_regraded_v4/{arm}_answers.jsonl'
        assert sha(source) == meta['original_file_sha256']
        assert sha(frozen) == meta['sha256']
        originals = {r['row_id']: r for r in read(source)}
        frozen_rows = {r['row_id']: r for r in read(frozen)}
        rows = read(graded)
        assert len(rows) == len({r['row_id'] for r in rows}) == 2024
        assert set(originals) == set(frozen_rows) == {r['row_id'] for r in rows} == set(questions)
        for row in rows:
            rid = row['row_id']
            for key, value in frozen_rows[rid].items():
                assert originals[rid][key] == value, (arm, rid, key)
                if key not in ['correct', 'verdict']:
                    assert row[key] == value, (arm, rid, key)
            assert row['gold'] == questions[rid]['answer']
            assert row['original_correct'] == originals[rid]['correct']
            assert row['original_verdict'] == originals[rid]['verdict']
        assert sum(r['correct'] for r in rows) == history['table'][arm]['correct']
        arms[arm] = rows
        evidence[arm] = dict(original_sha256=sha(source), input_sha256=sha(frozen), regraded_sha256=sha(graded))
    arms['grpo2'] = read(current_path)
    assert len(arms['grpo2']) == len({r['row_id'] for r in arms['grpo2']}) == 2024
    assert {r['row_id'] for r in arms['grpo2']} == set(questions)
    table = {}
    for arm, rows in arms.items():
        for row in rows:
            assert row['gold'] == questions[row['row_id']]['answer']
            assert row['verdict']['schema'] == 'math-verify-final-answer-v4'
            assert row['correct'] == row['verdict']['correct']
        correct = sum(r['correct'] for r in rows)
        table[arm] = dict(questions=2024, correct=correct, accuracy=correct / 2024,
            statuses=dict(Counter(r['verdict']['status'] for r in rows)),
            timeout_count=sum('timeout' in str(r['verdict'].get('error') or '').lower() for r in rows),
            by_level={level:dict(questions=sum(r['level'] == level for r in rows),
                correct=sum(r['correct'] for r in rows if r['level'] == level)) for level in ['Level 2', 'Level 3']})
        if arm in history['table']:
            table[arm]['original_v2_correct'] = history['table'][arm]['original_correct']
            table[arm]['grader_only_net_questions'] = correct - table[arm]['original_v2_correct']
    assert table['grpo2']['correct'] == retrieval['test']['correct'] == 1724
    old = {r['row_id']: r for r in arms['grpo1']}
    new = {r['row_id']: r for r in arms['grpo2']}
    gains = [rid for rid in old if not old[rid]['correct'] and new[rid]['correct']]
    losses = [rid for rid in old if old[rid]['correct'] and not new[rid]['correct']]
    net = len(gains) - len(losses)
    assert net == table['grpo2']['correct'] - table['grpo1']['correct']
    caveats = ['历史回答只重新评分，未重新生成；原v2报告保留。所有模型分母固定2024，未删除无法核验的回答。',
        '统一评分器代码与合同依赖；历史复评在Windows，本轮原评分在Linux。未记录TimeoutError；Base有1份AttributeError仍记为未通过，不声称该答案数学错误。',
        '未成功提取、解析或核验的回答计为未通过；格式合格率与最终答案正确率分开，测试不调用API。',
        '该结果是同一测试集上的最终答案得分比较，不代表推导逐步正确，也不是控制所有训练变量的因果实验。']
    result = dict(status='complete',checked_at_utc=datetime.now(timezone.utc).isoformat(),
        schema='math-verify-final-answer-v4',grader_sha256=plan['grader_sha256'],
        grader_dependencies=history['grader_runtime']['dependencies'],
        test_sha256=plan['test_sha256'],table=table,paired=dict(gained=len(gains),regressed=len(losses),
            net_questions=net,percentage_point_change=100*net/2024,gained_row_ids=gains,regressed_row_ids=losses),
        source_hashes=evidence,current_answers_sha256=sha(current_path),
        provenance_checks_passed=common_keys+['unique_ids','gold','verbatim_completions','original_scores','model_lineage'],
        caveats=caveats,retrieval=dict(files_verified=2631,archive_sha256=retrieval['archive_sha256'],
            adapter_sha256=retrieval['adapter_sha256']),server_required=False)
    save(root / 'final_comparison_v4.json', result)
    labels = dict(base='Base',sft='SFT',grpo1='上一轮 GRPO',grpo2='本轮 GRPO（549题）')
    lines = ['# 统一v4评分的MATH Level 2–3结果', '',
        f'本轮答对1724/2024（85.18%）；上一轮统一复评后答对1628/2024（80.43%）。净增加{net}题，提升{100*net/2024:.2f}个百分点。', '',
        '| 模型 | 答对/总题数 | 正确率 | Level 2 | Level 3 |', '|---|---:|---:|---:|---:|']
    for arm, row in table.items():
        lv = row['by_level']
        lines.append(f"| {labels[arm]} | {row['correct']}/2024 | {row['accuracy']:.2%} | {lv['Level 2']['correct']}/{lv['Level 2']['questions']} | {lv['Level 3']['correct']}/{lv['Level 3']['questions']} |")
    lines += ['', f'逐题比较：上一轮未通过、本轮通过{len(gains)}题；上一轮通过、本轮未通过{len(losses)}题，净增{net}题。', '',
        '统一标准：同一官方2024题、同一题目及参考答案、同一提示词、贪心n=1/temperature=0/max_tokens=3894/seed=3407；评分器v4固定SHA，Math-Verify 0.8.0与ANTLR 4.13.2。格式单独统计，测试不使用DeepSeek或其他API。', '',
        '原始v2分数保留：Base1397（69.02%）、SFT1562（77.17%）、上一轮GRPO1618（79.94%）。评分器变化本身使三者净增20、8、10题；所以不能把旧79.94%到本轮85.18%的全部差值算作训练收益。', '',
        '训练549次更新，用时约2小时12分；549题/2196答，481组参与、68组屏蔽、413组非零优势。', '',
        '模型、数据、配置、代码、训练日志、API提取记录、逐题测试共2631文件已取回且全部SHA256核验成功。模型目录：remote_results/training/adapter/；回执：remote_results/final_retrieval_verification.json；逐题变化及来源核验：final_comparison_v4.json。服务器不再需要运行。', '',
        '核验范围与限制：'] + ['- ' + s for s in caveats]
    (root / 'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(root / 'comparison_verification.json',dict(status='verified',questions_per_model=2024,
        historical_answers=6072,final_comparison_sha256=sha(root / 'final_comparison_v4.json'),
        report_sha256=sha(root / 'RESULTS.md'),original_results_preserved=True))
    print(json.dumps(dict(table=table,paired={k:v for k,v in result['paired'].items() if not k.endswith('_ids')}),ensure_ascii=False))


if __name__ == '__main__':
    main()
