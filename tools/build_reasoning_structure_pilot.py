"""Export direct, evidence-linked structural diagnoses and a small GRPO bridge set.

These decisions were written by the primary assistant after reading all 22
problems, reference solutions, four extracted finals, and the full sample-0
trace. They do not claim a complete audit of all 88 reasoning traces.
"""
import collections
import hashlib
import json
from pathlib import Path

from prepare_math_self600 import canonical

ROOT = Path(__file__).resolve().parents[1] / 'datasets/math/base_self600_20261015'
OUT = ROOT / 'reasoning_structure_audit'

FAMILIES = {
    'objects_and_constraints': '对象与约束建模：可区分性、位置限制、中位数和奇偶条件',
    'exhaustive_cases': '分类计数：互斥且穷尽的分支、严格不等号边界、分类求和',
    'diagram_grounding': '图形关系识别与分解：目标区域、边角归属、重叠和面积组合',
    'geometry_dependency_chain': '几何多步依赖：先求中间量，再用相似、勾股或分点关系',
    'global_constraint_extremum': '整体约束与最值：不变量、下界及取等可达性',
    'equivalence_and_complete_solutions': '等价变形与解集完整性：合法约分、定义域、验根和找全根',
    'structural_transform_and_execution': '结构化换表示与可靠执行：旋转/复数/矩阵变换及系数检查',
}

# index: (family, observed failure in the fully inspected sample 0, repair target)
DECISIONS = {
    0: ('objects_and_constraints', '把不可区分的盒子当成带编号变量；并错误地用减1处理不可区分性。', '先明确等价配置；按非增三元组穷举整数分拆，而非直接套星与杠。'),
    1: ('exhaustive_cases', '列出50种组合后把满足乘积<30的组合数误数成19。', '固定第一个转盘数，分别计10、10、9、7、5；相加41，再除50；检查严格边界。'),
    2: ('diagram_grounding', '凭28条边便假定内部有28个小矩形，面积错误乘以28。', '按水平条带宽1、3、5、7、5、3、1分解，共25格；边长2，面积100。'),
    3: ('diagram_grounding', '把目标多边形面积写成两个等面积正方形之差；还误认HG为半条边。', '先识别目标为两正方形的并集，减去重叠区域8，得到16+16-8。'),
    4: ('geometry_dependency_chain', '引入不支持的辅助点和相似关系，并把原题AB=24改成64。', '保留题设量：sin A求BD=16，sin C求BC=48，再勾股求DC。'),
    5: ('geometry_dependency_chain', '已经算出等边三角形高，却又无依据把等边三角形DBC称为直角三角形。', '识别AD为共边BC两侧的两条高之和；不能在等边三角形上误用直角勾股。'),
    6: ('diagram_grounding', '第一步把阴影四边形认成整个正六边形。', '先确定阴影是矩形，再求其边长10与10√3；识别目标先于代面积公式。'),
    7: ('global_constraint_extremum', '正确算出焦距c=3后，直接声称内切圆半径等于焦距。', '半径由焦点到椭圆边界的最短距离约束，证明最短距离a-c=2可达。'),
    8: ('equivalence_and_complete_solutions', '有理根候选漏分数；已知x=2为根，综合除法却给出非零余数，仍继续使用错误商。', '逐步保等价并验证除法余数；可用倒数对称降次，但不强制唯一解法；最终找全四根。'),
    9: ('global_constraint_extremum', '局部最大化单个分母后，选同一整行的8格，违反每行每列各1格。', '先编码排列约束，利用分母和恒为64，由倒数和下界1并给出可行取等排列。'),
    10: ('equivalence_and_complete_solutions', '把(x²-6x+8)/(x-3)错误约成x-4；验出错误候选不成立后便宣称原方程无解。', '维护x≠2,3,4，合法清分母得x²-6x+6=0；验根并取较小根。'),
    11: ('objects_and_constraints', '对百位、十位、个位的限制未建模，补集数量甚至超过总样本空间500。', '先锁定百位5到9，十个位只能34或43，共5×2；用总数界检查结果。'),
    12: ('diagram_grounding', '将题目明确共线的PQ和QR说成垂直，并虚构90°。', '先记录共线与外角136°，再用外角定理136=64+x。'),
    13: ('diagram_grounding', '引用平行线角关系时选错了对应的目标角，把补角当成同角。', '定位目标为30°的补角，再计算180-30；画图编码解读与几何推理需分开诊断。'),
    14: ('diagram_grounding', '先认错12√2所标的斜边，随后把45°当成边长45。', '将标签绑定到边/角对象，明确45-45-90比例，再由斜边除√2。'),
    16: ('objects_and_constraints', '擅自假定奇数个奇整数，却把偶数138当作集合中的中间项；把数值差7当成项数。', '承认偶数项的中位数可为偶数；等差数列首末对称，首项=2×138-145=131。'),
    17: ('diagram_grounding', '把左上矩形横边5认成6，后续又添加无依据的重叠矩形。', '从边标签推缺失高度，作互不重叠的矩形分解：5×6+3×4+4×5。'),
    18: ('diagram_grounding', '把B点外角108°直接代作三角形ABD的内角。', '先取补角72°，再用三角形内角和180-72-26=82。'),
    20: ('geometry_dependency_chain', '把边上的角平分线分点比7/5直接套给内心在BE上的分点比；还混同内心与重心。', '用角平分线定理后，继续通过重心坐标/质量点求交点分点比，不跳过第二层依赖。'),
    21: ('structural_transform_and_execution', '已正确求出D⁶=-64I和P逆，最后矩阵乘法又多引入一次1/2，变成-32I。', '已有对角化/复数方法；补系数保持与A⁶=-64PP逆的检查，不把它诊断成完全不懂矩阵幂。'),
}

# Manually chosen from inspected problems/reference solutions, not from test scores.
BRIDGES = [
    ('math:prealgebra/train-00000-of-00001.parquet:184', 'objects_and_constraints', [16], '将中位数作为有序位置，并构造可达上界5。'),
    ('math:prealgebra/train-00000-of-00001.parquet:204', 'objects_and_constraints', [16], '中位数确定中间项，均值确定总和，再反求首项。'),
    ('math:counting_and_probability/train-00000-of-00001.parquet:165', 'exhaustive_cases', [1], '四种同色事件互斥穷尽，概率平方后求和。'),
    ('math:counting_and_probability/train-00000-of-00001.parquet:645', 'exhaustive_cases', [1], '两轮状态路径分类，分别相乘再相加；不把边缘概率直接混算。'),
    ('math:geometry/train-00000-of-00001.parquet:132', 'diagram_grounding', [2,3,6,17], '由图形约束求大圆半径，再用大面积减两个不重叠的小圆。'),
    ('math:geometry/train-00000-of-00001.parquet:14', 'diagram_grounding', [3,6], '将阴影:内圆=3:1转成总圆:内圆=4:1，再把面积比转换为直径比。'),
    ('math:geometry/train-00000-of-00001.parquet:88', 'geometry_dependency_chain', [4,5], '先用勾股补短边20，再使用相似比3得到60。'),
    ('math:geometry/train-00000-of-00001.parquet:735', 'geometry_dependency_chain', [5], '先比较平方关系，再识别勾股逆定理，判断最大角90°。'),
    ('math:intermediate_algebra/train-00000-of-00001.parquet:1075', 'global_constraint_extremum', [9], '六项乘积不变，用AM-GM给下界，再构造a=b=c取等。'),
    ('math:intermediate_algebra/train-00000-of-00001.parquet:714', 'global_constraint_extremum', [9], '固定乘积后AM-GM求3/4，验证三个量同时等于1/4可实现。'),
    ('math:intermediate_algebra/train-00000-of-00001.parquet:781', 'equivalence_and_complete_solutions', [7,8,10], '代入消元保等价，将相切转化为重根条件，判别式为0。'),
    ('math:intermediate_algebra/train-00000-of-00001.parquet:105', 'equivalence_and_complete_solutions', [10], '因式分解约分前保留定义域；清分母后排除原分母为0的候选根。'),
    ('math:precalculus/train-00000-of-00001.parquet:413', 'structural_transform_and_execution', [21], '将复数幂转成模的幂和角度倍增，分别保持64及120°；参考解首行有等号笔误，未用作SFT目标。'),
]


def read(p):
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def write(p, rows):
    p.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def main():
    packets = read(OUT / 'strict_zero_review_packets.jsonl')
    source = {r['row_id']: r for r in read(ROOT / 'remote_results/grpo_reserved.jsonl')}
    scores = {r['row_id']: r for r in read(ROOT / 'grpo_supplement_v3/v3_scores.jsonl')}
    old_ids = {r['row_id'] for r in read(ROOT / 'remote_results/grpo_selected_direct367_v1/train.jsonl')}
    expanded_ids = {r['row_id'] for r in read(ROOT / 'grpo_supplement_v3/train_expanded.jsonl')}
    reviews = []
    for p in packets:
        i = p['audit_index']
        if i in DECISIONS:
            family, error, target = DECISIONS[i]
            reviews.append(dict(audit_index=i, row_id=p['row_id'], structure=family,
                disposition='remedial_instruction_candidate', observed_sample0_error=error,
                repair_target=target, reference_final_answer_review='consistent_with_problem',
                review_scope='Full problem, reference solution, four extracted finals, full sample 0 only',
                capability_inference='Observed failure, not proof this model cannot perform this structure',
                sft_training_ready=False))
        elif i == 15:
            reviews.append(dict(audit_index=i, row_id=p['row_id'], structure=None,
                disposition='hold_scorer_text_answer_mismatch',
                finding='样本0、1、2的最终排序与题目和参考一致；样本0完整通分推导正确。自动incorrect不能解释为推理失败。',
                reference_final_answer_review='consistent_with_problem', sft_training_ready=False))
        elif i == 19:
            reviews.append(dict(audit_index=i, row_id=p['row_id'], structure=None,
                disposition='hold_source_gold_incomplete', original_gold=p['answer'],
                reviewed_answer='12, 18', finding='题目要两个最小正整数；原gold仅18，参考解明确12与18；样本2、3最终值正确。原始gold不覆盖。',
                verification='k²+36 divisible by 180; among positive integers <=18 precisely 12 and18 satisfy it',
                reference_final_answer_review='gold_omits_first_requested_solution', sft_training_ready=False))
        else:
            raise AssertionError(i)
    assert [k for k in range(1,19) if (k*k+36)%180 == 0] == [12,18]
    seeds = [dict(**source[r['row_id']], structure_audit=r) for r in reviews if r['structure']]
    bridge_rows, bridge_provenance = [], []
    for row_id, family, evidence, explanation in BRIDGES:
        row, score = source[row_id], scores[row_id]
        assert score['eligible'] and row_id not in old_ids
        bridge_rows.append(dict(row_id=row_id, problem=row['problem'], answer=row['answer'],
            prompt=row['prompt'], source='math', split='train', level=row['level'], type=row['type'],
            difficulty=score['difficulty'], screen_correct_count=score['k'],
            screen_grading_schema='math-verify-final-answer-v3', reasoning_structure=family))
        bridge_provenance.append(dict(row_id=row_id, structure=family, linked_failure_indices=evidence,
            selection_rationale=explanation, already_in_expanded_1200=row_id in expanded_ids,
            reference_review_scope='Problem requirement and final-answer derivation checked; cached reasoning traces not certified'))
    assert len(seeds) == 20 and len(bridge_rows) == 13
    assert len({r['row_id'] for r in bridge_rows}) == len(bridge_rows)
    assert not {r['row_id'] for r in seeds} & {r['row_id'] for r in bridge_rows}
    keys = {canonical(r['problem']) for r in seeds+bridge_rows}
    assert len(keys) == len(seeds)+len(bridge_rows)
    for name in ['reviewed_direct/train.jsonl','development.jsonl','official_test/questions.jsonl']:
        assert not keys & {canonical(r['problem']) for r in read(ROOT/'remote_results'/name)}
    for r in bridge_rows:
        assert [m['role'] for m in r['prompt']] == ['system','user']
        assert r['prompt'][1]['content'] == r['problem']
        assert 'reference_solution' not in r and 'completion' not in r
    write(OUT / 'direct_reviews.jsonl', reviews)
    write(OUT / 'remedial_sft_candidates.jsonl', seeds)
    write(OUT / 'grpo_bridge_train.jsonl', bridge_rows)
    write(OUT / 'bridge_provenance.jsonl', bridge_provenance)
    summary = dict(status='pilot_data_prepared', automatic_four_incorrect_reviewed_questions=22,
        full_sample0_traces_read=22, other_samples_scope='Extracted final regions only; not full trace audits',
        remedial_candidates=20, scorer_issue_holds=1, source_gold_issue_holds=1,
        seed_structure_counts=dict(collections.Counter(r['structure_audit']['structure'] for r in seeds)),
        bridge_questions=13, bridge_difficulty=dict(collections.Counter(r['difficulty'] for r in bridge_rows)),
        bridge_structure_counts=dict(collections.Counter(r['reasoning_structure'] for r in bridge_rows)),
        bridge_already_in_expanded_1200=sum(r['already_in_expanded_1200'] for r in bridge_provenance),
        newly_trained_question_count=0, bridge_repeats_original_699=False,
        exact_normalized_sft_dev_test_overlap=False,
        difficulty_basis='SFT step64 frozen G4, not the current GRPO model',
        all_2138_structures_annotated=False, capability_gap_proven=False,
        server_started=False, training_started=False,
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            OUT/'direct_reviews.jsonl',OUT/'remedial_sft_candidates.jsonl',OUT/'grpo_bridge_train.jsonl',OUT/'bridge_provenance.jsonl']})
    (OUT/'pilot_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
