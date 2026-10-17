import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'datasets/grpo_candidates_gsm8k_math_l2_l3/training_audit_1536'
R = ROOT / 'datasets/gsm8k/base_sftv3_test/results'
OUT = R / 'math1536_error_classification'
OUT.mkdir(exist_ok=True)

def read(path):
    return [json.loads(x) for x in path.read_bytes().splitlines() if x.strip()]

pool = {x['row_id']: x for x in read(T.parent / 'expanded_math_v3/train_candidates.jsonl')}
groups = read(T / 'per_question_training_correctness.jsonl')
agg, stages, effective = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
for g in groups:
    p = pool[g['row_id']]
    ds = 'GSM8K' if g['source'] == 'gsm8k' else 'MATH ' + p['level']
    g['dataset_group'] = ds
    cat = '0/4' if g['correct'] == 0 else '4/4' if g['correct'] == 4 else '1-3/4'
    for target in [agg[ds], stages[(g['stage'], ds)]]:
        target[cat] += 1
        target['total'] += 1
        target['correct_answers'] += g['correct']
        target['dropped'] += int(g['dropped'])
    if not g['dropped']:
        effective[ds][cat] += 1
        effective[ds]['total'] += 1

stats = {'all_sampled': dict(agg), 'effective_not_dropped': dict(effective),
         'by_stage': [{'stage': s, 'dataset': d, **v} for (s, d), v in sorted(stages.items(), key=lambda x: (int(x[0][0].split('–')[0]), x[0][1]))]}
(T / 'correctness_groups_by_dataset.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
(T / 'per_question_correctness_with_level.jsonl').write_text(''.join(json.dumps(g, ensure_ascii=False)+'\n' for g in groups), encoding='utf-8')

def table(items):
    rows = ['| 数据 | 题数 | 全错 0/4 | 混合 1–3/4 | 全对 4/4 | 丢弃题组 |', '|---|---:|---:|---:|---:|---:|']
    for ds, c in items:
        rows.append(f"| {ds} | {c['total']} | {c['0/4']} ({c['0/4']/c['total']:.2%}) | {c['1-3/4']} ({c['1-3/4']/c['total']:.2%}) | {c['4/4']} ({c['4/4']/c['total']:.2%}) | {c['dropped']} |")
    return '\n'.join(rows)

text = '# GRPO 0–1536 步：训练正确性组\n\n口径：每道训练题当次采样 4 个回答，以日志中的正确性奖励（0/1）计数；格式奖励不参与分组。不是独立测试集准确率。原评分器存在已确认的漏判，未在本统计中回改奖励。\n\n'
text += table(sorted(agg.items()))
text += '\n\n丢弃题组包含在上表中；因截断等原因被整体屏蔽，不等于实际参与策略更新。有效题组计数另存 JSON。\n\n'
for stage in sorted({g['stage'] for g in groups}, key=lambda s: int(s.split('–')[0])):
    text += '## 更新阶段 ' + stage + '\n\n' + table(sorted((d,c) for (s,d),c in stages.items() if s == stage)) + '\n\n'
text += '每道题仅出现于一个采样组，共 3,072 道不同题；各阶段题目不同，不能把阶段分布当作相同题目的正确率变化。每批生成的回答被复用于后续 4 次更新。\n'
(T / 'CORRECTNESS_GROUPS_BY_DATASET.md').write_text(text, encoding='utf-8')

# Primary categories are mutually exclusive; notes retain secondary causes.
notes = [
('循环/截断','反复计算剩余面积，没有按剩余面积不超过 1 的条件停止；参考答案 8。'),
('建模/题意','需要的是 1600−1444=156，却改成 1764−1444，求错目标。'),
('代数变形','联立方程时替换关系错误，错误地判断无解；参考答案 2。'),
('代数变形','二次式错误因式分解；预测 6，参考答案 8。'),
('建模/题意','面积为 3 的格点矩形可为 1×3 或 3×1，错误地判断不存在；共 48 个。'),
('建模/题意','最近点的分界应为 2 和 5，使用了 2 和 4；正确概率 1/2。'),
('建模/题意','样本空间是 4 张纸的排列 4!，误用 20 的四次方；概率 1/24。'),
('算术执行','质因数分解后乘积算错：2^6×3^2×5=2880，不是 5760。'),
('建模/题意','读错骰子的奇偶面数，实际 4 奇 2 偶；正确概率 5/9。'),
('建模/题意','相邻三角形新增边数规律错误；参考答案 51。'),
('算术执行','85×14×83=98770，模型写为 100140。'),
('答案单位','推导得到 100% 概率，最终写 100，缺少百分号；原判错保留，单独列为单位表达问题。'),
('代数变形','阶乘约分将 11! 当成 10!，丢失因子；正确答案 12。'),
('公式/概念/方法','误用帕斯卡行的对称关系；正确答案 6435。'),
('计数/条件/边界','整数区间端点计数错误并伴随算术错误；参考答案 149。'),
('计数/条件/边界','未检查可行性：只有两张非零卡却要求选三张，结果应为 0。'),
('建模/题意','空间距离漏掉一个长度为 8 的方向；应为 sqrt(8²+8²+4²)=12。'),
('建模/题意','嵌套正方形与阴影区域关系理解错误；面积应为 4。'),
('公式/概念/方法','圆周角关系使用错误；正确角度为 39°。'),
('循环/截断','反复尝试高度计算，达到 token 上限，未完成答案；参考答案 80。'),
('代数变形','解周长与面积方程时错误因式分解，所得边长不满足原周长；参考答案 12。'),
('公式/概念/方法','正切边比与斜边比例混淆；KL=8 时 JL=15。'),
('建模/题意','公切线的直角三角形及位置关系错误；参考答案 7。'),
('公式/概念/方法','未识别 8、15、17 直角三角形，外接圆半径应为 17/2。'),
('循环/截断','在重复三角恒等式中耗尽 token，没有完成角平分线计算；参考答案 14。'),
('计数/条件/边界','三角形不等式须严格满足 2<x<6，错误地包含边界；概率 1/2。'),
('建模/题意','引入不成立的角度差关系，并取负角绝对值；参考答案 45°。'),
('代数变形','2sqrt(501)=sqrt(2004)，系数移入根号时没有平方；正确为 (46,2004)。'),
('公式/概念/方法','复合函数奇偶性判断错误；应为偶函数。'),
('代数变形','去分母时多引入因子，得到错误根；正确解为 1、3。'),
('公式/概念/方法','韦达公式及对称式处理错误；正确值 1/3。另有结束标签写成 </start_working_out> 的格式错误。'),
('算术执行','−1+177147−177147+12+1=12，误算为 14。'),
('代数变形','已推出 x²=162y，后续错误替换成 x=162y；正确结果 270。'),
('建模/题意','引入题目未给出的等差关系，误判无解；正确结果 12。'),
('代数变形','嵌套根式方程展开及求根错误，声称检验通过；正确解为 ±sqrt(5)。'),
('公式/概念/方法','根的正负及区间比较错误；应选择 B，预测 E。'),
('循环/截断','望远镜级数已可相消，却持续重复展开直至截断；正确值 1/2。'),
('算术执行','128×2^−3=16，误写 8；还伴随导数处理错误，正确结果 48。'),
('代数变形','将 y(y+2) 错当作 (y+2)²；正确结果 −1。'),
('公式/概念/方法','n/10^n 不是等比数列，误套等比求和公式；正确值 10/81。'),
('公式/概念/方法','使用无效的均值不等式变形，得到不成立的下界；最小值应为 2。'),
('建模/题意','滚动后的周长余数与落边位置映射错误；正确边 CD，预测 DE。'),
('计数/条件/边界','遗漏 k=0 的合法情形，非相交情况计数错误。'),
('公式/概念/方法','x=C/log(1+a) 的单调性方向判断错误；应选 B，预测 E。'),
('循环/截断','未利用函数方程中 y=0 的直接约束，持续枚举直到截断；正确值 4/3。'),
('代数变形','识别了复合函数的禁值条件，但解 f(x)=5/2 时算错；应排除 27/10。'),
('计数/条件/边界','小于 1000 的 45 的倍数应有 22 个，模型计为 20 个；正确概率 1/11。'),
('算术执行','循环小数相加和位数索引错误，将整数部分用于小数位求和；正确结果 30。'),
('循环/截断','重复展开 1/19 的循环节，没有按周期 18 定位；正确数字 2。'),
('代数变形','阶乘约分错误，多保留一个因子 7；正确结果 144。'),
('算术执行','平方数检验错误，例如把 1616 当成 40²；正确平方数为 1681。'),
('公式/概念/方法','周期定位错误，202 除以 20 余 2，应再等 18，预测 20。'),
('评分误判','最终答案 4:30 p.m. 与标准一致，时间字符串解析失败导致判错。'),
('计数/条件/边界','2011 项交错和的配对与剩余项处理错误；正确值 −1。'),
('循环/截断','4/15 的循环小数持续输出 6 直至截断，没有完成最终答案；所求数字为 6。'),
('公式/概念/方法','三角方程的周期及角度换算错误。'),
('公式/概念/方法','半角/倍角推导陷入恒等关系，没有求出数值；cos36°=(1+sqrt(5))/4。'),
('公式/概念/方法','极坐标圆锥曲线转换及椭圆半轴公式错误；半轴 5、3，面积 15π。'),
('循环/截断','未识别对数矩阵秩为 1、行列式为 0，反复展开直到截断。'),
('算术执行','−1+3(x−8)/2=(3x−26)/2，误写常数 −25/2；正确参数 (3/2,−13)。'),
('公式/概念/方法','把 ABM 误作在 M 处直角的三角形，余弦应使用余弦定理，结果 5/13。'),
('评分误判','负虚部的复数答案与标准等价；解析后的虚数单位假设不一致导致评分失败。'),
('循环/截断','角度与弧度分支处理错误后持续重复，最终截断；参考答案 900。'),
('公式/概念/方法','变量替换后面积的缩放因子及边界处理错误；正确面积 π²/6。'),
('代数变形','四次式因式分解错误，所得参数还违反正值条件；正确参数 11。'),
]
answers = read(R / 'grpo1536_official_eval/results/math/grpo1536/answers.jsonl')
refs = {x['row_id']: x for x in read(ROOT / 'datasets/math/test_l2_l3_200/questions.jsonl')}
wrong = [x for x in answers if not x['correct']]
assert len(wrong) == len(notes) == 65
reviews = []
for i, (a, (cat, note)) in enumerate(zip(wrong, notes), 1):
    reviews.append({**a, 'review_index': i, 'primary_category': cat, 'review_note': note,
                    'reference_solution': refs[a['row_id']]['solution'], 'subject': refs[a['row_id']]['type']})
assert sum(x['truncated'] for x in reviews) == 9
assert all(x['truncated'] == (x['primary_category']=='循环/截断') for x in reviews)
(OUT / 'reviewed_errors.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in reviews), encoding='utf-8')
counts = Counter(x['primary_category'] for x in reviews)
by_level = {lv: dict(Counter(x['primary_category'] for x in reviews if x['level']==lv)) for lv in ['Level 2','Level 3']}
summary = {'checkpoint':1536,'total':len(answers),'original_correct':sum(x['correct'] for x in answers), 'original_wrong':65,
           'categories':dict(counts),'by_level':by_level,'confirmed_false_negatives':2,
           'unit_omissions':1,'correct_after_two_confirmed_fixes':137,
           'note':'Original scoring is preserved. Two confirmed false negatives raise 135/200 to 137/200; percent omission remains separate.'}
(OUT / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
report = '# GRPO 1536：MATH Level 2/3 测试错误逐题分类\n\n范围：官方 MATH test 固定 200 题（Level 2、3 各 100），每题 1 条确定性回答。原始判对 135/200，判错 65/200。每条错误只分配一个主要类别，次要问题写入说明。人工分类依据模型完整回答和官方参考解答；保留原始评分。\n\n| 主要类别 | 条数 |\n|---|---:|\n'
for cat,n in counts.most_common(): report += f'| {cat} | {n} |\n'
report += '\n两条确定评分漏判分别是时间表达和等价复数。只修正这两条后为 137/200（68.5%）；概率答案缺百分号的 1 条另外列出，未计入修正正确数。9 条循环/截断均达到生成长度上限，不能按推导中出现过正确数值就判为完整答对。\n\n'
report += '## 逐题清单\n\n| 编号 | 题目 ID | 难度 | 类别 | 判错原因 / 复核结论 |\n|---:|---|---|---|---|\n'
for x in reviews:
    short=x['row_id'].replace('math:','').replace('/test-00000-of-00001.parquet', '')
    report += f"| {x['review_index']} | {short} | {x['level']} | {x['primary_category']} | {x['review_note']} |\n"
report += '\n完整题目、模型回答、原评分、标准答案和官方参考解答见同目录 reviewed_errors.jsonl。\n'
(OUT / 'ANALYSIS.md').write_text(report,encoding='utf-8')
print(json.dumps({'training':stats,'math':summary},ensure_ascii=False,indent=2))
