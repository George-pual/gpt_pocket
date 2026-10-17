"""Analyze saved G4 rewards and consumption logs without changing frozen results."""
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'datasets/math/base_self600_20261015'
SOURCE = ROOT / 'remote_results/grpo_direct367_v1'
OUT = ROOT / 'grpo_log_analysis'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def mean(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return statistics.mean(values) if values else None


def summarize(rows):
    active = [r for r in rows if not r['dropped_group']]
    return dict(start_step=rows[0]['step'] + 1, end_step=rows[-1]['step'] + 1,
                groups=len(rows), active=len(active), dropped=len(rows)-len(active),
                correctness_contrast=sum(r['mixed_correctness'] for r in active),
                format_only=sum(r['format_only_signal'] for r in active),
                total_zero=sum(r['total_zero'] for r in active),
                nonzero_advantage=sum(r['nonzero_advantage'] for r in rows),
                mean_reward_std_active=mean(active, 'reward_std'),
                reconstructed_mean_abs_advantage_active=mean(active, 'adv_abs'),
                mean_learning_rate=mean(rows, 'learning_rate'),
                mean_grad_norm_active=mean(active, 'grad_norm'),
                median_grad_norm_active=statistics.median(r['grad_norm'] for r in active),
                mean_kl_active=mean(active, 'kl'),
                median_kl_active=statistics.median(r['kl'] for r in active if r['kl'] is not None),
                mean_entropy_active=mean(active, 'entropy'),
                online_k_active=dict(collections.Counter(r['online_k'] for r in active)),
                mean_online_correctness_active=mean(active, 'online_k')/4)


def main():
    consumed=read(SOURCE/'train_consumption.jsonl')
    rollouts=read(SOURCE/'rollouts.jsonl')
    trainer=load(SOURCE/'trainer_log.json')
    logs={r['step']-1:r for r in trainer if 'grad_norm' in r}
    chosen=read(ROOT/'remote_results/grpo_selected_direct367_v1/train.jsonl')
    by_id={r['row_id']:r for r in chosen}
    groups=collections.defaultdict(list)
    for r in rollouts:
        groups[r['step']].append(r)
    assert len(consumed)==len(groups)==len(logs)==699 and len(rollouts)==2796
    assert {r['step'] for r in consumed}==set(range(699))
    assert len({r['row_id'] for r in consumed})==699
    rows=[];std_errors=[]
    for c in consumed:
        step=c['step']; samples=groups[step]; log=logs[step]
        assert len(samples)==4 and {s['row_id'] for s in samples}=={c['row_id']}
        rewards=[s['rewards'][0]+.1*s['rewards'][1] for s in samples]
        k=int(sum(s['rewards'][0] for s in samples))
        sd=statistics.stdev(rewards); avg=statistics.mean(rewards)
        advantage=[(x-avg)/(sd+1e-4) for x in rewards]
        std_errors.append(abs(sd-log['reward_std']))
        assert c['nonzero_advantage']==(not c['dropped_group'] and sd>1e-8)
        row=by_id[c['row_id']]
        rows.append(dict(**c, subject=row['type'],level=row['level'],difficulty=row['difficulty'],
                         offline_k=row['screen_correct_count'],online_k=k,
                         mixed_correctness=0<k<4,format_only_signal=k in (0,4) and sd>1e-8,
                         total_zero=sd<1e-8,reward_std=sd,
                         adv_abs=statistics.mean(abs(x) for x in advantage) if not c['dropped_group'] else 0,
                         learning_rate=log['learning_rate'],grad_norm=log['grad_norm'],
                         kl=log.get('kl'),entropy=log.get('entropy')))
    assert max(std_errors)<1e-6
    overall=summarize(rows)
    assert overall['active']==548 and overall['nonzero_advantage']==533
    assert overall['correctness_contrast']+overall['format_only']+overall['total_zero']==548
    windows={f'{a+1}-{b}':summarize(rows[a:b]) for a,b in
             [(0,174),(174,349),(349,524),(524,699),(0,100),(599,699)]}
    unknown_ids={r['row_id'] for r in rows if r['unconfirmed_grading']}
    errors=[r for r in rollouts if r['row_id'] in unknown_ids
            and r['verdict']['status'] not in ('correct','incorrect','extraction_failed')]
    assert {r['row_id'] for r in errors}==unknown_ids
    a=read(ROOT/'remote_results/test_direct367_v1/sft/answers.jsonl')
    b=read(ROOT/'remote_results/test_after_grpo_direct367_v1/grpo/answers.jsonl')
    assert [r['row_id'] for r in a]==[r['row_id'] for r in b] and len(a)==2024
    gain=[x for x,y in zip(a,b) if not x['correct'] and y['correct']]
    regress=[y for x,y in zip(a,b) if x['correct'] and not y['correct']]
    inputs=[SOURCE/n for n in ['train_consumption.jsonl','rollouts.jsonl','trainer_log.json','grpo_config.json']]
    result=dict(verification_status='ANALYZED',overall=overall,windows=windows,
                reconstruction='Sample std (G=4), (r-mean)/(std+1e-4), then same recorded group mask; derived, not a saved raw advantage tensor',
                max_reward_std_reconstruction_error=max(std_errors),
                unconfirmed_groups=len(unknown_ids),unconfirmed_samples=len(errors),
                unconfirmed_statuses=dict(collections.Counter(r['verdict']['status'] for r in errors)),
                gained_questions=len(gain),regressed_questions=len(regress),
                gained_previous_statuses=dict(collections.Counter(r['verdict']['status'] for r in gain)),
                regressed_new_statuses=dict(collections.Counter(r['verdict']['status'] for r in regress)),
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                gradient_check=dict(active_zero_grad=sum(not r['dropped_group'] and r['grad_norm']==0 for r in rows),
                                    dropped_nonzero_grad=sum(r['dropped_group'] and r['grad_norm']!=0 for r in rows),
                                    maximum_raw_grad_norm=max(r['grad_norm'] for r in rows)),
                clipped_policy_ratio_max=max(r.get('clip_ratio/region_mean',0) for r in logs.values()))
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'groups.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['## Material Passport','','- Origin Skill: academic-research-suite / experiment-agent',
          '- Origin Mode: validate','- Origin Date: 2026-10-16','- Verification Status: ANALYZED',
          '- Version Label: grpo_log_validation_v1','','## 结论','',
          '日志不支持训练后期组内优势大面积消失。更明确的现象是：699题一遍训练中，151组被屏蔽，102组仅有格式奖励差异，431组有正确性差异；后期学习率明显降低。单次实验不能确定哪一个因素造成最终增益大小。',
          '', '## 全程信号构成','','| 类型 | 组数 | 占全部699组 |','|---|---:|---:|']
    for name,n in [('正确性有差异',overall['correctness_contrast']),('只有格式奖励有差异',overall['format_only']),
                   ('组内总奖励完全相同',overall['total_zero']),('因结束异常或核验未确认整组屏蔽',overall['dropped'])]:
        text.append(f'| {name} | {n} | {n/699:.2%} |')
    text+=['','这里的“正确性有差异”是在线四答奖励中同时出现0和1。静态筛选时1–3/4并不保证在线重新生成仍是混合组。',
           '548个未屏蔽组中，431组有正确性对比，102组只有格式对比，15组总奖励相同。15组仍可能提供KL梯度；不能把总奖励相同等同于完全没有梯度。屏蔽组没有该组策略或KL梯度，但调度器继续前进，Adam动量及权重衰减也可能改变参数。',
           '', '## 随训练进度变化','','| 步数 | 未屏蔽 | 正确性差异组 | 格式差异组 | 总奖励相同组 | 平均绝对优势（未屏蔽，重算） | 平均学习率 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for key in ['1-174','175-349','350-524','525-699','1-100','600-699']:
        s=windows[key];text.append(f"| {key} | {s['active']} | {s['correctness_contrast']} | {s['format_only']} | {s['total_zero']} | {s['reconstructed_mean_abs_advantage_active']:.3f} | {s['mean_learning_rate']:.3g} |")
    text+=['','前100步：60组有正确性差异、79组非零优势；最后100步：71组有正确性差异、83组非零优势。最后100步总奖励相同仅1组。窗口数据没有显示后期信号耗尽。不同窗口是不同题目，不能将训练奖励趋势直接当作固定测试集的学习曲线。',
           '', '## 优势的含义与格式权重','',
           '本次scale_rewards=group，总奖励=正确性+0.1×格式。TRL先减组内均值，再除组内样本标准差加1e-4；因此原始奖励标准差变小不必然使归一化优势变小。四份最终奖励完全一致才会使该组策略优势归零。',
           '当正确性奖励四份全相同时，只剩格式差异；格式项的0.1系数在组内标准化中近似抵消。因此“格式权重0.1”不代表这些组的优势或梯度一定只有正确性组的十分之一。102组格式信号不能当作102组直接数学正确性训练信号。',
           '[TRL v0.23.1 官方实现](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_trainer.py)。',
           '', '重算使用保存的四份reward，并应用记录的整组屏蔽状态。未保存原始优势张量；重算标准差与日志最大差约'+f'{max(std_errors):.2g}，非零优势组数与消费日志完全匹配。',
           '', '## 更值得关注的限制','',
           '1. 151/699=21.60%的组被屏蔽。其中128组涉及prediction_parse_failed，共139份答案未被表达式解析器接受；23组仅异常结束，另1组两种问题兼有。未确认不等于数学错误。某些答案将正确表达式置于完整自然语言句子中，仍会解析失败；这类格式与解析接口问题可丢失整组信号。',
           '2. 全部699题只生成训练一遍，实际有正确性对比的仅431组。它们提供监督的是在线输出间的相对偏好，不保证覆盖测试中的所有薄弱题型。',
           '3. 后100步平均学习率约3.66e-7，前100步约4.43e-6，约低12倍；优势没有消失，但同类梯度能产生的参数更新幅度受学习率限制。不能据此保证延长训练或增大学习率一定更好。',
           '4. 梯度日志未发现未屏蔽组的梯度范数为0。存在个别大梯度/KL离群值，配置max_grad_norm=1提供裁剪；不能仅用平均KL或接近0的带符号loss判断没有学习。',
           '', '## 与最终测试得失的关系','',
           'SFT→GRPO：140道从未判对变为判对，84道从判对变为未判对，净增56道，即2.77个百分点。并非只学会56道，而是收益与退化相互抵消。新增判对中118道原先为incorrect，22道原先提取或解析失败；退化中68道变为incorrect、16道变为提取或解析失败。',
           'SFT未判对462道，GRPO未判对406道，未判对数相对减少12.12%。本分析不预设2.77个百分点一定大或小，也不从单个随机种子实验归因某一超参数。',
           '', '## 下一轮可检验的方向','',
           '优先做离线解析失败审计，检查最终答案输出协议；如改变解析规则，需固定新的规则并对Base/SFT/GRPO一致重评，不能只放宽GRPO分数。其次可比较固定“有效正确性混合组”预算与本次固定699生成组预算，显式报告补采样成本。学习率或多步数对照应单独做，固定起点、参考模型、数据与评分，保存完整优化器和调度器状态。这里只提出可检验方向，没有启动新训练或开启服务器。',
           '', '## 统计解释检查（11/11）','',
           '| 检查 | 本次判断 |','|---|---|',
           '| Simpson悖论 | 未将不同题目窗口的总体趋势当作各学科均有同样趋势；因果解释仍受题目构成影响。 |',
           '| 生态谬误 | 组级信号比例不能推出每道测试题的收益。 |',
           '| Berkson选择偏差 | 题池经G4混合筛选，结论仅描述本训练池。 |',
           '| Collider偏差 | 再按结束/解析过滤会改变题型构成，不能把保留组当全池无偏样本。 |',
           '| 基础率忽略 | 同时列示全部699组、548未屏蔽组及正确性/格式信号类别。 |',
           '| 均值回归 | 离线按1–3/4选题后，在线重新采样可能自然得到0/4或4/4，不能全部归因退化或进步。 |',
           '| 幸存者偏差 | 显式报告151屏蔽组，未将其从699总预算中隐藏。 |',
           '| 多重搜寻 | 窗口为探索性描述，列示四段与首尾100步，不做挑选显著窗口的推断。 |',
           '| 分析路径自由度 | 已提供原始来源SHA、逐组CSV和复算脚本；未宣称预注册验证。 |',
           '| 相关不等于因果 | 没有消融或多种子对照，不能确认屏蔽、格式项或学习率哪项导致增益大小。 |',
           '| 反向因果 | 策略变化与题目构成都能改变在线reward差异，不能从窗口关系单独确定方向。 |',
           '', '## 复现范围','',
           'Verification Status: ANALYZED。已从本地保存日志重算与核对，未重新执行GPU训练；服务器保持关机。逐组数据见groups.csv，汇总与输入SHA256见summary.json。分析入口为项目tools/analyze_direct_grpo_logs.py。所有训练及测试原始文件保持不变。']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(overall=overall,first100=windows['1-100'],last100=windows['600-699'],output=str(OUT)),ensure_ascii=False))


if __name__ == '__main__':
    main()
