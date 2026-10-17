"""Record manual review notes; preserve original benchmark scores unchanged."""
import json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/gsm8k/base_sftv3_test/results/grpo1024_error_audit'
def read(name):return [json.loads(s) for s in (OUT/name).read_text(encoding='utf-8').splitlines()]
math_notes=[
('nontermination','未使用剩余面积≤1的停止条件，逐日除二到第50天'),
('target','应算1600−1444，却算1764−1444'),
('algebra','代入时把2e=d−9错误替换为2(d−9)'),
('algebra','二次式因式分解错误，未回代核验'),
('counting','面积3只可能1×3或3×1，模型无依据地选边界和扣除'),
('counting','最近点的边界应是中点2和5，模型用2和4'),
('counting','四张未署名作业为4!个一一对应，不是独立从20张选'),
('arithmetic','2^6×3^2×5=2880，最终乘法算错'),
('counting','第一枚骰子把3同时当作奇数和偶数，频数统计错误'),
('modeling','相邻三角形共享边，每次增加2条边，却臆造公差1'),
('arithmetic','组合公式正确，85×84×83/6计算错误'),
('answer_unit','推理明确100%，答案只写100；百分数语义/输出单位与评分需分开'),
('counting','倒序名次应201−53+1，未统一计数起点'),
('constraints','只有两张非零卡，无法取三张；忽略可行性'),
('geometry','中心到角点应使用半边长8、8及厚度4，误用16、16、0.5'),
('geometry','误判内接正方形面积及阴影结构'),
('geometry','错误使用圆周角与圆心角关系'),
('geometry','未使用V−E+F=2，臆造棱锥不可行'),
('nontermination','反复说明找梯形高的方法，不执行勾股计算'),
('algebra','二次式分解及回代错误，所得边长甚至不符合周长'),
('geometry','正切对边/邻边及勾股的斜边混淆'),
('geometry','两圆内公切线应使用相似三角形半径比，构造错误方程'),
('geometry','角平分线定理及图中量对应错误，出现sin值大于1仍继续'),
('constraints','非退化三角形要求2<x<6，却把端点算入'),
('geometry','凭空拼接角，负角取绝对值没有几何依据'),
('algebra','2√501=√2004，嵌套根式系数未移入根号'),
('algebra','复合函数奇偶性规则错误'),
('algebra','清分母多乘x+2，后续声称错误代入结果为0'),
('algebra','未正确使用韦达公式，冗长错误推导且结束标签错误'),
('arithmetic','余式定理应用正确，但−1+177147−177147+12+1算成14'),
('nontermination','对数等差数列化简后循环输出同一式子'),
('modeling','五个边和成等差数列，模型臆造关系；应利用总和与中项'),
('algebra','根式方程化简和求根错误，虚构回代均满足'),
('algebra','按多项式次数直接比较实根大小，缺少符号和单调性分析'),
('arithmetic','在x=2时128×2^−3算成8，实际16'),
('algebra','y(y+2)误写为(y+2)^2，且无依据使用AM-GM'),
('nontermination','不等式错误变形后循环输出同一表达式'),
('periodicity','滚动问题未按周长23取模定位边'),
('constraints','约去/除以变量时遗漏k=0特例'),
('algebra','指数方程解的大小与对数分母大小方向弄反'),
('nontermination','函数方程可令y=0直接解，却枚举f(64)、f(66)…'),
('constraints','复合函数定义域漏掉正确第二禁值27/10，并错误比较大小'),
('counting','1000以内45的倍数应22个，误数20且终点写945'),
('arithmetic','循环小数相加与小数位索引均错误'),
('nontermination','不断写1/19循环节，未用周期18取模定位'),
('nontermination','反复复制题目，未使用ab=gcd×lcm'),
('arithmetic','声称1616=40²等错误平方关系，未实际验证'),
('constraints','最大三位数问题漏掉因数6，选431而不是621'),
('periodicity','下一班车应按202 mod 20=2求等待18，模型答20'),
('grader_false_negative','4:30 p.m.与标准答案一致，时间字符串解析失败'),
('counting','2011项奇偶交替求和，误计项数/漏掉末项'),
('nontermination','不断写4/15的小数6，未识别第2位起全为6'),
('constraints','把复数3i称为实数，忽略k∈R'),
('trigonometry','cos4x通解缩放及±形式转换错误'),
('grader_false_negative','2±sqrt(6)与标准根集合一致，解析失败'),
('trigonometry','cos18°的值记错，后续化简也错误'),
('trigonometry','椭圆半短轴公式错误'),
('nontermination','对数换底/行列式结构未识别，循环同一公式'),
('constraints','要求正整数n，模型直接给180/84=15/7'),
('modeling','几乎只复制题目，未经行列式推导直接给1,3'),
('arithmetic','−1+3(x−8)/2常数项算错，实际−13'),
('geometry','ABM并非在M处直角，错误套用余弦邻边/斜边'),
('grader_false_negative','−128−128i√3与标准表达式等价，评分器未识别'),
('nontermination','重复度数/弧度同余的说法，没有解sin等式两分支'),
('trigonometry','积化和差循环回原式，最终答案无推导依据'),
('trigonometry','sin变量替换后把原平面面积当变换后椭圆面积'),
('arithmetic','正交条件两方程列对，二元线性方程求解错误'),
('algebra','复数四次式因式分解错误，后续根及面积均错误'),
]
math=read('math_wrong.jsonl');assert len(math)==len(math_notes)==68
for x,(cat,note) in zip(math,math_notes):x.update(review_category=cat,review_note=note,review_basis='manual reading of completion; reference problem and solution available')
gsm_notes={
3:('target','已算出最高利润125，却把两方案利润差29当答案'),6:('target','已算儿子14岁，又多做31−14'),7:('units','金额差30当成30套积木，出现负数量仍不核验'),11:('missing_condition','遗漏每周3次'),15:('modeling','三倍应对应Marcia的6，不是题中差值2'),29:('units','half a dozen误当0.5件而不是6件'),30:('missing_condition','忽略已经做4次保姆收入20'),34:('modeling','额外通话300分钟应扣除，却加回'),37:('grader_false_negative','118 000带空格千位分组，末数字抽取误得000'),39:('units','一张纸四页而非两页'),46:('modeling','30%的指代应覆盖三项活动总时长，误只覆盖看演出'),54:('units','把每周工资当每月工资，遗漏月内周数；题目需约定4周/月'),59:('modeling','两位主人加6朋友应8人，误作7人'),60:('modeling','出发时间、行程时长、到达提前量混淆'),61:('units','1小时20分误作70分，且胜负时间方向也反了'),62:('modeling','省2元列成多花2元'),65:('units','推理得100美分，答案100未写单位；基准按美元1评分'),67:('modeling','人塔高度应按层数，误把每层人数计入'),71:('units','日期间隔算错'),75:('target','已得额外能买11张，又重复减已有1张'),77:('modeling','5年后27岁误作当前27，再次加5'),81:('modeling','两天共50、共120误作每天各50、120'),83:('units','盒数、单双数量和每盒折扣混淆'),85:('modeling','配偶人数重复计数，且漏主人；人数范围错误'),88:('modeling','每周两次应乘2，误除2'),92:('modeling','奖金使用加薪前/后的基数选择与标准解不同'),93:('reference_ambiguity','70 times as many可指总数或新增数；模型160000与标准162000的口径有歧义，暂不当纯能力错'),100:('modeling','5棒棒糖+4糖的总价3.20误作仅4糖的价格'),113:('units','小时工资、课次数、每天奖金的计费单位混淆'),114:('missing_condition','遗漏香型种类数量4和8'),117:('modeling','两个人的骑行总次数漏乘人数；需明确题目计数口径'),130:('target','已得每周40，又重复除4'),132:('modeling','成本=销售额−利润，符号及总额/单价均错'),136:('missing_condition','没有正确先扣15杯肉桂茶'),138:('modeling','each person包含Maria，应5人，误作4人'),143:('missing_condition','已赠送4箱没有从剩余量扣除'),150:('modeling','每年按原价增加5%应线性，误用复利'),163:('units','4美元应16枚25美分硬币，误作400枚'),166:('target','方程已解Sam20，又改用剩余80作答'),167:('units','三层高度应3×24，误仅24')}
gsm=read('gsm8k_wrong.jsonl');sample=json.loads((OUT/'gsm8k_manual_sample_indices.json').read_text());assert set(sample)==set(gsm_notes)
reviews=[]
for i in sample:
 x=dict(gsm[i]);cat,note=gsm_notes[i];x.update(wrong_list_index=i,review_category=cat,review_note=note);reviews.append(x)
for i,x in enumerate(gsm):
 if x['truncated']:
  z=dict(x,wrong_list_index=i,review_category='nontermination',review_note='截断尾部复核：重复题目、重复推导或无效枚举；应教会有限推导与停止条件')
  reviews.append(z)
for name,rs in [('math_reviewed.jsonl',math),('gsm8k_reviewed_sample.jsonl',reviews)]:
 (OUT/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs),encoding='utf-8')
print('MATH review categories',dict(Counter(x['review_category'] for x in math)))
print('GSM nontruncated sample categories',dict(Counter(x['review_category'] for x in reviews if not x['truncated'])))
