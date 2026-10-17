"""Audit applied reward weights and observed format failures, without retraining."""
import json,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'datasets/grpo_candidates_gsm8k_math_l2_l3/reward_pilot_v3/results'
OUT=P/'format_audit';OUT.mkdir(exist_ok=True)
read=lambda p:[json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]
def text(r):
 c=r['completion'];return c if isinstance(c,str) else '\n'.join(x['content'] for x in c)
def format_ok(c,e):
 c=c.strip();c=re.sub(r'(?:<\|endoftext\|>|<\|im_end\|>)\s*$','',c).strip()
 return bool(e and c.count('<end_working_out>')==1 and c.count('<SOLUTION>')==1 and c.count('</SOLUTION>')==1 and re.fullmatch(r'.+<end_working_out>\s*<SOLUTION>[^<>]+</SOLUTION>',c,re.S) and '<start_working_out>' not in c)
def reasons(c,e):
 flags=[]
 if not e:flags.append('no_eos')
 if '</start_working_out>' in c:flags.append('wrong_end_tag')
 if c.count('<end_working_out>')!=1:flags.append('missing_or_multiple_end_tag')
 if c.count('<SOLUTION>')!=1 or c.count('</SOLUTION>')!=1:flags.append('missing_or_multiple_solution_tags')
 if '</SOLUTION>' in c and not c.strip().endswith('</SOLUTION>'):flags.append('content_after_solution')
 if '<start_working_out>' in c:flags.append('repeated_start_tag')
 return flags or ['other_structure_failure']
stats={};examples=[]
sets={'training':read(P/'B_correctness_format/rollouts.jsonl'),'before':read(P/'baseline_eval/answers.jsonl'),'after':read(P/'B_correctness_format_eval/answers.jsonl')}
for name,rows in sets.items():
 counts=Counter();groups=defaultdict(list);ok=0;by_source={};mismatches=0
 for r in rows:
  c=text(r);ended=r.get('ended',r.get('eos'));fmt=format_ok(c,ended)
  saved=bool(r['rewards'][1]) if name=='training' else r['format_eos']
  mismatches+=fmt!=saved;ok+=fmt
  groups[(r.get('step'),r['row_id'])].append((int(r.get('correct',r.get('verdict',{}).get('correct'))),int(fmt)))
  if not fmt:
   fs=reasons(c,ended);counts.update(fs)
   if sum(x['set']==name for x in examples)<6:examples.append(dict(set=name,row_id=r['row_id'],reasons=fs,completion=c))
 for src in ['gsm8k','math']:
  part=[r for r in rows if r['source']==src]
  by_source[src]=dict(answers=len(part),format_ok=sum(format_ok(text(r),r.get('ended',r.get('eos'))) for r in part))
 group_stats=Counter()
 for pairs in groups.values():
  assert len(pairs)==4
  cs=[x[0] for x in pairs];fs=[x[1] for x in pairs]
  group_stats['groups']+=1
  group_stats['format_differs']+=len(set(fs))>1
  group_stats['correctness_differs']+=len(set(cs))>1
  group_stats['format_only_differs']+=len(set(fs))>1 and len(set(cs))==1
  group_stats['all_formats_wrong']+=sum(fs)==0
  group_stats['all_formats_right']+=sum(fs)==4
  # Same correctness but differing format provides an unconfounded format ranking.
  group_stats['same_correctness_format_pairs']+=sum(cs[i]==cs[j] and fs[i]!=fs[j] for i in range(4) for j in range(i+1,4))
 stats[name]=dict(answers=len(rows),format_ok=ok,failures=len(rows)-ok,reason_counts_nonexclusive=dict(counts),recomputed_format_mismatches=mismatches,groups=dict(group_stats),by_source=by_source)
 assert mismatches==0
logs=json.loads((P/'B_correctness_format/trainer_log.json').read_text());logs=[r for r in logs if 'loss' in r]
stats['weighted_reward_max_error']=max(abs(r['reward']-(r['pilot/correct_rate']+.1*r['pilot/format_eos_rate'])) for r in logs)
stats['training_format_by_quarter']=[sum(r['pilot/format_eos_rate'] for r in logs[i:i+8])/8 for i in range(0,32,8)]
stats['dropped_groups_max_fraction']=max(r['pilot/dropped_group_fraction'] for r in logs)
stats['validation_format_change_ci']=json.loads((P/'comparison.json').read_text())['metrics']['format_eos']
(OUT/'statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
