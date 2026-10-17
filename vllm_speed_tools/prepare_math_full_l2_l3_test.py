"""Build full official MATH L2/L3 test and an independent overlap-clean view."""
import hashlib,json,re,sys,unicodedata
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'closure_grpo_experiment'))
from math_verification import extract_final_answer
OUT=ROOT/'datasets/math/test_l2_l3_full'
def read(p):return [json.loads(x) for x in p.read_bytes().splitlines() if x.strip()]
def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).lower()
def write(p,rs):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs),encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    parent=ROOT/'datasets/math/test_l2_l3_200'
    oldmanifest=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    expectedhash={p:h for p,h,_ in oldmanifest['downloads']}
    trains=read(ROOT/'datasets/math/train.jsonl')
    sft=read(ROOT/'sft_quality_audit_v2/merged_supplied_answers/train_merged_505.jsonl')
    fingerprints={}
    for name,rows in [('MATH_train',trains),('SFT505',sft)]:
        for r in rows:fingerprints.setdefault(norm(r['problem']),set()).add(name)
    full=[];sourcecounts=Counter();overlaps=[];gold_normalizations=[]
    for p in sorted((parent/'raw').glob('*/*.parquet')):
        rel=p.relative_to(parent/'raw').as_posix()
        assert digest(p)==expectedhash[rel]
        for idx,r in enumerate(pq.read_table(p).to_pylist()):
            sourcecounts[r['level']]+=1
            if r['level'] not in ['Level 2','Level 3']:continue
            answer,origin,error=extract_final_answer(r['solution'])
            assert not error and origin=='boxed',(rel,idx,error)
            overlap=sorted(fingerprints.get(norm(r['problem']),set()))
            rid=f'math:{rel}:{idx}';raw_answer=answer;reason=None
            # Two audited reference-only presentation defects. Retain the
            # original boxed answer and full source solution for inspection.
            if rid=='math:intermediate_algebra/test-00000-of-00001.parquet:129':
                assert answer==r'\frac{7}{5} + \frac{1}{5} i.'
                answer=answer[:-1];reason='Remove terminal prose period from complex expression.'
            if rid=='math:prealgebra/test-00000-of-00001.parquet:551':
                assert answer=='90\\text{ square\nunits}'
                answer='90';reason='Reference requests area; canonical numeric answer omits generic square units.'
            if reason:gold_normalizations.append(dict(row_id=rid,raw=raw_answer,normalized=answer,reason=reason))
            x=dict(r,row_id=rid,answer=answer,answer_raw=raw_answer,answer_normalization=reason,source='math',split='test',
                   overlaps_training_exact_normalized=bool(overlap),overlap_sources=overlap)
            full.append(x)
            if overlap:overlaps.append(x)
    clean=[x for x in full if not x['overlaps_training_exact_normalized']]
    assert len(full)==2025 and Counter(x['level'] for x in full)=={'Level 2':894,'Level 3':1131}
    assert len({x['row_id'] for x in full})==len(full)
    assert {x['row_id'] for x in read(parent/'questions.jsonl')} <= {x['row_id'] for x in clean}
    write(OUT/'questions_all_official.jsonl',full)
    write(OUT/'questions.jsonl',clean)
    write(OUT/'overlaps_training.jsonl',overlaps)
    manifest=dict(repository=oldmanifest['repository'],revision=oldmanifest['revision'],split='test',
        source_test_questions=sum(sourcecounts.values()),official_levels=dict(Counter(x['level'] for x in full)),
        official_questions=len(full),independent_questions=len(clean),
        independent_levels=dict(Counter(x['level'] for x in clean)),overlap_exclusions=[x['row_id'] for x in overlaps],
        overlap_check_against=['all7500MATHtrain','SFT505'],
        official_data_sha256=digest(OUT/'questions_all_official.jsonl'),data_sha256=digest(OUT/'questions.jsonl'),
        previous200_included=True,reference_answer_normalizations=gold_normalizations,
        scoring='Same versioned scorer for every arm; report Level 2 and Level 3 separately.',
        note='All official L2/L3 rows retained in questions_all_official.jsonl. Main independent evaluation uses questions.jsonl after exact normalized overlap exclusion, not model-score filtering.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text(f"# MATH 官方 Level 2/3 全量测试\n\n官方 test 共 5000 题，其中 Level 2 为 894 题、Level 3 为 1131 题，合计 2025 题，完整保存在 questions_all_official.jsonl。\n\n与 7500 条 MATH train 及 505 条 SFT 题目做 NFKC、忽略空白和大小写的精确重叠核查后，主独立评测文件 questions.jsonl 共 {len(clean)} 题，难度计数 {manifest['independent_levels']}。重叠题单独保存在 overlaps_training.jsonl，原始完整 test 未删改。该检查不能排除近重复或模型预训练污染。\n\n原固定 200 题全部包含在主评测文件中。所有模型沿用同一 Prompt、template、EOS、max_tokens=3894、temperature=0、每题一回答；统一评分版本，按难度分别报告正确率、格式率、截断率。此次扩大测试范围不引入任何训练题。\n",encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
