"""Present frozen records and persist explicit per-item reasoning audit notes."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'datasets/math/sft_v3_supplement_unseen_grpo1536_v2/supplement_records.jsonl'
AUDIT=ROOT/'datasets/math/sft_v3_supplement_unseen_grpo1536_v2/remaining_reasoning_audit'


def rows():
    return [json.loads(s) for s in SOURCE.read_text('utf-8').splitlines()]


def save(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['show','record','status'])
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--count',type=int,default=50)
    p.add_argument('--notes',type=Path)
    p.add_argument('--diagram',type=int)
    p.add_argument('--compact',action='store_true',help='Omit solution illustration code only; question figures remain visible')
    a=p.parse_args()
    AUDIT.mkdir(exist_ok=True)
    records=rows();target=[r for r in records if not r['mathematical_steps_independently_reviewed']]
    manifest=AUDIT/'manifest.json'
    expected=dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),target_count=len(target),
        previously_reviewed_count=len(records)-len(target),target_indices=[r['supplement_index'] for r in target],
        scope='Review mathematical reasoning, not merely agreement with the saved reference answer. Notes are explicit per record. No training.')
    if manifest.exists():
        assert json.loads(manifest.read_text('utf-8'))==expected
    else:save(manifest,expected)
    reviewed_file=AUDIT/'reviews.jsonl'
    reviewed=[json.loads(s) for s in reviewed_file.read_text('utf-8').splitlines()] if reviewed_file.exists() else []
    seen={r['supplement_index'] for r in reviewed}
    if a.action=='show':
        selected=target[a.start:a.start+a.count]
        if a.diagram is not None:
            selected=[next(r for r in target if r['supplement_index']==a.diagram)]
        for r in selected:
            print(f"\n#{r['supplement_index']} {r['type']} {r['level']}"+(' REVIEWED' if r['supplement_index'] in seen else ''))
            for key in ('problem','solution'):
                text=r[key]
                if a.compact and key=='solution' and a.diagram is None:
                    text=re.sub(r'\[asy\].*?\[/asy\]',' [solution illustration omitted; use --diagram INDEX to inspect] ',text,flags=re.S)
                if a.diagram is None:
                    text=re.sub(r'\[asy\](.*?)\[/asy\]',lambda m:' [ASY: '+re.sub(r'\s+',' ',m[1]).strip()+'] ',text,flags=re.S)
                print(key[0].upper()+': '+text)
    elif a.action=='record':
        assert a.notes
        by_id={r['supplement_index']:r for r in target}
        for line in a.notes.read_text('utf-8-sig').splitlines():
            if not line.strip():continue
            index,status,note=line.split('|',2);index=int(index)
            assert index in by_id and index not in seen,index
            assert status in ('pass','needs_fix','needs_diagram','uncertain')
            assert note.strip()
            r=by_id[index]
            reviewed.append(dict(supplement_index=index,row_id=r['row_id'],verdict=status,
                reasoning_check=note.strip(),source_solution_sha256=hashlib.sha256(r['solution'].encode()).hexdigest()))
            seen.add(index)
        reviewed.sort(key=lambda r:r['supplement_index'])
        reviewed_file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviewed),encoding='utf-8')
    print('\nPROGRESS',len(reviewed),'/',len(target))


if __name__=='__main__':main()
