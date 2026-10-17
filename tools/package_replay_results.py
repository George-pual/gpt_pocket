"""Archive a completed replay run with a checksum for every saved file."""
import argparse,collections,hashlib,json,zipfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
    root=Path(args.root).resolve()
    assert root.name=='grpo_replay549_extract_v4_20261016_batch12'
    status=json.loads((root/'status.json').read_text());assert status['stage']=='complete_awaiting_retrieval'
    plan=json.loads((root/'plan.json').read_text())
    training=json.loads((root/'training/completion_summary.json').read_text())
    assert training['status']=='complete' and training['updates']==549 and training['parameter_max_delta']>0
    assert sha(root/'training/adapter/adapter_model.safetensors')==training['adapter_sha256']
    train=read(root/'data/train.jsonl');consumed=read(root/'training/train_consumption.jsonl')
    assert len(train)==len(consumed)==len({x['row_id'] for x in consumed})==549
    assert {x['row_id'] for x in train}=={x['row_id'] for x in consumed}
    rollouts=read(root/'training/rollouts.jsonl');assert len(rollouts)==2196
    groups=collections.defaultdict(list)
    for x in rollouts:groups[x['row_id']].append(x)
    assert len(groups)==549 and all(len(v)==4 and {x['sample_index'] for x in v}==set(range(4)) for v in groups.values())
    test=json.loads((root/'test_after_grpo/comparison.json').read_text())['grpo_replay549']
    answers=read(root/'test_after_grpo/grpo_replay549/answers.jsonl');questions=read(root/'data/official_test.jsonl')
    assert len(answers)==len(questions)==test['questions']==2024
    assert len({x['row_id'] for x in answers})==2024 and {x['row_id'] for x in answers}=={x['row_id'] for x in questions}
    assert sum(x['correct'] for x in answers)==test['correct']
    runtime=json.loads((root/'test_after_grpo/manifest.json').read_text())
    assert runtime['grader_sha256']==plan['grader_sha256'] and runtime['data_sha256']==plan['test_sha256']
    assert runtime['adapter_sha256']['grpo_replay549']==training['adapter_sha256']
    archive=root/'replay_results.zip';temporary=root/'replay_results.zip.partial';receipt_path=root/'package_receipt.json'
    assert not archive.exists() and not receipt_path.exists(),'Use existing verified archive instead of overwriting it'
    excluded={archive,temporary,receipt_path}
    files=sorted(p for p in root.rglob('*') if p.is_file() and p not in excluded)
    manifest={}
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_STORED) as z:
        for p in files:
            name=p.relative_to(root).as_posix();digest=hashlib.sha256()
            with p.open('rb') as source,z.open(name,'w',force_zip64=True) as target:
                for block in iter(lambda:source.read(8*1024*1024),b''):target.write(block);digest.update(block)
            manifest[name]=digest.hexdigest()
        z.writestr('all_results_manifest.json',json.dumps(manifest,indent=2))
    temporary.replace(archive)
    receipt=dict(status='packaged',archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files=len(manifest),
        training=training,test=test,train_sha256=plan['train_sha256'],test_sha256=plan['test_sha256'],
        adapter_sha256=training['adapter_sha256'],shutdown=False)
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
