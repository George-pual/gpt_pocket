"""Verify and extract the replay archive within a chosen local output folder."""
import argparse,hashlib,json,os,shutil,zipfile
from pathlib import Path
from package_replay_results import sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--receipt',required=True)
    ap.add_argument('--output',required=True);args=ap.parse_args()
    archive=Path(args.archive);receipt=json.loads(Path(args.receipt).read_text(encoding='utf-8'))
    assert archive.stat().st_size==receipt['archive_bytes'] and sha(archive)==receipt['archive_sha256']
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('all_results_manifest.json'))
        assert len(manifest)==receipt['files'] and set(z.namelist())==set(manifest)|{'all_results_manifest.json'}
        assert len(z.namelist())==len(set(z.namelist()))
        for name,expected in manifest.items():
            relative=Path(name);path=(output/relative).resolve()
            assert not relative.is_absolute() and '..' not in relative.parts and path.is_relative_to(output)
            if os.name=='nt' and not str(path).startswith('\\\\?\\'):
                path=Path('\\\\?\\'+str(path))
            digest=hashlib.sha256()
            with z.open(name) as source:
                for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
            assert digest.hexdigest()==expected,name
            if path.exists():assert sha(path)==expected,('Existing local file differs',name)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as source,path.open('wb') as target:shutil.copyfileobj(source,target,8*1024*1024)
            assert sha(path)==expected,name
    (output/'all_results_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    train=json.loads((output/'training/completion_summary.json').read_text())
    test=json.loads((output/'test_after_grpo/comparison.json').read_text())['grpo_replay549']
    assert train['status']=='complete' and train['updates']==549 and test['questions']==2024
    assert sha(output/'training/adapter/adapter_model.safetensors')==receipt['adapter_sha256']==train['adapter_sha256']
    report=dict(status='verified',files_verified=len(manifest),archive_sha256=receipt['archive_sha256'],
        adapter_sha256=receipt['adapter_sha256'],training=train,test=test,shutdown_authorized=False)
    (output/'final_retrieval_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
