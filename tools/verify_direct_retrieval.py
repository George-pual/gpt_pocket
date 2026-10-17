"""Extract in the project and verify the complete remote artifact manifest before shutdown."""
import argparse,hashlib,json,zipfile
from pathlib import Path


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    archive=Path(a.archive);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    receipt=json.loads(Path(a.receipt).read_text(encoding='utf-8'))
    assert digest(archive)==receipt['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():assert (out/info.filename).resolve().is_relative_to(out.resolve()),'Unsafe archive path'
        z.extractall(out)
    manifest=json.loads((out/'all_results_manifest.json').read_text(encoding='utf-8'))
    assert len(manifest)==receipt['total_manifest_files']
    for name,expected in manifest.items():assert digest(out/name)==expected,name
    for name in ['sft_direct367_v1','grpo_direct367_v1']:
        assert json.loads((out/name/'completion_summary.json').read_text())['status']=='complete'
        assert (out/name/'adapter/adapter_model.safetensors').stat().st_size>0
    for name in ['test_direct367_v1','test_after_grpo_direct367_v1']:
        assert json.loads((out/name/'progress.json').read_text())['status']=='complete'
        for arm in json.loads((out/name/'comparison.json').read_text()).values():assert arm['questions']==2024
    result=dict(status='verified',complete_remote_manifest_files=len(manifest),archive_sha256=receipt['archive_sha256'],
        all_models_data_and_logs_retrieved=True,shutdown_ready=True,shutdown_performed=False)
    (out/'final_retrieval_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
