"""Package completed experiment evidence; models already retrieved in SFT backup are referenced by hash."""
import hashlib,json,zipfile
from pathlib import Path


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    status=json.loads((root/'direct_pipeline_status.json').read_text())
    assert status['stage']=='awaiting_verified_local_retrieval' and status['status']=='training_and_evaluation_complete'
    with zipfile.ZipFile(root/'sft_direct367_backup.zip') as z:
        prior=json.loads(z.read('sft_backup_manifest.json'))
    names=['sft_direct367_v1','dev_direct367_v1','test_direct367_v1','grpo_screen_direct367_v1',
        'grpo_selected_direct367_v1','grpo_direct367_v1','test_after_grpo_direct367_v1','reviewed_direct',
        'official_test','tools','questions.jsonl','development.jsonl','grpo_reserved.jsonl','selection_manifest.json',
        'cold_start_selection.json','direct_pipeline_status.json']
    files=[]
    for name in names:
        p=root/name
        assert p.exists(),name
        files.extend([p] if p.is_file() else [f for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts])
    files.extend(root.glob('*.log'))
    files=sorted(set(files));manifest={};added=[]
    archive=root/'direct_final_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files:
            name=p.relative_to(root).as_posix();h=digest(p);manifest[name]=h
            if prior.get(name)==h:continue
            z.write(p,name);added.append(name)
        z.writestr('all_results_manifest.json',json.dumps(manifest,indent=2))
    receipt=dict(status='packaged',archive=archive.name,bytes=archive.stat().st_size,archive_sha256=digest(archive),
        total_manifest_files=len(manifest),new_archive_files=len(added),
        prior_archive='sft_direct367_backup.zip',prior_archive_sha256=digest(root/'sft_direct367_backup.zip'),
        prior_archive_files_reused=len(manifest)-len(added),shutdown_performed=False)
    (root/'direct_final_package_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
