"""Wait for this inference job, score on CPU, and archive its results. No shutdown."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def save(p, obj):
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(p)


def main():
    root=Path(sys.argv[1]).resolve()
    status=root/'postprocess_status.json'
    lock=(root/'postprocess.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(stage,**kw):
        item=dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw)
        save(status,item)
        print(json.dumps(item),flush=True)
    try:
        state('waiting_for_generation')
        while True:
            progress=json.loads((root/'results/status.json').read_text())
            if progress['stage']=='generation_complete':
                break
            if progress['stage']=='failed':
                raise RuntimeError('Generation failed: '+repr(progress))
            proc=Path('/proc')/str(progress['pid'])/'stat'
            if not proc.exists() or proc.read_text().split(') ',1)[1].split()[0] in ['Z','X']:
                raise RuntimeError('Generation process ended without completion')
            time.sleep(15)
        state('scoring')
        tools=Path(__file__).resolve().parent
        subprocess.run([sys.executable,str(tools/'test_audited_answer_scoring.py')],check=True)
        subprocess.run([sys.executable,'-u',str(tools/'math_remaining_eval.py'),'score',
            '--bundle',str(root/'bundle'),'--output',str(root/'results'),
            '--score-output',str(root/'scored_v31')],check=True)
        state('archiving')
        archive=root/'results_and_scores.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for name in ['results','scored_v31','generation.log','postprocess.log','bundle/manifest.json']:
                tar.add(root/name,arcname=name)
        sha=hashlib.sha256(archive.read_bytes()).hexdigest()
        state('complete',archive=str(archive),archive_sha256=sha,report=str(root/'scored_v31/REPORT.md'))
    except BaseException as exc:
        state('failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
