"""Evaluate GRPO1024 on the exact completed Base/SFT test inputs and settings."""
import json,os,subprocess
from pathlib import Path
ROOT=Path('/root/autodl-tmp/gsm8k_base_sftv3_eval')
HASH='3a159f8045efc10bc1aae34901abd200751a16936d2798cfcce258101874112c'

def main():
    out=ROOT/'grpo1024_test'
    assert not out.exists(),'Refuse overwrite'
    out.mkdir()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',VLLM_CACHE_ROOT=str(ROOT/'cache/vllm'))
    results={}
    for kind,data,reference in [('gsm8k','data/test.jsonl','results_vllm'),('math','data/math_l2_l3/questions.jsonl','results_math_l2_l3')]:
        (out/'stage.txt').write_text('evaluating_'+kind)
        cmd=['/root/autodl-tmp/vllm_speed_env/bin/python','-u',str(ROOT/'tools/eval_grpo1024_test.py'),
             '--model','/root/autodl-tmp/qwen3_hf/model','--adapter',str(ROOT/'grpo1024_adapter'),
             '--data',str(ROOT/data),'--output',str(out/kind),'--dataset-kind',kind,
             '--arms','grpo1024','--expected-adapter-sha256',HASH]
        with (out/f'{kind}.log').open('w') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((out/kind/'grpo1024/summary.json').read_text())
        old=json.loads((ROOT/reference/'comparison.json').read_text())
        assert summary['status']=='complete'
        for arm in ('base','sft_v3'):
            for key in ('questions','data_sha256','prompt_tokens_sha256','sampling'):
                assert summary[key]==old[arm][key],(kind,arm,key)
        results[kind]=dict(old,grpo1024=summary)
        (out/'comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        print('DATASET_COMPLETE',kind,json.dumps(summary),flush=True)
    (out/'stage.txt').write_text('complete')
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('PIPELINE_FAILED',repr(exc),flush=True)
        raise
