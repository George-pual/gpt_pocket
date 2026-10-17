"""Audit full token lengths and TRL collator labels on CPU, without model weights."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import statistics
import sys


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--common-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Explicitly disable GPUs for this audit with CUDA_VISIBLE_DEVICES=""')
    if a.output.exists(): raise RuntimeError('Refuse overwrite: '+str(a.output))
    assert metadata.version('trl')=='0.23.1'
    sys.path.insert(0,str(a.common_dir))
    from common import tokenizer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    import torch
    torch.set_num_threads(2)
    t=tokenizer(str(a.model)); t.padding_side='right'
    assert t.eos_token_id==151643
    prepared=read(a.data/'supplement_prepared.jsonl')
    records=read(a.data/'supplement_records.jsonl')
    assert len(prepared)==len(records)==1390
    encoded=[]; lengths=[]
    for row,record in zip(prepared,records):
        assert (row['problem'],row['answer'])==(record['problem'],record['answer'])
        prompt=t.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True)
        prefix=t(prompt,add_special_tokens=False)['input_ids']
        completion=t(row['completion'],add_special_tokens=False)['input_ids']
        assert completion and completion[-1]==t.eos_token_id
        assert completion.count(t.eos_token_id)==1
        ids=prefix+completion; mask=[0]*len(prefix)+[1]*len(completion)
        encoded.append(dict(input_ids=ids,completion_mask=mask))
        lengths.append(dict(row_id=record['row_id'],level=record['level'],
                            prompt_tokens=len(prefix),completion_tokens=len(completion),
                            full_tokens=len(ids),exceeds_2048=len(ids)>2048,exceeds_4096=len(ids)>4096))
    collator=DataCollatorForLanguageModeling(pad_token_id=t.pad_token_id,completion_only_loss=True)
    checked=0
    for start in range(0,len(encoded),32):
        batch=encoded[start:start+32]
        output=collator(batch)
        for index,row in enumerate(batch):
            wanted=[token if mask else -100 for token,mask in zip(row['input_ids'],row['completion_mask'])]
            actual=output['labels'][index].tolist()
            assert actual[:len(wanted)]==wanted
            assert all(v==-100 for v in actual[len(wanted):])
            assert actual[len(wanted)-1]==151643
            checked+=1
    a.output.mkdir(parents=True)
    (a.output/'sequence_lengths.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in lengths),encoding='utf-8')
    report=dict(status='cpu_tokenizer_and_collator_audit_passed_no_training',records=len(records),
                full_tokens_min=min(r['full_tokens'] for r in lengths),
                full_tokens_max=max(r['full_tokens'] for r in lengths),
                full_tokens_mean=statistics.mean(r['full_tokens'] for r in lengths),
                full_tokens_median=statistics.median(r['full_tokens'] for r in lengths),
                over_2048=[r['row_id'] for r in lengths if r['exceeds_2048']],
                over_4096=[r['row_id'] for r in lengths if r['exceeds_4096']],
                no_truncation=True,eos_id=151643,final_eos_supervised=checked,
                prompt_labels_all_masked=checked,padding_labels_all_masked=True,
                collator='trl.trainer.sft_trainer.DataCollatorForLanguageModeling',
                scope='Exact tokenizer and original explicit token/mask preparation plus TRL0.23.1 collator. No model/SFTTrainer instantiated; no training or mathematical-step certification.',
                packages={name:metadata.version(name) for name in ('transformers','trl','torch')},
                input_sha256={name:sha(a.data/name) for name in ('supplement_prepared.jsonl','supplement_records.jsonl')},
                tokenizer_sha256={name:sha(a.model/name) for name in ('tokenizer.json','tokenizer_config.json')},
                common_sha256=sha(a.common_dir/'common.py'),script_sha256=sha(Path(__file__)))
    (a.output/'tokenizer_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    main()
