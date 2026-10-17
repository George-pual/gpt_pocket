"""Pinned TRL 0.23.1 SFT from Base with a fresh LoRA; preserve messages JSONL."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import time


def convert_messages(row):
    from common import messages, END, SOL_START, SOL_END
    from math_verification import extract_final_answer
    msgs = row.get('messages', [])
    if len(msgs) != 2 or [m['role'] for m in msgs] != ['user','assistant']:
        raise ValueError('Expected exactly user/assistant messages')
    problem, solution = (m['content'].strip() for m in msgs)
    if not problem or not solution:
        raise ValueError('Empty problem/solution')
    markers = list(re.finditer(r'(?:Final\s+answer|Answer)\s*:',solution,re.I))
    if len(markers) != 1:
        raise ValueError('Exactly one final-answer marker required')
    boundary = markers[0].start()
    reasoning = solution[:boundary].strip()
    final, source, error = extract_final_answer(solution[boundary:])
    if error or not final or not reasoning:
        raise ValueError(f'Incomplete reasoning/final answer: {error}')
    if any(tag in solution for tag in ['<SOLUTION>','<end_working_out>','<|endoftext|>']):
        raise ValueError('Unexpected embedded training tags')
    return {'problem':problem,'answer':final,'prompt':messages(problem),
            'completion':reasoning+'\n'+END+SOL_START+final+SOL_END+'<|endoftext|>'}


def main():
    p = argparse.ArgumentParser()
    for name in ['model','data','validation','output']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--prepare-only',action='store_true',help='CPU tokenizer validation, no model/GPU load')
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--gradient-accumulation',type=int,default=4)
    p.add_argument('--max-length',type=int,default=2048)
    a = p.parse_args()
    if a.microbatch*a.gradient_accumulation != 16:
        raise ValueError('Effective batch is fixed to 16; use 4x4 or 8x2')
    from common import read_jsonl, save_jsonl, save_json, tokenizer, file_hash
    from prepare_bc import canonical
    source = read_jsonl(a.data)
    train = [convert_messages(row) for row in source]
    validation = read_jsonl(a.validation)
    train_keys = [canonical(row['problem']) for row in train]
    if len(set(train_keys)) != len(train_keys):
        raise ValueError('Duplicate training questions')
    if set(train_keys) & {canonical(row['problem']) for row in validation}:
        raise ValueError('Training/validation overlap')
    out = Path(a.output)
    if out.exists():
        raise ValueError('Use a fresh output directory; never overwrite old experiments')
    out.mkdir(parents=True)
    model_dir = Path(a.model)
    if (model_dir/'adapter_config.json').exists() or (model_dir/'adapter_model.safetensors').exists():
        raise ValueError('--model must point to the original Base checkpoint, not an adapter directory')
    model_config = json.loads((model_dir/'config.json').read_text(encoding='utf-8'))
    if model_config.get('model_type') != 'qwen3':
        raise ValueError('Expected the Qwen3 Base checkpoint')
    t = tokenizer(a.model)
    t.padding_side = 'right'
    def tokenize(rows):
        prepared = []
        for index,row in enumerate(rows):
            prompt = row['prompt']
            if isinstance(prompt,list):
                prompt = t.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
            prefix = t(prompt,add_special_tokens=False)['input_ids']
            completion = t(row['completion'],add_special_tokens=False)['input_ids']
            ids = prefix+completion
            if len(ids)>a.max_length:
                raise ValueError(f'Example {index} needs {len(ids)} tokens; no truncation permitted')
            if not completion or completion[-1] != t.eos_token_id:
                raise ValueError('EOS missing')
            prepared.append({'input_ids':ids,'completion_mask':[0]*len(prefix)+[1]*len(completion)})
        return prepared
    encoded_train,encoded_val = tokenize(train),tokenize(validation)
    save_jsonl(out/'data/train.jsonl',train)
    save_jsonl(out/'data/validation.jsonl',validation)
    save_json(out/'data_audit.json',{'train_questions':len(train),'validation_questions':len(validation),
        'source_sha256':file_hash(a.data),'validation_sha256':file_hash(a.validation),
        'initialization':'Qwen3-4B-Base with newly initialized LoRA; no prior adapter loaded',
        'base_config_sha256':file_hash(model_dir/'config.json'),
        'sequence_length_min':min(len(r['input_ids']) for r in encoded_train+encoded_val),
        'sequence_length_max':max(len(r['input_ids']) for r in encoded_train+encoded_val),
        'no_truncation':True,'no_canonical_train_validation_overlap':True,
        'final_answers_from':'Supplied final-answer fields, not replaced with old reference answers',
        'scope':'Structure/tokenizer checks only, no mathematical proof certification',
        'settings':vars(a)})
    print('DATA_READY',len(train),len(validation),flush=True)
    if a.prepare_only:
        return
    if importlib.metadata.version('trl') != '0.23.1':
        raise ValueError('Pinned environment requires TRL 0.23.1')
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTConfig,SFTTrainer
    from common import load_model,csv_callback
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    free,total = torch.cuda.mem_get_info()
    if free/total < .8:
        raise RuntimeError('GPU memory occupied; inspect existing jobs before launch')
    set_seed(3407)
    model,dtype = load_model(a.model,None,training=True)
    config = SFTConfig(output_dir=str(out),max_length=a.max_length,
        per_device_train_batch_size=a.microbatch,gradient_accumulation_steps=a.gradient_accumulation,
        per_device_eval_batch_size=2,learning_rate=2e-5,num_train_epochs=1,
        warmup_ratio=.1,weight_decay=.01,lr_scheduler_type='linear',optim='adamw_8bit',
        bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        packing=False,completion_only_loss=True,logging_steps=1,
        save_strategy='epoch',eval_strategy='epoch',save_total_limit=1,
        report_to=['tensorboard'],seed=3407,data_seed=3407)
    trainer = SFTTrainer(model=model,args=config,processing_class=t,
        train_dataset=Dataset.from_list(encoded_train),eval_dataset=Dataset.from_list(encoded_val),
        callbacks=[csv_callback(out)])
    for raw,ds in [(encoded_train,trainer.train_dataset),(encoded_val,trainer.eval_dataset)]:
        for expected,actual in zip(raw,ds):
            if actual['input_ids'] != expected['input_ids']:
                raise RuntimeError('Trainer changed prepared tokens')
            labels = trainer.data_collator([actual])['labels'][0].tolist()
            wanted = [token if mask else -100 for token,mask in zip(expected['input_ids'],expected['completion_mask'])]
            if labels != wanted:
                raise RuntimeError('Prompt masking or completion/EOS supervision failure')
        # Check padding specifically: pad ID equals EOS; the real final EOS must survive.
        batch = trainer.data_collator([ds[0],ds[-1]])
        for i in range(2):
            size = len(ds[0 if i==0 else len(ds)-1]['input_ids'])
            if batch['labels'][i,size-1].item()!=t.eos_token_id:
                raise RuntimeError('Real EOS masked as padding')
            if not (batch['labels'][i,size:]==-100).all():
                raise RuntimeError('Padding contributes to loss')
    save_json(out/'sft_config.json',config.to_dict())
    save_json(out/'supervision_audit.json',{'checked':len(train)+len(validation),
        'prompt_masked':True,'completion_and_real_EOS_supervised':True,'padding_masked':True})
    initial_parameters = {n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    before = trainer.evaluate()
    start = time.time()
    result = trainer.train()
    after = trainer.evaluate()
    trainer.save_model(str(out/'adapter'))
    t.save_pretrained(out/'adapter')
    final_hash = file_hash(out/'adapter/adapter_model.safetensors')
    deltas = {n:float((p.detach().float().cpu()-initial_parameters[n]).abs().max())
              for n,p in model.named_parameters() if n in initial_parameters}
    if not deltas or not any(delta>0 for delta in deltas.values()):
        raise RuntimeError('Fresh adapter parameters did not update')
    save_json(out/'trainer_log.json',trainer.state.log_history)
    save_json(out/'completion_summary.json',{'status':'complete','steps':trainer.state.global_step,
        'train_questions':len(train),'epochs':1,'runtime_seconds':time.time()-start,
        'before_validation_loss':before,'after_validation_loss':after,'train_metrics':result.metrics,
        'initialization':'Base plus fresh LoRA; no old adapters',
        'updated_trainable_tensors':sum(delta>0 for delta in deltas.values()),
        'final_adapter_sha256':final_hash})
    import os
    os.sync()
    print('SFT_SAVED',out/'adapter',flush=True)


if __name__=='__main__':main()
