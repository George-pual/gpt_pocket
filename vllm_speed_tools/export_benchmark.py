"""Export a shared frozen BF16 checkpoint derived from C's decoded NF4 weights."""
import argparse,json
from pathlib import Path
import torch
from common import file_hash,tokenizer,save_json
from dense_rollout import build_dense_rollout


def main():
    p=argparse.ArgumentParser()
    for n in ['model','adapter','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh export directory required')
    before=file_hash(Path(a.adapter)/'adapter_model.safetensors')
    wrapped,_=build_dense_rollout(a.model,a.adapter)
    for h in wrapped._projection_dtype_handles:h.remove()
    base=wrapped.base_model.model
    count=0
    for name,module in list(base.named_modules()):
        if hasattr(module,'lora_A') and hasattr(module,'get_base_layer'):
            layer=module.get_base_layer()
            assert isinstance(layer,torch.nn.Linear)
            with torch.no_grad():layer.weight.add_(module.get_delta_weight('default').to(layer.weight.dtype))
            parent_name,leaf=name.rsplit('.',1);setattr(base.get_submodule(parent_name),leaf,layer);count+=1
    assert count==252 and not any('lora_' in n or 'base_layer' in n for n,_ in base.named_parameters())
    base.to(torch.bfloat16);base.requires_grad_(False);base.eval()
    for n in ['quantization_config','_pre_quantization_dtype']:
        if hasattr(base.config,n):delattr(base.config,n)
    for n in ['hf_quantizer','quantization_method']:
        if hasattr(base,n):delattr(base,n)
    base.is_quantized=False
    t=tokenizer(a.model)
    base.generation_config.eos_token_id=t.eos_token_id;base.generation_config.pad_token_id=t.pad_token_id
    base.save_pretrained(out,safe_serialization=True,max_shard_size='3GB');t.save_pretrained(out)
    assert file_hash(Path(a.adapter)/'adapter_model.safetensors')==before
    save_json(out/'benchmark_export.json',{'source_adapter_sha256':before,'merged_layers':count,
        'representation':'NF4-decoded C base plus merged C LoRA, all parameters cast to BF16',
        'warning':'A shared benchmark derivative, not bitwise identical to mixed-dtype NF4 training policy',
        'files':{x.name:file_hash(x) for x in out.glob('*.safetensors')}})
    print('EXPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
