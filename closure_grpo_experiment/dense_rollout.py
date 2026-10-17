"""Inference-only decoded NF4 weights; training policy stays quantized."""
import torch
from types import SimpleNamespace
from common import load_model

def build_dense_rollout(model_path, adapter):
    from trl.trainer.grpo_trainer import prepare_peft_model
    model, dtype = load_model(model_path, adapter, training=False)
    prepare_peft_model(model, None, SimpleNamespace(bf16=True, gradient_checkpointing=False))
    model.dequantize()
    # Linear4bit computes in BF16 then casts back to the activation input dtype.
    # Match that contract after replacing its base layers with decoded Linear.
    def projection_dtype(module, inputs, output):
        return output.to(inputs[0].dtype)
    handles = []
    for name, module in model.named_modules():
        if name.endswith('.base_layer') and isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(projection_dtype))
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = True
    model._projection_dtype_handles = handles
    return model, dtype

def sync_adapter(policy, rollout):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    state = get_peft_model_state_dict(policy, save_embedding_layers=False)
    result = set_peft_model_state_dict(rollout, state)
    if result.unexpected_keys:
        raise RuntimeError(f'Unexpected adapter keys: {result.unexpected_keys}')
