import csv
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

START = "<start_working_out>"
END = "<end_working_out>"
SOL_START = "<SOLUTION>"
SOL_END = "</SOLUTION>"
SYSTEM = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    f"Place it between {START} and {END}.\n"
    f"Then, provide your solution between {SOL_START}{SOL_END}"
)
# Original single-turn template; no additional special tokens.
TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}{{ " + repr(SYSTEM) + " + eos_token }}"
    "{% set loop_messages = messages %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ " + repr(START) + " }}{% endif %}"
)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]

def save_json(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def save_jsonl(file, rows):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(file):
    with Path(file).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def messages(problem):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": str(problem)}]

def tokenizer(model):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(model, local_files_only=True)
    eid = t.convert_tokens_to_ids("<|endoftext|>")
    if eid is None or t.convert_ids_to_tokens(eid) != "<|endoftext|>":
        raise ValueError("Expected Qwen endoftext token not found")
    t.eos_token = "<|endoftext|>"
    t.pad_token = t.eos_token
    t.chat_template = TEMPLATE
    return t

def load_model(model_path, adapter=None, training=False):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        device_map={"": 0}, attn_implementation="sdpa")
    if training:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter:
        model = PeftModel.from_pretrained(
            model, adapter, is_trainable=training, local_files_only=True)
    elif training:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=32, lora_alpha=64,
            lora_dropout=0.0, bias="none", target_modules=TARGETS))
    model.config.use_cache = not training
    return model, dtype

def extract_answer(text):
    found = re.findall(r"<SOLUTION>(.*?)</SOLUTION>", text, re.S)
    return found[0].strip() if len(found) == 1 else None

def numeric(value):
    if value is None:
        return None
    s = str(value).strip()
    if len(s) > 128 or not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s):
        return None
    try:
        result = Decimal(s)
        return result if result.is_finite() else None
    except InvalidOperation:
        return None

def numeric_equal(prediction, answer):
    p, a = numeric(prediction), numeric(answer)
    return p is not None and a is not None and p == a

def file_hash(file):
    h = hashlib.sha256()
    with Path(file).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def runtime_manifest(args, output, data=None):
    import importlib.metadata
    import torch
    versions = {}
    for name in ["torch", "transformers", "trl", "peft", "accelerate",
                 "bitsandbytes", "datasets", "huggingface-hub", "math-verify",
                 "latex2sympy2_extended", "antlr4-python3-runtime", "sympy"]:
        versions[name] = importlib.metadata.version(name)
    save_json(Path(output) / "run_manifest.json", {
        "arguments": vars(args), "versions": versions,
        "gpu": torch.cuda.get_device_name(0),
        "data_sha256": file_hash(data) if data else None, "template": TEMPLATE})

def csv_callback(output):
    from transformers import TrainerCallback
    class Logger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not state.is_world_process_zero:
                return
            p = Path(output) / "metrics.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            exists = p.exists()
            with p.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(["step", "epoch", "metric", "value"])
                for k, v in logs.items():
                    w.writerow([state.global_step, state.epoch, k, v])
    return Logger()
