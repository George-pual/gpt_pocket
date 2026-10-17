"""Controlled 256-update LR comparison from GRPO 1024, one arm per GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TOOLS = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp/qwen3_hf/runs/E1_grpo1024_lr_ab256_v1')
MODEL = '/root/autodl-tmp/qwen3_hf/model'
INITIAL = Path('/root/autodl-tmp/qwen3_hf/runs/E1_sft_v3_half_math1024_fast_v2/B_components/checkpoint-1024')
REFERENCE = Path('/root/autodl-tmp/qwen3_hf/runs/E1_sft_v3/adapter')
DATA = TOOLS.parent / 'data'
TEST = Path('/root/autodl-tmp/grpo1536_official_eval/data')
INITIAL_HASH = '3a159f8045efc10bc1aae34901abd200751a16936d2798cfcce258101874112c'
REFERENCE_HASH = '5461e88ae8dba6a3fb536e2bd93bdd8e6f3098a3234943eaef28bc0196017b9b'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def command(arm):
    return [sys.executable, '-u', str(TOOLS/'train_reward_scale128_v3.py'),
            '--model', MODEL, '--adapter', str(INITIAL), '--reference-adapter', str(REFERENCE),
            '--expected-adapter-sha256', INITIAL_HASH, '--base-step', '1024',
            '--learning-rate', '2e-6', '--lr-scheduler', 'linear' if arm == 'A_linear' else 'constant',
            '--warmup-steps', '2' if arm == 'A_linear' else '0',
            '--save-training-state', '--data-dir', str(DATA), '--format-weight', '.1',
            '--format-mode', 'components', '--prepared-data', '--microbatch', '2',
            '--generation-batch', '32', '--vllm-memory', '.5', '--steps', '256',
            '--checkpoint-steps', '128,256', '--output', str(ROOT/arm)]


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               MASTER_ADDR='127.0.0.1', MASTER_PORT=str(29560+gpu),
               OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               VLLM_CACHE_ROOT=str(ROOT/f'cache_gpu{gpu}/vllm'),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT/f'cache_gpu{gpu}/inductor'),
               TRITON_CACHE_DIR=str(ROOT/f'cache_gpu{gpu}/triton'))
    return env


def run_arm(arm, gpu):
    (ROOT/f'{arm}.stage.txt').write_text('training')
    with (ROOT/f'{arm}.log').open('w') as log:
        subprocess.run(command(arm), cwd=TOOLS, env=environment(gpu), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    summary = json.loads((ROOT/arm/'completion_summary.json').read_text())
    assert summary['status'] == 'complete' and summary['steps'] == 256
    assert summary['cumulative_steps'] == 1280 and summary['optimizer_reset']
    # Same official data and deterministic single-answer protocol for both arms.
    adapter = ROOT/arm/'checkpoint-256'
    for kind, filename in [('gsm8k', 'gsm8k_test.jsonl'), ('math', 'math_l2_l3_200.jsonl')]:
        (ROOT/f'{arm}.stage.txt').write_text('evaluating_'+kind)
        cmd = [sys.executable, '-u', str(TOOLS/'eval_gsm8k_base_sftv3.py'),
               '--model', MODEL, '--adapter', str(adapter), '--data', str(TEST/filename),
               '--dataset-kind', kind, '--arms', 'grpo1280',
               '--expected-adapter-sha256', digest(adapter/'adapter_model.safetensors'),
               '--output', str(ROOT/f'{arm}_eval_{kind}')]
        with (ROOT/f'{arm}_eval_{kind}.log').open('w') as log:
            subprocess.run(cmd, cwd=TOOLS, env=environment(gpu), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    (ROOT/f'{arm}.stage.txt').write_text('complete')


def main():
    if len(sys.argv) == 3 and sys.argv[1] in ('A_linear', 'B_constant'):
        run_arm(sys.argv[1], int(sys.argv[2]))
        return
    ROOT.mkdir(parents=True, exist_ok=False)
    (ROOT/'stage.txt').write_text('preflight')
    try:
        assert digest(INITIAL/'adapter_model.safetensors') == INITIAL_HASH
        assert digest(REFERENCE/'adapter_model.safetensors') == REFERENCE_HASH
        assert all((TEST/f).is_file() for f in ('gsm8k_test.jsonl', 'math_l2_l3_200.jsonl'))
        from math_verification import SCHEMA
        assert SCHEMA == 'math-verify-final-answer-v2'
        with (ROOT/'reward_tests.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_math_verification_v2',
                            'test_pilot_rewards_v3', 'test_dense_format_reward_v3', 'test_trimmed_pilot'],
                           cwd=TOOLS, stdout=log, stderr=subprocess.STDOUT, check=True)
        save(ROOT/'experiment.json', dict(status='prepared', initial_adapter_sha256=INITIAL_HASH,
             reference_adapter_sha256=REFERENCE_HASH, base_step=1024, additional_steps=256,
             optimizer='reset identically in both arms', seed=3407, data_seed=3407,
             reward='v2 correctness + original format components capped at 0.1',
             warmup_steps={'A_linear': 2, 'B_constant': 0},
             note='A retains original 2-step warmup; B is constant from first update. A decays over the new 256-step horizon.',
             inputs={str(p): digest(p) for p in list(DATA.glob('*.jsonl')) + list(TEST.glob('*.jsonl'))},
             code={p.name: digest(p) for p in TOOLS.glob('*.py')},
             commands={a: command(a) for a in ('A_linear', 'B_constant')}))
        children = []
        for gpu, arm in enumerate(('A_linear', 'B_constant')):
            log = (ROOT/f'{arm}.pipeline.log').open('w')
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), arm, str(gpu)],
                                    cwd=TOOLS, stdout=log, stderr=subprocess.STDOUT)
            children.append((arm, proc, log))
        save(ROOT/'processes.json', {a: p.pid for a, p, _ in children})
        (ROOT/'stage.txt').write_text('training_and_evaluating')
        codes = {a: p.wait() for a, p, _ in children}
        save(ROOT/'exitcodes.json', codes)
        assert all(v == 0 for v in codes.values()), codes
        # Check actual question order, including repetitions per generation group.
        def trace(arm):
            return [(r['step'], r['row_id']) for line in (ROOT/arm/'rollouts.jsonl').read_bytes().splitlines()
                    if line.strip() for r in [json.loads(line)]]
        assert trace('A_linear') == trace('B_constant'), 'Question order mismatch'
        comparisons = {}
        for kind in ('gsm8k', 'math'):
            pair = {a: json.loads((ROOT/f'{a}_eval_{kind}/grpo1280/summary.json').read_text())
                    for a in ('A_linear', 'B_constant')}
            for key in ('questions', 'data_sha256', 'prompt_tokens_sha256', 'sampling'):
                assert pair['A_linear'][key] == pair['B_constant'][key], key
            comparisons[kind] = pair
        save(ROOT/'comparison.json', dict(status='complete', actual_question_order_identical=True,
                                         datasets=comparisons))
        (ROOT/'stage.txt').write_text('complete')
    except Exception as exc:
        save(ROOT/'failure.json', dict(error=repr(exc), time=time.time()))
        (ROOT/'stage.txt').write_text('failed')
        raise


if __name__ == '__main__':
    main()
