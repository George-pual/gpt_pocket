"""Fail closed if Python resolves a different grader than the pinned artifact."""
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path


def load_verified_online_reward(expected_sha256=None):
    root = Path(__file__).resolve().parent
    contract = json.loads((root / 'online_reward_contract.json').read_text(encoding='utf-8'))
    expected_path = (root / contract['module_filename']).resolve()
    module = importlib.import_module('online_grpo_reward')
    if Path(module.__file__).resolve() != expected_path:
        raise RuntimeError('Wrong online reward module loaded')
    if Path(inspect.getsourcefile(inspect.unwrap(module.OnlineRewardScorer.score_one))).resolve() != expected_path:
        raise RuntimeError('Online reward callable came from a different source file')
    actual_sha = hashlib.sha256(expected_path.read_bytes()).hexdigest()
    if actual_sha != contract['sha256'] or (expected_sha256 and actual_sha != expected_sha256):
        raise RuntimeError('Online reward SHA256 mismatch')
    if module.POLICY != contract['policy'] or module.FORMAT_WEIGHT != contract['format_weight']:
        raise RuntimeError('Online reward policy mismatch')
    return module, dict(module_path=str(expected_path), sha256=actual_sha,
                        policy=module.POLICY, format_weight=module.FORMAT_WEIGHT)


def load_verified_scorer(expected_sha256=None, run_preflight=True):
    root = Path(__file__).resolve().parent
    contract = json.loads((root / 'scorer_v4_contract.json').read_text(encoding='utf-8'))
    expected_path = (root / contract['module_filename']).resolve()
    module = importlib.import_module('math_verification')
    loaded_path = Path(module.__file__).resolve()
    if loaded_path != expected_path:
        raise RuntimeError(f'Wrong grader module loaded: {loaded_path}; expected {expected_path}')
    function_path = Path(inspect.getsourcefile(module.verify_final_answer)).resolve()
    if function_path != expected_path:
        raise RuntimeError('Grader callable came from a different source file')
    actual_sha = hashlib.sha256(loaded_path.read_bytes()).hexdigest()
    if actual_sha != contract['sha256'] or (expected_sha256 and actual_sha != expected_sha256):
        raise RuntimeError('Grader SHA256 mismatch; refusing to score')
    if module.SCHEMA != contract['schema']:
        raise RuntimeError('Grader schema mismatch')
    versions = {package:importlib.metadata.version(package) for package in
                ['math-verify','antlr4-python3-runtime']}
    if versions['math-verify'] != contract['math_verify_version'] or versions['antlr4-python3-runtime'] != contract['antlr4_python3_runtime_version']:
        raise RuntimeError(f'Pinned grader dependency mismatch: {versions}')
    if run_preflight:
        checks = [('<SOLUTION>1/2</SOLUTION>', r'\frac12', None, True),
                  ('<SOLUTION>3</SOLUTION>', '2', None, False),
                  ('<SOLUTION>2 or 3</SOLUTION>', '2', None, False),
                  ('<SOLUTION>2</SOLUTION>actually 3', '2', None, False),
                  ('<SOLUTION>Softball, Kickball, Picnic</SOLUTION>', r'\text{Softball, Kickball, Picnic}',
                   'List the names of the choices from most popular to least popular, separated by commas.', True),
                  ('<SOLUTION>Kickball, Picnic, Softball</SOLUTION>', r'\text{Softball, Kickball, Picnic}',
                   'List the names of the choices from most popular to least popular, separated by commas.', False)]
        for completion,gold,question,correct in checks:
            result = module.verify_final_answer(completion,gold,question=question)
            if result['correct'] != correct or result['schema'] != contract['schema']:
                raise RuntimeError('Grader preflight failed; refusing GPU generation')
    return module, dict(module_path=str(loaded_path), callable_source_path=str(function_path),
                        sha256=actual_sha, schema=module.SCHEMA, dependencies=versions,
                        preflight_passed=run_preflight)
