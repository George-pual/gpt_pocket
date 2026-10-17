import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from scorer_runtime_guard import load_verified_scorer, load_verified_online_reward


class ScorerLoadingTests(unittest.TestCase):
    def test_real_local_module(self):
        module,record=load_verified_scorer()
        self.assertEqual(record['schema'],'math-verify-final-answer-v4')
        self.assertTrue(record['preflight_passed'])
        self.assertEqual(Path(record['module_path']).resolve(), Path(module.__file__).resolve())

    def test_wrong_expected_hash_stops(self):
        with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):
            load_verified_scorer('0'*64)

    def test_shadowed_python_import_stops(self):
        fake=types.SimpleNamespace(__file__=str(Path(__file__).with_name('unrelated.py')))
        with patch.dict(sys.modules,{'math_verification':fake}):
            with self.assertRaisesRegex(RuntimeError,'Wrong grader module'):
                load_verified_scorer()

    def test_wrong_dependency_version_stops(self):
        with patch('scorer_runtime_guard.importlib.metadata.version',return_value='9.9.9'):
            with self.assertRaisesRegex(RuntimeError,'dependency mismatch'):
                load_verified_scorer()

    def test_real_online_reward_and_wrong_hash(self):
        module, record = load_verified_online_reward()
        self.assertEqual(record['policy'], 'correctness_gated_strict_format_v1')
        self.assertEqual(record['format_weight'], .1)
        self.assertEqual(Path(module.__file__).resolve(), Path(record['module_path']))
        with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
            load_verified_online_reward('0' * 64)

    def test_shadowed_reward_import_stops(self):
        fake = types.SimpleNamespace(__file__=str(Path(__file__).with_name('unrelated.py')))
        with patch.dict(sys.modules, {'online_grpo_reward': fake}):
            with self.assertRaisesRegex(RuntimeError, 'Wrong online reward module'):
                load_verified_online_reward()


if __name__=='__main__':
    unittest.main()
