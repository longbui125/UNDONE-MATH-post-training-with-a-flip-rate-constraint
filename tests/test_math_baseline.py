"""Checks for the current multi-topic math baseline."""
import unittest
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plmco.config import ExperimentConfig
from plmco.math_data import (MathCase, _eligible, integer_answer,
                             last_boxed, load_splits, prepare_splits)
from plmco.math_trainer import METHODS, _rl_loss
from plmco.retention import prepare_retention_anchors, update_multiplier


class MathBaselineTests(unittest.TestCase):
    def test_grpo_loss_uses_mean_token_objective(self):
        config = replace(ExperimentConfig.from_json(
            Path(__file__).resolve().parents[1] / "configs" / "math_retention.json"),
            max_completion_tokens=4)
        rollout = SimpleNamespace(ids=None, prompt_length=0,
                                  old_logprobs=torch.zeros(2))
        with patch("plmco.math_trainer.completion_logprobs",
                   return_value=torch.zeros(2)):
            ordinary = _rl_loss(None, [rollout], [torch.tensor(1.0)], config)
        self.assertAlmostEqual(ordinary.item(), -1.0)

    def test_matched_methods_and_equal_generation_limits(self):
        config = ExperimentConfig.from_json(
            Path(__file__).resolve().parents[1] / "configs" / "math_retention.json")
        self.assertEqual(METHODS, ("grpo", "flip_constrained_grpo"))
        self.assertEqual(config.model_name, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(config.max_completion_tokens, config.eval_max_new_tokens)
        self.assertEqual(config.max_completion_tokens, config.retention_eval_tokens)
        self.assertEqual(config.max_completion_tokens, 1024)

    def test_long_gold_solution_does_not_remove_test_question(self):
        class Tokenizer:
            eos_token = "<eos>"

            def apply_chat_template(self, messages, **_kwargs):
                return " ".join(message["content"] for message in messages)

            def __call__(self, text, **_kwargs):
                return SimpleNamespace(input_ids=text.split())

        config = replace(ExperimentConfig.from_json(
            Path(__file__).resolve().parents[1] / "configs" / "math_retention.json"),
            max_sft_tokens=80)
        solution = ("Reasoning " * 150) + r"\boxed{7}"
        cases, audit = _eligible([{"problem": "Find seven", "solution": solution}],
                                 "algebra", "test", Tokenizer(), config)
        self.assertEqual(len(cases), 1)
        self.assertEqual(audit["eligible"], 1)

    def test_dual_weight_responds_to_observed_flip_rate(self):
        config = ExperimentConfig.from_json(
            Path(__file__).resolve().parents[1] / "configs" / "math_retention.json")
        self.assertGreater(update_multiplier(0.0, 3, 8, config), 0.0)
        self.assertEqual(update_multiplier(0.0, 0, 8, config), 0.0)
        self.assertLess(update_multiplier(0.5, 0, 8, config), 0.5)
        self.assertEqual(update_multiplier(0.5, 0, 0, config), 0.5)

    def test_only_baseline_correct_anchors_are_selected(self):
        config = ExperimentConfig.from_json(
            Path(__file__).resolve().parents[1] / "configs" / "math_retention.json")
        cases = [MathCase("a", "algebra", "p1", 1, "\\boxed{1}"),
                 MathCase("b", "algebra", "p2", 2, "\\boxed{2}")]
        fake_model = object()
        with tempfile.TemporaryDirectory() as temp, \
             patch("plmco.retention.torch.cuda.is_available", return_value=True), \
             patch("plmco.retention.load_evaluation_model", return_value=fake_model), \
             patch("plmco.retention.evaluate_case",
                   side_effect=[{"correct": True}, {"correct": False}]) as evaluator:
            path = Path(temp) / "baseline.json"
            selected = prepare_retention_anchors(config, object(), cases, path)
            self.assertEqual(selected["algebra"], ["a"])
            self.assertEqual(selected["geometry"], [])
            self.assertEqual(evaluator.call_count, 2)
            self.assertEqual(prepare_retention_anchors(config, object(), cases, path), selected)
            self.assertEqual(evaluator.call_count, 2)

    def test_integer_verifier_uses_final_box(self):
        self.assertEqual(last_boxed(r"Earlier \boxed{3}, final \boxed{-12}"), "-12")
        self.assertEqual(integer_answer(r"Reasoning... \boxed{-12}"), -12)
        self.assertIsNone(integer_answer(r"\boxed{\frac{1}{2}}"))
        self.assertIsNone(integer_answer("Answer: 12"))

    def test_frozen_splits_are_disjoint_and_config_locked(self):
        class Tokenizer:
            eos_token = "<eos>"

            def apply_chat_template(self, messages, **_kwargs):
                return " ".join(message["content"] for message in messages)

            def __call__(self, text, **_kwargs):
                return SimpleNamespace(input_ids=text.split())

        config = replace(
            ExperimentConfig.from_json(Path(__file__).resolve().parents[1] / "configs" / "math_retention.json"),
            train_per_topic=1, anchor_per_topic=1, validation_per_topic=1,
            test_per_topic=1,
        )

        def source(_name, topic, split, **_kwargs):
            return [{"problem": f"{split} {topic} problem {index}",
                     "solution": rf"Answer is \boxed{{{index}}}"}
                    for index in range(6)]

        with tempfile.TemporaryDirectory() as temp, patch("datasets.load_dataset", side_effect=source):
            path = Path(temp) / "splits.json"
            splits = prepare_splits(config, Tokenizer(), path)
            self.assertEqual({name: len(cases) for name, cases in splits.items()},
                             {name: len(config.topics) for name in splits})
            self.assertEqual(splits, load_splits(config, path))
            with self.assertRaises(RuntimeError):
                load_splits(replace(config, data_seed=999), path)


if __name__ == "__main__":
    unittest.main()
