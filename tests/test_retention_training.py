"""Regression tests for the retained window feedback and pre-training precision audit."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from plmco.config import ExperimentConfig
from plmco.math_data import MathCase
from plmco.math_trainer import train_method
from plmco.modeling import load_evaluation_model
from plmco.utils import write_json


class RetentionTrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig.from_json(ROOT / "configs/math_retention.json")

    def test_evaluation_matches_training_precision(self):
        class Model:
            def eval(self):
                return self
        base = Model()
        with patch("plmco.modeling._base_model", return_value=base), \
             patch("plmco.modeling.prepare_model_for_kbit_training", return_value=base) as prepare:
            self.assertIs(load_evaluation_model(self.config, None), base)
            prepare.assert_called_once_with(base, use_gradient_checkpointing=False)

    def test_initial_audit_failure_stops_before_sampling(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
        splits = {name: [MathCase(f"{name}:{t}", t, f"{name} {t}", 1, "1")
                        for t in self.config.topics] for name in ("train", "anchor")}
        selected = {c.topic: [c.uid] for c in splits["anchor"]}
        with tempfile.TemporaryDirectory() as temp, \
             patch("plmco.math_trainer.load_trainable_model", return_value=Model()), \
             patch("plmco.math_trainer.evaluate_case", side_effect=lambda _m,_t,c,*a,**k: {"uid": c.uid,"correct": False,"tokens":1}), \
             patch("plmco.math_trainer.sample_group") as sample:
            folder = Path(temp)
            with self.assertRaisesRegex(RuntimeError,"before training"):
                train_method("flip_constrained_grpo", self.config, object(), splits, folder, 42, selected)
            sample.assert_not_called()
            self.assertFalse((folder / "summary.json").exists())

    def test_runner_preserves_stage_order(self):
        import run_replication
        stages = []
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(run_replication,"ROOT",Path(temp)), \
             patch.object(run_replication,"active_config",return_value=self.config), \
             patch.object(run_replication,"prepare",side_effect=lambda:stages.append("prepare")), \
             patch.object(run_replication,"train",side_effect=lambda:stages.append("train")), \
             patch.object(run_replication,"evaluate",side_effect=lambda:stages.append("evaluate")), \
             patch.object(run_replication,"compare",side_effect=lambda:stages.append("compare")), \
             patch.object(run_replication,"screen_report",side_effect=lambda **k:stages.append("report")):
            run_replication.run()
            self.assertEqual(stages,["prepare","train","evaluate","compare","report"])

    def test_old_feedback_activates_after_32_not_on_first_flip(self):
        config = replace(self.config, max_steps=33, policy_epochs=1)
        splits = {name: [MathCase(f"{name}:{t}", t, f"{name} {t}", 1, "1")
                        for t in config.topics] for name in ("train", "anchor")}
        selected = {c.topic: [c.uid] for c in splits["anchor"]}
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
            def save_pretrained(self, path):
                write_json(path / "adapter_config.json", {})
        model = Model()
        calls = 0
        def observe(_model, _tokenizer, case, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            return {"uid": case.uid, "correct": calls <= 4, "tokens": 1}
        rollouts = [SimpleNamespace(length=1, text="1", reward=0., capped=False, has_box=True, parsed_answer=0)
                    for _ in range(config.group_size)]
        with tempfile.TemporaryDirectory() as temp, \
             patch("plmco.math_trainer.load_trainable_model", return_value=model), \
             patch("plmco.math_trainer.evaluate_case", side_effect=observe), \
             patch("plmco.math_trainer.sample_group", return_value=rollouts), \
             patch("plmco.math_trainer.gold_solution_loss", side_effect=lambda *_a: model.weight.square()), \
             patch("plmco.math_trainer.torch.cuda.empty_cache"):
            folder = Path(temp)
            train_method("flip_constrained_grpo", config, SimpleNamespace(save_pretrained=lambda p: None),
                         splits, folder, 42, selected)
            rows = [json.loads(line) for line in (folder / "train_metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["retention_weight_used"] == 0 for row in rows[:32]))
            self.assertAlmostEqual(rows[31]["retention_multipliers"]["algebra"], .27)
            self.assertAlmostEqual(rows[32]["retention_weight_used"], .27)
            self.assertEqual(rows[32]["optimizer_updates"], 1)


if __name__ == "__main__":
    unittest.main()
