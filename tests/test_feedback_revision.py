"""Regression tests for false initial flips, delayed feedback and frozen-data reuse."""
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
from plmco.math_data import MathCase, config_digest, load_splits
from plmco.math_trainer import train_method
from plmco.modeling import load_evaluation_model
from plmco.replication import prepare_feedback_revision
from plmco.retention import RecentAnchorFeedback, prepare_retention_anchors
from plmco.utils import write_json


class FeedbackRevisionTests(unittest.TestCase):
    def setUp(self):
        self.legacy = ExperimentConfig.from_json(ROOT / "configs/general_math_baseline.json")
        self.revised = ExperimentConfig.from_json(ROOT / "configs/general_math_feedback_v2.json")

    def test_first_flip_activates_immediately_and_duplicate_ids_have_one_vote(self):
        feedback = RecentAnchorFeedback(replace(self.revised, retention_recent_checks=4))
        weight, wrong, total = feedback.observe("geometry", "a", True, 0.0)
        self.assertGreater(weight, 0)
        self.assertEqual((wrong, total), (1, 1))
        weight, wrong, total = feedback.observe("geometry", "a", False, weight)
        self.assertEqual((wrong, total), (0, 1))
        self.assertAlmostEqual(weight, 0.24)
        for uid in ("b", "c", "d", "e"):
            weight, wrong, total = feedback.observe("geometry", uid, False, weight)
        self.assertEqual(total, 4)
        self.assertNotIn("a", feedback.latest["geometry"])
        self.assertEqual(len(feedback.latest["algebra"]), 0)

    def test_evaluation_matches_training_precision_only_in_new_version(self):
        class Model:
            def eval(self):
                return self
        for config, expected in ((self.legacy, 0), (self.revised, 1)):
            base = Model()
            with patch("plmco.modeling._base_model", return_value=base), \
                 patch("plmco.modeling.prepare_model_for_kbit_training", return_value=base) as prepare:
                self.assertIs(load_evaluation_model(config, None), base)
                self.assertEqual(prepare.call_count, expected)
                if expected:
                    prepare.assert_called_once_with(base, use_gradient_checkpointing=False)

    def test_expanded_revision_excludes_inspected_questions_and_preserves_source(self):
        class Tokenizer:
            eos_token = "<eos>"

            def apply_chat_template(self, messages, **_kwargs):
                return " ".join(message["content"] for message in messages)

            def __call__(self, text, **_kwargs):
                return SimpleNamespace(input_ids=text.split())

        config = replace(self.revised, train_per_topic=1, anchor_per_topic=1,
                         validation_per_topic=1, test_per_topic=1, max_steps=4)

        def source_rows(_dataset, topic, split, **_kwargs):
            return [{"problem": f"  {split}\n{topic}  ", "solution": r"\boxed{1}"}] + [
                {"problem": f"new {split} {topic} {index}", "solution": rf"\boxed{{{index}}}"}
                for index in range(6)]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "configs/general_math_baseline.json", json.loads(
                (ROOT / "configs/general_math_baseline.json").read_text()))
            payload = {"config_sha256": config_digest(self.legacy),
                       "source": {"dataset": "HuggingFaceH4/MATH",
                                  "revision": "9bbe1fc38f097a38e3bdf5bbec8d6feee21318c9"},
                       "splits": {name: [dict(uid=f"{name}:{topic}", topic=topic,
                                                   problem=f"{name} {topic}", answer=1,
                                                   solution=r"\boxed{1}")
                                         for topic in self.legacy.topics]
                                  for name in ("train", "anchor", "validation", "test")}}
            source = root / "outputs" / self.legacy.run_name / "splits.json"
            write_json(source, payload)
            before = source.read_bytes()
            with patch("datasets.load_dataset", side_effect=source_rows):
                splits = prepare_feedback_revision(root, config, Tokenizer())
            destination = root / "outputs" / config.run_name
            self.assertEqual(splits, load_splits(config, destination / "splits.json"))
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse((destination / "retention_baseline.json").exists())
            old_problems = {row["problem"] for rows in payload["splits"].values() for row in rows}
            self.assertFalse(old_problems & {case.problem for cases in splits.values() for case in cases})
            audit = json.loads((destination / "splits.json").read_text())["selection_audit"]
            self.assertTrue(all(row["train"]["previously_inspected_excluded"] == 1 for row in audit.values()))
            self.assertEqual(splits, prepare_feedback_revision(root, config, Tokenizer()))
            with self.assertRaises(ValueError):
                prepare_feedback_revision(root, replace(config, learning_rate=1e-4), Tokenizer())

    def test_new_anchor_scan_rejects_legacy_precision_cache(self):
        case = MathCase("a", "algebra", "p", 1, "1")
        with tempfile.TemporaryDirectory() as temp, \
             patch("plmco.retention.torch.cuda.is_available", return_value=True):
            path = Path(temp) / "retention_baseline.json"
            write_json(path, {"predictions": {"a": {"correct": True}}})
            with self.assertRaisesRegex(RuntimeError, "precision setup"):
                prepare_retention_anchors(self.revised, object(), [case], path)

    def test_runner_trains_and_evaluates_complete_pairs_before_next_seed(self):
        import run_replication
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(run_replication, "ROOT", Path(temp)), \
             patch.object(run_replication.ExperimentConfig, "from_json", return_value=self.revised), \
             patch.object(run_replication, "prepare_feedback_revision", return_value={"anchor": []}), \
             patch.object(run_replication, "load_tokenizer", return_value=object()), \
             patch.object(run_replication, "prepare_retention_anchors", return_value={topic: [] for topic in self.revised.topics}), \
             patch.object(run_replication, "train_method") as train, \
             patch.object(run_replication, "evaluate") as evaluate, \
             patch.object(run_replication, "compare") as compare, \
             patch.object(run_replication, "screen_report"), \
             patch.dict("os.environ"):
            run_replication.run()
            self.assertEqual([(call.args[5], call.args[0]) for call in train.call_args_list],
                             [(42, "grpo"), (42, "flip_constrained_grpo"),
                              (43, "grpo"), (43, "flip_constrained_grpo"),
                              (44, "grpo"), (44, "flip_constrained_grpo")])
            self.assertEqual([call.kwargs["include_initial"] for call in evaluate.call_args_list], [True, False, False])
            self.assertEqual([call.kwargs["seeds"] for call in compare.call_args_list], [(42,), (42, 43), (42, 43, 44)])
            self.assertTrue(all(call.kwargs["initial_seed"] == 42 for call in compare.call_args_list))

    def test_trainer_uses_feedback_on_same_step_and_stops_on_initial_mismatch(self):
        config = replace(self.revised, max_steps=4, policy_epochs=1)
        splits = {name: [MathCase(f"{name}:{topic}", topic, f"{name} {topic}", 1, "1")
                        for topic in config.topics] for name in ("train", "anchor")}
        selected = {c.topic: [c.uid] for c in splits["anchor"]}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def save_pretrained(self, path):
                write_json(path / "adapter_config.json", {})

        tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
        for mismatch in (False, True):
            model = Model()
            calls = 0

            def observe(_model, _tokenizer, case, *_args, **_kwargs):
                nonlocal calls
                calls += 1
                # The fifth check is the first actual step, after the 4-anchor audit.
                return {"uid": case.uid, "correct": not (mismatch or calls == 5), "tokens": 1}

            rollouts = [SimpleNamespace(length=1, text="1", reward=0.0, capped=False,
                                        has_box=True, parsed_answer=0) for _ in range(config.group_size)]
            with tempfile.TemporaryDirectory() as temp, \
                 patch("plmco.math_trainer.load_trainable_model", return_value=model), \
                 patch("plmco.math_trainer.evaluate_case", side_effect=observe), \
                 patch("plmco.math_trainer.sample_group", return_value=rollouts) as sample, \
                 patch("plmco.math_trainer.gold_solution_loss", side_effect=lambda *_a: model.weight.square()), \
                 patch("plmco.math_trainer.torch.cuda.empty_cache"):
                folder = Path(temp)
                if mismatch:
                    with self.assertRaisesRegex(RuntimeError, "before training"):
                        train_method("flip_constrained_grpo", config, tokenizer, splits,
                                     folder, 43, selected)
                    sample.assert_not_called()
                    self.assertFalse((folder / "summary.json").exists())
                else:
                    train_method("flip_constrained_grpo", config, tokenizer, splits,
                                 folder, 43, selected)
                    rows = [json.loads(line) for line in (folder / "train_metrics.jsonl").read_text().splitlines()]
                    self.assertAlmostEqual(rows[0]["retention_weight_used"], 0.27)
                    self.assertEqual(rows[0]["optimizer_updates"], 1)
                    self.assertEqual(rows[0]["retention_feedback_total"], 1)
                    summary = json.loads((folder / "summary.json").read_text())
                    self.assertEqual(summary["stats"]["initial_anchor_audit_tokens"], 4)
                    self.assertEqual(summary["stats"]["dual_adjustments"], 4)


if __name__ == "__main__":
    unittest.main()
