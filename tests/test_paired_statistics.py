"""Validate denominators and pairing when the same test is used for several seeds."""
import sys
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plmco.paired_statistics import paired_bootstrap_summary
from plmco.config import ExperimentConfig
from plmco.utils import write_json


class PairedStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.initial = [{"uid": str(i), "topic": "a" if i < 2 else "b", "correct": i == 0}
                        for i in range(4)]

    def test_identical_methods_have_zero_paired_differences_and_intervals(self):
        evaluations = {seed: {"grpo": self.initial,
                              "flip_constrained_grpo": list(reversed(self.initial))}
                       for seed in (42, 43, 44)}
        rows = paired_bootstrap_summary(self.initial, evaluations, ["a", "b"], resamples=200)
        overall = next(row for row in rows if row["scope"] == "all_topics" and row["seed"] == "all")
        self.assertEqual(overall["unique_test_questions"], 4)
        self.assertEqual(overall["initially_correct_unique"], 1)
        for column in ("accuracy_difference_pp", "accuracy_ci_low_pp", "accuracy_ci_high_pp",
                       "flip_rate_difference_pp", "flip_ci_low_pp", "flip_ci_high_pp"):
            self.assertEqual(overall[column], 0.0)

    def test_flip_difference_uses_initially_correct_denominator_and_correct_sign(self):
        incorrect = [{**row, "correct": False} for row in self.initial]
        evaluations = {42: {"grpo": incorrect, "flip_constrained_grpo": self.initial}}
        rows = paired_bootstrap_summary(self.initial, evaluations, ["a", "b"], resamples=200)
        overall = next(row for row in rows if row["scope"] == "all_topics" and row["seed"] == "all")
        self.assertEqual(overall["accuracy_difference_pp"], 25.0)
        self.assertEqual(overall["flip_rate_difference_pp"], -100.0)
        self.assertLessEqual(overall["accuracy_ci_low_pp"], 25.0)
        self.assertGreaterEqual(overall["accuracy_ci_high_pp"], 25.0)

    def test_unpaired_questions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unpaired test IDs"):
            paired_bootstrap_summary(self.initial, {42: {"grpo": self.initial[:-1],
                                                         "flip_constrained_grpo": self.initial}}, ["a", "b"], resamples=200)

    def test_compare_exports_uncertainty_for_new_protocol(self):
        import compare
        from dataclasses import replace
        config = replace(ExperimentConfig.from_json(
            Path(__file__).resolve().parents[1] / "configs/general_math_feedback_v2.json"),
            topics=["a", "b"], training_seeds=[42])
        predictions = [{**row, "capped": False} for row in self.initial]
        split = {"predictions": predictions, "capped_rate": 0., "no_box_rate": 0.,
                 "by_topic": {"a": .5, "b": 0.}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "outputs" / config.run_name
            for method in ("initial", "grpo", "flip_constrained_grpo"):
                folder = output / "seed_42" / method
                write_json(folder / "evaluation.json", {"validation": split, "test": split})
                write_json(folder / "summary.json", {"seconds": 1., "stats": {}})
            with patch.object(compare, "ROOT", root), patch.object(compare, "active_config", return_value=config):
                compare.run(seeds=(42,), methods=("grpo", "flip_constrained_grpo"), initial_seed=42)
            with (output / "paired_statistics.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            overall = next(row for row in rows if row["scope"] == "all_topics" and row["seed"] == "all")
            self.assertEqual(overall["unique_test_questions"], "4")
            self.assertEqual(float(overall["accuracy_difference_pp"]), 0.)


if __name__ == "__main__":
    unittest.main()
