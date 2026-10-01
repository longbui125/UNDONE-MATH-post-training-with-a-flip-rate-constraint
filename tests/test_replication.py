"""Checks that paired seed runs reuse frozen data without mixing test IDs."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import compare
from plmco.config import ExperimentConfig
from plmco.math_data import MathCase
from plmco.replication import REPLICATION_METHODS, verify_replication_inputs
from plmco.utils import write_json


class ReplicationTests(unittest.TestCase):
    def test_frozen_seed_plan_and_reused_initial_evaluation(self):
        config = ExperimentConfig.from_json(ROOT / "configs" / "general_math_baseline.json")
        splits = {
            "train": [MathCase(f"{topic}:train:{index}", topic,
                               f"{topic} train {index}", index, str(index))
                      for topic in config.topics for index in range(16)],
            "anchor": [MathCase(f"{topic}:anchor", topic, topic, 1, "1")
                       for topic in config.topics],
            "validation": [MathCase(f"{topic}:validation", topic, topic, 1, "1")
                           for topic in config.topics],
            "test": [MathCase(f"{topic}:test", topic, topic, 1, "1")
                     for topic in config.topics],
        }

        def evaluation(flipped_uid=None):
            result = {}
            for split_name in ("validation", "test"):
                predictions = [{"uid": case.uid, "topic": case.topic,
                                "correct": case.uid != flipped_uid, "capped": False}
                               for case in splits[split_name]]
                result[split_name] = {
                    "predictions": predictions, "capped_rate": 0.0,
                    "no_box_rate": 0.0,
                    "by_topic": {topic: next(row["correct"] for row in predictions
                                             if row["topic"] == topic)
                                 for topic in config.topics},
                }
            return result

        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            output = workspace / "outputs" / config.run_name
            initial = evaluation()
            write_json(output / "seed_42" / "initial" / "evaluation.json", initial)
            write_json(output / "retention_baseline.json", {
                "predictions": {case.uid: {"correct": True} for case in splits["anchor"]}
            })
            for seed in (42, 43, 44):
                for method in REPLICATION_METHODS:
                    folder = output / f"seed_{seed}" / method
                    write_json(folder / "summary.json", {
                        "seed": seed, "steps": config.max_steps,
                        "seconds": 1.0, "stats": {},
                    })
                    write_json(folder / "evaluation.json", evaluation(
                        splits["test"][0].uid if seed == 43 and method == "grpo" else None
                    ))
                    write_json(folder / "adapter" / "adapter_config.json", {})

            selected = verify_replication_inputs(config, splits, output)
            plan = json.loads((output / "seed_replication_plan.json").read_text(encoding="utf-8"))
            self.assertEqual({topic: len(uids) for topic, uids in selected.items()},
                             {topic: 1 for topic in config.topics})
            self.assertEqual(plan["train_topic_counts"], {topic: 16 for topic in config.topics})
            self.assertEqual({len(set(order)) for order in plan["train_question_order"].values()}, {64})
            self.assertEqual(len({tuple(order) for order in plan["train_question_order"].values()}), 3)
            self.assertEqual(verify_replication_inputs(config, splits, output), selected)

            with patch.object(compare, "ROOT", workspace), \
                 patch.object(compare, "active_config", return_value=config):
                compare.run(seeds=(42, 43, 44), methods=REPLICATION_METHODS,
                            initial_seed=42, filename="replication_test.csv")
            with (output / "replication_test.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertEqual(next(row for row in rows if row["seed"] == "43"
                                  and row["method"] == "grpo")["correct_to_wrong"], "1")
            self.assertEqual(next(row for row in rows if row["seed"] == "44"
                                  and row["method"] == "initial")["initially_correct_test"], "4")


if __name__ == "__main__":
    unittest.main()
