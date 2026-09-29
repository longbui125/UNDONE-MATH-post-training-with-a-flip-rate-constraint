"""VS Code entry point 4: compare accuracy, actual flips, constraint, and cost."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from sys import path as _sys_path
_sys_path.insert(0, str(ROOT / "src"))
from plmco.config import active_config
from plmco.math_trainer import METHODS


def run() -> None:
    config = active_config(ROOT)
    output = ROOT / "outputs" / config.run_name
    rows = []
    for seed in config.training_seeds:
        initial = json.loads((output / f"seed_{seed}" / "initial" / "evaluation.json").read_text(encoding="utf-8"))
        before = {row["uid"]: row["correct"] for row in initial["test"]["predictions"]}
        before_validation = {row["uid"]: row["correct"]
                             for row in initial["validation"]["predictions"]}
        for method in ("initial", *METHODS):
            folder = output / f"seed_{seed}" / method
            evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
            summary = (json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                       if method != "initial" else {"seconds": 0, "stats": {}})
            after = evaluation["test"]["predictions"]
            validation_after = evaluation["validation"]["predictions"]
            row = {
                "seed": seed, "method": method,
                "test_macro_accuracy": sum(evaluation["test"]["by_topic"].values()) / len(config.topics),
                "test_worst_topic_accuracy": min(evaluation["test"]["by_topic"].values()),
                "validation_macro_accuracy": sum(evaluation["validation"]["by_topic"].values()) / len(config.topics),
                "test_capped_rate": evaluation["test"]["capped_rate"],
                "test_no_box_rate": evaluation["test"]["no_box_rate"],
                "validation_capped_rate": evaluation["validation"]["capped_rate"],
                "correct_to_wrong": sum(before[item["uid"]] and not item["correct"] for item in after),
                "initially_correct_test": sum(before.values()),
                "correct_to_wrong_rate": sum(before[item["uid"]] and not item["correct"] for item in after)
                / max(1, sum(before.values())),
                "validation_correct_to_wrong_rate": sum(
                    before_validation[item["uid"]] and not item["correct"]
                    for item in validation_after
                ) / max(1, sum(before_validation.values())),
                "wrong_to_correct": sum(not before[item["uid"]] and item["correct"] for item in after),
                "rollout_tokens": summary["stats"].get("rollout_tokens", 0),
                "mixed_group_rate": summary["stats"].get("mixed_groups", 0)
                / max(1, summary["stats"].get("sampled_groups", 0)),
                "train_capped_rate": summary["stats"].get("capped_rollouts", 0)
                / max(1, summary["stats"].get("sampled_groups", 0) * config.group_size),
                "optimizer_updates": summary["stats"].get("updates", 0),
                "retention_checks": summary["stats"].get("retention_checks", 0),
                "retention_flip_rate_train": summary["stats"].get("retention_flips", 0)
                / max(1, summary["stats"].get("retention_checks", 0)),
                "retention_active_updates": summary["stats"].get("retention_active_updates", 0),
                "train_seconds": summary["seconds"],
            }
            for topic in config.topics:
                row[f"test_{topic}"] = evaluation["test"]["by_topic"][topic]
                row[f"validation_{topic}"] = evaluation["validation"]["by_topic"][topic]
                baseline_topic = {item["uid"]: item["correct"] for item in initial["test"]["predictions"]
                                  if item["topic"] == topic}
                topic_after = [item for item in after if item["topic"] == topic]
                row[f"test_flip_rate_{topic}"] = sum(
                    baseline_topic[item["uid"]] and not item["correct"] for item in topic_after
                ) / max(1, sum(baseline_topic.values()))
                row[f"initially_correct_test_{topic}"] = sum(baseline_topic.values())
            rows.append(row)
    destination = output / "comparison.csv"
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison: {destination}")


if __name__ == "__main__":
    run()
