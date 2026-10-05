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
from plmco.paired_statistics import paired_bootstrap_summary


def run(*, seeds: tuple[int, ...] | None = None,
        methods: tuple[str, ...] | None = None,
        initial_seed: int | None = None,
        filename: str = "comparison.csv") -> None:
    config = active_config(ROOT)
    output = ROOT / "outputs" / config.run_name
    seeds = tuple(config.training_seeds) if seeds is None else seeds
    methods = METHODS if methods is None else methods
    if not seeds or not methods or len(set(seeds)) != len(seeds) or any(
        method not in METHODS for method in methods
    ):
        raise ValueError("Invalid comparison seeds or methods")
    if Path(filename).name != filename or not filename.endswith(".csv"):
        raise ValueError("Comparison filename must be a CSV in the run folder")
    rows = []
    paired_evaluations = {}
    paired_initial = None
    for seed in seeds:
        baseline_seed = seed if initial_seed is None else initial_seed
        initial = json.loads((output / f"seed_{baseline_seed}" / "initial" / "evaluation.json").read_text(encoding="utf-8"))
        before_rows = {row["uid"]: row for row in initial["test"]["predictions"]}
        before = {uid: row["correct"] for uid, row in before_rows.items()}
        if config.align_kbit_evaluation:
            if paired_initial is not None and before != {row["uid"]: row["correct"] for row in paired_initial}:
                raise RuntimeError("Paired statistics require the same initial predictions across seeds")
            paired_initial = initial["test"]["predictions"]
        before_validation = {row["uid"]: row["correct"]
                             for row in initial["validation"]["predictions"]}
        for method in ("initial", *methods):
            folder = output / f"seed_{seed}" / method
            evaluation = (initial if method == "initial" else json.loads(
                (folder / "evaluation.json").read_text(encoding="utf-8")))
            summary = (json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                       if method != "initial" else {"seconds": 0, "stats": {}})
            after = evaluation["test"]["predictions"]
            if method in ("grpo", "flip_constrained_grpo"):
                paired_evaluations.setdefault(seed, {})[method] = after
            if {item["uid"] for item in after} != set(before):
                raise RuntimeError(f"Seed {seed}, {method}: test IDs differ from the initial model")
            paired_uncapped = [item for item in after
                               if not item["capped"] and not before_rows[item["uid"]]["capped"]]
            validation_after = evaluation["validation"]["predictions"]
            if {item["uid"] for item in validation_after} != set(before_validation):
                raise RuntimeError(f"Seed {seed}, {method}: validation IDs differ from the initial model")
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
                "correct_to_wrong_capped_either": sum(
                    before[item["uid"]] and not item["correct"]
                    and (item["capped"] or before_rows[item["uid"]]["capped"])
                    for item in after
                ),
                "paired_uncapped_test": len(paired_uncapped),
                "initially_correct_paired_uncapped": sum(before[item["uid"]] for item in paired_uncapped),
                "correct_to_wrong_paired_uncapped": sum(
                    before[item["uid"]] and not item["correct"] for item in paired_uncapped
                ),
                "validation_correct_to_wrong_rate": sum(
                    before_validation[item["uid"]] and not item["correct"]
                    for item in validation_after
                ) / max(1, sum(before_validation.values())),
                "wrong_to_correct": sum(not before[item["uid"]] and item["correct"] for item in after),
                "rollout_tokens": summary["stats"].get("rollout_tokens", 0),
                "retention_eval_tokens": summary["stats"].get("retention_eval_tokens", 0),
                "initial_anchor_audit_tokens": summary["stats"].get("initial_anchor_audit_tokens", 0),
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
    destination = output / filename
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison: {destination}")
    if config.align_kbit_evaluation and {"grpo", "flip_constrained_grpo"} <= set(methods):
        statistics = paired_bootstrap_summary(paired_initial, paired_evaluations, config.topics)
        statistics_path = output / "paired_statistics.csv"
        with statistics_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(statistics[0]))
            writer.writeheader()
            writer.writerows(statistics)
        print(f"Paired uncertainty estimates: {statistics_path}")


if __name__ == "__main__":
    run()
