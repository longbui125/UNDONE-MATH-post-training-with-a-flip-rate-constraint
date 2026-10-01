"""Validate the frozen seed-42 experiment before paired seed replications."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .config import ExperimentConfig
from .math_data import MathCase, config_digest
from .math_trainer import _ordered_cases
from .utils import write_json


BASELINE_SEED = 42
REPLICATION_SEEDS = (43, 44)
REPLICATION_METHODS = ("grpo", "flip_constrained_grpo")
COMPARISON_FILE = "seed_replication_comparison.csv"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required seed-42 artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_replication_inputs(config: ExperimentConfig,
                              splits: dict[str, list[MathCase]],
                              output: Path) -> dict[str, list[str]]:
    """Keep data, starting model, reward and evaluation fixed across seeds."""
    if config.training_seeds != [BASELINE_SEED]:
        raise ValueError("The frozen source config must contain only seed 42")
    if BASELINE_SEED in REPLICATION_SEEDS or len(set(REPLICATION_SEEDS)) != len(REPLICATION_SEEDS):
        raise ValueError("Replication seeds must be unique and different from seed 42")

    initial = _read_json(output / "seed_42" / "initial" / "evaluation.json")
    for split_name in ("validation", "test"):
        expected = {case.uid for case in splits[split_name]}
        observed = {row["uid"] for row in initial[split_name]["predictions"]}
        if observed != expected:
            raise RuntimeError(f"Seed-42 initial {split_name} does not match frozen splits")
    for method in REPLICATION_METHODS:
        folder = output / "seed_42" / method
        summary = _read_json(folder / "summary.json")
        evaluation = _read_json(folder / "evaluation.json")
        for split_name in ("validation", "test"):
            expected = {case.uid for case in splits[split_name]}
            observed = {row["uid"] for row in evaluation[split_name]["predictions"]}
            if observed != expected:
                raise RuntimeError(f"Seed-42 {method} {split_name} does not match frozen splits")
        if summary["seed"] != BASELINE_SEED or summary["steps"] != config.max_steps:
            raise RuntimeError(f"Seed-42 {method} is not the expected completed baseline")
        if not (folder / "adapter" / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Seed-42 {method} adapter is missing")

    baseline = _read_json(output / "retention_baseline.json")
    anchor_ids = {case.uid for case in splits["anchor"]}
    if set(baseline["predictions"]) != anchor_ids:
        raise RuntimeError("Baseline-correct anchors do not match frozen splits")
    selected = {
        topic: [case.uid for case in splits["anchor"]
                if case.topic == topic and baseline["predictions"][case.uid]["correct"]]
        for topic in config.topics
    }
    if not any(selected.values()):
        raise RuntimeError("No baseline-correct anchors for the flip constraint")

    orders = {}
    for seed in (BASELINE_SEED, *REPLICATION_SEEDS):
        cases_by_topic = _ordered_cases(splits["train"], config.topics, seed + 11)
        schedule = []
        for step in range(config.max_steps):
            topic = config.topics[step % len(config.topics)]
            pool = cases_by_topic[topic]
            schedule.append(pool[(step // len(config.topics)) % len(pool)].uid)
        if len(set(schedule)) != config.max_steps:
            raise RuntimeError(f"Seed {seed} repeats train questions within this run")
        orders[str(seed)] = schedule
    if len({tuple(order) for order in orders.values()}) != len(orders):
        raise RuntimeError("Training orders unexpectedly coincide across seeds")
    counts = Counter(case.topic for case in splits["train"])
    if len(set(counts.values())) != 1 or set(counts) != set(config.topics):
        raise RuntimeError("The four training topics are not balanced")

    plan = {
        "source_run": config.run_name,
        "source_config_sha256": config_digest(config),
        "source_seed": BASELINE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "methods": list(REPLICATION_METHODS),
        "steps_per_method": config.max_steps,
        "rollouts_per_step": config.group_size,
        "train_topic_counts": dict(counts),
        "train_question_order": orders,
        "initial_evaluation_reused_from_seed": BASELINE_SEED,
    }
    plan_path = output / "seed_replication_plan.json"
    if plan_path.exists():
        if _read_json(plan_path) != plan:
            raise RuntimeError("Stored replication plan differs from the frozen experiment")
    else:
        write_json(plan_path, plan)
    return selected
