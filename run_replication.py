"""VS Code: rerun matched GRPO/constraint on seeds 43 and 45 with corrected feedback."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

from compare import run as compare
from evaluate import run as evaluate
from screen_report import run as screen_report
from plmco.config import ExperimentConfig
from plmco.math_trainer import train_method
from plmco.modeling import load_tokenizer
from plmco.replication import REPLICATION_METHODS, prepare_feedback_revision
from plmco.retention import prepare_retention_anchors
from plmco.utils import append_jsonl, write_json


def run() -> None:
    started = time.time()
    # Explicit selection also routes evaluate/compare/report to the same run.
    os.environ["PLMCO_CONFIG"] = "general_math_feedback_v2.json"
    config = ExperimentConfig.from_json(ROOT / "configs" / os.environ["PLMCO_CONFIG"])
    output = ROOT / "outputs" / config.run_name
    splits = prepare_feedback_revision(ROOT, config)
    completed = False
    try:
        tokenizer = load_tokenizer(config.model_name)
        retention_ids = prepare_retention_anchors(
            config, tokenizer, splits["anchor"], output / "retention_baseline.json")
        for seed in config.training_seeds:
            for method in REPLICATION_METHODS:
                train_method(method, config, tokenizer, splits,
                             output / f"seed_{seed}" / method, seed,
                             retention_ids)
            # Complete a pair before starting the next seed: partial results available sooner.
            evaluate(seeds=(seed,), methods=REPLICATION_METHODS,
                     include_initial=seed == config.training_seeds[0])
            completed_seeds = tuple(config.training_seeds[:config.training_seeds.index(seed) + 1])
            compare(seeds=completed_seeds, methods=REPLICATION_METHODS,
                    initial_seed=config.training_seeds[0], filename="comparison.csv")
            screen_report(filename="comparison.csv", show_baseline_timing=False)
        completed = True
    finally:
        sessions_path = output / "seed_replication_sessions.jsonl"
        append_jsonl(sessions_path, {
            "seeds": config.training_seeds, "methods": list(REPLICATION_METHODS),
            "seconds": time.time() - started, "completed": completed,
        })
        sessions = [json.loads(line) for line in sessions_path.read_text(encoding="utf-8").splitlines()]
        total = sum(item["seconds"] for item in sessions)
        write_json(output / "seed_replication_timing.json", {
            "cumulative_seconds": total, "cumulative_hours": total / 3600,
            "complete": completed,
        })
        print(f"Replication elapsed across launches: {total / 3600:.2f} hours", flush=True)


if __name__ == "__main__":
    run()
