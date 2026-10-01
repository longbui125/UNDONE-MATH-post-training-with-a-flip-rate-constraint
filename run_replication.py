"""VS Code: replicate GRPO versus flip-constrained GRPO on seeds 43 and 44."""
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
from plmco.config import active_config
from plmco.math_data import load_splits
from plmco.math_trainer import train_method
from plmco.modeling import load_tokenizer
from plmco.replication import (BASELINE_SEED, COMPARISON_FILE,
                               REPLICATION_METHODS, REPLICATION_SEEDS,
                               verify_replication_inputs)
from plmco.utils import append_jsonl, write_json


def run() -> None:
    started = time.time()
    config = active_config(ROOT)
    output = ROOT / "outputs" / config.run_name
    splits = load_splits(config, output / "splits.json")
    retention_ids = verify_replication_inputs(config, splits, output)
    completed = False
    try:
        tokenizer = load_tokenizer(config.model_name)
        for seed in REPLICATION_SEEDS:
            for method in REPLICATION_METHODS:
                train_method(method, config, tokenizer, splits,
                             output / f"seed_{seed}" / method, seed,
                             retention_ids)
        evaluate(seeds=REPLICATION_SEEDS, methods=REPLICATION_METHODS,
                 include_initial=False)
        compare(seeds=(BASELINE_SEED, *REPLICATION_SEEDS),
                methods=REPLICATION_METHODS,
                initial_seed=BASELINE_SEED, filename=COMPARISON_FILE)
        screen_report(filename=COMPARISON_FILE, show_baseline_timing=False)
        completed = True
    finally:
        sessions_path = output / "seed_replication_sessions.jsonl"
        append_jsonl(sessions_path, {
            "seeds": list(REPLICATION_SEEDS), "methods": list(REPLICATION_METHODS),
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
