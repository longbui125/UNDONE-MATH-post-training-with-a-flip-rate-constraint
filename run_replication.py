"""One-click runner for the retained seed-42 GRPO/constraint experiment."""
from __future__ import annotations

import time
from pathlib import Path

from prepare_math_data import run as prepare
from train import run as train
from evaluate import run as evaluate
from compare import run as compare
from screen_report import run as screen_report
from plmco.config import active_config
from plmco.utils import append_jsonl, write_json

ROOT = Path(__file__).resolve().parent


def run() -> None:
    started = time.time()
    config = active_config(ROOT)
    output = ROOT / "outputs" / config.run_name
    completed = False
    try:
        prepare()
        train()
        evaluate()
        compare()
        screen_report(show_baseline_timing=False)
        completed = True
    finally:
        # Keep the original completed-run timing; rerendering tables must not overwrite it.
        elapsed = time.time() - started
        append_jsonl(output / "maintenance_sessions.jsonl", {"seconds": elapsed, "completed": completed})
        if completed and not (output / "seed_replication_timing.json").exists():
            write_json(output / "seed_replication_timing.json", {
                "cumulative_seconds": elapsed, "cumulative_hours": elapsed / 3600, "complete": True})


if __name__ == "__main__":
    run()
