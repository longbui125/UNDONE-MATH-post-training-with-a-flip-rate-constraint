"""VS Code entry point 5: concise held-out comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from sys import path as _sys_path

ROOT = Path(__file__).resolve().parent
_sys_path.insert(0, str(ROOT / "src"))

from plmco.config import active_config


def run(*, filename: str = "comparison.csv", show_baseline_timing: bool = True) -> None:
    config = active_config(ROOT)
    comparison = ROOT / "outputs" / config.run_name / filename
    with comparison.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    print(f"Experiment: {config.run_name}")
    print("seed method                  macro  flips  gains  capped(test/train)  capped-flips  RL/anchor/audit tokens  hours")
    for row in rows:
        print(f"{row['seed']:>4} {row['method']:<23} "
              f"{float(row['test_macro_accuracy']):>5.1%} "
              f"{int(float(row['correct_to_wrong'])):>6} "
              f"{int(float(row['wrong_to_correct'])):>6} "
              f"{float(row['test_capped_rate']):>5.1%}/{float(row['train_capped_rate']):>5.1%} "
              f"{int(float(row['correct_to_wrong_capped_either'])):>12} "
              f"{int(float(row['rollout_tokens'])):>7}/"
              f"{int(float(row['retention_eval_tokens'])):>7}/"
              f"{int(float(row['initial_anchor_audit_tokens'])):>7} "
              f"{float(row['train_seconds']) / 3600:>5.2f}")
    constrained = [row for row in rows if row["method"] == "flip_constrained_grpo"]
    if constrained and all(float(row["retention_active_updates"]) == 0 for row in constrained):
        print("Constraint was never active; do not attribute any difference to it.")
    timing = comparison.parent / "run_timing.json"
    if show_baseline_timing and timing.exists():
        hours = json.loads(timing.read_text(encoding="utf-8"))["total_seconds"] / 3600
        print(f"Whole pipeline including preparation and evaluation: {hours:.2f} hours")
    print("Compare macro accuracy and flips together; a lower flip rate alone is insufficient.")


if __name__ == "__main__":
    run()
