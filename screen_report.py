"""VS Code entry point 5: concise held-out comparison."""
from __future__ import annotations

import csv
from pathlib import Path
from sys import path as _sys_path

ROOT = Path(__file__).resolve().parent
_sys_path.insert(0, str(ROOT / "src"))

from plmco.config import active_config


def run() -> None:
    config = active_config(ROOT)
    comparison = ROOT / "outputs" / config.run_name / "comparison.csv"
    with comparison.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    print(f"Experiment: {config.run_name}")
    print("seed method                  macro  flips  gains  active/checks  hours")
    for row in rows:
        active = int(float(row["retention_active_updates"]))
        checks = int(float(row["retention_checks"]))
        print(f"{row['seed']:>4} {row['method']:<23} "
              f"{float(row['test_macro_accuracy']):>5.1%} "
              f"{int(float(row['correct_to_wrong'])):>6} "
              f"{int(float(row['wrong_to_correct'])):>6} "
              f"{active:>6}/{checks:<6} "
              f"{float(row['train_seconds']) / 3600:>5.2f}")
    constrained = [row for row in rows if row["method"] == "flip_constrained_grpo"]
    if constrained and all(float(row["retention_active_updates"]) == 0 for row in constrained):
        print("Constraint was never active; do not attribute any difference to it.")
    print("Compare macro accuracy and flips together; a lower flip rate alone is insufficient.")


if __name__ == "__main__":
    run()
