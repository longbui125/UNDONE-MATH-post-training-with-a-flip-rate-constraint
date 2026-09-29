"""VS Code entry point for the matched 64-step pilot."""
from __future__ import annotations

import os

os.environ["PLMCO_CONFIG"] = "deadline_pilot.json"

from prepare_math_data import run as prepare
from train import run as train
from evaluate import run as evaluate
from compare import run as compare
from screen_report import run as report


if __name__ == "__main__":
    prepare()
    train()
    evaluate()
    compare()
    report()
