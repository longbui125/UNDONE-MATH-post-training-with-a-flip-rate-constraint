"""One-click VS Code run of the general-model, long-completion experiment."""
import time
from pathlib import Path

from prepare_math_data import run as prepare
from train import run as train
from evaluate import run as evaluate
from compare import run as compare
from screen_report import run as screen_report
from plmco.config import active_config
from plmco.utils import write_json


if __name__ == "__main__":
    started = time.time()
    prepare()
    train()
    evaluate()
    compare()
    config = active_config(Path(__file__).resolve().parent)
    write_json(Path(__file__).resolve().parent / "outputs" / config.run_name / "run_timing.json",
               {"total_seconds": time.time() - started})
    screen_report()
