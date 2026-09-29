"""VS Code entry point 2: train matched GRPO and retention-constrained arms."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

from plmco.config import active_config
from plmco.math_data import load_splits
from plmco.math_trainer import METHODS, train_method
from plmco.modeling import load_tokenizer
from plmco.retention import prepare_retention_anchors


def run() -> None:
    config = active_config(ROOT)
    folder = ROOT / "outputs" / config.run_name
    splits_path = folder / "splits.json"
    if not splits_path.exists():
        raise RuntimeError("Run prepare_math_data.py first")
    splits = load_splits(config, splits_path)
    tokenizer = load_tokenizer(config.model_name)
    retention_ids = prepare_retention_anchors(
        config, tokenizer, splits["anchor"], folder / "retention_baseline.json")
    for seed in config.training_seeds:
        for method in METHODS:
            train_method(method, config, tokenizer, splits,
                         folder / f"seed_{seed}" / method, seed, retention_ids)
    print(f"Training complete: {folder}")


if __name__ == "__main__":
    run()
