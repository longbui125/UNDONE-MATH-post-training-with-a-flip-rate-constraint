"""VS Code entry point 1: download and freeze disjoint multi-topic MATH splits."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

from plmco.config import active_config
from plmco.math_data import prepare_splits
from plmco.modeling import load_tokenizer


def run() -> None:
    config = active_config(ROOT)
    tokenizer = load_tokenizer(config.model_name)
    path = ROOT / "outputs" / config.run_name / "splits.json"
    splits = prepare_splits(config, tokenizer, path)
    for name, cases in splits.items():
        print(f"{name}: {len(cases)} cases ({', '.join(config.topics)})")
    print(f"Frozen splits: {path}")


if __name__ == "__main__":
    run()
