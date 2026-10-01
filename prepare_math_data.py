"""VS Code entry point 1: download and freeze disjoint multi-topic MATH splits."""
from __future__ import annotations

import os
import sys
import json
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
    audit = json.loads(path.read_text(encoding="utf-8")).get("selection_audit", {})
    for topic, counts in audit.items():
        print(f"[{topic}] train source={counts['train']['source_rows']} "
              f"integer+prompt eligible={counts['train']['eligible']} "
              f"test source={counts['test']['source_rows']} "
              f"integer+prompt eligible={counts['test']['eligible']} "
              f"anchor gold fits={counts['anchor_gold_fits']}")
    print(f"Frozen splits: {path}")


if __name__ == "__main__":
    run()
