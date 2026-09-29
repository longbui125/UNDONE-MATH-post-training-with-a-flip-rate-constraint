"""VS Code entry point 3: run deterministic validation and sealed test evaluation."""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

from plmco.config import active_config
from plmco.math_data import load_splits
from plmco.math_model import evaluate_case
from plmco.math_trainer import METHODS
from plmco.modeling import load_evaluation_model, load_tokenizer
from plmco.utils import write_json


def run() -> None:
    config = active_config(ROOT)
    output = ROOT / "outputs" / config.run_name
    splits = load_splits(config, output / "splits.json")
    jobs = [(seed, method, output / f"seed_{seed}" / method)
            for seed in config.training_seeds for method in METHODS]
    if any(not (folder / "summary.json").exists() for _, _, folder in jobs):
        raise RuntimeError("All training arms must finish before opening the held-out test")
    tokenizer = load_tokenizer(config.model_name)
    for seed in config.training_seeds:
        for method in ("initial", *METHODS):
            folder = output / f"seed_{seed}" / method
            if (folder / "evaluation.json").exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            adapter = None if method == "initial" else str(folder / "adapter")
            model = load_evaluation_model(config, adapter)
            evaluation = {}
            for split_name in ("validation", "test"):
                predictions = []
                cases = splits[split_name]
                for index, case in enumerate(cases, 1):
                    predictions.append(evaluate_case(model, tokenizer, case, config))
                    if index % 8 == 0 or index == len(cases):
                        print(f"[{seed}/{method}/{split_name}] {index}/{len(cases)} "
                              f"correct={sum(row['correct'] for row in predictions)} "
                              f"capped={sum(row['capped'] for row in predictions)}", flush=True)
                by_topic = {
                    topic: sum(row["correct"] for row in predictions if row["topic"] == topic)
                    / sum(row["topic"] == topic for row in predictions)
                    for topic in config.topics
                }
                evaluation[split_name] = {
                    "overall": sum(row["correct"] for row in predictions) / len(predictions),
                    "by_topic": by_topic,
                    "mean_output_tokens": sum(row["tokens"] for row in predictions) / len(predictions),
                    "capped_rate": sum(row["capped"] for row in predictions) / len(predictions),
                    "no_box_rate": sum(not row["has_box"] for row in predictions) / len(predictions),
                    "predictions": predictions,
                }
            write_json(folder / "evaluation.json", evaluation)
            print(f"[{seed}/{method}] test={evaluation['test']['overall']:.3f} "
                  f"capped={evaluation['test']['capped_rate']:.1%} "
                  f"no_box={evaluation['test']['no_box_rate']:.1%} "
                  f"topics={evaluation['test']['by_topic']}", flush=True)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
