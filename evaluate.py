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


def run(*, seeds: tuple[int, ...] | None = None,
        methods: tuple[str, ...] | None = None,
        include_initial: bool = True) -> None:
    config = active_config(ROOT)
    output = ROOT / "outputs" / config.run_name
    splits = load_splits(config, output / "splits.json")
    seeds = tuple(config.training_seeds) if seeds is None else seeds
    methods = METHODS if methods is None else methods
    if not seeds or not methods or len(set(seeds)) != len(seeds) or any(
        method not in METHODS for method in methods
    ):
        raise ValueError("Invalid evaluation seeds or methods")
    jobs = [(seed, method, output / f"seed_{seed}" / method)
            for seed in seeds for method in methods]
    if any(not (folder / "summary.json").exists()
           or not (folder / "adapter" / "adapter_config.json").exists()
           for _, _, folder in jobs):
        raise RuntimeError("Selected training arms must finish before evaluation")
    tokenizer = load_tokenizer(config.model_name)
    for seed in seeds:
        selected = (("initial", *methods) if include_initial else methods)
        for method in selected:
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
