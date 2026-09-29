"""Baseline-correct anchors and a per-topic empirical flip-rate constraint."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import ExperimentConfig
from .math_data import MathCase
from .math_model import evaluate_case
from .modeling import load_evaluation_model
from .utils import write_json


def prepare_retention_anchors(config: ExperimentConfig, tokenizer,
                              cases: list[MathCase], path: Path) -> dict[str, list[str]]:
    """Freeze which separate anchor problems the untouched model solves correctly."""
    if not torch.cuda.is_available():
        raise RuntimeError("Anchor scan and training require a CUDA GPU")
    expected = {case.uid for case in cases}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload["predictions"]) != expected:
            raise RuntimeError("Retention anchors do not match frozen splits")
    else:
        model = load_evaluation_model(config, None)
        predictions = {}
        for index, case in enumerate(cases, 1):
            predictions[case.uid] = evaluate_case(
                model, tokenizer, case, config,
                max_new_tokens=config.retention_eval_tokens,
            )
            if index % 16 == 0 or index == len(cases):
                print(f"[retention baseline] {index}/{len(cases)}", flush=True)
        write_json(path, {"predictions": predictions})
        payload = {"predictions": predictions}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected = {
        topic: [case.uid for case in cases
                if case.topic == topic and payload["predictions"][case.uid]["correct"]]
        for topic in config.topics
    }
    for topic, uids in selected.items():
        print(f"[retention baseline] {topic}: {len(uids)} initially correct anchors", flush=True)
    if not any(selected.values()):
        raise RuntimeError("Base model solved no anchor problems; flip-rate constraint is undefined")
    return selected


def update_multiplier(current: float, wrong: int, total: int,
                      config: ExperimentConfig) -> float:
    if total == 0:
        return current
    observed = wrong / total
    return min(config.max_retention_weight, max(
        0.0, current + config.dual_learning_rate * (observed - config.target_flip_rate)
    ))
