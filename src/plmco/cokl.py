"""Finite-group CoKL surrogate (Eq. 17 of Wang et al., arXiv:2608.01743).

Reference-correct responses are cached once. Current-policy responses are
sampled on the same anchor prompt at each CoKL step. The normalization
correction is retained; omitting it would produce correct-only replay.
"""
from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import torch

from .config import ExperimentConfig
from .math_data import MathCase, config_digest
from .math_model import Rollout, completion_logprobs, sample_group
from .modeling import load_evaluation_model
from .utils import set_seed, write_json


def prepare_reference_buffer(config: ExperimentConfig, tokenizer,
                             cases: list[MathCase], path: Path
                             ) -> tuple[dict[str, list[Rollout]], dict]:
    """Generate reference-correct groups and freeze them before all seeds."""
    import json

    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["config_sha256"] != config_digest(config) or set(payload["groups"]) != {
            case.uid for case in cases
        }:
            raise RuntimeError("CoKL reference buffer does not match frozen experiment")
    else:
        started = time.time()
        set_seed(config.data_seed + 5917)
        model = load_evaluation_model(config, None)
        groups = {}
        generated_tokens = 0
        for index, case in enumerate(cases, 1):
            rollouts = sample_group(model, tokenizer, case, config,
                                    group_size=config.cokl_reference_group_size)
            generated_tokens += sum(item.length for item in rollouts)
            groups[case.uid] = [{"ids": item.ids.tolist(),
                                 "prompt_length": item.prompt_length}
                                for item in rollouts if item.reward == 1.0]
            if index % 4 == 0 or index == len(cases):
                retained = sum(bool(group) for group in groups.values())
                print(f"[CoKL reference] {index}/{len(cases)} prompts; "
                      f"{retained} have a verified-correct response", flush=True)
        payload = {
            "config_sha256": config_digest(config), "reference_model": config.model_name,
            "group_size": config.cokl_reference_group_size,
            "generated_tokens": generated_tokens,
            "preparation_seconds": time.time() - started, "groups": groups,
        }
        write_json(path, payload)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    buffer = {
        uid: [Rollout(torch.tensor(item["ids"], dtype=torch.long),
                      item["prompt_length"], "", 1.0)
              for item in group]
        for uid, group in payload["groups"].items() if group
    }
    if not buffer:
        raise RuntimeError("CoKL found no verified-correct reference responses")
    return buffer, {"prompts_with_correct": len(buffer),
                    "reference_generated_tokens": payload["generated_tokens"],
                    "preparation_seconds": payload.get("preparation_seconds")}


def conditional_weights(current_scores: torch.Tensor, old_scores: torch.Tensor,
                        epsilon: float) -> torch.Tensor:
    """Eq. 15-16: sequence importance weights among verified-correct samples."""
    if current_scores.numel() == 0:
        return current_scores
    lower, upper = math.log1p(-epsilon), math.log1p(epsilon)
    ratios = (current_scores.detach() - old_scores).clamp(lower, upper).exp()
    return ratios / ratios.sum()


def backward_cokl(model, reference_correct: list[Rollout],
                  current_correct: list[Rollout], beta: float,
                  importance_clip: float) -> None:
    """Backpropagate both terms separately to limit peak activation memory."""
    for item in reference_correct:
        log_probability = completion_logprobs(model, item.ids, item.prompt_length).sum()
        (-beta * log_probability / len(reference_correct)).backward()
    if not current_correct:
        return
    with torch.no_grad():
        new_scores = torch.stack([
            completion_logprobs(model, item.ids, item.prompt_length).sum()
            for item in current_correct
        ])
    old_scores = torch.tensor([item.old_logprobs.sum().item()
                               for item in current_correct],
                              device=new_scores.device)
    weights = conditional_weights(new_scores, old_scores, importance_clip)
    for item, weight in zip(current_correct, weights):
        log_probability = completion_logprobs(model, item.ids, item.prompt_length).sum()
        (beta * weight * log_probability).backward()
