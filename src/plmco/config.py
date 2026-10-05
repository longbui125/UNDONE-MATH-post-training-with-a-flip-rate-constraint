"""Configuration for the multi-topic MATH retention experiment."""
from __future__ import annotations

import json
import os
from dataclasses import MISSING, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    run_name: str
    data_seed: int
    training_seeds: list[int]
    topics: list[str]
    train_per_topic: int
    anchor_per_topic: int
    validation_per_topic: int
    test_per_topic: int
    max_steps: int
    group_size: int
    max_prompt_tokens: int
    max_completion_tokens: int
    max_sft_tokens: int
    eval_max_new_tokens: int
    temperature: float
    top_p: float
    learning_rate: float
    target_flip_rate: float
    dual_learning_rate: float
    max_retention_weight: float
    retention_window_steps: int
    retention_eval_tokens: int
    clip_epsilon: float
    policy_epochs: int
    max_grad_norm: float
    dtype: str
    gradient_checkpointing: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: list[str]
    log_every: int
    quantization_4bit: bool = False
    # Unused historical fields remain only to preserve the completed run's config hash.
    reference_kl_beta: float = 0.04
    cokl_beta: float = 0.001
    cokl_reference_group_size: int = 4
    cokl_is_epsilon: float = 0.2
    align_kbit_evaluation: bool = False
    retention_feedback: str = "window"
    retention_recent_checks: int = 4

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(cls)}
        required = {field.name for field in fields(cls)
                    if field.default is MISSING and field.default_factory is MISSING}
        if required - set(payload) or set(payload) - allowed:
            raise ValueError(f"Config missing={sorted(required - set(payload))}, extra={sorted(set(payload) - allowed)}")
        result = cls(**payload)
        if not result.topics or len(set(result.topics)) != len(result.topics):
            raise ValueError("topics must be nonempty and unique")
        if not result.training_seeds or len(set(result.training_seeds)) != len(result.training_seeds):
            raise ValueError("training_seeds must be nonempty and unique")
        if result.group_size < 2 or result.max_steps < 1 or result.policy_epochs < 1:
            raise ValueError("Invalid group_size, max_steps, or policy_epochs")
        if min(result.train_per_topic, result.anchor_per_topic,
               result.validation_per_topic, result.test_per_topic) < 1:
            raise ValueError("Each split needs at least one problem per topic")
        if not 0 < result.clip_epsilon < 1:
            raise ValueError("Invalid clip_epsilon")
        if result.reference_kl_beta < 0 or result.cokl_beta < 0:
            raise ValueError("KL coefficients must be nonnegative")
        if result.cokl_reference_group_size < 2 or not 0 < result.cokl_is_epsilon < 1:
            raise ValueError("Invalid CoKL reference group or importance clip")
        if not isinstance(result.quantization_4bit, bool):
            raise ValueError("quantization_4bit must be a boolean")
        if result.max_sft_tokens < result.max_prompt_tokens or min(
            result.max_completion_tokens, result.eval_max_new_tokens,
            result.retention_eval_tokens) < 1:
            raise ValueError("Invalid token limits")
        if not 0 <= result.target_flip_rate < 1:
            raise ValueError("target_flip_rate must be in [0, 1)")
        if result.dual_learning_rate <= 0 or result.max_retention_weight <= 0:
            raise ValueError("Dual hyperparameters must be positive")
        if result.retention_window_steps < len(result.topics):
            raise ValueError("retention_window_steps must cover every topic")
        if result.retention_eval_tokens < 1:
            raise ValueError("retention_eval_tokens must be positive")
        if not isinstance(result.align_kbit_evaluation, bool):
            raise ValueError("align_kbit_evaluation must be a boolean")
        if result.retention_feedback != "window":
            raise ValueError("Unknown retention_feedback")
        if result.retention_recent_checks < 1:
            raise ValueError("retention_recent_checks must be positive")
        return result


def active_config(root: Path) -> ExperimentConfig:
    """Select an immutable config for an experiment run."""
    name = os.environ.get("PLMCO_CONFIG", "math_retention.json")
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("PLMCO_CONFIG must name a JSON file in configs/")
    return ExperimentConfig.from_json(root / "configs" / name)
