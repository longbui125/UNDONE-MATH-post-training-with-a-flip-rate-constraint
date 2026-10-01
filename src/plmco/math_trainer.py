"""Matched GRPO, published KL baselines, and the flip-rate constraint."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

from .config import ExperimentConfig
from .cokl import backward_cokl
from .math_data import MathCase
from .math_model import completion_logprobs, evaluate_case, gold_solution_loss, sample_group
from .modeling import load_trainable_model, trainable_parameters
from .retention import update_multiplier
from .utils import append_jsonl, set_seed, write_json

METHODS = ("grpo", "grpo_reference_kl", "cokl_grpo", "flip_constrained_grpo")


def _ordered_cases(cases: list[MathCase], topics: list[str], seed: int) -> dict[str, list[MathCase]]:
    import random

    result = {topic: [case for case in cases if case.topic == topic] for topic in topics}
    for index, topic in enumerate(topics):
        random.Random(seed + 1009 * index).shuffle(result[topic])
        if not result[topic]:
            raise ValueError(f"No cases for {topic}")
    return result


def _rl_loss(model, rollouts, advantages, config: ExperimentConfig) -> torch.Tensor:
    terms = []
    for rollout, advantage in zip(rollouts, advantages):
        new = completion_logprobs(model, rollout.ids, rollout.prompt_length)
        old = rollout.old_logprobs.to(new.device)
        ratio = (new - old).clamp(-20, 20).exp()
        clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
        token_objective = torch.minimum(ratio * advantage, clipped * advantage)
        terms.append(-token_objective.mean())
    return torch.stack(terms).mean()


def _backward_policy(model, rollouts, advantages, config: ExperimentConfig,
                     use_reference_kl: bool) -> None:
    """Accumulate one rollout at a time so long completions do not stack graphs."""
    for index, rollout in enumerate(rollouts):
        current = completion_logprobs(model, rollout.ids, rollout.prompt_length)
        loss = None
        if advantages is not None:
            old = rollout.old_logprobs.to(current.device)
            ratio = (current - old).clamp(-20, 20).exp()
            clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
            advantage = advantages[index]
            loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
        if use_reference_kl and config.reference_kl_beta > 0:
            reference = rollout.reference_logprobs.to(current.device)
            log_ratio = reference - current
            kl = (log_ratio.clamp(-20, 20).exp() - log_ratio - 1).mean()
            loss = (loss if loss is not None else 0) + config.reference_kl_beta * kl
        if loss is not None:
            (loss / len(rollouts)).backward()


def train_method(method: str, config: ExperimentConfig, tokenizer,
                 splits: dict[str, list[MathCase]], folder: Path, seed: int,
                 retention_ids: dict[str, list[str]],
                 cokl_buffer: dict[str, list] | None = None) -> None:
    if method not in METHODS:
        raise ValueError(method)
    if (folder / "summary.json").exists() and (folder / "adapter" / "adapter_config.json").exists():
        print(f"[{seed}/{method}] already complete", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    # An interrupted arm restarts from its seed; partial logs are not resumed.
    (folder / "train_metrics.jsonl").unlink(missing_ok=True)
    (folder / "rollouts.jsonl").unlink(missing_ok=True)
    set_seed(seed)
    model = load_trainable_model(config)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    train_cases = _ordered_cases(splits["train"], config.topics, seed + 11)
    anchors = _ordered_cases(splits["anchor"], config.topics, seed + 23)
    correct_anchors = {topic: [case for case in anchors[topic]
                               if case.uid in set(retention_ids[topic])]
                       for topic in config.topics}
    eligible_topics = [topic for topic in config.topics if correct_anchors[topic]]
    if method == "flip_constrained_grpo" and not eligible_topics:
        raise RuntimeError("No initially correct retention anchors")
    cokl_cases = {topic: [case for case in anchors[topic]
                          if cokl_buffer is not None and case.uid in cokl_buffer]
                  for topic in config.topics}
    cokl_topics = [topic for topic in config.topics if cokl_cases[topic]]
    if method == "cokl_grpo" and not cokl_topics:
        raise RuntimeError("CoKL has no reference-correct anchor prompts")
    multipliers = {topic: 0.0 for topic in config.topics}
    window = {topic: {"wrong": 0, "total": 0} for topic in config.topics}
    cumulative = {topic: {"wrong": 0, "total": 0} for topic in config.topics}
    stats = {"sampled_groups": 0, "mixed_groups": 0, "updates": 0,
             "rollout_tokens": 0, "capped_rollouts": 0, "no_box_rollouts": 0,
             "retention_checks": 0, "retention_flips": 0,
             "retention_active_updates": 0, "dual_adjustments": 0,
             "cokl_groups": 0, "cokl_current_correct": 0,
             "cokl_generated_tokens": 0, "retention_eval_tokens": 0}
    started = time.time()
    for step in range(config.max_steps):
        topic_index = step % len(config.topics)
        topic = config.topics[topic_index]
        case = train_cases[topic][(step // len(config.topics)) % len(train_cases[topic])]
        retention_topic = None
        retention_case = None
        flipped = None
        if method == "flip_constrained_grpo":
            retention_topic = eligible_topics[step % len(eligible_topics)]
            pool = correct_anchors[retention_topic]
            retention_case = pool[(step // len(eligible_topics)) % len(pool)]
            observed = evaluate_case(
                model, tokenizer, retention_case, config,
                max_new_tokens=config.retention_eval_tokens,
            )
            flipped = int(not observed["correct"])
            window[retention_topic]["wrong"] += flipped
            window[retention_topic]["total"] += 1
            cumulative[retention_topic]["wrong"] += flipped
            cumulative[retention_topic]["total"] += 1
            stats["retention_checks"] += 1
            stats["retention_flips"] += flipped
            stats["retention_eval_tokens"] += observed["tokens"]

        cokl_case = None
        cokl_rollouts = []
        if method == "cokl_grpo":
            cokl_topic = cokl_topics[step % len(cokl_topics)]
            pool = cokl_cases[cokl_topic]
            cokl_case = pool[(step // len(cokl_topics)) % len(pool)]
            cokl_rollouts = sample_group(model, tokenizer, cokl_case, config)
            stats["cokl_groups"] += 1
            stats["cokl_generated_tokens"] += sum(item.length for item in cokl_rollouts)
            current_correct = [item for item in cokl_rollouts if item.reward == 1.0]
            stats["cokl_current_correct"] += len(current_correct)
            with torch.no_grad():
                for item in current_correct:
                    item.old_logprobs = completion_logprobs(
                        model, item.ids, item.prompt_length).detach().cpu()

        rollouts = sample_group(model, tokenizer, case, config)
        stats["sampled_groups"] += 1
        stats["rollout_tokens"] += sum(rollout.length for rollout in rollouts)
        capped = sum(rollout.capped for rollout in rollouts)
        no_box = sum(not rollout.has_box for rollout in rollouts)
        stats["capped_rollouts"] += capped
        stats["no_box_rollouts"] += no_box
        append_jsonl(folder / "rollouts.jsonl", {
            "step": step + 1, "uid": case.uid, "topic": topic, "answer": case.answer,
            "rollouts": [{"reward": item.reward, "prediction": item.parsed_answer,
                          "tokens": item.length, "capped": item.capped,
                          "has_box": item.has_box, "text": item.text}
                         for item in rollouts],
        })
        rewards = torch.tensor([rollout.reward for rollout in rollouts], device=parameters[0].device)
        mixed = bool(rewards.max().item() != rewards.min().item())
        stats["mixed_groups"] += int(mixed)
        if mixed:
            with torch.no_grad():
                for rollout in rollouts:
                    rollout.old_logprobs = completion_logprobs(
                        model, rollout.ids, rollout.prompt_length).detach().cpu()
            centered = rewards - rewards.mean()
            advantages = centered / (rewards.std(unbiased=False) + 1e-8)
        else:
            advantages = None

        if method == "grpo_reference_kl":
            with torch.no_grad(), model.disable_adapter():
                for rollout in rollouts:
                    rollout.reference_logprobs = completion_logprobs(
                        model, rollout.ids, rollout.prompt_length).detach().cpu()

        for _ in range(config.policy_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if mixed or method == "grpo_reference_kl":
                _backward_policy(model, rollouts, advantages, config,
                                 method == "grpo_reference_kl")
            if method == "cokl_grpo" and config.cokl_beta > 0:
                backward_cokl(model, cokl_buffer[cokl_case.uid], current_correct,
                              config.cokl_beta, config.cokl_is_epsilon)
            retention_weight = (multipliers[retention_topic]
                                if method == "flip_constrained_grpo" else 0.0)
            if retention_weight > 0:
                (retention_weight * gold_solution_loss(
                    model, tokenizer, retention_case, config)).backward()
                stats["retention_active_updates"] += 1
            if (mixed or retention_weight > 0
                    or method == "grpo_reference_kl" and config.reference_kl_beta > 0
                    or method == "cokl_grpo" and config.cokl_beta > 0):
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
                optimizer.step()
                stats["updates"] += 1
        if method == "flip_constrained_grpo" and (step + 1) % config.retention_window_steps == 0:
            for monitored_topic in eligible_topics:
                counts = window[monitored_topic]
                if counts["total"]:
                    multipliers[monitored_topic] = update_multiplier(
                        multipliers[monitored_topic], counts["wrong"],
                        counts["total"], config,
                    )
                    stats["dual_adjustments"] += 1
                window[monitored_topic] = {"wrong": 0, "total": 0}
        append_jsonl(folder / "train_metrics.jsonl", {
            "step": step + 1, "topic": topic, "reward_mean": float(rewards.mean()),
            "mixed": mixed, "capped_in_group": capped, "no_box_in_group": no_box,
            "rollout_tokens": stats["rollout_tokens"],
            "retention_topic": retention_topic,
            "retention_uid": retention_case.uid if retention_case else None,
            "retention_flipped": flipped,
            "retention_multipliers": multipliers.copy(),
            "retention_checks": stats["retention_checks"],
            "retention_flips": stats["retention_flips"],
            "retention_active_updates": stats["retention_active_updates"],
            "cokl_generated_tokens": stats["cokl_generated_tokens"],
            "elapsed_seconds": time.time() - started,
        })
        if (step + 1) % config.log_every == 0 or step == 0:
            projected_hours = (time.time() - started) / (step + 1) * config.max_steps / 3600
            print(f"[{seed}/{method}] {step + 1}/{config.max_steps} "
                  f"reward={rewards.mean().item():.2f} "
                  f"mixed={stats['mixed_groups']}/{stats['sampled_groups']} "
                  f"capped={stats['capped_rollouts']}/{stats['sampled_groups'] * config.group_size} "
                  f"retention_flips={stats['retention_flips']}/{stats['retention_checks']} "
                  f"active={stats['retention_active_updates']} "
                  f"projected_arm_hours={projected_hours:.1f}", flush=True)
    adapter = folder / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    write_json(folder / "summary.json", {
        "method": method, "seed": seed, "steps": config.max_steps,
        "seconds": time.time() - started, "stats": stats,
        "retention_cumulative": cumulative,
        "final_multipliers": multipliers,
        "eligible_topics": eligible_topics,
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()
