"""Matched GRPO and the window-based per-topic flip-rate constraint."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

from .config import ExperimentConfig
from .math_data import MathCase
from .math_model import completion_logprobs, evaluate_case, gold_solution_loss, sample_group
from .modeling import load_trainable_model, trainable_parameters
from .retention import update_multiplier
from .utils import append_jsonl, set_seed, write_json

METHODS = ("grpo", "flip_constrained_grpo")


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


def _backward_policy(model, rollouts, advantages, config: ExperimentConfig) -> None:
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
        if loss is not None:
            (loss / len(rollouts)).backward()


def train_method(method: str, config: ExperimentConfig, tokenizer,
                 splits: dict[str, list[MathCase]], folder: Path, seed: int,
                 retention_ids: dict[str, list[str]]) -> None:
    if method not in METHODS:
        raise ValueError(method)
    if (folder / "summary.json").exists() and (folder / "adapter" / "adapter_config.json").exists():
        print(f"[{seed}/{method}] already complete", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    # An interrupted arm restarts from its seed; partial logs are not resumed.
    (folder / "train_metrics.jsonl").unlink(missing_ok=True)
    (folder / "rollouts.jsonl").unlink(missing_ok=True)
    (folder / "initial_anchor_audit.json").unlink(missing_ok=True)
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
    multipliers = {topic: 0.0 for topic in config.topics}
    window = {topic: {"wrong": 0, "total": 0} for topic in config.topics}
    cumulative = {topic: {"wrong": 0, "total": 0} for topic in config.topics}
    stats = {"sampled_groups": 0, "mixed_groups": 0, "updates": 0,
             "rollout_tokens": 0, "capped_rollouts": 0, "no_box_rollouts": 0,
             "unique_completions": 0,
             "retention_checks": 0, "retention_flips": 0,
             "retention_active_updates": 0, "dual_adjustments": 0,
             "retention_eval_tokens": 0}
    started = time.time()
    if config.align_kbit_evaluation:
        # A zero-initialized LoRA must solve the anchors BEFORE any optimizer update.
        # Reject loader/precision drift instead of calling it training regression.
        audit = []
        for monitored_topic in eligible_topics:
            for anchor in correct_anchors[monitored_topic]:
                observed = evaluate_case(model, tokenizer, anchor, config,
                                         max_new_tokens=config.retention_eval_tokens)
                audit.append(observed)
        write_json(folder / "initial_anchor_audit.json", {
            "predictions": audit, "seconds": time.time() - started,
            "tokens": sum(row["tokens"] for row in audit),
            "mismatches": [row["uid"] for row in audit if not row["correct"]],
        })
        if any(not row["correct"] for row in audit):
            raise RuntimeError("Fresh trainable model fails baseline-correct anchors before "
                               "training. See initial_anchor_audit.json; RL was not started.")
        stats["initial_anchor_audit_tokens"] = sum(row["tokens"] for row in audit)
        # Anchor inference must not shift the random stream used for RL rollouts.
        set_seed(seed)
    for step in range(config.max_steps):
        topic_index = step % len(config.topics)
        topic = config.topics[topic_index]
        case = train_cases[topic][(step // len(config.topics)) % len(train_cases[topic])]
        retention_topic = None
        retention_case = None
        flipped = None
        feedback_wrong = feedback_total = None
        retention_weight_used = 0.0
        retention_gold_loss = None
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
        rollouts = sample_group(model, tokenizer, case, config)
        stats["sampled_groups"] += 1
        stats["rollout_tokens"] += sum(rollout.length for rollout in rollouts)
        unique_completions = len({rollout.text for rollout in rollouts})
        stats["unique_completions"] += unique_completions
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

        for _ in range(config.policy_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if mixed:
                _backward_policy(model, rollouts, advantages, config)
            retention_weight = (multipliers[retention_topic]
                                if method == "flip_constrained_grpo" else 0.0)
            retention_weight_used = retention_weight
            if retention_weight > 0:
                anchor_loss = gold_solution_loss(model, tokenizer, retention_case, config)
                retention_gold_loss = float(anchor_loss.detach())
                (retention_weight * anchor_loss).backward()
                stats["retention_active_updates"] += 1
            if mixed or retention_weight > 0:
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
                optimizer.step()
                stats["updates"] += 1
        if (method == "flip_constrained_grpo" and config.retention_feedback == "window"
                and (step + 1) % config.retention_window_steps == 0):
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
            "unique_completions": unique_completions,
            "rollout_tokens": stats["rollout_tokens"],
            "retention_topic": retention_topic,
            "retention_uid": retention_case.uid if retention_case else None,
            "retention_flipped": flipped,
            "retention_feedback_wrong": feedback_wrong,
            "retention_feedback_total": feedback_total,
            "retention_weight_used": retention_weight_used,
            "retention_gold_loss": retention_gold_loss,
            "optimizer_updates": stats["updates"],
            "retention_multipliers": multipliers.copy(),
            "retention_checks": stats["retention_checks"],
            "retention_flips": stats["retention_flips"],
            "retention_active_updates": stats["retention_active_updates"],
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
        "mean_unique_completions": stats["unique_completions"] / stats["sampled_groups"],
        "retention_cumulative": cumulative,
        "final_multipliers": multipliers,
        "eligible_topics": eligible_topics,
        "align_kbit_evaluation": config.align_kbit_evaluation,
        "retention_feedback": config.retention_feedback,
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()
