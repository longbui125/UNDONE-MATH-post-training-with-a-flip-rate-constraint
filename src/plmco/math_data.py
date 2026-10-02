"""Frozen, disjoint MATH splits for four-topic integer-answer training."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ExperimentConfig

DATASET_ID = "HuggingFaceH4/MATH"
REVISION = "9bbe1fc38f097a38e3bdf5bbec8d6feee21318c9"


@dataclass(frozen=True)
class MathCase:
    uid: str
    topic: str
    problem: str
    answer: int
    solution: str


def last_boxed(text: str) -> str | None:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    opening = start + len(r"\boxed{")
    depth = 1
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening:index].strip()
    return None


def integer_answer(solution: str) -> int | None:
    answer = last_boxed(solution)
    if answer is None or not re.fullmatch(r"[+-]?\d+", answer):
        return None
    return int(answer)


def prompt_text(problem: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "Solve the mathematics problem with concise reasoning. Keep the solution short and end with the final integer answer in \\boxed{...}."},
         {"role": "user", "content": problem}],
        tokenize=False, add_generation_prompt=True,
    )


def _eligible(source, topic: str, split: str, tokenizer,
              config: ExperimentConfig) -> tuple[list[MathCase], dict[str, int]]:
    cases = []
    seen = set()
    audit = {"source_rows": 0, "unsupported_answer": 0,
             "duplicate_problem": 0, "prompt_over_limit": 0}
    for index, row in enumerate(source):
        audit["source_rows"] += 1
        problem, solution = row["problem"], row["solution"]
        answer = integer_answer(solution)
        fingerprint = " ".join(problem.split()).casefold()
        if answer is None:
            audit["unsupported_answer"] += 1
            continue
        if fingerprint in seen:
            audit["duplicate_problem"] += 1
            continue
        prompt_length = len(tokenizer(prompt_text(problem, tokenizer), add_special_tokens=False).input_ids)
        if prompt_length > config.max_prompt_tokens:
            audit["prompt_over_limit"] += 1
            continue
        case = MathCase(f"{topic}:{split}:{index}", topic, problem, answer, solution)
        seen.add(fingerprint)
        cases.append(case)
    audit["eligible"] = len(cases)
    return cases, audit


def _solution_fits(case: MathCase, tokenizer, config: ExperimentConfig) -> bool:
    prefix = tokenizer(prompt_text(case.problem, tokenizer), add_special_tokens=False).input_ids
    completion = tokenizer(case.solution + tokenizer.eos_token,
                           add_special_tokens=False).input_ids
    return len(prefix) + len(completion) <= config.max_sft_tokens


def config_digest(config: ExperimentConfig) -> str:
    payload = asdict(config)
    # Preserve the hash of already frozen v1 experiments when using legacy defaults.
    legacy_defaults = {"align_kbit_evaluation": False,
                       "retention_feedback": "window", "retention_recent_checks": 4}
    for key, default in legacy_defaults.items():
        if payload[key] == default:
            payload.pop(key)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_splits(config: ExperimentConfig, tokenizer, path: Path) -> dict[str, list[MathCase]]:
    from datasets import load_dataset

    if path.exists():
        return load_splits(config, path)
    splits: dict[str, list[MathCase]] = {name: [] for name in ("train", "anchor", "validation", "test")}
    data_audit = {}
    for topic_index, topic in enumerate(config.topics):
        source_train = load_dataset(DATASET_ID, topic, split="train", revision=REVISION)
        source_test = load_dataset(DATASET_ID, topic, split="test", revision=REVISION)
        train_pool, train_audit = _eligible(source_train, topic, "train", tokenizer, config)
        test_pool, test_audit = _eligible(source_test, topic, "test", tokenizer, config)
        train_problems = {" ".join(case.problem.split()).casefold() for case in train_pool}
        test_pool = [case for case in test_pool
                     if " ".join(case.problem.split()).casefold() not in train_problems]
        rng = random.Random(config.data_seed + 104729 * topic_index)
        rng.shuffle(train_pool)
        rng.shuffle(test_pool)
        anchor_candidates = [case for case in train_pool
                             if _solution_fits(case, tokenizer, config)]
        anchors = anchor_candidates[:config.anchor_per_topic]
        anchor_ids = {case.uid for case in anchors}
        remaining = [case for case in train_pool if case.uid not in anchor_ids]
        need = config.train_per_topic + config.validation_per_topic
        if len(anchors) < config.anchor_per_topic or len(remaining) < need or len(test_pool) < config.test_per_topic:
            raise RuntimeError(
                f"Too few eligible {topic} cases: anchors={len(anchors)}, "
                f"train+validation={len(remaining)}, test={len(test_pool)}. "
                "Revise the per-topic counts or token limits in a new run config."
            )
        splits["anchor"].extend(anchors)
        splits["train"].extend(remaining[:config.train_per_topic])
        splits["validation"].extend(remaining[config.train_per_topic:need])
        splits["test"].extend(test_pool[:config.test_per_topic])
        data_audit[topic] = {
            "train": train_audit,
            "test": test_audit,
            "anchor_gold_fits": len(anchor_candidates),
            "selected": {"train": config.train_per_topic,
                         "anchor": config.anchor_per_topic,
                         "validation": config.validation_per_topic,
                         "test": config.test_per_topic},
        }
    payload = {
        "config_sha256": config_digest(config),
        "source": {"dataset": DATASET_ID, "revision": REVISION},
        "selection_audit": data_audit,
        "splits": {name: [asdict(case) for case in rows] for name, rows in splits.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits


def load_splits(config: ExperimentConfig, path: Path) -> dict[str, list[MathCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["config_sha256"] != config_digest(config):
        raise RuntimeError("Config changed after data preparation; use a new run_name")
    if payload["source"] != {"dataset": DATASET_ID, "revision": REVISION}:
        raise RuntimeError("Dataset provenance mismatch")
    splits = {name: [MathCase(**row) for row in payload["splits"][name]]
              for name in ("train", "anchor", "validation", "test")}
    identifiers = [case.uid for rows in splits.values() for case in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Dataset splits overlap")
    problems = [" ".join(case.problem.split()).casefold()
                for rows in splits.values() for case in rows]
    if len(problems) != len(set(problems)):
        raise RuntimeError("Duplicate problem appears in two splits")
    return splits
