"""Paired, topic-stratified bootstrap; repeated questions are not independent seeds."""
from __future__ import annotations

import numpy as np


def paired_bootstrap_summary(initial: list[dict], evaluations: dict[int, dict[str, list[dict]]],
                             topics: list[str], *, resamples: int = 2000,
                             random_seed: int = 20261002) -> list[dict]:
    """Compare constraint minus GRPO, retaining question/seed pairing.

    Overall intervals resample both training seeds and shared questions by topic.
    Per-seed intervals resample questions only. With three seeds these are
    exploratory percentile intervals, not a guarantee of statistical power.
    """
    if resamples < 100 or not evaluations:
        raise ValueError("Need evaluations and at least 100 bootstrap resamples")
    uids = [row["uid"] for row in initial]
    if len(uids) != len(set(uids)):
        raise ValueError("Duplicate initial test IDs")
    before = np.array([row["correct"] for row in initial], dtype=bool)
    topic_ids = {topic: np.array([i for i, row in enumerate(initial) if row["topic"] == topic])
                 for topic in topics}
    if any(len(ids) == 0 for ids in topic_ids.values()):
        raise ValueError("Every requested topic needs test predictions")
    seeds = sorted(evaluations)
    outcomes = {}
    for method in ("grpo", "flip_constrained_grpo"):
        rows = []
        for seed in seeds:
            predictions = evaluations[seed][method]
            lookup = {row["uid"]: bool(row["correct"]) for row in predictions}
            if len(predictions) != len(lookup) or set(lookup) != set(uids):
                raise ValueError(f"Unpaired test IDs: {seed}/{method}")
            rows.append([lookup[uid] for uid in uids])
        outcomes[method] = np.array(rows, dtype=np.int8)
    accuracy_delta = outcomes["flip_constrained_grpo"] - outcomes["grpo"]
    flip_delta = -accuracy_delta * before[None, :]
    rng = np.random.default_rng(random_seed)
    result = []
    for scope in ("all_topics", *topics):
        groups = list(topic_ids.values()) if scope == "all_topics" else [topic_ids[scope]]
        indices = np.concatenate(groups)
        for seed_label in ("all", *seeds):
            selected_seeds = np.arange(len(seeds)) if seed_label == "all" else np.array([seeds.index(seed_label)])
            sampled_questions = np.concatenate([
                rng.choice(ids, size=(resamples, len(ids)), replace=True) for ids in groups
            ], axis=1)
            sampled_seeds = rng.choice(selected_seeds, size=(resamples, len(selected_seeds)), replace=True)
            correct_denominators = before[sampled_questions].sum(axis=1)
            accuracy_draws = accuracy_delta[sampled_seeds[:, :, None], sampled_questions[:, None, :]].mean(axis=(1, 2)) * 100
            flip_draws = flip_delta[sampled_seeds[:, :, None], sampled_questions[:, None, :]].mean(axis=1).sum(axis=1)
            flip_draws = np.divide(flip_draws * 100, correct_denominators,
                                   out=np.full(resamples, np.nan), where=correct_denominators > 0)
            denom = int(before[indices].sum())
            row = {"scope": scope, "seed": seed_label,
                   "completed_seeds": ",".join(map(str, seeds)),
                   "unique_test_questions": len(indices), "initially_correct_unique": denom,
                   "accuracy_difference_pp": float(accuracy_delta[np.ix_(selected_seeds, indices)].mean() * 100),
                   "accuracy_ci_low_pp": float(np.quantile(accuracy_draws, .025)),
                   "accuracy_ci_high_pp": float(np.quantile(accuracy_draws, .975)),
                   "flip_rate_difference_pp": float(flip_delta[np.ix_(selected_seeds, indices)].mean(axis=0).sum() * 100 / denom) if denom else float("nan"),
                   "flip_ci_low_pp": float(np.nanquantile(flip_draws, .025)) if denom else float("nan"),
                   "flip_ci_high_pp": float(np.nanquantile(flip_draws, .975)) if denom else float("nan"),
                   "bootstrap_resamples": resamples}
            result.append(row)
    return result
