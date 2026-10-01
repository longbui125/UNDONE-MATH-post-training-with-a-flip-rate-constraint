"""Generation, exact integer grading, and anchor gold-solution loss."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .math_data import MathCase, integer_answer, last_boxed, prompt_text


@dataclass
class Rollout:
    ids: torch.Tensor
    prompt_length: int
    text: str
    reward: float
    capped: bool = False
    parsed_answer: int | None = None
    has_box: bool = False
    old_logprobs: torch.Tensor | None = None
    reference_logprobs: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return int(self.ids.numel() - self.prompt_length)


def prompt_ids(tokenizer, case: MathCase, config: ExperimentConfig) -> torch.Tensor:
    ids = tokenizer(prompt_text(case.problem, tokenizer), add_special_tokens=False).input_ids
    if len(ids) > config.max_prompt_tokens:
        raise ValueError(f"Prompt exceeds max_prompt_tokens: {case.uid}")
    return torch.tensor(ids, dtype=torch.long)


def completion_logprobs(model, ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    ids = ids.to(next(model.parameters()).device).unsqueeze(0)
    if ids.numel() <= prompt_length:
        raise ValueError("Completion is empty")
    logits = model(input_ids=ids, use_cache=False).logits[:, prompt_length - 1:-1, :]
    targets = ids[:, prompt_length:]
    return -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), reduction="none"
    )


def gold_solution_loss(model, tokenizer, case: MathCase, config: ExperimentConfig) -> torch.Tensor:
    prefix = prompt_ids(tokenizer, case, config)
    answer_ids = tokenizer(case.solution + tokenizer.eos_token,
                           add_special_tokens=False).input_ids
    ids = torch.cat((prefix, torch.tensor(answer_ids, dtype=torch.long)))
    if ids.numel() > config.max_sft_tokens:
        raise ValueError(f"Gold solution exceeds max_sft_tokens: {case.uid}")
    return -completion_logprobs(model, ids, prefix.numel()).mean()


def was_truncated(full: torch.Tensor, prompt_length: int, limit: int,
                  tokenizer) -> bool:
    """Distinguish a token-budget stop from an EOS emitted on the last token."""
    if full.numel() - prompt_length < limit:
        return False
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    return int(full[-1].item()) not in eos_ids


@torch.no_grad()
def sample_group(model, tokenizer, case: MathCase, config: ExperimentConfig,
                 group_size: int | None = None) -> list[Rollout]:
    model.eval()
    prefix = prompt_ids(tokenizer, case, config)
    on_device = prefix.unsqueeze(0).to(next(model.parameters()).device)
    result = []
    for _ in range(group_size or config.group_size):
        full = model.generate(
            on_device, do_sample=True, temperature=config.temperature,
            top_p=config.top_p, max_new_tokens=config.max_completion_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, use_cache=True,
        )[0]
        text = tokenizer.decode(full[prefix.numel():], skip_special_tokens=True)
        prediction = integer_answer(text)
        length = int(full.numel() - prefix.numel())
        result.append(Rollout(full.cpu(), prefix.numel(), text,
                              float(prediction == case.answer),
                              capped=was_truncated(full, prefix.numel(),
                                                   config.max_completion_tokens, tokenizer),
                              parsed_answer=prediction, has_box=last_boxed(text) is not None))
    return result


@torch.no_grad()
def evaluate_case(model, tokenizer, case: MathCase, config: ExperimentConfig,
                  max_new_tokens: int | None = None) -> dict:
    model.eval()
    prefix = prompt_ids(tokenizer, case, config)
    limit = max_new_tokens or config.eval_max_new_tokens
    full = model.generate(
        prefix.unsqueeze(0).to(next(model.parameters()).device),
        do_sample=False, max_new_tokens=limit,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, use_cache=True,
    )[0]
    text = tokenizer.decode(full[prefix.numel():], skip_special_tokens=True)
    prediction = integer_answer(text)
    return {
        "uid": case.uid, "topic": case.topic, "answer": case.answer,
        "prediction": prediction, "correct": prediction == case.answer,
        "tokens": int(full.numel() - prefix.numel()),
        "capped": was_truncated(full, prefix.numel(), limit, tokenizer),
        "has_box": last_boxed(text) is not None,
        "text": text,
    }
