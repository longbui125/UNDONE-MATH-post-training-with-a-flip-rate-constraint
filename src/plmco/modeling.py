"""Load the same base model and LoRA settings for every pilot arm."""
from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
import transformers.utils.import_utils as import_utils

# The existing tf_gpu environment has an optional audio import conflict.
import_utils._librosa_available = False
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if name in {"bfloat16", "float16"} and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _base_model(config: ExperimentConfig):
    return AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=resolve_dtype(config.dtype),
        trust_remote_code=True, low_cpu_mem_usage=True,
    )


def load_trainable_model(config: ExperimentConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("Training this pilot requires a CUDA GPU")
    model = _base_model(config).cuda()
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=config.lora_targets,
        bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def load_evaluation_model(config: ExperimentConfig, adapter: str | None):
    model = _base_model(config)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found")
    return params
