#!/usr/bin/env python3
import argparse
import os
import textwrap
from typing import Iterable

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import modeling_utils as transformers_modeling_utils


class _SafeOpenWithPyTorchMetadataFallback:
    """Wrap safetensors.safe_open and supply missing HF format metadata."""

    def __init__(self, safe_open_impl, *args, **kwargs):
        self._inner = safe_open_impl(*args, **kwargs)
        self._reader = None

    def __enter__(self):
        self._reader = self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._inner.__exit__(exc_type, exc_value, traceback)

    def metadata(self):
        metadata = self._reader.metadata()
        if metadata is None:
            print("[WARN] Safetensors metadata is missing; assuming format='pt'.")
            return {"format": "pt"}
        return metadata

    def __getattr__(self, name):
        return getattr(self._reader, name)


def load_causal_lm(path, *, torch_dtype):
    """Load an HF model while accepting metadata-less PyTorch safetensors."""
    original_safe_open = transformers_modeling_utils.safe_open

    def safe_open_with_fallback(*args, **kwargs):
        return _SafeOpenWithPyTorchMetadataFallback(original_safe_open, *args, **kwargs)

    transformers_modeling_utils.safe_open = safe_open_with_fallback
    try:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype)
    finally:
        transformers_modeling_utils.safe_open = original_safe_open


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Approximate a full fine-tuned HF checkpoint as a PEFT LoRA adapter "
            "by SVD-compressing W_finetuned - W_base for selected Linear modules."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Convert the target full fine-tuned model into a target PEFT LoRA anchor.
              python3 tools/full_model_delta_to_lora.py \\
                --base-model /path/to/target_base_model \\
                --trained-model /path/to/target_trained_full_model \\
                --output-dir adapters/target_svd_lora \\
                --rank 16 \\
                --lora-alpha 16

              # Convert the source full fine-tuned model into a source PEFT LoRA adapter.
              python3 tools/full_model_delta_to_lora.py \\
                --base-model /path/to/source_base_model \\
                --trained-model /path/to/source_trained_full_model \\
                --output-dir adapters/source_svd_lora \\
                --rank 16 \\
                --lora-alpha 16

              # Then use the generated adapters in a HeteroFusion config:
              #   base_model_path: /path/to/target_base_model
              #   initial_target_lora: adapters/target_svd_lora
              #   source_lora_paths:
              #     - adapters/source_svd_lora

            Requirements:
              --trained-model must be a full checkpoint fine-tuned from --base-model.
              The output directory is a PEFT LoRA adapter directory containing
              adapter_config.json and adapter_model.safetensors.
            """
        ),
    )
    parser.add_argument("--base-model", required=True, help="Base HF model path or repo id.")
    parser.add_argument("--trained-model", required=True, help="Full fine-tuned HF model path or repo id.")
    parser.add_argument("--output-dir", required=True, help="Directory for the generated PEFT LoRA adapter.")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank used for SVD compression.")
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha. Defaults to rank, so PEFT scaling alpha/r is 1.",
    )
    parser.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated Linear module suffixes to convert.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Load dtype. SVD is still computed in float32.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model loading and SVD, e.g. cpu, cuda, cuda:0. CPU is safest but memory-heavy.",
    )
    parser.add_argument("--save-tokenizer", action="store_true", help="Also save the base tokenizer to output-dir.")
    return parser.parse_args()


def dtype_from_name(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def split_csv(value: str):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def is_target_module(module_name: str, target_modules: Iterable[str]):
    leaf = module_name.rsplit(".", 1)[-1]
    return leaf in target_modules


def collect_linear_weights(model: nn.Module, target_modules: Iterable[str]):
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and is_target_module(name, target_modules):
            weights[name] = module.weight.detach()
    return weights


def low_rank_lora_from_delta(delta: torch.Tensor, rank: int, scale: float):
    rows, cols = delta.shape
    svd_rank = min(rank, rows, cols)
    if svd_rank <= 0:
        raise ValueError(f"Invalid rank={rank} for delta shape={tuple(delta.shape)}.")

    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    u = u[:, :svd_rank]
    s = s[:svd_rank]
    vh = vh[:svd_rank, :]

    # PEFT applies scaling * (B @ A). Build B/A so that scaling * B @ A
    # approximates the full fine-tuning delta.
    s_sqrt = torch.sqrt(s / scale)
    lora_b = u * s_sqrt.unsqueeze(0)
    lora_a = s_sqrt.unsqueeze(1) * vh

    if svd_rank < rank:
        padded_a = torch.zeros((rank, cols), dtype=lora_a.dtype, device=lora_a.device)
        padded_b = torch.zeros((rows, rank), dtype=lora_b.dtype, device=lora_b.device)
        padded_a[:svd_rank, :] = lora_a
        padded_b[:, :svd_rank] = lora_b
        lora_a, lora_b = padded_a, padded_b

    return lora_a, lora_b


def get_peft_lora_module(peft_model: nn.Module, original_name: str):
    candidates = (
        f"base_model.model.{original_name}",
        f"base_model.model.model.{original_name}",
        original_name,
    )
    for candidate in candidates:
        try:
            module = peft_model.get_submodule(candidate)
        except AttributeError:
            continue
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            return module
    raise KeyError(f"Could not find PEFT LoRA module for original module: {original_name}")


def main():
    args = parse_args()
    target_modules = split_csv(args.target_modules)
    lora_alpha = float(args.lora_alpha if args.lora_alpha is not None else args.rank)
    scale = lora_alpha / float(args.rank)
    if scale <= 0:
        raise ValueError(f"lora_alpha/rank must be positive, got {scale}.")

    torch_dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device)

    print(f"[INFO] Loading base model: {args.base_model}")
    base_model = load_causal_lm(args.base_model, torch_dtype=torch_dtype).to(device)
    print(f"[INFO] Loading fine-tuned model: {args.trained_model}")
    trained_model = load_causal_lm(args.trained_model, torch_dtype=torch_dtype).to(device)

    base_weights = collect_linear_weights(base_model, target_modules)
    trained_weights = collect_linear_weights(trained_model, target_modules)
    common_names = sorted(set(base_weights) & set(trained_weights))
    if not common_names:
        raise RuntimeError(f"No common Linear modules found for target_modules={target_modules}.")

    missing_in_trained = sorted(set(base_weights) - set(trained_weights))
    missing_in_base = sorted(set(trained_weights) - set(base_weights))
    if missing_in_trained or missing_in_base:
        print(
            "[WARN] Module name mismatch. "
            f"missing_in_trained={len(missing_in_trained)}, missing_in_base={len(missing_in_base)}"
        )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=args.base_model,
    )
    peft_model = get_peft_model(base_model, lora_config)

    converted = 0
    skipped_shape = 0
    with torch.no_grad():
        for name in common_names:
            base_w = base_weights[name]
            trained_w = trained_weights[name]
            if base_w.shape != trained_w.shape:
                skipped_shape += 1
                print(f"[WARN] Skipping shape mismatch: {name}: {tuple(base_w.shape)} vs {tuple(trained_w.shape)}")
                continue

            delta = trained_w.float() - base_w.float()
            lora_a, lora_b = low_rank_lora_from_delta(delta, args.rank, scale)
            lora_module = get_peft_lora_module(peft_model, name)
            lora_module.lora_A["default"].weight.copy_(lora_a.to(lora_module.lora_A["default"].weight.device))
            lora_module.lora_B["default"].weight.copy_(lora_b.to(lora_module.lora_B["default"].weight.device))
            converted += 1

    os.makedirs(args.output_dir, exist_ok=True)
    peft_model.save_pretrained(args.output_dir, safe_serialization=True)

    if args.save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.save_pretrained(args.output_dir)

    print(f"[INFO] Converted Linear modules: {converted}")
    print(f"[INFO] Skipped shape mismatches: {skipped_shape}")
    print(f"[INFO] Wrote PEFT LoRA adapter to: {args.output_dir}")
    print("[INFO] Use this directory as initial_target_lora or one entry in source_lora_paths.")


if __name__ == "__main__":
    main()
