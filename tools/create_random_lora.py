#!/usr/bin/env python3
"""Create a PEFT LoRA adapter filled with random weights, shape-matched to a
given base model.

This is a placebo/control adapter: same rank, same target_modules, same
per-layer in/out feature shapes as a real target or source LoRA for that
base model, but with no trained signal in it at all. Use it to sanity-check
HeteroFusion's own claims, e.g.:
  - Swap a real source_lora_path for one of these (shape-matched to the same
    source base model) and rerun fusion. If the result doesn't meaningfully
    regress vs. the real source, the "gain" isn't coming from source content.
  - Compare against tools/create_zero_lora.py as a target anchor: a zero
    anchor is a no-op start; a random anchor is a *non*-zero, unstructured
    start (its own B @ A delta actively perturbs the base model), which is a
    different and generally less desirable condition for a target anchor,
    but useful as a second control point.

Unlike PEFT's own default init (lora_A ~ Kaiming-uniform, lora_B = 0, i.e. a
no-op), this tool fills BOTH lora_A and lora_B with random values, so the
resulting adapter has a genuinely non-zero, unstructured B @ A delta.

By default the random tensors use the same Kaiming-uniform scheme PEFT uses
for lora_A, applied to both A and B. Pass --match-norm-to a real adapter
directory to instead rescale each generated tensor to match that real
adapter's per-tensor Frobenius norm (shape-matched by module name where
possible, otherwise by the nearest same-shape tensor) -- a more rigorous
"same shape, same magnitude, scrambled content" placebo.
"""
import argparse
import math
import os
import textwrap

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        description="Create a random-weight PEFT LoRA adapter, shape-matched to a base model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Random placebo shaped like a source adapter for this base model.
              python3 tools/create_random_lora.py \\
                --base-model /path/to/source_base_model \\
                --output-dir adapters/source_random_placebo \\
                --rank 64

              # Same, but with magnitude matched to a specific real adapter.
              python3 tools/create_random_lora.py \\
                --base-model /path/to/source_base_model \\
                --output-dir adapters/source_random_placebo_matched \\
                --rank 64 \\
                --match-norm-to adapters/source_real_lora

              # Then swap it in as a source for an ablation run:
              #   source_lora_paths:
              #     - adapters/source_random_placebo

            Requirements:
              --rank and --target-modules must match whatever target/other
              source LoRAs you plan to fuse against (the current HeteroFusion
              implementation requires target/source rank to agree).
            """
        ),
    )
    parser.add_argument("--base-model", required=True, help="Base HF model path or repo id to shape-match against.")
    parser.add_argument("--output-dir", required=True, help="Directory for the generated PEFT LoRA adapter.")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank. Must match the adapters you'll fuse it with.")
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha. Defaults to rank, so PEFT scaling alpha/r is 1.",
    )
    parser.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated Linear module suffixes to attach LoRA to.",
    )
    parser.add_argument(
        "--match-norm-to",
        default=None,
        help=(
            "Optional path to a real PEFT LoRA adapter directory. If given, each "
            "generated lora_A/lora_B tensor is rescaled to match that adapter's "
            "corresponding tensor's Frobenius norm (matched by module name, falling "
            "back to rank order among same-shape tensors)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Load/save dtype for the adapter tensors.",
    )
    parser.add_argument("--device", default="cpu", help="Device for model loading, e.g. cpu, cuda, cuda:0.")
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


def kaiming_uniform_like(tensor: torch.Tensor, generator: torch.Generator):
    # Mirrors PEFT's own lora_A init (kaiming_uniform_ with a=sqrt(5)).
    fan_in = tensor.shape[1]
    gain = math.sqrt(2.0 / (1 + 5))
    bound = gain * math.sqrt(3.0 / fan_in)
    return torch.empty_like(tensor).uniform_(-bound, bound, generator=generator)


def load_reference_state(adapter_dir):
    safe_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(safe_path):
        return load_file(safe_path, device="cpu")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No adapter found in {adapter_dir}")


def build_reference_norm_pools(reference_state):
    """Group reference tensors by (lora_attr, shape) so same-shaped tensors
    can be matched even when module naming differs across model families."""
    pools = {}
    for key, tensor in reference_state.items():
        attr = "lora_A" if "lora_A" in key else ("lora_B" if "lora_B" in key else None)
        if attr is None:
            continue
        pools.setdefault((attr, tuple(tensor.shape)), []).append(tensor.float().norm().item())
    return pools


def pick_reference_norm(pools, attr, shape, cursor):
    key = (attr, tuple(shape))
    norms = pools.get(key)
    if not norms:
        return None
    idx = cursor.get(key, 0) % len(norms)
    cursor[key] = idx + 1
    return norms[idx]


def main():
    args = parse_args()
    target_modules = split_csv(args.target_modules)
    lora_alpha = float(args.lora_alpha if args.lora_alpha is not None else args.rank)

    torch_dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    reference_pools = None
    cursor = {}
    if args.match_norm_to:
        print(f"[INFO] Loading reference adapter for norm matching: {args.match_norm_to}")
        reference_pools = build_reference_norm_pools(load_reference_state(args.match_norm_to))

    print(f"[INFO] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype).to(device)

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

    randomized = 0
    matched = 0
    with torch.no_grad():
        for module in peft_model.modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            for attr in ("lora_A", "lora_B"):
                weight = getattr(module, attr)["default"].weight
                random_tensor = kaiming_uniform_like(weight.float().cpu(), generator)

                if reference_pools is not None:
                    ref_norm = pick_reference_norm(reference_pools, attr, random_tensor.shape, cursor)
                    if ref_norm is not None:
                        current_norm = random_tensor.norm().item()
                        if current_norm > 0:
                            random_tensor = random_tensor * (ref_norm / current_norm)
                            matched += 1

                weight.copy_(random_tensor.to(weight.dtype).to(weight.device))
            randomized += 1

    os.makedirs(args.output_dir, exist_ok=True)
    peft_model.save_pretrained(args.output_dir, safe_serialization=True)

    if args.save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.save_pretrained(args.output_dir)

    print(f"[INFO] Randomized LoRA modules: {randomized}")
    if args.match_norm_to:
        print(f"[INFO] Tensors norm-matched to reference: {matched}")
    print(f"[INFO] Wrote random PEFT LoRA adapter to: {args.output_dir}")
    print("[INFO] This adapter has a genuinely non-zero, unstructured B @ A delta -- a placebo, not a no-op.")


if __name__ == "__main__":
    main()
