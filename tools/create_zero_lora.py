#!/usr/bin/env python3
"""Create a no-op ("zero") PEFT LoRA adapter directly on a base model.

Use this when you have no trained target adapter at all and want a clean,
uncontaminated `initial_target_lora` anchor for HeteroFusion: a LoRA whose
B @ A product is exactly zero, so the wrapped model behaves identically to
the raw base model until HeteroFusion (or any other trainer) actually learns
something on top of it.

Two variants, controlled by --mode:
  both    (default) lora_A and lora_B are both all-zero. The cleanest anchor
          (no contamination from any prior training), but every block's raw
          content collapses to the same LayerNorm-bias constant inside the
          HeteroFusion transfer net's block encoder, so the cross-attention
          loses the ability to distinguish one target layer/module from
          another by content (position embeddings are all that's left).
  b_only  lora_A keeps PEFT's standard random (Kaiming-uniform) init, only
          lora_B is zeroed. Still a no-op adapter (B @ A == 0 regardless of
          A), but the encoder sees non-degenerate (if meaningless, random)
          per-layer content on the A-derived blocks. If you plan to fuse with
          update_mode: b_only, this still leaves A untrained forever — use
          update_mode: ab_joint so the hypernet can actually shape A.
"""
import argparse
import os
import textwrap

import torch
from peft import LoraConfig, get_peft_model
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
        description="Create a no-op (zero) PEFT LoRA adapter for a base model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Fully zero (A and B both zero) anchor for a target base model.
              python3 tools/create_zero_lora.py \\
                --base-model /path/to/target_base_model \\
                --output-dir adapters/target_zero_lora \\
                --rank 64 --mode both

              # B-only zero, keeping A at PEFT's standard random init, so the
              # transfer net's cross-attention still sees non-degenerate
              # (if meaningless) per-layer query content.
              python3 tools/create_zero_lora.py \\
                --base-model /path/to/target_base_model \\
                --output-dir adapters/target_zero_lora_b_only \\
                --rank 64 --mode b_only

              # Then use it in a HeteroFusion config:
              #   base_model_path: /path/to/target_base_model
              #   initial_target_lora: adapters/target_zero_lora

            Requirements:
              --rank and --target-modules must match whatever source LoRAs
              you plan to fuse against (the current HeteroFusion implementation
              requires target/source rank to agree).
            """
        ),
    )
    parser.add_argument("--base-model", required=True, help="Base HF model path or repo id.")
    parser.add_argument("--output-dir", required=True, help="Directory for the generated PEFT LoRA adapter.")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank. Must match your source LoRAs' rank.")
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
        "--mode",
        choices=("both", "b_only"),
        default="both",
        help="'both': zero lora_A and lora_B. 'b_only': zero lora_B only, leave lora_A at PEFT's random init.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Load/save dtype for the adapter tensors.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model loading, e.g. cpu, cuda, cuda:0.",
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


def main():
    args = parse_args()
    target_modules = split_csv(args.target_modules)
    lora_alpha = float(args.lora_alpha if args.lora_alpha is not None else args.rank)

    torch_dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device)

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

    zeroed_a = 0
    zeroed_b = 0
    with torch.no_grad():
        for module in peft_model.modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if args.mode == "both":
                module.lora_A["default"].weight.zero_()
                zeroed_a += 1
            module.lora_B["default"].weight.zero_()
            zeroed_b += 1

    os.makedirs(args.output_dir, exist_ok=True)
    peft_model.save_pretrained(args.output_dir, safe_serialization=True)

    if args.save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.save_pretrained(args.output_dir)

    print(f"[INFO] mode={args.mode}  lora_A zeroed={zeroed_a}  lora_B zeroed={zeroed_b}")
    print(f"[INFO] Wrote zero PEFT LoRA adapter to: {args.output_dir}")
    print("[INFO] Use this directory as initial_target_lora. B @ A == 0, so this is a no-op on the base model.")


if __name__ == "__main__":
    main()
