#!/usr/bin/env python3
"""Merge one PEFT LoRA adapter into its base Hugging Face model."""

import argparse
import os
import textwrap

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge a PEFT LoRA adapter into a base causal language model and "
            "save a standalone full-weight Hugging Face checkpoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              python3 tools/merge_lora_to_full_model.py \\
                --base-model /path/to/base_model \\
                --adapter /path/to/peft_adapter \\
                --output-dir /path/to/merged_full_model \\
                --dtype bfloat16 \\
                --device cuda:0

            The output is a standalone Hugging Face model. It no longer needs
            the base model or adapter directory when loaded for inference.
            """
        ),
    )
    parser.add_argument("--base-model", required=True, help="Base HF model path or repo id.")
    parser.add_argument("--adapter", required=True, help="PEFT LoRA adapter directory or repo id.")
    parser.add_argument("--output-dir", required=True, help="New directory for the merged full model.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Load and output dtype. Default: bfloat16.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Merge device, for example cpu, cuda, or cuda:0. Default: cpu.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum output safetensors shard size. Default: 5GB.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path or repo id. Defaults to --base-model.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Do not save tokenizer files to the output directory.",
    )
    parser.add_argument(
        "--no-safe-merge",
        action="store_true",
        help="Disable PEFT's NaN validation while merging (not recommended).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code from a Hugging Face repository.",
    )
    return parser.parse_args()


def dtype_from_name(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def validate_output_dir(output_dir, base_model, adapter):
    output_abs = os.path.abspath(os.path.expanduser(output_dir))

    for label, source in (("base model", base_model), ("adapter", adapter)):
        source_expanded = os.path.expanduser(source)
        if os.path.exists(source_expanded) and os.path.abspath(source_expanded) == output_abs:
            raise ValueError(f"Output directory must not overwrite the {label}: {output_abs}")

    if os.path.exists(output_abs) and not os.path.isdir(output_abs):
        raise NotADirectoryError(f"Output path exists and is not a directory: {output_abs}")
    if os.path.isdir(output_abs) and os.listdir(output_abs):
        raise FileExistsError(
            f"Output directory is not empty: {output_abs}. "
            "Choose a new directory to avoid mixing or overwriting checkpoint files."
        )
    return output_abs


def main():
    args = parse_args()
    output_dir = validate_output_dir(args.output_dir, args.base_model, args.adapter)
    torch_dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device)

    print(f"[1/5] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)

    print(f"[2/5] Loading PEFT adapter: {args.adapter}")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    peft_model.eval()

    print("[3/5] Merging adapter weights into the base model")
    merged_model = peft_model.merge_and_unload(safe_merge=not args.no_safe_merge)
    remaining_lora = [name for name, _ in merged_model.named_parameters() if "lora_" in name.lower()]
    if remaining_lora:
        preview = ", ".join(remaining_lora[:5])
        raise RuntimeError(f"LoRA parameters remain after merge: {preview}")

    print(f"[4/5] Saving full-weight model: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    if args.skip_tokenizer:
        print("[5/5] Skipping tokenizer")
    else:
        tokenizer_path = args.tokenizer or args.base_model
        print(f"[5/5] Saving tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer.save_pretrained(output_dir)

    print(f"[DONE] Standalone full-weight model written to: {output_dir}")


if __name__ == "__main__":
    main()
