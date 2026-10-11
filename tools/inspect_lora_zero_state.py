#!/usr/bin/env python3
"""Inspect whether a PEFT LoRA adapter has zero A/B matrices."""

import argparse
import os

import torch
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect LoRA A/B zero state.")
    parser.add_argument("adapter", help="PEFT adapter directory containing adapter_model.safetensors/bin.")
    parser.add_argument("--tol", type=float, default=0.0, help="Treat abs values <= tol as zero.")
    parser.add_argument("--show", type=int, default=8, help="Number of module rows to print.")
    return parser.parse_args()


def load_adapter_state(adapter_dir):
    safe_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(safe_path):
        return load_file(safe_path, device="cpu")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {adapter_dir}")


def is_zero(tensor, tol):
    if tensor.numel() == 0:
        return True
    return bool(torch.all(tensor.float().abs() <= tol).item())


def main():
    args = parse_args()
    state = load_adapter_state(args.adapter)

    rows = []
    sum_a_norm = 0.0
    sum_b_norm = 0.0
    sum_ba_norm = 0.0
    zero_a = 0
    zero_b = 0

    for key in sorted(state):
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in state:
            continue

        a = state[key].float()
        b = state[b_key].float()
        a_zero = is_zero(a, args.tol)
        b_zero = is_zero(b, args.tol)
        ba_norm = (b @ a).norm().item()

        zero_a += int(a_zero)
        zero_b += int(b_zero)
        sum_a_norm += a.norm().item()
        sum_b_norm += b.norm().item()
        sum_ba_norm += ba_norm
        rows.append((key.replace(".lora_A.weight", ""), a.norm().item(), b.norm().item(), ba_norm, a_zero, b_zero))

    total = len(rows)
    print(f"adapter: {args.adapter}")
    print(f"pairs: {total}")
    print(f"zero A: {zero_a}/{total}")
    print(f"zero B: {zero_b}/{total}")
    print(f"sum ||A||: {sum_a_norm:.8g}")
    print(f"sum ||B||: {sum_b_norm:.8g}")
    print(f"sum ||B@A||: {sum_ba_norm:.8g}")

    if total and zero_a == total and zero_b == total:
        verdict = "both-zero: lora_A and lora_B are all zero"
    elif total and zero_a == 0 and zero_b == total:
        verdict = "B-only-zero: lora_A is nonzero, lora_B is all zero"
    elif total and zero_b == total:
        verdict = "B-zero with some zero A"
    else:
        verdict = "nonzero adapter"
    print(f"verdict: {verdict}")

    if args.show > 0:
        print("\nmodule preview:")
        print("module ||A|| ||B|| ||B@A|| A_zero B_zero")
        for row in rows[: args.show]:
            name, a_norm, b_norm, ba_norm, a_zero, b_zero = row
            print(f"{name} {a_norm:.6g} {b_norm:.6g} {ba_norm:.6g} {a_zero} {b_zero}")


if __name__ == "__main__":
    main()
