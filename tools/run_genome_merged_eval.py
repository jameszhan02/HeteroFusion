#!/usr/bin/env python3
"""Evaluate one full model or one LoRA adapter on one GENOME task."""

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENOME_ROOT = Path(os.environ.get("GENOME_ROOT", REPO_ROOT / "external" / "GENOME"))


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", default="base")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def result_file_for_task(work_dir: Path, task: str, split: str, lora_path: str) -> Path:
    result_task_map = {
        "mmlupro": "mmlu_pro",
        "flores37": "flores37",
    }
    result_task = result_task_map.get(task, task)
    result_name = Path(lora_path).name if lora_path != "base" else "base"
    return work_dir / "results" / result_task / f"{result_name}_{split}_results.json"


def normalize_score(score):
    if isinstance(score, dict):
        return score
    return {"score": score}


def main():
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    lora_path = args.lora_path
    resolved_lora_path = str(Path(lora_path).resolve()) if lora_path != "base" else "base"
    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    task = args.task.lower()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    datas_link = work_dir / "datas"
    if not datas_link.exists():
        datas_link.symlink_to(GENOME_ROOT / "datas", target_is_directory=True)

    sys.path.insert(0, str(GENOME_ROOT))
    os.environ.setdefault("GENOME_MODEL_PATH", str(model_path))
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from src.evaluate.eval import Evaluator, Method
    from src.evaluate.factory import EvaluatorFactory
    from vllm import LLM

    def load_model_compatible(self, model_name_or_path: str) -> LLM:
        return LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            trust_remote_code=True,
            enable_lora=True,
            enforce_eager=True,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=0.90,
        )

    # GENOME was written against an older vLLM API that accepted device="auto".
    # The local infer_train env uses vLLM 0.10, so patch only this wrapper process.
    Evaluator.load_model = load_model_compatible

    with pushd(work_dir):
        evaluator = EvaluatorFactory(model_name_or_path=str(model_path)).get_evaluator(task)
        score = evaluator.evaluate(
            method=Method.LOCAL,
            model_name_or_path=str(model_path),
            lora_path=resolved_lora_path,
            split=args.split,
            max_samples=args.max_samples,
        )

    predictions_src = result_file_for_task(work_dir, task, args.split, resolved_lora_path)
    predictions_dst = output_dir / f"{task}_{args.split}_predictions.json"
    if predictions_src.exists():
        shutil.copy2(predictions_src, predictions_dst)

    summary = {
        "model_path": str(model_path),
        "lora_path": resolved_lora_path,
        "task": task,
        "split": args.split,
        "max_samples": args.max_samples,
        "score": normalize_score(score),
        "predictions_path": str(predictions_dst) if predictions_dst.exists() else None,
    }
    with open(output_dir / f"{task}_{args.split}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
