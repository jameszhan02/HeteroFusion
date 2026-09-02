#!/usr/bin/env python3
"""Convert GSM8K JSON/JSONL records to LLaMA-Factory SFT format."""

import argparse
import json
import random
import re
from pathlib import Path


DEFAULT_INSTRUCTION = """Solve the following math problem step by step.

{question}

Give your reasoning first. Put the final answer on its own last line using this exact format:
Answer: $ANSWER"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw GSM8K question/answer records into the "
            "instruction/input/output format used by LLaMA-Factory."
        )
    )
    parser.add_argument("--input", required=True, help="Input GSM8K JSON or JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSON array path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep at most this many examples. Default: keep all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used with --shuffle. Default: 42.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle records before applying --limit.",
    )
    parser.add_argument(
        "--keep-calculator-tags",
        action="store_true",
        help="Keep GSM8K tags such as <<3*2=6>> in the reasoning output.",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Write only the final answer instead of the worked reasoning.",
    )
    parser.add_argument(
        "--dataset-info",
        default=None,
        help="Optional dataset_info.json to update with the output dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name to register with --dataset-info.",
    )
    parser.add_argument(
        "--overwrite-dataset-entry",
        action="store_true",
        help="Allow replacing an existing entry with --dataset-info.",
    )
    return parser.parse_args()


def load_records(path: Path):
    """Load either a JSON array or newline-delimited JSON records."""
    text = path.read_text(encoding="utf-8")
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array or JSONL file: {path}")
    return records


def convert_record(item, index, *, keep_calculator_tags, include_reasoning):
    if "question" not in item or "answer" not in item:
        raise KeyError(f"Record {index} must contain 'question' and 'answer'.")

    raw_answer = str(item["answer"])
    if "####" not in raw_answer:
        raise ValueError(f"Record {index} does not contain GSM8K final-answer marker '####'.")

    reasoning, final_answer = raw_answer.rsplit("####", 1)
    reasoning = reasoning.strip()
    final_answer = final_answer.strip()

    if not keep_calculator_tags:
        reasoning = re.sub(r"<<[^<>]*>>", "", reasoning)
        reasoning = re.sub(r"[ \t]{2,}", " ", reasoning)
        reasoning = re.sub(r"\n{3,}", "\n\n", reasoning).strip()

    instruction = DEFAULT_INSTRUCTION.format(question=str(item["question"]).strip())
    if include_reasoning and reasoning:
        output = f"{reasoning}\nAnswer: {final_answer}"
    else:
        output = f"Answer: {final_answer}"

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "task": "gsm8k",
        "source_index": index,
    }


def register_dataset(dataset_info_path, dataset_name, output_path, *, overwrite):
    info_path = Path(dataset_info_path).resolve()
    output_abs = Path(output_path).resolve()
    try:
        relative_file = output_abs.relative_to(info_path.parent)
    except ValueError as exc:
        raise ValueError(
            "Output must be inside the dataset_info.json directory when registering it. "
            f"output={output_abs}, dataset_info_dir={info_path.parent}"
        ) from exc

    data = json.loads(info_path.read_text(encoding="utf-8"))
    if dataset_name in data and not overwrite:
        raise ValueError(
            f"Dataset entry already exists: {dataset_name}. "
            "Use --overwrite-dataset-entry to replace it."
        )

    data[dataset_name] = {
        "file_name": relative_file.as_posix(),
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    info_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_records(input_path)
    indexed_records = list(enumerate(records))
    if args.shuffle:
        random.Random(args.seed).shuffle(indexed_records)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        indexed_records = indexed_records[: args.limit]

    converted = [
        convert_record(
            item,
            index,
            keep_calculator_tags=args.keep_calculator_tags,
            include_reasoning=not args.no_reasoning,
        )
        for index, item in indexed_records
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.dataset_info:
        if not args.dataset_name:
            raise ValueError("--dataset-name is required when --dataset-info is provided.")
        register_dataset(
            args.dataset_info,
            args.dataset_name,
            output_path,
            overwrite=args.overwrite_dataset_entry,
        )

    print(f"[DONE] Input records: {len(records)}")
    print(f"[DONE] Output records: {len(converted)}")
    print(f"[DONE] Wrote: {output_path}")
    if args.dataset_info:
        print(f"[DONE] Registered dataset: {args.dataset_name}")


if __name__ == "__main__":
    main()
