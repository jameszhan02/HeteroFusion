#!/usr/bin/env python3
"""Build the 6-task GENOME validation replay set for HeteroFusion."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENOME_DATA = REPO_ROOT / "data" / "genome_tasks"
OUT_DIR = REPO_ROOT / "data" / "genome_valid_mix_1p9"
OUT_FILE = OUT_DIR / "genome_valid_mix_6tasks.json"
META_FILE = OUT_DIR / "metadata.json"


MMLUPRO_PROMPT = """Answer the following {subject} question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {candidates}.

{question}

{options}

Let's think step by step."""

GSM8K_PROMPT = """Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

MBPP_PROMPT = """You are an expert Python programmer, and here is your task:
{question}

Your code should pass these tests:
{test}
""".strip()

DROP_PROMPT = """You will be asked to read a passage and answer a question.
Write a line of the form "Answer: $ANSWER" at the end of your response.

{context}
""".strip()

FLORES_PROMPT = """Translate the following sentence from English to {language}. Your output should be formatted as follows:

Translation: $SENTENCE

(where $SENTENCE is the translated version of the sentence into {language}).

Below are examples to guide the translation task:

{examples}

Now translate the following sentence:

{sentence}
""".strip()

EMORY_PROMPT = """Given a conversation history and a current utterance, follow these steps to identify the emotion of the current utterance from the given options. The emotion should be determined based on both the conversation context and the current utterance.
The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDEFG. Let's think step by step.

History:
{history}

Utterance:
{utterance}

Options:
{options}"""


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return list(obj.values())
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sample(task: str, instruction: str, output: str, idx: int) -> dict:
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "task": task,
        "source_index": idx,
    }


def build_mmlupro() -> list[dict]:
    rows = load_records(GENOME_DATA / "mmlu_pro/valid.json")
    out = []
    for idx, item in enumerate(rows):
        prompt = MMLUPRO_PROMPT.format(
            subject=item["category"],
            question=item["question"],
            candidates="".join(item["options"].keys()),
            options="\n".join(f"{key}. {value}" for key, value in item["options"].items()),
        )
        out.append(sample("mmlupro", prompt, f"Answer: {item['answer']}", idx))
    return out


def build_gsm8k() -> list[dict]:
    rows = load_records(GENOME_DATA / "gsm8k/valid.json")
    out = []
    for idx, item in enumerate(rows):
        answer = item["answer"].split("#### ")[-1].strip()
        prompt = GSM8K_PROMPT.format(Question=item["question"])
        out.append(sample("gsm8k", prompt, f"Answer: {answer}", idx))
    return out


def build_mbpp() -> list[dict]:
    rows = load_records(GENOME_DATA / "mbpp/valid.json")
    out = []
    for idx, item in enumerate(rows):
        prompt = MBPP_PROMPT.format(
            question=item["text"],
            test="\n".join(item["test_list"]),
        )
        out.append(sample("mbpp", prompt, item["code"].strip(), idx))
    return out


def build_drop() -> list[dict]:
    rows = load_records(GENOME_DATA / "drop/drop_v0_validation.jsonl")
    out = []
    for idx, item in enumerate(rows):
        prompt = DROP_PROMPT.format(context=item["context"])
        answer = item.get("completion") or item.get("ref_text") or ""
        out.append(sample("drop", prompt, f"Answer: {answer.strip()}", idx))
    return out


def build_flores37() -> list[dict]:
    rows = load_records(GENOME_DATA / "flores101/flores37_template_valid.json")
    out = []
    for idx, item in enumerate(rows):
        examples = ""
        for ex_idx, (reference, candidate) in enumerate(
            zip(item["template_reference_list"], item["template_candidate_list"])
        ):
            examples += f"{reference}\nTranslation: {candidate}\n\n"
            if ex_idx == 2:
                break
        prompt = FLORES_PROMPT.format(
            sentence=item["reference"],
            language=item["language"],
            examples=examples,
        )
        out.append(sample("flores37", prompt, f"Translation: {item['candidate']}", idx))
    return out


def build_emorynlp() -> list[dict]:
    rows = load_records(GENOME_DATA / "emorynlp/valid.json")
    out = []
    for idx, item in enumerate(rows):
        prompt = EMORY_PROMPT.format(
            history="- " + "\n- ".join(item["history"]),
            utterance=item["utterance"],
            options="\n".join(f"{key}. {value}" for key, value in item["candidate"].items()),
        )
        out.append(sample("emorynlp", prompt, f"Answer: {item['answer']}", idx))
    return out


def main() -> None:
    builders = [
        build_mmlupro,
        build_gsm8k,
        build_mbpp,
        build_drop,
        build_flores37,
        build_emorynlp,
    ]
    records: list[dict] = []
    counts: dict[str, int] = {}
    for builder in builders:
        part = builder()
        records.extend(part)
        for row in part:
            counts[row["task"]] = counts.get(row["task"], 0) + 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    META_FILE.write_text(
        json.dumps(
            {
                "dataset_name": "genome_valid_mix_6tasks",
                "output_file": str(OUT_FILE.relative_to(REPO_ROOT)),
                "total": len(records),
                "counts": counts,
                "source_split": "valid",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT_FILE), "total": len(records), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
