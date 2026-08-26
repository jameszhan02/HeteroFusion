import copy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "heterofusion"
    / "paper_experiments"
    / "single_source_qwen_to_llama"
    / "llama_target_mit_movie.yaml"
)
TARGET_DIR = (
    PROJECT_ROOT
    / "configs"
    / "heterofusion"
    / "paper_experiments"
    / "ablation_main_qwen_to_llama"
)


VARIANTS = [
    {
        "name": "update_b_only_tail",
        "description": "Current method: tail alignment + update B only",
        "layer_alignment": "tail",
        "update_mode": "b_only",
    },
    {
        "name": "update_a_only_tail",
        "description": "Tail alignment + update A only",
        "layer_alignment": "tail",
        "update_mode": "a_only",
    },
    {
        "name": "update_ab_joint_tail",
        "description": "Tail alignment + update A and B jointly",
        "layer_alignment": "tail",
        "update_mode": "ab_joint",
    },
    {
        "name": "align_head_b_only",
        "description": "Head alignment + update B only",
        "layer_alignment": "head",
        "update_mode": "b_only",
    },
    {
        "name": "align_uniform_b_only",
        "description": "Uniform interpolation + update B only",
        "layer_alignment": "uniform",
        "update_mode": "b_only",
    },
    {
        "name": "align_naive_b_only",
        "description": "Naive layer-index matching + update B only",
        "layer_alignment": "naive",
        "update_mode": "b_only",
    },
]


def build_config(base_config, variant):
    cfg = copy.deepcopy(base_config)
    cfg["experiment_name"] = "paper_main_ablation_qwen_to_llama"
    cfg["output_dir"] = f"outputs/paper/ablation_main_qwen_to_llama/{variant['name']}"

    if len(cfg.get("tasks", [])) != 1:
        raise ValueError("Expected the main ablation anchor config to contain exactly one task.")

    task = cfg["tasks"][0]
    task["task_name"] = variant["name"]
    task["seed"] = int(task.get("seed", task.get("training", {}).get("seed", cfg.get("seed", 42))))
    task["layer_alignment"] = variant["layer_alignment"]

    task.setdefault("training", {})
    task["training"]["fusion_group_name"] = f"heterofusion_transfer_{variant['name']}"
    task["training"]["update_mode"] = variant["update_mode"]
    task["training"]["seed"] = task["seed"]
    task["training"]["variant_note"] = variant["description"]
    return cfg


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for stale in TARGET_DIR.glob("*.yaml"):
        stale.unlink()

    with SOURCE_CONFIG.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    for variant in VARIANTS:
        cfg = build_config(base_config, variant)
        out_path = TARGET_DIR / f"{variant['name']}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    print(f"Generated {len(VARIANTS)} main ablation configs in: {TARGET_DIR}")


if __name__ == "__main__":
    main()
