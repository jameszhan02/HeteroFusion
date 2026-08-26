# HeteroFusion Sanitized Release

This package is prepared for paper code submission.

## Included

- Core HeteroFusion training code.
- LLaMA-Factory data utilities required by the trainer.
- The 1+9 multi-task transfer configuration:
  `configs/heterofusion/genome_1p9/llama_code_target_gemma9_sources.yaml`
- The six-task validation/test data used in the experiment:
  `data/genome_tasks/`
- The generated mixed replay data:
  `data/genome_valid_mix_1p9/`
- Helper scripts:
  `tools/build_genome_valid_mix_dataset.py`
  `tools/run_genome_merged_eval.py`
  `run_genome_heterofusion_1p9.sh`

## Excluded

- Adapter weights.
- Base model weights.
- Training outputs, logs, inference outputs, and caches.
- Local absolute paths and machine-specific result files.

## Reproduction Layout

Set `MODEL_ROOT` to the directory containing `llama-3.1-8b-instruct`.
Place adapters under `adapters/` or set `ADAPTER_ROOT` externally.

```bash
export MODEL_ROOT=/path/to/models
export ADAPTER_ROOT=/path/to/adapters
CUDA_VISIBLE_DEVICES=0 bash run_genome_heterofusion_1p9.sh
```

Evaluation with `tools/run_genome_merged_eval.py` expects the GENOME evaluator
repository via `GENOME_ROOT`, unless it is placed at `external/GENOME`.
