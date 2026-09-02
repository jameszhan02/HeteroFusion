# HeteroFusion

**Can Heterogeneous Language Models Be Fused?**

Paper: [arXiv:2604.01674](https://arxiv.org/abs/2604.01674)

HeteroFusion is a research code release for adapter-space fusion across
heterogeneous language-model families. It learns structured updates for a
target LoRA adapter by reading one or more source LoRA adapters from different
backbones, aligning modules by layer topology, and optimizing the fused adapter
on lightweight replay data.

The current repository is a sanitized release: it contains the fusion code,
experiment configs, replay/evaluation utilities, figures, and small released
datasets. It does not include base model weights, LoRA adapter weights, training
outputs, logs, or private machine-specific artifacts.

<p align="center">
  <img src="assets/framework.png" alt="HeteroFusion framework" width="88%">
</p>

## What Is Included

- HeteroFusion entry point: `main.py`
- Fusion model, trainer, and regularizer: `src/model.py`, `src/trainer.py`,
  `src/losses.py`
- Dataset loader wrapper around the vendored LLaMA-Factory utilities: `data.py`
- GENOME 1+9 reproduction config:
  `configs/heterofusion/genome_1p9/llama_code_target_gemma9_sources.yaml`
- GENOME six-task validation/test data under `data/genome_tasks/`
- Generated GENOME mixed validation replay set under `data/genome_valid_mix_1p9/`
- Broader paper experiment configs under
  `configs/heterofusion/paper_experiments/`
- Sample replay data under `data/sample/`
- Helper scripts for GENOME replay construction, fusion, and evaluation
- README figures under `assets/`

## What Is Not Included

- Base model checkpoints
- PEFT LoRA adapter weights
- Fused/merged output adapters
- Optimizer states or trainer checkpoints
- Inference results, logs, caches, and local absolute paths
- The external GENOME evaluator repository

Adapters are intentionally absent. The `adapters/` directory is kept only as a
documented mount point; see `adapters/README.md` for the expected checkpoint
layout.

## Repository Layout

```text
HeteroFusion/
├── adapters/                         # Mount point for external LoRA adapters
├── assets/                           # Figures used in this README
├── configs/heterofusion/
│   ├── genome_1p9/                   # Sanitized GENOME 1+9 config
│   └── paper_experiments/            # Paper configs and ablation sweeps
├── data/
│   ├── genome_tasks/                 # Released GENOME valid/test task data
│   ├── genome_valid_mix_1p9/          # Generated six-task replay mix
│   └── sample/                       # Lightweight sample replay data
├── llamafactory/                     # Vendored utilities used by this release
├── src/                              # HeteroFusion transfer network/trainer
├── tools/                            # Dataset and GENOME evaluation helpers
├── data.py                           # Mixed-dataset builder
├── main.py                           # Fusion pipeline entry point
├── run_genome_heterofusion_1p9.sh     # End-to-end GENOME helper script
└── run_paper_*.sh                    # Paper experiment launch helpers
```

## Method Components In This Code

`HeteroFusionTransferNet` in `src/model.py` is the learnable transfer module. It
encodes LoRA `A` and `B` blocks, applies SVD-guided conflict-aware gating, uses
topology-aligned attention from target blocks to source blocks, and decodes
adapter deltas.

`HeteroFusionTrainer` in `src/trainer.py` loads target/source LoRA states,
aligns heterogeneous source layers to target layers with `tail`, `head`,
`uniform`, or `naive` strategies, dynamically patches predicted LoRA weights
into the target model during optimization, and exports a final PEFT adapter to
`merged_lora/`.

The supported update modes are:

- `b_only`: preserve target LoRA `A` and update `B`
- `a_only`: update target LoRA `A`
- `ab_joint`: update both target LoRA matrices

## Environment

This release is not packaged as a pip library. A practical starting environment
is:

```bash
conda create -n infer_train python=3.10 -y
conda activate infer_train
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers peft safetensors pyyaml fire tqdm numpy datasets accelerate sentencepiece scipy einops
```

The `run_genome_heterofusion_1p9.sh` script uses `conda run -n infer_train`.
Rename the environment in that script or create the environment with this name.

GENOME evaluation additionally expects a working vLLM-based GENOME evaluator
environment. Put that repository at `external/GENOME` or set `GENOME_ROOT`.

## Required External Paths

Set a base-model root:

```bash
export MODEL_ROOT=/path/to/base_models
```

The GENOME 1+9 config expects:

```text
${MODEL_ROOT}/llama-3.1-8b-instruct
```

Set an adapter root, or place adapters inside this repo's `adapters/` directory:

```bash
export ADAPTER_ROOT=/path/to/heterofusion_adapters
```

If `ADAPTER_ROOT` is unset, `main.py` resolves `${ADAPTER_ROOT}` to
`./adapters`.

Each adapter directory must contain a standard PEFT LoRA checkpoint:

```text
adapter_config.json
adapter_model.safetensors   # or adapter_model.bin
```

For the released GENOME 1+9 config, the required adapter layout is:

```text
${ADAPTER_ROOT}/
  llama3.1-8b-instruct/
    GENOME/
      code_alpaca_fast/
  gemma-2-2b-it/
    GENOME/
      cot/
      flan_v2/
      gpt4_alpaca/
      lima/
      oasst1/
      open_orca/
      science_literature/
      sharegpt/
      wizardlm/
```

## GENOME 1+9 Quick Start

Build or refresh the released six-task validation replay mix:

```bash
python tools/build_genome_valid_mix_dataset.py
```

Run HeteroFusion training for the GENOME 1+9 config:

```bash
python main.py configs/heterofusion/genome_1p9/llama_code_target_gemma9_sources.yaml
```

The fused adapter is written to:

```text
outputs/genome_1p9/llama_code_target_gemma9_sources/genome_1p9_valid_mix_tail_b_only/merged_lora/
```

To run the bundled end-to-end helper, including replay construction, training,
and six GENOME test evaluations:

```bash
export MODEL_ROOT=/path/to/base_models
export ADAPTER_ROOT=/path/to/heterofusion_adapters
export GENOME_ROOT=/path/to/GENOME
CUDA_VISIBLE_DEVICES=0 bash run_genome_heterofusion_1p9.sh
```

The script writes logs under `logs/genome_1p9/` and evaluation outputs under
`infer_results/GENOME_1P9_LLAMA_CODE_GEMMA9_TEST6/`.

## Running Other Paper Configs

The broader paper configs are available under:

```text
configs/heterofusion/paper_experiments/
```

They cover:

- single-source Qwen-to-Llama transfer
- multi-source cross-family fusion
- noisy-source robustness
- GLUE cross-family transfer
- alignment and update-mode ablations
- `alpha` and `mu_gate` sensitivity sweeps

Run any config with:

```bash
python main.py path/to/config.yaml
```

These configs use the same `${MODEL_ROOT}` and `${ADAPTER_ROOT}` convention.
Only the model and adapter paths referenced by the config you run are required.
Some paper configs also depend on the sample replay datasets in `data/sample/`.

The shell helpers are:

```bash
bash run_paper_single_source_qwen_to_llama.sh
bash run_paper_glue_cross_family.sh
bash run_paper_hyperparameter_sweep.sh
bash run_paper_main_ablation_qwen_to_llama.sh
```

Review each script before launching on a new machine, because GPU assignment,
environment names, output directories, and skip/dry-run behavior are controlled
inside the scripts and by environment variables.

## Checkpoint Conversion Utilities

Approximate a full fine-tuned checkpoint as a rank-16 PEFT LoRA adapter:

```bash
python tools/full_model_delta_to_lora.py \
  --base-model /path/to/base_model \
  --trained-model /path/to/full_finetuned_model \
  --output-dir adapters/converted_r16 \
  --rank 16 \
  --lora-alpha 16 \
  --dtype bfloat16 \
  --device cuda:0
```

Merge one PEFT LoRA adapter back into its base model and save a standalone
full-weight Hugging Face checkpoint:

```bash
python tools/merge_lora_to_full_model.py \
  --base-model /path/to/base_model \
  --adapter adapters/converted_r16 \
  --output-dir /path/to/merged_full_model \
  --dtype bfloat16 \
  --device cuda:0
```

The merge utility saves safetensors shards and tokenizer files. For safety, it
refuses to write into a non-empty output directory or overwrite the base model
or adapter directory.

## GENOME Evaluation Helper

Evaluate a base model or fused LoRA on one GENOME task:

```bash
python tools/run_genome_merged_eval.py \
  --model-path "${MODEL_ROOT}/llama-3.1-8b-instruct" \
  --lora-path outputs/genome_1p9/llama_code_target_gemma9_sources/genome_1p9_valid_mix_tail_b_only/merged_lora \
  --task gsm8k \
  --split test \
  --output-dir infer_results/gsm8k \
  --work-dir /tmp/heterofusion_genome_eval_gsm8k
```

Supported released GENOME task names are:

```text
mmlupro, gsm8k, mbpp, drop, flores37, emorynlp
```

## Notes On Reproducibility

This repository preserves code and config structure, but exact reproduction
requires compatible external checkpoints and evaluator versions. In particular:

- `MODEL_ROOT` must point to the target base model used by the config.
- `ADAPTER_ROOT` must contain the target and source LoRA adapters named in the
  config.
- GENOME evaluation requires the external GENOME repository and its vLLM
  dependencies.
- Outputs are intentionally ignored by Git through `.gitignore`.

## Results Figures

The repository includes static figures used for documentation.

<p align="center">
  <img src="assets/noise_avg_metrics.png" alt="Noise robustness" width="47%">
  <img src="assets/glue_performance.png" alt="GLUE performance" width="47%">
</p>

<p align="center">
  <img src="assets/sensitivity.png" alt="Sensitivity analysis" width="70%">
</p>

## Citation

```bibtex
@article{heterofusion2026,
  title   = {Can Heterogeneous Language Models Be Fused?},
  author  = {Chen, Shilian and Zhou, Jie and Chen, Qin and Wu, Wen and Li, Xin and Feng, Qi and He, Liang},
  journal = {arXiv preprint arXiv:2604.01674},
  year    = {2026}
}
```

## Acknowledgements

- LLaMA-Factory for the data/model utility foundation vendored in this release
- Hugging Face Transformers and PEFT for model and adapter tooling
- The GENOME benchmark/evaluator for the six-task evaluation workflow
