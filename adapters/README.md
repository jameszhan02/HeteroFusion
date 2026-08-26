# Adapter Checkpoints

This sanitized release does not include adapter weights. The directory is kept
as a documented mount point for PEFT LoRA checkpoints used by the released
configs.

All HeteroFusion configs reference adapters through `${ADAPTER_ROOT}`. If
`ADAPTER_ROOT` is not set, `main.py` resolves it to this repo-local
`adapters/` directory. You can either place checkpoints here or point
`ADAPTER_ROOT` to an external adapter store.

```bash
export ADAPTER_ROOT=/path/to/heterofusion_adapters
```

Each adapter directory should contain a standard PEFT LoRA checkpoint:

```text
adapter_config.json
adapter_model.safetensors   # or adapter_model.bin
```

Base model checkpoints are separate and should be provided through
`MODEL_ROOT`.

## GENOME 1+9 Experiment

The GENOME 1+9 config is:

```text
configs/heterofusion/genome_1p9/llama_code_target_gemma9_sources.yaml
```

It uses one Llama target adapter and nine Gemma source adapters. To reproduce
that setting, provide the following layout under `${ADAPTER_ROOT}`:

```text
adapters/
  llama3.1-8b-instruct/
    GENOME/
      code_alpaca_fast/              # target adapter
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

The Gemma `code_alpaca` adapter is intentionally excluded from the source pool
because the experiment uses the Llama `code_alpaca_fast` adapter as the target
anchor.

## Paper Config Adapter Families

The broader paper configs under `configs/heterofusion/paper_experiments/`
expect the same `${ADAPTER_ROOT}` convention. Depending on the config, the
adapter store may include these families:

```text
adapters/
  llama_instruct/
    NER_mit-movie_sft/
    NER_TweetNER7_sample_15000_sft/
    RE_conll04/
    RE_New-York-Times-RE_sample_30000_sft/
    ET_FabNER_sft/
    ET_FindVehicle_sft/
    NER_CrossNER_science/
    NER_CrossNER_politics/
    NER_bc4chemd/
    NER_bc2gm/

  Qwen2.5/
    UIE/
      NER_mit-movie/
      NER_TweetNER7_sample_15000/
      RE_conll04/
      RE_New-York-Times-RE_sample_30000/
      ET_FabNER/
      ET_FindVehicle/

  mistral/
    ET_Fab/sft/
    ET_FindVehicle/sft/

  glue/
    llama3/
      cola_qa_train/
      mrpc_qa_train/
      qnli_qa_train/
      rte_qa_train/
      sst2_qa_train/
    Qwen2.5/
      cola_qa_train/
      mrpc_qa_train/
      qnli_qa_train/
      rte_qa_train/
      sst2_qa_train/
```

Only the paths referenced by the config you run are required. If your local
checkpoint names differ, either rename the directories to match the YAML files
or update the config paths.

## What Is Not Included

This release intentionally excludes:

- adapter weight files
- base model weights
- optimizer states and trainer checkpoints
- merged/fused output adapters
- local caches, logs, and machine-specific absolute paths

Before redistributing any adapter checkpoints, verify that their original model,
dataset, and fine-tuning licenses allow redistribution.

## Quick Sanity Check

For the GENOME 1+9 setting, this command lists missing adapter configs:

```bash
for path in \
  llama3.1-8b-instruct/GENOME/code_alpaca_fast \
  gemma-2-2b-it/GENOME/cot \
  gemma-2-2b-it/GENOME/flan_v2 \
  gemma-2-2b-it/GENOME/gpt4_alpaca \
  gemma-2-2b-it/GENOME/lima \
  gemma-2-2b-it/GENOME/oasst1 \
  gemma-2-2b-it/GENOME/open_orca \
  gemma-2-2b-it/GENOME/science_literature \
  gemma-2-2b-it/GENOME/sharegpt \
  gemma-2-2b-it/GENOME/wizardlm
do
  test -f "${ADAPTER_ROOT:-$PWD/adapters}/$path/adapter_config.json" || echo "missing: $path"
done
```
