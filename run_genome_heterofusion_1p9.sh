#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

HF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HF_ROOT}"
CONFIG="${HF_ROOT}/configs/heterofusion/genome_1p9/llama_code_target_gemma9_sources.yaml"
BASE_MODEL="${BASE_MODEL_PATH:-${MODEL_ROOT:-}/llama-3.1-8b-instruct}"
RUN_NAME="GENOME_1P9_LLAMA_CODE_GEMMA9_TEST6"
RESULT_ROOT="${HF_ROOT}/infer_results/${RUN_NAME}"
WORK_ROOT="/dev/shm/genome_1p9_eval_workspace"
LOG_DIR="${HF_ROOT}/logs/genome_1p9"
TASKS=(mmlupro gsm8k mbpp drop flores37 emorynlp)

mkdir -p "${RESULT_ROOT}" "${WORK_ROOT}" "${LOG_DIR}"

if [[ "${BASE_MODEL}" == "/llama-3.1-8b-instruct" ]]; then
  echo "Please set MODEL_ROOT or BASE_MODEL_PATH before running this script." >&2
  exit 1
fi

echo "[1/4] Building GENOME valid replay dataset..."
conda run --no-capture-output -n infer_train python "${HF_ROOT}/tools/build_genome_valid_mix_dataset.py"

echo "[2/4] Running HeteroFusion training on GPU ${CUDA_VISIBLE_DEVICES}..."
cd "${HF_ROOT}"
conda run --no-capture-output -n infer_train python main.py "${CONFIG}" 2>&1 | tee "${LOG_DIR}/train_${RUN_NAME}.log"

FUSED_LORA="${HF_ROOT}/outputs/genome_1p9/llama_code_target_gemma9_sources/genome_1p9_valid_mix_tail_b_only/merged_lora"
if [[ ! -f "${FUSED_LORA}/adapter_model.safetensors" ]]; then
  echo "Missing fused LoRA: ${FUSED_LORA}/adapter_model.safetensors" >&2
  exit 1
fi

echo "[3/4] Evaluating fused LoRA on GENOME test split..."
for TASK in "${TASKS[@]}"; do
  TASK_OUT="${RESULT_ROOT}/heterofusion_1p9/${TASK}"
  TASK_WORK="${WORK_ROOT}/${TASK}"
  mkdir -p "${TASK_OUT}" "${TASK_WORK}"
  echo "  - ${TASK}"
  conda run --no-capture-output -n infer_train python "${HF_ROOT}/tools/run_genome_merged_eval.py" \
    --model-path "${BASE_MODEL}" \
    --lora-path "${FUSED_LORA}" \
    --task "${TASK}" \
    --split test \
    --output-dir "${TASK_OUT}" \
    --work-dir "${TASK_WORK}" \
    2>&1 | tee "${LOG_DIR}/eval_${TASK}.log"
done

echo "[4/4] Done."
echo "Fused LoRA: ${FUSED_LORA}"
echo "Results: ${RESULT_ROOT}"
