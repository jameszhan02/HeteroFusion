#!/usr/bin/env bash

set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-csl}"
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONFIG_DIR="$WORK_DIR/configs/heterofusion/paper_experiments/ablation_main_qwen_to_llama"
BASE_CONFIG_ROOT="$CONFIG_DIR"

LOG_ROOT="$WORK_DIR/logs"
RUN_NAME="paper_main_ablation_qwen_to_llama_$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_ROOT/$RUN_NAME"

STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
SKIP_FINISHED="${SKIP_FINISHED:-1}"
DRY_RUN="${DRY_RUN:-0}"
GPU_LIST="${GPU_LIST:-0 4 6}"
GPU_MAP_FILE="${GPU_MAP_FILE:-}"

declare -A CONFIG_GPU_MAP=()

print_header() {
  echo "============================================================"
  echo ">>> $1"
  echo "============================================================"
}

activate_env() {
  print_header "Initialize Environment"

  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
  set -u

  cd "$WORK_DIR"
  mkdir -p "$RUN_LOG_DIR"

  echo "ENV: $CONDA_ENV_NAME"
  echo "WORK_DIR: $(pwd)"
  echo "RUN_LOG_DIR: $RUN_LOG_DIR"
  echo "GPU_LIST: $GPU_LIST"
}

load_gpu_map_file() {
  if [[ -n "$GPU_MAP_FILE" && -f "$GPU_MAP_FILE" ]]; then
    while read -r rel gpu _; do
      [[ -z "${rel:-}" ]] && continue
      [[ "$rel" =~ ^# ]] && continue
      [[ -z "${gpu:-}" ]] && continue
      CONFIG_GPU_MAP["$rel"]="$gpu"
    done < "$GPU_MAP_FILE"
    echo "Loaded GPU map file: $GPU_MAP_FILE"
  fi
}

collect_configs() {
  if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "ERROR: missing config dir: $CONFIG_DIR"
    exit 1
  fi

  mapfile -t cfgs < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
  if [[ ${#cfgs[@]} -eq 0 ]]; then
    echo "ERROR: no yaml configs found in $CONFIG_DIR"
    exit 1
  fi
  printf '%s\n' "${cfgs[@]}"
}

get_output_dir_from_yaml() {
  local cfg="$1"
  awk '
    /^[[:space:]]*output_dir:[[:space:]]*/ {
      sub(/^[[:space:]]*output_dir:[[:space:]]*/, "", $0)
      sub(/[[:space:]]+#.*/, "", $0)
      gsub(/"/, "", $0)
      gsub(/\047/, "", $0)
      print $0
      exit
    }
  ' "$cfg"
}

abs_path_from_config_output_dir() {
  local out_dir="$1"
  if [[ "$out_dir" = /* ]]; then
    echo "$out_dir"
  else
    echo "$WORK_DIR/$out_dir"
  fi
}

run_one_config_on_gpu() {
  local cfg="$1"
  local idx="$2"
  local gpu="$3"
  local queue_id="$4"

  local rel cfg_name output_dir abs_output done_marker log_file
  cfg_name="$(basename "$cfg" .yaml)"
  rel="${cfg#$BASE_CONFIG_ROOT/}"
  output_dir="$(get_output_dir_from_yaml "$cfg")"

  if [[ -z "$output_dir" ]]; then
    echo "[$queue_id:$idx][GPU $gpu] ERROR: cannot parse output_dir in $cfg"
    return 1
  fi

  abs_output="$(abs_path_from_config_output_dir "$output_dir")"
  done_marker="$(find "$abs_output" -path '*/merged_lora/adapter_model.safetensors' -print -quit 2>/dev/null || true)"
  log_file="$RUN_LOG_DIR/gpu${gpu}_$(printf '%02d' "$idx")_${cfg_name}.log"

  echo "[$queue_id:$idx][GPU $gpu] config=$rel"
  echo "[$queue_id:$idx][GPU $gpu] output_dir=$output_dir"
  echo "[$queue_id:$idx][GPU $gpu] log=$log_file"

  if [[ "$SKIP_FINISHED" == "1" && -f "$done_marker" ]]; then
    echo "[$queue_id:$idx][GPU $gpu] SKIP: already finished"
    return 2
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$queue_id:$idx][GPU $gpu] DRY_RUN=1"
    return 0
  fi

  local start_ts end_ts dur
  start_ts="$(date +%s)"
  if CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python main.py --config_path "$cfg" > "$log_file" 2>&1; then
    end_ts="$(date +%s)"
    dur="$((end_ts - start_ts))"
    echo "[$queue_id:$idx][GPU $gpu] OK (${dur}s)"
    return 0
  else
    end_ts="$(date +%s)"
    dur="$((end_ts - start_ts))"
    echo "[$queue_id:$idx][GPU $gpu] FAIL (${dur}s)"
    return 1
  fi
}

worker() {
  local gpu="$1"
  local list_file="$2"

  local queue_id="W${gpu}"
  local summary_file="$RUN_LOG_DIR/summary_gpu${gpu}.txt"
  local worker_log="$RUN_LOG_DIR/worker_gpu${gpu}.log"
  local ok=0 skip=0 fail=0 idx=0 rc

  {
    echo "[$queue_id] start worker on GPU $gpu"
    while IFS= read -r cfg; do
      [[ -z "$cfg" ]] && continue
      idx=$((idx + 1))
      if run_one_config_on_gpu "$cfg" "$idx" "$gpu" "$queue_id"; then
        ok=$((ok + 1))
      else
        rc=$?
        if [[ "$rc" -eq 2 ]]; then
          skip=$((skip + 1))
        else
          fail=$((fail + 1))
          if [[ "$STOP_ON_ERROR" == "1" ]]; then
            echo "[$queue_id] STOP_ON_ERROR=1, stop this worker"
            break
          fi
        fi
      fi
    done < "$list_file"
    echo "[$queue_id] done: ok=$ok skip=$skip fail=$fail"
  } > >(tee "$worker_log") 2>&1

  {
    echo "gpu=$gpu"
    echo "ok=$ok"
    echo "skip=$skip"
    echo "fail=$fail"
  } > "$summary_file"

  [[ "$fail" -eq 0 ]]
}

aggregate_and_print_summary() {
  local total_ok=0 total_skip=0 total_fail=0
  local f ok skip fail
  for f in "$RUN_LOG_DIR"/summary_gpu*.txt; do
    [[ -f "$f" ]] || continue
    ok="$(awk -F= '/^ok=/{print $2}' "$f")"
    skip="$(awk -F= '/^skip=/{print $2}' "$f")"
    fail="$(awk -F= '/^fail=/{print $2}' "$f")"
    total_ok=$((total_ok + ok))
    total_skip=$((total_skip + skip))
    total_fail=$((total_fail + fail))
  done

  print_header "Parallel Run Finished"
  echo "success: $total_ok"
  echo "skipped: $total_skip"
  echo "failed:  $total_fail"
  echo "run logs: $RUN_LOG_DIR"

  [[ "$total_fail" -eq 0 ]]
}

main() {
  activate_env
  load_gpu_map_file

  mapfile -t CONFIG_LIST < <(collect_configs)
  read -r -a GPUS <<< "$GPU_LIST"
  if [[ ${#GPUS[@]} -eq 0 ]]; then
    echo "ERROR: GPU_LIST is empty"
    exit 1
  fi

  print_header "Build GPU Queues"
  echo "config_count=${#CONFIG_LIST[@]}"
  echo "STOP_ON_ERROR=$STOP_ON_ERROR SKIP_FINISHED=$SKIP_FINISHED DRY_RUN=$DRY_RUN"

  local queue_dir="$RUN_LOG_DIR/queues"
  mkdir -p "$queue_dir"

  local cfg rel gpu gpu_from qf i
  for i in "${!CONFIG_LIST[@]}"; do
    cfg="${CONFIG_LIST[$i]}"
    rel="${cfg#$BASE_CONFIG_ROOT/}"
    if [[ -n "${CONFIG_GPU_MAP[$rel]+x}" ]]; then
      gpu="${CONFIG_GPU_MAP[$rel]}"
      gpu_from="config"
    else
      gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
      gpu_from="round-robin"
    fi
    qf="$queue_dir/gpu${gpu}.list"
    echo "$cfg" >> "$qf"
    echo "assign: $rel -> GPU $gpu (from $gpu_from)"
  done

  print_header "Start Workers"
  local qfile gpu_name pid any_fail=0
  local -a pids=()
  for qfile in "$queue_dir"/gpu*.list; do
    [[ -f "$qfile" ]] || continue
    gpu_name="$(basename "$qfile" .list)"
    gpu_name="${gpu_name#gpu}"
    worker "$gpu_name" "$qfile" &
    pid=$!
    pids+=("$pid")
    echo "worker started: GPU $gpu_name pid=$pid"
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      any_fail=1
    fi
  done

  if ! aggregate_and_print_summary; then
    exit 1
  fi
  if [[ "$any_fail" -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
