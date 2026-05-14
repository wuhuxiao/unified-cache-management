#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENARIO="${SCENARIO:-ucm}"
MODEL_PATH="${MODEL_PATH:-/home/models/DeepSeek-V4-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10035}"
UCM_CONFIG_FILE="${UCM_CONFIG_FILE:-${REPO_ROOT}/examples/ucm_config_example.yaml}"
RUN_TS="$(date +%Y-%m-%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/results/llmperf_vllm_ucm/vllm_${SCENARIO}.${RUN_TS}.log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
export VLLM_CPU_AFFINITY="${VLLM_CPU_AFFINITY:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${LOG_FILE}")"

common_args=(
  vllm serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --kv-cache-dtype fp8
  --block-size 256
  --enable-expert-parallel
  --tensor-parallel-size 4
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}'
  --attention-config '{"use_fp4_indexer_cache": false}'
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --no-disable-hybrid-kv-cache-manager
  --reasoning-parser deepseek_v4
  --max-model-len 50000
)

scenario_args=()
case "${SCENARIO}" in
  baseline)
    scenario_args+=(--no-enable-prefix-caching)
    ;;
  hbm)
    scenario_args+=(--enable-prefix-caching)
    ;;
  ucm)
    kv_transfer_json="$(
      UCM_CONFIG_FILE="${UCM_CONFIG_FILE}" python3 -c 'import json, os; print(json.dumps({
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "UCM_CONFIG_FILE": os.environ["UCM_CONFIG_FILE"]
        }
      }))'
    )"
    scenario_args+=(--no-enable-prefix-caching --kv-transfer-config "${kv_transfer_json}")
    ;;
  *)
    echo "Unsupported SCENARIO=${SCENARIO}. Use baseline, hbm, or ucm." >&2
    exit 2
    ;;
esac

extra_args=()
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  read -r -a extra_args <<<"${EXTRA_VLLM_ARGS}"
fi

exec >>"${LOG_FILE}" 2>&1
echo "======== $(date '+%Y-%m-%d %H:%M:%S %z') start vllm serve ========"
echo "scenario=${SCENARIO}"
echo "model_path=${MODEL_PATH}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "host=${HOST}"
echo "port=${PORT}"
echo "ucm_config_file=${UCM_CONFIG_FILE}"
echo "log_file=${LOG_FILE}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "repo_root=${REPO_ROOT}"
echo "==================================================================="

exec "${common_args[@]}" "${scenario_args[@]}" "${extra_args[@]}"
