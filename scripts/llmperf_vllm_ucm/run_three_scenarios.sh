#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/models/DeepSeek-V4-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10035}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:${PORT}}"
UCM_CONFIG_FILE="${UCM_CONFIG_FILE:-${REPO_ROOT}/examples/ucm_config_example.yaml}"
INPUT_TOKENS="${INPUT_TOKENS:-16000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
CONCURRENCY="${CONCURRENCY:-8}"
REQUESTS="${REQUESTS:-8}"
HIT_RATE="${HIT_RATE:-50}"
RANDOM_SEED="${RANDOM_SEED:-42}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
PREFILL_SLEEP_S="${PREFILL_SLEEP_S:-2}"
KEEP_SERVER_ON_FAILURE="${KEEP_SERVER_ON_FAILURE:-1}"
REUSE_RUNNING_SERVER="${REUSE_RUNNING_SERVER:-0}"
RUN_TS="$(date +%Y-%m-%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/llmperf_vllm_ucm/${RUN_TS}}"

mkdir -p "${RESULTS_DIR}"

server_pid=""

stop_server() {
  if [[ -x "${REPO_ROOT}/stop_vllm.sh" ]]; then
    GRACE_SECONDS="${GRACE_SECONDS:-20}" "${REPO_ROOT}/stop_vllm.sh" || true
  elif [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
}

wait_for_server() {
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  local health_url="${SERVER_URL%/}/health"
  echo "[INFO] Waiting for ${health_url}"
  until env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}" \
    curl -fsS "${health_url}" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
      echo "[ERROR] vLLM server did not become ready within ${SERVER_READY_TIMEOUT}s" >&2
      return 1
    fi
    sleep 5
  done
  echo "[INFO] vLLM server is ready"
}

start_server() {
  local scenario="$1"
  local log_file="${RESULTS_DIR}/vllm_${scenario}.log"
  if [[ "${REUSE_RUNNING_SERVER}" == "1" ]]; then
    echo "[INFO] Reusing existing vLLM server for scenario=${scenario}"
    wait_for_server
    return
  fi
  echo "[INFO] Starting scenario=${scenario}, log=${log_file}"
  SCENARIO="${scenario}" \
    MODEL_PATH="${MODEL_PATH}" \
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
    HOST="${HOST}" \
    PORT="${PORT}" \
    UCM_CONFIG_FILE="${UCM_CONFIG_FILE}" \
    LOG_FILE="${log_file}" \
    "${SCRIPT_DIR}/start_vllm_perf_server.sh" &
  server_pid="$!"
  wait_for_server
}

run_case() {
  local scenario="$1"
  local hit_rate="$2"
  echo "[INFO] Running llmperf scenario=${scenario} hit_rate=${hit_rate}"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}" \
    python3 "${SCRIPT_DIR}/run_llmperf_case.py" \
    --scenario "${scenario}" \
    --server-url "${SERVER_URL}" \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer-path "${MODEL_PATH}" \
    --input-tokens "${INPUT_TOKENS}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --concurrency "${CONCURRENCY}" \
    --requests "${REQUESTS}" \
    --hit-rate "${hit_rate}" \
    --random-seed "${RANDOM_SEED}" \
    --prefill-sleep-s "${PREFILL_SLEEP_S}" \
    --results-dir "${RESULTS_DIR}"
}

run_scenario() {
  local scenario="$1"
  local hit_rate="$2"
  if [[ "${REUSE_RUNNING_SERVER}" != "1" ]]; then
    stop_server
  fi
  start_server "${scenario}"
  run_case "${scenario}" "${hit_rate}"
  if [[ "${REUSE_RUNNING_SERVER}" != "1" ]]; then
    stop_server
  fi
}

on_exit() {
  local rc="$?"
  if ((rc != 0)) && [[ "${KEEP_SERVER_ON_FAILURE}" == "1" ]] && [[ -n "${server_pid}" ]]; then
    echo "[WARN] Benchmark failed; leaving vLLM server running for debugging because KEEP_SERVER_ON_FAILURE=1."
    echo "[WARN] Stop it manually with ${REPO_ROOT}/stop_vllm.sh when finished."
  else
    stop_server
  fi
}

trap on_exit EXIT

echo "[INFO] Results dir: ${RESULTS_DIR}"
echo "[INFO] Config: input=${INPUT_TOKENS}, output=${OUTPUT_TOKENS}, concurrency=${CONCURRENCY}, requests=${REQUESTS}, hit_rate=${HIT_RATE}"

run_scenario baseline 0
run_scenario hbm "${HIT_RATE}"
run_scenario ucm "${HIT_RATE}"

python3 "${SCRIPT_DIR}/summarize_results.py" --results-dir "${RESULTS_DIR}"
