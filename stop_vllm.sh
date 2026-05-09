#!/usr/bin/env bash
set -euo pipefail

GRACE_SECONDS="${GRACE_SECONDS:-10}"

patterns=(
  "vllm.entrypoints.openai.api_server"
  "vllm serve"
  "vllm.entrypoints"
  "api_server"
  "VLLM::"
  "offline_inference.py"
  "run_vllm_ucm.sh"
)

worker_patterns=(
  "vllm_worker"
  "multiproc_worker_utils"
  "ray::"
  "raylet"
  "torchrun"
)

current_pid="$$"

find_pids() {
  local pattern="$1"
  pgrep -f "$pattern" 2>/dev/null | awk -v self="$current_pid" '$1 != self'
}

collect_pids() {
  local -n _patterns="$1"
  local pids=()
  local pid pattern

  for pattern in "${_patterns[@]}"; do
    while read -r pid; do
      [[ -n "$pid" ]] && pids+=("$pid")
    done < <(find_pids "$pattern")
  done

  if ((${#pids[@]} == 0)); then
    return 0
  fi

  printf "%s\n" "${pids[@]}" | sort -n | uniq
}

terminate_pids() {
  local signal="$1"
  shift
  local pids=("$@")

  if ((${#pids[@]} == 0)); then
    return 0
  fi

  echo "Sending ${signal} to: ${pids[*]}"
  kill "-${signal}" "${pids[@]}" 2>/dev/null || true
}

wait_for_exit() {
  local pids=("$@")
  local deadline=$((SECONDS + GRACE_SECONDS))
  local alive=()
  local pid

  while ((SECONDS < deadline)); do
    alive=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive+=("$pid")
      fi
    done
    ((${#alive[@]} == 0)) && return 0
    sleep 1
  done

  printf "%s\n" "${alive[@]}"
}

main() {
  local pids=()
  local remaining=()
  local worker_pids=()

  mapfile -t pids < <(collect_pids patterns)
  if ((${#pids[@]} == 0)); then
    echo "No matching vLLM server process found."
  else
    terminate_pids TERM "${pids[@]}"
    mapfile -t remaining < <(wait_for_exit "${pids[@]}")
    if ((${#remaining[@]} > 0)); then
      terminate_pids KILL "${remaining[@]}"
    fi
  fi

  mapfile -t worker_pids < <(collect_pids worker_patterns)
  if ((${#worker_pids[@]} > 0)); then
    terminate_pids TERM "${worker_pids[@]}"
    mapfile -t remaining < <(wait_for_exit "${worker_pids[@]}")
    if ((${#remaining[@]} > 0)); then
      terminate_pids KILL "${remaining[@]}"
    fi
  fi

  if command -v ray >/dev/null 2>&1; then
    echo "Running ray stop --force"
    ray stop --force >/dev/null 2>&1 || true
  fi

  echo "vLLM stop routine completed."
}

main "$@"
