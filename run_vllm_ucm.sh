#!/usr/bin/env bash
# DeepSeek-V4-Flash + UCM：续行时反斜杠必须是行尾最后一个字符，后面不能有空格。

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
export VLLM_CPU_AFFINITY=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 可按部署修改模型路径、日志路径与 UCM 配置
MODEL_PATH="${MODEL_PATH:-/home/models/DeepSeek-V4-Flash}"
# 未指定 LOG_FILE 时：文件名带启动时间戳，避免多次运行混在同一文件
RUN_TS="$(date +%Y-%m-%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/vllmdsv4ucm.${RUN_TS}.log}"

# 如需换机器，只改下面 JSON 里 UCM_CONFIG_FILE 的绝对路径即可
KV_TRANSFER_JSON=$(
  python3 -c 'import json; print(json.dumps({
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"
    }
  }))'
)

exec >>"${LOG_FILE}" 2>&1
echo "======== $(date '+%Y-%m-%d %H:%M:%S %z') 启动 vllm serve ========="
echo "log_file=${LOG_FILE}  model_path=${MODEL_PATH}  pid=$$"
echo "CUDA_HOME=${CUDA_HOME}  nvcc=$(command -v nvcc || true)"
echo "========================================================="

exec vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 4 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
  --attention-config '{"use_fp4_indexer_cache": false}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser deepseek_v4 \
  --max-model-len 50000 \
  --kv-transfer-config "${KV_TRANSFER_JSON}"
