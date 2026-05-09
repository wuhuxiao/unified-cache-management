---
name: vllm-ucm-llmperf
description: Run vLLM online serving performance tests for Unified Cache Management using the project's internal llmperf tool, including baseline, HBM prefix-cache hit-rate, and UCM external-cache hit-rate scenarios with TTFT, TPOT, and E2E comparisons.
---

# vLLM UCM LLMPerf

Use the repo scripts in `scripts/llmperf_vllm_ucm/` when asked to benchmark vLLM serving with UCM or prefix-cache hit rates.

## Default Benchmark

Run the complete three-scenario benchmark from the repo root:

```bash
scripts/llmperf_vllm_ucm/run_three_scenarios.sh
```

Defaults:

- `MODEL_PATH=/home/models/DeepSeek-V4-Flash`
- `INPUT_TOKENS=16000`
- `OUTPUT_TOKENS=512`
- `CONCURRENCY=8`
- `REQUESTS=8`
- `HIT_RATE=50`
- `PORT=10035`

The script starts and stops a vLLM online server for each scenario:

- `baseline`: no UCM connector, prefix caching disabled, hit rate 0.
- `hbm`: no UCM connector, vLLM prefix caching enabled, 50% warmup prefix.
- `ucm`: UCM connector enabled, vLLM prefix caching disabled, 50% warmup prefix.

It writes `baseline_summary.json`, `hbm_summary.json`, `ucm_summary.json`, `summary.json`, and `summary.md` under `results/llmperf_vllm_ucm/<timestamp>/`.

By default `KEEP_SERVER_ON_FAILURE=1`, so a benchmark failure leaves the active vLLM server running for debugging instead of repeatedly restarting it. Stop it with `stop_vllm.sh` after inspection.

## Useful Overrides

Set environment variables before running:

```bash
MODEL_PATH=/path/to/model \
UCM_CONFIG_FILE=/path/to/ucm_config.yaml \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
scripts/llmperf_vllm_ucm/run_three_scenarios.sh
```

Use `SERVER_READY_TIMEOUT` for slow model startup and `RESULTS_DIR` to force an output directory.
Use `REUSE_RUNNING_SERVER=1` only when a matching server is already running and you want to rerun the llmperf request phase without starting a new server.

## Single Scenario

Start one server:

```bash
SCENARIO=ucm scripts/llmperf_vllm_ucm/start_vllm_perf_server.sh
```

Run one llmperf case against an existing server:

```bash
python3 scripts/llmperf_vllm_ucm/run_llmperf_case.py \
  --scenario ucm \
  --server-url http://127.0.0.1:10035 \
  --model DeepSeek-V4-Flash \
  --tokenizer-path /home/models/DeepSeek-V4-Flash \
  --input-tokens 16000 \
  --output-tokens 512 \
  --concurrency 8 \
  --requests 8 \
  --hit-rate 50
```

Use `stop_vllm.sh` to clean up running vLLM services.
