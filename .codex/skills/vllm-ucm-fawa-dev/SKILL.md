---
name: vllm-ucm-fawa-dev
description: Develop, debug, and validate the UCMFAWAConnector HMA integration for vLLM Unified Cache Management, including FA/WA two-store behavior, kv_cache_group layout handling, chunk-prefill edge cases, simulated tests, and UCM 50% hit llmperf validation.
---

# vLLM UCM FAWA Development

Use this skill when working on `ucm/integration/vllm/hma_connector.py` or the `UCMConnector` HMA wrapper in `ucm/integration/vllm/ucm_connector.py`.

## Core Files

- Main implementation: `ucm/integration/vllm/hma_connector.py`
- Wrapper delegation: `ucm/integration/vllm/ucm_connector.py`
- Focused simulated tests: `test/test_hma_connector_chunk_prefill.py`
- Accelerator end-to-end simulations:
  - `test/test_hma_connector_ascend_tp4_e2e.py`
  - `test/test_hma_connector_gpu_tp4_e2e.py`
- Online perf scripts: `scripts/llmperf_vllm_ucm/`
- vLLM lifecycle references:
  - `/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
  - `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py`
  - `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py`
  - `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py`

## Invariants

- Keep the implementation general. Do not add DeepSeek-specific class or variable names to FAWA logic.
- Preserve two stores:
  - `fa_store`: full-attention groups. Store/load every reusable prefix block.
  - `wa_store`: window-attention groups. Store tail blocks at each prefix boundary, load only the final matched boundary.
- `hash_block_size` comes from the FA kv-cache group block size. For current DeepSeek V4-style layouts this is usually `256`.
- Remote keys currently use `generate_hash(self.hash_block_size, request.all_token_ids, self._seed)` plus `_block_key()`. Do not switch to `request.block_hashes` unless explicitly requested.
- Keep the full-hit `external_hit_tokens -= 1` behavior unless the user explicitly asks to change it.
- `group_tail_blocks` for compressor state groups depends on model ratio:
  - if `window_tokens <= compress_ratio`, tail blocks are `0`.
  - otherwise `ceil((window_tokens - compress_ratio) / group_block_size)`.
  - non-compressor window groups use `max(1, ceil(window_tokens / group_block_size))`.
- `update_state_after_alloc()` must record allocated group block ids for all groups and immediately derive any contiguous `group_block_ids` that are now recoverable.
- Chunk prefill may provide partial `new_block_ids`; only create dump rows for contiguous fully recorded canonical blocks.
- Chunk prefill dump progress is governed by `store_block_cursor`, not only `token_processed`. A partial allocation can advance `token_processed` to request length before all group rows are recorded; later allocation completion must still be able to dump remaining contiguous rows.
- For external hits, `build_connector_meta()` must only build a load plan after `group_block_ids` exist for the full external-hit prefix. Missing rows usually means `update_state_after_alloc()` failed to record allocated groups.

## Ascend FAWA Layout Notes

- Use `UCMAscendFAWAConnector` for vLLM Ascend KV configs detected by `FAWABlockSpanLayout.is_ascend_kv_cache_config()`.
- Ascend canonical hash blocks are 512 tokens. Compressed FA groups can store a 512-token canonical segment inside larger physical pages:
  - `Compress4AttentionSpec`: physical 512 tokens.
  - `C4IndexerSpec`: physical 4096 tokens, so 8 canonical segments can map to one physical block with different offsets.
  - `Compress128AttentionSpec`: physical 16384 tokens, so 32 canonical segments can map to one physical block with different offsets.
- Ascend registered KV tensors use `[num_blocks, block_size, num_head, head_dim]` views. `C4IndexerSpec` contributes two tensors per layer; the group/layer tensor-index mapping must follow vllm-ascend tuple order.
- Ascend window/state groups may have trimmed tail spans:
  - SWA groups keep normal sliding-window blocks.
  - C4 state groups use the final `window_tokens - compress_ratio` tokens when a compress ratio applies.
  - C128 state groups can have zero WA tail when `window_tokens <= compress_ratio`.
- For WA external hits, still load only the final matched boundary. Earlier missing WA tails may be scratch-loaded and must not force a full WA load for every external-hit block.

## GPU FAWA Layout Notes

- GPU layouts may register KV tensors as `[num_blocks, 2, block_size, ...]`; split the K/V axis before deriving token block size and pointer rows.
- Keep support for legacy `[2, num_blocks, block_size, ...]` tensors and mixed 3D group tensors such as `[num_blocks, 64, ...]` plus `[num_blocks, 2, ...]`.
- Avoid assuming one tensor block size across a group. Use `tensor_block_sizes`, `token_strides`, and `segment_tensor_size_list()` for segmented loads/dumps.
- FA and WA stores can have different row byte sizes for the same canonical key. Tests should verify the two stores do not overwrite each other.

## Scheduler/Worker Lifecycle

Expected scheduler side:

1. `get_num_new_matched_tokens()` queries `fa_store` and `wa_store`, returns the minimum continuous external hit.
2. `update_state_after_alloc()` converts HMA `KVCacheBlocks` to group block id rows and records contiguous canonical block rows.
3. `build_connector_meta()` emits per-request load/dump plans.
4. `request_finished_all_groups()` clears request state and returns `False` unless the connector owns async block release.

Expected worker side:

1. `bind_connector_metadata()` stores metadata from scheduler.
2. `start_load_kv()` loads FA rows for every external-hit block and WA rows only for the last external-hit block.
3. `wait_for_save()` dumps both FA and WA rows for completed dump blocks.
4. `get_finished()` and `build_connector_worker_meta()` are no-ops for the current synchronous FAWA path unless async completion is implemented.

## Common Debugging Checks

When an online test fails, first inspect the vLLM log instead of restarting:

```bash
rg -n "Traceback|RuntimeError|EngineCore|FAWA|load plan|record FAWA|dump FAWA" results/llmperf_vllm_ucm/vllm_ucm.*.log
```

If `/health` returns 502 or fails through a proxy, disable proxies for localhost:

```bash
curl --noproxy '*' -i http://127.0.0.1:10035/health
```

If the log shows:

```text
FAWA load plan is missing group block ids
```

check that `_record_allocated_group_block_ids()` calls `_record_ready_group_block_ids()` for first, replace, and append cases. A common bug is returning immediately after replacing `allocated_group_block_ids`, leaving `group_block_ids` empty for external-hit load plans.

For chunk prefill issues, check:

- `scheduled_cached_reqs.new_block_ids` may be `None` or partial.
- append vs replace must follow `request_id in resumed_req_ids`.
- `store_block_cursor` should prevent re-dumping externally loaded blocks.
- `token_processed` tracks scheduled token progress, while `store_block_cursor` tracks persisted canonical blocks.
- If `dump_block_ids` stays empty after the final chunk allocation arrives, inspect whether `_make_dispatch_meta()` is gating on `token_processed < num_token_ids`; the correct guard is whether `store_block_cursor` has remaining full canonical blocks.

## End-to-End Simulation Expectations

- TP4 simulations should build one scheduler connector and four worker connectors with `tp_rank=0..3`.
- HBM tensors in accelerator E2E tests must be allocated on the current accelerator: prefer NPU when `torch.npu.is_available()`, otherwise CUDA when `torch.cuda.is_available()`. Skip accelerator E2E tests when neither exists.
- Worker tests should exercise real connector lifecycle calls: `bind_connector_metadata()`, `start_load_kv()`, and `wait_for_save()`. Use a store double only behind the connector API; do not bypass connector pointer extraction.
- MLA TP4 save behavior is rank-0-only for this FAWA path. Nonzero TP workers should load successfully but not dump.
- Simulate partial external hits by pre-seeding both FA and WA stores for the prefix. Pre-seed FA for each matched canonical block and WA for each boundary, then assert WA load uses only the last matched boundary.
- For chunk-prefill allocation tests, first provide only a partial `new_block_ids` update and assert no dump occurs; then provide the remaining groups and assert only contiguous complete canonical rows are dumped.

## Verification

Always run focused checks after editing FAWA logic:

```bash
python3 -m py_compile \
  ucm/integration/vllm/hma_connector.py \
  ucm/integration/vllm/ucm_connector.py \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py

python3 -m pytest \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py
```

For online UCM 50% hit validation, use the `vllm-ucm-llmperf` skill. Preferred single-scenario flow:

```bash
SCENARIO=ucm PORT=10035 scripts/llmperf_vllm_ucm/start_vllm_perf_server.sh

python3 scripts/llmperf_vllm_ucm/run_llmperf_case.py \
  --scenario ucm \
  --server-url http://127.0.0.1:10035 \
  --model DeepSeek-V4-Flash \
  --tokenizer-path /home/models/DeepSeek-V4-Flash \
  --input-tokens 16000 \
  --output-tokens 512 \
  --concurrency 8 \
  --requests 8 \
  --hit-rate 50 \
  --results-dir results/llmperf_vllm_ucm/ucm_50_<timestamp>
```

Do not repeatedly relaunch vLLM for request-tool failures. If the server is still healthy, rerun only the llmperf request phase. Relaunch only after a real EngineCore fatal or code change that requires a new process.
