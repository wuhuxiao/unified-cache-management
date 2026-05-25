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
- Load-error and async-dump lifecycle tests: `test/test_hma_connector_load_errors.py`
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
- Keep block-size naming scoped to three concepts:
  - `hash_block_size`: connector hash/store-key block size.
  - `group_token_block_sizes`: per-KV-group logical token block size.
  - `group_tensor_block_sizes`: per-KV-group HBM tensor block span.
- Preserve two stores:
  - `fa_store`: full-attention groups. Store/load every reusable prefix block.
  - `wa_store`: window-attention groups. Store tail blocks at each prefix boundary, load only the final matched boundary.
- `hash_block_size` defaults to `256` for GPU FAWA and `512` for Ascend FAWA.
- Remote keys currently use `generate_hash(self.hash_block_size, request.all_token_ids, self._seed)`. Do not switch to `request.block_hashes` unless explicitly requested.
- Keep the full-hit `external_hit_tokens -= 1` behavior unless the user explicitly asks to change it.
- Current `KVCacheGroupMeta.tail_blocks` is derived as `max(tail_tokens // token_block_size, 1)` after `tail_tokens` is selected. Compressor state groups use `window_size - layer_compress_ratios[layer_index]` for `tail_tokens`; SWA groups use `window_size`; FA groups use `hash_block_size`.
- `update_state_after_alloc()` is intentionally a no-op in the current range-metadata implementation; `build_connector_meta()` owns block-id accumulation from `scheduled_new_reqs.block_ids` and `scheduled_cached_reqs.new_block_ids`.
- `FAWARequestMeta.vllm_block_ids` stores accumulated per-KV-group block ids. Cached request updates append `new_block_ids`; preemption resume resets the accumulated rows before appending the resumed allocation.
- `FAWARequestMeta.token_processed` drives dump progress. `_generate_dispatch_meta()` computes dump ranges from `token_processed` and `num_scheduled_tokens`, then advances `token_processed` to `min(num_token_ids, token_processed + new_tokens)`.
- Window boundary token indices are inclusive end-of-hash-block positions: `(hash_index + 1) * hash_block_size - 1`. Keep this for both load and dump slicing so WA tail rows match the final token of each canonical block.
- Load failures reported through `get_block_ids_with_load_errors()` must use vLLM/HMA block ids from the first KV-cache group anchor row. vLLM's scheduler currently matches invalid ids against the first block-id list from `kv_cache_manager.get_block_ids(req_id)`, so FAWA must not report WA/state group block ids, hash block indices, or canonical row indices. Ignore scratch segments with `block_id=-1`.
- Current FAWA dump completion is task-based: `wait_for_save()` submits FA/WA dump tasks into `tp_dump_tasks`; `handle_preemptions()` drains all pending dump tasks; `request_finished_all_groups()` waits tasks associated with the request and still returns `(False, None)`.

## Ascend FAWA Layout Notes

- `UCMFAWAConnector.can_handle_ascend_kv_cache_config()` handles Ascend FAWA configs when the first KV-cache group type starts with `Ascend` and the specs include `Compress4AttentionSpec`, `C4IndexerSpec`, and `Compress128AttentionSpec`.
- Ascend canonical hash blocks are 512 tokens. Compressed FA groups can store a 512-token canonical segment inside larger tensor blocks:
  - `Compress4AttentionSpec`: tensor block 512 tokens.
  - `C4IndexerSpec`: tensor block 4096 tokens, so 8 canonical segments can map to one tensor block with different offsets.
  - `Compress128AttentionSpec`: tensor block 16384 tokens, so 32 canonical segments can map to one tensor block with different offsets.
- Ascend registered KV tensors use `[num_blocks, block_size, num_head, head_dim]` views. `C4IndexerSpec` contributes two tensors per layer; the group/layer tensor-index mapping must follow vllm-ascend tuple order.
- Ascend window/state groups may have trimmed tail spans:
  - SWA groups keep normal sliding-window blocks.
  - C4 state groups use the final `window_tokens - compress_ratio` tokens when a compress ratio applies.
  - C128 state groups can have zero WA tail when `window_tokens <= compress_ratio`.
- For WA external hits, still load only the final matched boundary. Earlier missing WA tails may be scratch-loaded and must not force a full WA load for every external-hit block.

## GPU FAWA Layout Notes

- GPU layouts may register KV tensors as `[num_blocks, 2, block_size, ...]`; split the K/V axis before deriving token block size and pointer rows.
- Keep support for legacy `[2, num_blocks, block_size, ...]` tensors and mixed 3D group tensors such as `[num_blocks, 64, ...]` plus `[num_blocks, 2, ...]`.
- Avoid assuming one tensor-view block size across a group. Use `tensor_block_sizes`, `tensor_token_strides`, and `segment_tensor_size_list()` for segmented loads/dumps.
- FA and WA stores can have different row byte sizes for the same canonical key. Tests should verify the two stores do not overwrite each other.

## Scheduler/Worker Lifecycle

Expected scheduler side:

1. `get_num_new_matched_tokens()` queries `fa_store` and `wa_store`, returns the minimum continuous external hit.
2. `update_state_after_alloc()` is a lifecycle no-op for current FAWA range metadata.
3. `build_connector_meta()` emits per-request load/dump plans and accumulates block ids from scheduled new/cached request metadata.
4. Finished request ids are removed from `requests_meta` in `build_connector_meta()`.

Expected worker side:

1. `bind_connector_metadata()` stores metadata from scheduler.
2. `start_load_kv()` loads FA rows for every external-hit block and WA rows only for the last external-hit block.
3. `wait_for_save()` dumps FA rows for its TP slice and dumps WA rows by request-level TP ring assignment; submitted tasks are tracked in `tp_dump_tasks`.
4. `handle_preemptions()` waits all tracked dump tasks and clears `tp_dump_tasks`.
5. `request_finished_all_groups()` waits dump tasks whose tracked request tuple contains the finished request and returns `(False, None)`.
6. `get_finished()` and `build_connector_worker_meta()` are no-ops for the current FAWA path.

## Common Debugging Checks

When an online test fails, first inspect the vLLM log instead of restarting:

```bash
rg -n "Traceback|RuntimeError|EngineCore|FAWA|load|dump|Wait for dumping" results/llmperf_vllm_ucm/vllm_ucm.*.log
```

If `/health` returns 502 or fails through a proxy, disable proxies for localhost:

```bash
curl --noproxy '*' -i http://127.0.0.1:10035/health
```

For chunk prefill issues, check:

- `scheduled_cached_reqs.new_block_ids` may be `None` or partial.
- append vs replace must follow `scheduled_cached_reqs.resumed_from_preemption[i]` when available, otherwise `request_id in scheduled_cached_reqs.resumed_req_ids`.
- `new_block_ids is None` should behave as empty per-group updates.
- `token_processed` controls which canonical keys are dumped; partial allocation behavior depends on whether the accumulated `vllm_block_ids` have enough rows for `_slice_group_block_ids()`.
- If load or dump slices look shifted by one block, inspect the inclusive boundary expression in `_generate_dispatch_meta()`.
- If dump tasks appear lost, inspect `tp_dump_tasks` keys, `handle_preemptions()`, and `request_finished_all_groups()`.

## End-to-End Simulation Expectations

- TP4 simulations should build one scheduler connector and four worker connectors with `tp_rank=0..3`.
- HBM tensors in accelerator E2E tests must be allocated on the current accelerator: prefer NPU when `torch.npu.is_available()`, otherwise CUDA when `torch.cuda.is_available()`. Skip accelerator E2E tests when neither exists.
- Worker tests should exercise real connector lifecycle calls: `bind_connector_metadata()`, `start_load_kv()`, and `wait_for_save()`. Use a store double only behind the connector API; do not bypass connector pointer extraction.
- TP4 save behavior splits FA dumps by TP rank. WA dumps are assigned round-robin by request across TP ranks.
- Load-error tests should assert `get_block_ids_with_load_errors()` reports first-group anchor vLLM block ids exactly once.
- Simulate partial external hits by pre-seeding both FA and WA stores for the prefix. Pre-seed FA for each matched canonical block and WA for each boundary, then assert WA load uses only the last matched boundary.
- Chunk-prefill tests should cover cached `new_block_ids`, `new_block_ids=None`, preemption resume reset, and the inclusive WA boundary slicing for load and dump paths.

## Verification

Always run focused checks after editing FAWA logic:

```bash
python3 -m py_compile \
  ucm/integration/vllm/hma_connector.py \
  ucm/integration/vllm/ucm_connector.py \
  test/test_hma_connector_load_errors.py \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py

python3 -m pytest \
  test/test_hma_connector_load_errors.py \
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
