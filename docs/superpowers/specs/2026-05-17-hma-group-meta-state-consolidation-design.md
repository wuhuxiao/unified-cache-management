# HMA Group Meta State Consolidation Design

Date: 2026-05-17

## Goal

Consolidate FAWA per-KV-group derived state into `KVCacheGroupMeta`.
The connector should stop maintaining parallel tuple state for group block
sizes, tensor block spans, tensor ratios, tail blocks, and window spans.

This change does not rename existing meta fields. The current
`token_block_size` and `tensor_block_size` names stay in place for this step;
their naming will be handled separately.

## Current Problem

`UCMFAWAConnector` currently initializes several parallel tuples:

- `group_token_block_sizes`
- `group_tensor_block_sizes`
- `group_tensor_block_ratios`
- `group_tail_blocks`
- `group_window_spans`

Then `_init_group_metas()` copies those values into `KVCacheGroupMeta`.
This leaves two sources of truth. Tests and worker fixtures also mutate these
tuples directly, which makes the flow harder to reason about and makes later
renaming more risky.

`FAWABlockSpanLayout` has the same shape: it computes and stores group block
size tuples as attributes, then the Ascend connector copies them into connector
state before building metas.

## Design

`KVCacheGroupMeta` becomes the single source of truth for per-group derived
state after connector initialization.

`_init_group_metas()` is the only connector-level initialization function for
this state. It computes each group's:

- `token_block_size`
- `tensor_block_size`
- `logical_blocks_per_hash_block`
- `hash_blocks_per_tensor_block`
- `tail_blocks`
- `window_spans`

No extra connector helper functions should be added for this refactor. Existing
helper functions that only exist to compute these tuple fields should be
removed or folded into `_init_group_metas()`.

After `_init_group_metas()` finishes, runtime code reads:

```python
meta = self.group_metas[group_id]
```

and uses fields from `meta`, instead of reading parallel tuple attributes.

## Compatibility

During this refactor, compatibility accessors may remain for code and tests
that still read old names:

- `group_token_block_sizes`
- `group_tensor_block_sizes`
- `group_tensor_block_ratios`
- `group_tail_blocks`
- `group_window_spans`

If present, these should be read-only properties derived from `group_metas`.
They must not be independently assigned in normal connector initialization.

Test fixtures should prefer constructing `group_metas` through
`_init_group_metas()` or assigning a complete `group_metas` dict directly.

## FAWABlockSpanLayout

`FAWABlockSpanLayout` should no longer expose tuple attributes as the primary
data contract for connector initialization. It may use local intermediate
values while building Ascend group metadata, but the connector should consume
the resulting group metadata, not copy a set of tuple fields.

Ascend-specific behavior must remain unchanged:

- canonical hash block size is 512 tokens
- compressed FA groups can map canonical segments into larger tensor blocks
- SWA groups keep normal sliding-window spans
- C4/C128 state groups use trimmed or empty WA tails as before
- layer tensor-index mapping remains available for splitting registered cache
  tensors

## Runtime Paths To Update

The following paths should read from `self.group_metas[group_id]`:

- `_store_tensor_size_list()`
- `_slice_group_block_ids()`
- `_extract_fa_ptr()`
- `_extract_wa_ptr()`
- logging and test-facing compatibility paths

The already-approved range metadata behavior must not change:

- scheduler metadata contains hash ranges plus per-group vLLM block ids
- WA load uses only the final external-hit boundary
- WA dump uses tail candidates per dumped hash block
- zero-tail WA groups use empty candidate lists
- `update_state_after_alloc()` remains a no-op for this design
- old complete-candidate gating is not reintroduced

## Non-Goals

- Do not rename `token_block_size` or `tensor_block_size` in this step.
- Do not reintroduce `KVCacheSegment`, row metadata, or old cursor state.
- Do not add extra private helper functions for group meta construction.
- Do not change FA/WA store semantics.

## Testing

Focused validation:

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

Additional review checks:

- `hma_connector.py` has no `KVCacheSegment` or `KVCacheGroupRow` references.
- Connector initialization does not assign the old tuple fields as real state.
- Runtime logic reads per-group values from `group_metas`.
