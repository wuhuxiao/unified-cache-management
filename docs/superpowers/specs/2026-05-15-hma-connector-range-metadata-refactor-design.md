# HMA Connector Range Metadata Refactor Design

Date: 2026-05-15

## Context

`ucm/integration/vllm/hma_connector.py` is being refactored away from the old
`KVCacheSegment` and `KVCacheGroupRow` scheduler metadata model. The new model
keeps scheduler metadata compact: it sends canonical hash block ranges and
range-local vLLM block ids, while the worker computes KV cache pointers from
its local layouts.

The implementation must preserve FAWA behavior for both GPU and Ascend layouts,
including full-attention storage, window-attention boundary storage, Ascend
compressed tensor-block offsets, and existing vLLM/HMA lifecycle behavior.

## Goals

- Remove `KVCacheSegment`, `KVCacheGroupRow`, and allocation row metadata from
  the implementation.
- Make `FAWARequestDispatchMeta` explicit and range-based.
- Keep `build_connector_meta()` responsible for accumulating per-request vLLM
  block ids and emitting range-local candidate block ids.
- Move pointer computation to the worker side.
- Batch hash block pointer computation with numpy where practical.
- Keep the code style close to `ucm_connector.py`: clear lifecycle methods,
  minimal private helper proliferation, and simple flow in the main methods.

## Non-Goals

- Do not change hash generation. The connector still uses
  `generate_hash(self.hash_block_size, request.all_token_ids, self._seed)`.
- Do not change the full-hit `external_hit_tokens -= 1` behavior.
- Do not add scheduler-side checks that every hash block has a complete old-style
  per-group row. The current allocation path is assumed to provide usable block
  ids for the emitted range.
- Do not make FAWA own asynchronous block release. `request_finished_all_groups()`
  remains a no-op for release ownership.

## Metadata Boundary

`FAWARequestDispatchMeta` should use explicit fields instead of tuple-packed
load and dump metadata:

```python
@dataclass
class FAWARequestDispatchMeta:
    load_keys: list[bytes]
    load_hash_start: int
    load_hash_end: int
    load_vllm_block_ids: tuple[list[int], ...]

    dump_keys: list[bytes]
    dump_hash_start: int
    dump_hash_end: int
    dump_vllm_block_ids: tuple[list[int], ...]
```

Field contracts:

- `load_keys == req_meta.ucm_block_ids[load_hash_start:load_hash_end]`.
- `dump_keys == req_meta.ucm_block_ids[dump_hash_start:dump_hash_end]`.
- `load_vllm_block_ids` and `dump_vllm_block_ids` are tuples aligned by global
  KV cache group id. `candidate_vllm_ids[group_id]` directly addresses that
  group.
- Each FA group candidate list is already sliced to the same hash range as the
  corresponding keys. It contains the minimum physical vLLM block ids needed to
  cover the range.
- Each WA group candidate list is sliced further to only the tail blocks needed
  for each hash block in the range. It does not carry all logical WA blocks in
  the hash block.
- For WA load, `load_vllm_block_ids` carries only the final boundary hash block's
  WA tail blocks. FA groups still cover the full load hash range.
- For WA dump, `dump_vllm_block_ids` carries the WA tail blocks for every hash
  block in the dump range.
- WA/state groups with `tail_blocks == 0`, such as 128A state cache groups, use
  an empty candidate list for that group.
- New requests provide complete `request.block_ids`.
- Cached and chunk-prefill requests provide incremental `new_block_ids`.
- The connector accumulates block ids in `req_meta.vllm_block_ids`.
- `update_state_after_alloc()` does not participate in the new logic and should
  remain a lifecycle no-op.

`build_connector_meta()` handles the WA load special case when building
`load_vllm_block_ids`: FA groups use `[load_hash_start, load_hash_end)`, while WA
groups use only `[load_hash_end - 1, load_hash_end)`. This keeps
`_extract_wa_ptr()` simple because it receives only the final boundary's WA
candidate ids for load.

## Group Metadata

Group metadata should avoid fractional ratios. Use integer fields with clear
semantics:

```python
@dataclass(frozen=True)
class KVCacheGroupMeta:
    group_id: int
    token_block_size: int
    tensor_block_size: int
    logical_blocks_per_hash_block: int
    hash_blocks_per_tensor_block: int
    tail_blocks: int | None
    window_spans: tuple[int, ...]
```

Semantics:

- `logical_blocks_per_hash_block = hash_block_size // token_block_size`.
- `hash_blocks_per_tensor_block = max(1, tensor_block_size // hash_block_size)`.
- GPU FA groups normally have one logical block per hash block and one hash block
  per tensor block.
- GPU WA groups can have multiple logical blocks per hash block.
- Ascend compressed FA groups can have multiple hash blocks sharing one tensor
  block, for example C4 indexer and C128 groups.
- `tail_blocks` remains `None` for FA groups and an integer for WA/state groups.
- `window_spans` describes the persisted span sizes for each group. Ascend
  trimmed state groups use this to point to the tail span without carrying a
  per-segment length in the dispatch metadata.

The scheduler should compute group candidate slices through one range-to-slice
calculation rather than using `int(hash_idx * ratio)`. This prevents boundary
errors for Ascend groups where multiple hash blocks share one physical tensor
block. WA groups use a tail-slice variant of this calculation:

- For each hash block, compute that hash block's logical WA block range.
- Keep only the last `tail_blocks` physical blocks for the group.
- If `tail_blocks == 0`, append no ids for that group.
- For WA load, apply this only to the final loaded hash block.
- For WA dump, apply this to every dumped hash block and concatenate the results
  in hash block order.

## Worker Pointer Extraction

The worker keeps the public extraction shape simple:

```python
_extract_fa_ptr(keys, hash_start, hash_end, candidate_vllm_ids) -> np.ndarray
_extract_wa_ptr(keys, hash_start, hash_end, candidate_vllm_ids) -> np.ndarray
```

Both methods return a `np.uint64` pointer matrix with shape
`(len(keys), row_width)`.

The extraction input contract is uniform:

- `keys` correspond to `[hash_start, hash_end)`.
- FA `candidate_vllm_ids` are sliced to `[hash_start, hash_end)`.
- WA `candidate_vllm_ids` are sliced to the per-hash tail blocks for
  `[hash_start, hash_end)`. Groups with zero tail blocks have empty candidate
  lists.
- The method does not need to know about a larger original range.

Batch extraction should use only the values needed for pointer computation:

```python
row_ids: np.ndarray
block_ids: np.ndarray
offsets: np.ndarray
```

No `lengths` array is needed in the main path. Store row tensor sizes are already
fixed by `_store_tensor_size_list()` and `window_spans`.

`KVCacheGroupLayout` should add a vectorized pointer calculation method:

```python
extract_segment_addrs_batch(
    block_ids: np.ndarray,
    offsets: np.ndarray,
    group_tensor_block_size: int,
) -> np.ndarray
```

It computes tensor offsets and pointers with numpy broadcasting. The result is
`(num_segments, num_group_views)` and is then assembled into the final store row
matrix by `_extract_fa_ptr()` or `_extract_wa_ptr()`.

FA extraction:

- Produces one store row per hash block.
- Includes all FA groups.
- For Ascend compressed groups, multiple hash rows may use the same physical
  block id with different offsets.

WA extraction:

- Produces one store row per hash block in the passed range.
- Includes all window groups.
- For normal WA groups, candidate ids already contain only the last `tail_blocks`
  logical group blocks for each hash block.
- For Ascend trimmed state groups, offsets point to the final `window_spans`
  portion of the tail block.
- For zero-tail groups, no pointers are emitted for that group.

## FAWA Behavior

The refactor preserves the current FAWA store semantics:

- FA load covers every external-hit hash block in `[load_hash_start, load_hash_end)`.
- WA load covers only the final external-hit boundary
  `[load_hash_end - 1, load_hash_end)`, and the WA candidate ids contain only
  that boundary's tail blocks.
- FA dump covers every hash block in `[dump_hash_start, dump_hash_end)`.
- WA dump covers every hash block in `[dump_hash_start, dump_hash_end)`, and WA
  candidate ids contain only each hash block's tail blocks.
- Rank 0 owns dumping in `wait_for_save()`.
- Nonzero TP ranks return from `wait_for_save()` without dumping.
- `get_finished()` and `build_connector_worker_meta()` remain no-ops for this
  synchronous FAWA path.

## Error Handling

Load errors must still report vLLM/HMA block ids that vLLM can invalidate.
Because row objects are removed, anchor ids are derived from the first KV cache
group candidate ids:

- For FA load, anchors come from `load_vllm_block_ids[0]` for the FA range.
- For WA load, anchors come from the first group ids corresponding to the final
  loaded boundary.
- Negative or scratch ids must not be reported.
- `get_block_ids_with_load_errors()` still returns and clears the accumulated
  invalid block id set.

## Test Migration

Tests should move away from asserting `KVCacheSegment` rows. New focused tests
should cover:

- Flat `FAWARequestDispatchMeta` fields.
- Global KV group id alignment for candidate tuples.
- Range-to-candidate slicing for GPU FA, GPU WA, and Ascend compressed FA groups.
- WA group candidate slicing that keeps only tail blocks per hash block.
- WA load candidate slicing that keeps only the final boundary's tail blocks.
- Zero-tail 128A state cache groups represented by empty candidate lists.
- `_extract_fa_ptr()` producing correct pointers for Ascend shared tensor blocks.
- `_extract_wa_ptr()` producing correct tail pointers for normal WA and Ascend
  trimmed state groups.
- WA load using only the final boundary key while WA dump uses every dump key.
- Load-error anchors derived from first-group candidate ids.

Focused verification after implementation:

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

## Code Style Constraints

The implementation should follow the style of `ucm_connector.py`:

- Keep lifecycle methods readable and direct.
- Avoid creating many small private helper methods for one-line formulas.
- Keep only helpers with clear boundaries and reuse value, such as:
  - group metadata initialization,
  - range-to-candidate slice calculation,
  - `KVCacheGroupLayout.extract_segment_addrs_batch()`,
  - `_extract_fa_ptr()`,
  - `_extract_wa_ptr()`.
- Keep simple flow local to `_generate_dispatch_meta()`, `start_load_kv()`, and
  `wait_for_save()`.
- Use numpy batch operations for hash block pointer extraction where it keeps the
  implementation simpler or avoids repeated Python loops.
