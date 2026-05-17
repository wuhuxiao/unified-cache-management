# HMA Group Meta State Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `KVCacheGroupMeta` the single source of truth for FAWA per-group derived state.

**Architecture:** Keep `_init_group_metas()` as the only connector-level group meta initialization entry point. Fold the old tuple-building logic into that function and expose old tuple names only as derived read-only compatibility properties. Runtime code should read `self.group_metas[group_id]` directly.

**Tech Stack:** Python 3.12, vLLM connector APIs, NumPy, PyTorch, pytest.

---

## File Structure

- Modify `ucm/integration/vllm/hma_connector.py`
  - Expand `KVCacheGroupMeta` only if a derived field is missing.
  - Fold group block size, tensor span, ratio, tail, and window-span construction into `_init_group_metas()`.
  - Remove connector initialization assignments to old tuple fields.
  - Convert old tuple names into properties derived from `group_metas`.
  - Update runtime code to read `KVCacheGroupMeta`.
- Modify `test/test_hma_connector_chunk_prefill.py`
  - Update fixtures that currently assign old tuple fields.
  - Add tests that old tuple names are derived from `group_metas`.
- Modify `test/test_hma_connector_ascend_tp4_e2e.py`
  - Update worker fixture copying to copy `group_metas`, not tuple state.
- Modify `test/test_hma_connector_gpu_tp4_e2e.py`
  - Update helper reads where needed to prefer `group_metas`.

No new production helper functions should be added for group meta construction.

---

### Task 1: Add Regression Tests For Meta-Derived Compatibility State

**Files:**
- Modify: `test/test_hma_connector_chunk_prefill.py`

- [ ] **Step 1: Add a regression test for compatibility properties**

Add this test near `test_group_meta_uses_integer_ratios_and_zero_tail`:

```python
def test_group_tuple_compatibility_properties_are_derived_from_metas():
    connector = make_connector()
    connector.group_metas = {
        0: type(connector.group_metas[0])(
            group_id=0,
            token_block_size=256,
            tensor_block_size=256,
            logical_blocks_per_hash_block=1,
            hash_blocks_per_tensor_block=1,
            tail_blocks=None,
            window_spans=(256,),
        ),
        1: type(connector.group_metas[1])(
            group_id=1,
            token_block_size=64,
            tensor_block_size=512,
            logical_blocks_per_hash_block=4,
            hash_blocks_per_tensor_block=2,
            tail_blocks=2,
            window_spans=(64, 64),
        ),
    }

    assert connector.group_token_block_sizes == (256, 64)
    assert connector.group_tensor_block_sizes == (256, 512)
    assert connector.group_tensor_block_ratios == (1, 8)
    assert connector.group_tail_blocks == (None, 2)
    assert connector.group_window_spans == ((256,), (64, 64))
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_chunk_prefill.py::test_group_tuple_compatibility_properties_are_derived_from_metas \
  -q
```

Expected: FAIL because the old tuple attributes are normal mutable attributes, not properties derived from `group_metas`.

- [ ] **Step 3: Commit the failing test**

```bash
git add test/test_hma_connector_chunk_prefill.py
git commit -m "test: cover FAWA group meta compatibility state"
```

---

### Task 2: Make `_init_group_metas()` Build All Per-Group State

**Files:**
- Modify: `ucm/integration/vllm/hma_connector.py`
- Modify: `test/test_hma_connector_chunk_prefill.py`

- [ ] **Step 1: Update connector initialization**

In `UCMFAWAConnector.__init__`, replace:

```python
self.group_token_block_sizes = self._get_group_token_block_sizes()
self.group_tensor_block_sizes = self._get_group_tensor_block_sizes()
self.group_tensor_block_ratios = self._get_group_tensor_block_ratios()
self.group_tail_blocks = self._get_group_tail_blocks()
self.group_window_spans = self._get_group_window_spans()
self._init_group_metas()
```

with:

```python
self._init_group_metas()
```

- [ ] **Step 2: Fold group meta construction into `_init_group_metas()`**

Replace the body of `_init_group_metas()` with code that:

```python
groups = self._kv_cache_config.kv_cache_groups
if not groups:
    raise RuntimeError("FAWA connector found no KV cache groups.")

token_block_sizes = []
for group_id, group in enumerate(groups):
    token_block_size = self._spec_token_block_size(group.kv_cache_spec)
    if group_id in self.fa_group_ids:
        token_block_size = self.hash_block_size
    token_block_sizes.append(token_block_size)

self._validate_group_token_block_sizes(tuple(token_block_sizes))

tensor_block_sizes = list(token_block_sizes)
tail_blocks = [None] * len(token_block_sizes)
window_spans = [(size,) for size in token_block_sizes]

for group_id in self.window_group_ids:
    group_spec = groups[group_id]
    token_block_size = token_block_sizes[group_id]
    window_tokens = FAWABlockSpanLayout.group_window_tokens(group_spec)
    if window_tokens is None:
        tail_blocks[group_id] = self.hash_block_size // token_block_size
    elif self._is_compressor_state_group(group_id):
        tail_blocks[group_id] = self._compressor_state_tail_blocks(
            group_id,
            window_tokens,
            token_block_size,
        )
    else:
        tail_blocks[group_id] = max(1, math.ceil(window_tokens / token_block_size))
    window_spans[group_id] = (token_block_size,) * int(tail_blocks[group_id])

self.group_metas = {}
for group_id, token_block_size in enumerate(token_block_sizes):
    tensor_block_size = tensor_block_sizes[group_id]
    if tensor_block_size % token_block_size != 0:
        raise RuntimeError(
            f"FAWA group {group_id} logical block size "
            f"{token_block_size} must divide tensor block size "
            f"{tensor_block_size}."
        )
    self.group_metas[group_id] = KVCacheGroupMeta(
        group_id=group_id,
        token_block_size=token_block_size,
        tensor_block_size=tensor_block_size,
        logical_blocks_per_hash_block=self.hash_block_size // token_block_size,
        hash_blocks_per_tensor_block=max(1, tensor_block_size // self.hash_block_size),
        tail_blocks=tail_blocks[group_id],
        window_spans=tuple(window_spans[group_id]),
    )
```

Keep this logic inside `_init_group_metas()`. Do not create new connector helper functions.

- [ ] **Step 3: Add compatibility properties**

Add properties on `UCMFAWAConnector`:

```python
@property
def group_token_block_sizes(self) -> tuple[int, ...]:
    return tuple(meta.token_block_size for _, meta in sorted(self.group_metas.items()))

@property
def group_tensor_block_sizes(self) -> tuple[int, ...]:
    return tuple(meta.tensor_block_size for _, meta in sorted(self.group_metas.items()))

@property
def group_tensor_block_ratios(self) -> tuple[int, ...]:
    return tuple(
        meta.tensor_block_size // meta.token_block_size
        for _, meta in sorted(self.group_metas.items())
    )

@property
def group_tail_blocks(self) -> tuple[int | None, ...]:
    return tuple(meta.tail_blocks for _, meta in sorted(self.group_metas.items()))

@property
def group_window_spans(self) -> tuple[tuple[int, ...], ...]:
    return tuple(meta.window_spans for _, meta in sorted(self.group_metas.items()))
```

Do not add setters.

- [ ] **Step 4: Update simulated test fixtures**

Where tests currently do this:

```python
connector.group_token_block_sizes = (...)
connector.group_tensor_block_sizes = (...)
connector.group_tensor_block_ratios = (...)
connector.group_tail_blocks = (...)
connector.group_window_spans = (...)
connector._init_group_metas()
```

change the fixture to set `connector.group_metas` directly with `KVCacheGroupMeta(...)`, or provide enough `_kv_cache_config` data and call `_init_group_metas()`. For simple synthetic connectors, direct `group_metas` assignment is preferred.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python3 -m pytest test/test_hma_connector_chunk_prefill.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ucm/integration/vllm/hma_connector.py test/test_hma_connector_chunk_prefill.py
git commit -m "refactor: derive FAWA group state from metas"
```

---

### Task 3: Adapt Ascend Meta Construction Without Tuple State

**Files:**
- Modify: `ucm/integration/vllm/hma_connector.py`
- Modify: `test/test_hma_connector_chunk_prefill.py`
- Modify: `test/test_hma_connector_ascend_tp4_e2e.py`

- [ ] **Step 1: Override only `_init_group_metas()` in `UCMAscendFAWAConnector`**

Replace Ascend overrides for `_get_group_token_block_sizes()`,
`_get_group_tensor_block_sizes()`, `_get_group_tail_blocks()`, and
`_get_group_window_spans()` with a single `_init_group_metas()` override.

The override should:

```python
if self.block_span_layout is None:
    raise RuntimeError("Ascend FAWA connector requires block span layout.")
self._ascend_layout = self.block_span_layout.is_ascend

groups = self._kv_cache_config.kv_cache_groups
token_block_sizes = []
tensor_block_sizes = []
for group_id, group in enumerate(groups):
    token_block_size = self.block_span_layout._ascend_group_token_block_size(group)
    detected = {
        tensor_block_size
        for spec in self.block_span_layout.group_specs(group)
        if (
            tensor_block_size := self.block_span_layout.spec_tensor_block_size(spec)
        ) is not None
    }
    if len(detected) > 1:
        raise RuntimeError(
            f"FAWA Ascend group {group_id} has mixed tensor "
            f"block sizes: {sorted(detected)}."
        )
    tensor_block_size = detected.pop() if detected else token_block_size
    token_block_sizes.append(token_block_size)
    tensor_block_sizes.append(tensor_block_size)

self._validate_group_token_block_sizes(tuple(token_block_sizes))

tail_blocks = [None] * len(token_block_sizes)
window_spans = [(size,) for size in token_block_sizes]
for group_id in self.window_group_ids:
    token_block_size = token_block_sizes[group_id]
    window_tail_tokens = self._ascend_window_tail_tokens(group_id)
    if window_tail_tokens is None:
        tail_blocks[group_id] = self.hash_block_size // token_block_size
    elif window_tail_tokens == 0:
        tail_blocks[group_id] = 0
    elif self.block_span_layout.is_swa_group(group_id):
        tail_blocks[group_id] = max(1, math.ceil(window_tail_tokens / token_block_size))
    else:
        tail_blocks[group_id] = math.ceil(window_tail_tokens / token_block_size)

    if tail_blocks[group_id] == 0:
        window_spans[group_id] = ()
    elif (
        window_tail_tokens is None
        or self.block_span_layout.is_swa_group(group_id)
    ):
        window_spans[group_id] = (token_block_size,) * int(tail_blocks[group_id])
    else:
        spans = []
        remaining_tokens = window_tail_tokens
        while remaining_tokens > 0:
            segment_tokens = min(token_block_size, remaining_tokens)
            spans.append(segment_tokens)
            remaining_tokens -= segment_tokens
        window_spans[group_id] = tuple(reversed(spans))
```

Then build `self.group_metas` in the same style as the base method.

- [ ] **Step 2: Simplify `FAWABlockSpanLayout`**

Remove `FAWABlockSpanLayout.__init__` assignments for:

```python
self.group_token_block_sizes
self.group_tensor_block_sizes
self.group_tensor_block_ratios
```

Keep:

```python
self.hash_block_size
self.group_layer_tensor_indices
```

Keep `allocation_index()` working by computing the ratio locally from the
group spec when called, or by reading from connector metas if the call path has
been moved away. Do not add a new connector helper.

- [ ] **Step 3: Update Ascend tests and fixtures**

Update `make_ascend_connector()` and TP4 worker setup so they do not assign:

```python
group_token_block_sizes
group_tensor_block_sizes
group_tensor_block_ratios
group_tail_blocks
group_window_spans
```

They should call `_init_group_metas()` on connectors or copy
`scheduler.group_metas` to worker fixtures.

- [ ] **Step 4: Run Ascend-focused tests**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  ucm/integration/vllm/hma_connector.py \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py
git commit -m "refactor: initialize Ascend FAWA metas directly"
```

---

### Task 4: Remove Runtime Reads Of Parallel Tuple State

**Files:**
- Modify: `ucm/integration/vllm/hma_connector.py`
- Modify: `test/test_hma_connector_gpu_tp4_e2e.py`
- Modify: `test/test_hma_connector_ascend_tp4_e2e.py`

- [ ] **Step 1: Update runtime code to use meta fields**

Change these runtime paths:

```python
_store_tensor_size_list()
_slice_group_block_ids()
_extract_fa_ptr()
_extract_wa_ptr()
logger.info(...) group config summary
```

For each group, use:

```python
meta = self.group_metas[group_id]
token_blocks_per_tensor_block = meta.tensor_block_size // meta.token_block_size
```

Do not read `self.group_token_block_sizes`, `self.group_tensor_block_sizes`,
`self.group_tensor_block_ratios`, `self.group_tail_blocks`, or
`self.group_window_spans` in production runtime logic.

- [ ] **Step 2: Update E2E helper reads**

In E2E tests, prefer:

```python
meta = connector.group_metas[group_id]
```

and replace helper calculations that read old tuple fields.

- [ ] **Step 3: Verify old tuple assignment is gone**

Run:

```bash
rg -n "self\\.group_(token_block_sizes|tensor_block_sizes|tensor_block_ratios|tail_blocks|window_spans)\\s*=" ucm/integration/vllm/hma_connector.py
```

Expected: no output.

- [ ] **Step 4: Verify old tuple reads are only compatibility/test-facing**

Run:

```bash
rg -n "group_(token_block_sizes|tensor_block_sizes|tensor_block_ratios|tail_blocks|window_spans)" ucm/integration/vllm/hma_connector.py
```

Expected: matches only compatibility property definitions, logging if still
test-facing, or comments. Runtime methods should use `group_metas`.

- [ ] **Step 5: Run focused E2E tests**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  ucm/integration/vllm/hma_connector.py \
  test/test_hma_connector_gpu_tp4_e2e.py \
  test/test_hma_connector_ascend_tp4_e2e.py
git commit -m "refactor: read FAWA group state from metas"
```

---

### Task 5: Final Verification And Cleanup

**Files:**
- Modify only if verification finds small issues.

- [ ] **Step 1: Run compile verification**

Run:

```bash
python3 -m py_compile \
  ucm/integration/vllm/hma_connector.py \
  ucm/integration/vllm/ucm_connector.py \
  test/test_hma_connector_load_errors.py \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py
```

Expected: exit code 0.

- [ ] **Step 2: Run focused pytest suite**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_load_errors.py \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py
```

Expected: all tests pass.

- [ ] **Step 3: Run code hygiene checks**

Run:

```bash
git diff --check
rg -n "KVCacheSegment|KVCacheGroupRow|KVCacheGroupRows|record_block_cursor|store_block_cursor|allocated_group_block_ids" \
  ucm/integration/vllm/hma_connector.py \
  test/test_hma_connector_*.py
rg -n "self\\.group_(token_block_sizes|tensor_block_sizes|tensor_block_ratios|tail_blocks|window_spans)\\s*=" \
  ucm/integration/vllm/hma_connector.py
```

Expected:

- `git diff --check` exits 0.
- old segment/row/cursor grep has no output.
- old tuple assignment grep has no output.

- [ ] **Step 4: Commit final cleanup if needed**

If Step 3 required edits:

```bash
git add ucm/integration/vllm/hma_connector.py test/test_hma_connector_*.py
git commit -m "test: verify FAWA group meta consolidation"
```

If no edits were needed, do not create an empty commit.
