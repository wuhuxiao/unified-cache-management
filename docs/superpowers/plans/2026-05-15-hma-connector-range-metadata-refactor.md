# HMA Connector Range Metadata Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace old `KVCacheSegment`/`KVCacheGroupRow` FAWA metadata with explicit hash-range metadata and worker-side batch pointer extraction.

**Architecture:** The scheduler accumulates vLLM block ids per request and emits flat load/dump fields with range-local candidate ids aligned by global KV group id. FA groups carry the physical blocks covering the whole hash range; WA groups carry only per-hash tail blocks, with load restricted to the final boundary. Workers use `KVCacheGroupLayout` batch address calculation plus compact FA/WA extraction methods to build UCM store pointer rows.

**Tech Stack:** Python 3.12, dataclasses, numpy, torch, vLLM HMA connector lifecycle, pytest.

---

## File Structure

- Modify `ucm/integration/vllm/hma_connector.py`
  - Remove `KVCacheSegment`, `KVCacheGroupRow`, `KVCacheGroupRows`, and `KVCacheGroupAllocation`.
  - Replace tuple-packed `FAWARequestDispatchMeta` with flat fields.
  - Add `KVCacheGroupMeta`.
  - Add batch pointer extraction to `KVCacheGroupLayout`.
  - Keep `UCMFAWAConnector` helper count small: group meta setup, candidate slicing, `_extract_fa_ptr()`, `_extract_wa_ptr()`.
  - Keep `update_state_after_alloc()` as a lifecycle no-op.
- Modify `test/test_hma_connector_chunk_prefill.py`
  - Remove `KVCacheSegment` imports and row assertions.
  - Add focused tests for flat dispatch metadata, FA/WA candidate slicing, zero-tail WA groups, and pointer offsets.
- Modify `test/test_hma_connector_load_errors.py`
  - Remove row-anchor tests that depend on `KVCacheSegment`.
  - Test first-group load-error anchors from flat candidate ids.
- Modify `test/test_hma_connector_gpu_tp4_e2e.py`
  - Update metadata assertions from `load_block_ids`/`dump_block_ids` rows to flat field assertions and real store behavior.
- Modify `test/test_hma_connector_ascend_tp4_e2e.py`
  - Update metadata assertions to flat fields and ensure Ascend WA zero-tail and trimmed-state behavior remain covered.

Do not modify `ucm/integration/vllm/ucm_connector.py` unless a wrapper delegation compatibility issue appears during verification.

## Task 1: Data Structures And Batch Layout API

**Files:**
- Modify: `ucm/integration/vllm/hma_connector.py`
- Test: `test/test_hma_connector_chunk_prefill.py`

- [ ] **Step 1: Write failing layout batch pointer tests**

Add these tests near the existing `KVCacheGroupLayout` tests in `test/test_hma_connector_chunk_prefill.py`:

```python
def test_layout_extracts_segment_addresses_batch():
    tensor = torch.empty((4, 128, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    addrs = layout.extract_segment_addrs_batch(
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([4096, 8192], dtype=np.int64),
        group_tensor_block_size=16384,
    )

    assert addrs.shape == (2, 1)
    assert addrs[0, 0] == np.uint64(tensor[1, 32].data_ptr())
    assert addrs[1, 0] == np.uint64(tensor[2, 64].data_ptr())
```

Add this test near the connector helper tests:

```python
def test_group_meta_uses_integer_ratios_and_zero_tail():
    connector = make_connector()
    connector.group_token_block_sizes = (256, 64, 64)
    connector.group_tensor_block_sizes = (256, 64, 64)
    connector.group_tail_blocks = (None, 1, 0)
    connector.group_window_spans = ((256,), (64,), ())
    connector.window_group_ids = (1, 2)

    connector._init_group_metas()

    assert connector.group_metas[0].logical_blocks_per_hash_block == 1
    assert connector.group_metas[0].hash_blocks_per_tensor_block == 1
    assert connector.group_metas[1].logical_blocks_per_hash_block == 4
    assert connector.group_metas[1].tail_blocks == 1
    assert connector.group_metas[2].tail_blocks == 0
    assert connector.group_metas[2].window_spans == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_chunk_prefill.py::test_layout_extracts_segment_addresses_batch \
  test/test_hma_connector_chunk_prefill.py::test_group_meta_uses_integer_ratios_and_zero_tail \
  -q
```

Expected: FAIL because `extract_segment_addrs_batch()` and `_init_group_metas()` do not exist.

- [ ] **Step 3: Implement dataclasses and batch layout API**

In `ucm/integration/vllm/hma_connector.py`, replace the old `KVCacheGroupMeta` stub with:

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

Replace `FAWARequestMeta` and `FAWARequestDispatchMeta` with:

```python
@dataclass
class FAWARequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: tuple[list[int], ...] = field(default_factory=tuple)
    token_processed: int = 0


@dataclass
class FAWARequestDispatchMeta:
    load_keys: list[bytes] = field(default_factory=list)
    load_hash_start: int = 0
    load_hash_end: int = 0
    load_vllm_block_ids: tuple[list[int], ...] = field(default_factory=tuple)
    dump_keys: list[bytes] = field(default_factory=list)
    dump_hash_start: int = 0
    dump_hash_end: int = 0
    dump_vllm_block_ids: tuple[list[int], ...] = field(default_factory=tuple)
```

In `KVCacheGroupLayout`, add:

```python
    def _tensor_tokens_for_logical_batch(
        self,
        logical_offsets: np.ndarray,
        group_tensor_block_size: int,
    ) -> np.ndarray:
        logical_offsets = np.asarray(logical_offsets, dtype=np.uint64)
        scaled = logical_offsets[:, None] * self.view_tensor_block_sizes[None, :]
        group_size = np.uint64(group_tensor_block_size)
        misaligned = scaled % group_size
        if np.any(misaligned):
            raise ValueError(
                f"Logical offsets {logical_offsets.tolist()} do not align with "
                f"view tensor block sizes={self.view_tensor_block_sizes.tolist()} "
                f"and group tensor block size={group_tensor_block_size}."
            )
        return scaled // group_size

    def extract_segment_addrs_batch(
        self,
        block_ids: np.ndarray,
        offsets: np.ndarray,
        group_tensor_block_size: int,
    ) -> np.ndarray:
        signed_block_ids = np.asarray(block_ids, dtype=np.int64)
        if signed_block_ids.size == 0:
            return np.empty((0, len(self.base_ptrs)), dtype=np.uint64)
        if np.any(signed_block_ids < 0):
            raise ValueError("Negative KV cache block id needs a scratch target.")
        block_ids_np = signed_block_ids.astype(np.uint64, copy=False)
        tensor_offsets = self._tensor_tokens_for_logical_batch(
            np.asarray(offsets, dtype=np.uint64),
            group_tensor_block_size,
        )
        return (
            block_ids_np[:, None] * self.block_strides[None, :]
            + tensor_offsets * self.token_strides[None, :]
            + self.base_ptrs[None, :]
        ).astype(np.uint64, copy=False)
```

In `UCMFAWAConnector.__init__()`, call the existing size getters and initialize metas:

```python
        self.hash_block_size = self._get_hash_block_size()
        self.block_size = self.hash_block_size
        self.group_token_block_sizes = self._get_group_token_block_sizes()
        self.group_tensor_block_sizes = self._get_group_tensor_block_sizes()
        self.group_tensor_block_ratios = self._get_group_tensor_block_ratios()
        self.group_tail_blocks = self._get_group_tail_blocks()
        self.group_window_spans = self._get_group_window_spans()
        self.group_metas: dict[int, KVCacheGroupMeta] = {}
        self._init_group_metas()
```

Add one connector helper:

```python
    def _init_group_metas(self) -> None:
        self.group_metas = {}
        for group_id, token_block_size in enumerate(self.group_token_block_sizes):
            tensor_block_size = self.group_tensor_block_sizes[group_id]
            self.group_metas[group_id] = KVCacheGroupMeta(
                group_id=group_id,
                token_block_size=token_block_size,
                tensor_block_size=tensor_block_size,
                logical_blocks_per_hash_block=(
                    self.hash_block_size // token_block_size
                ),
                hash_blocks_per_tensor_block=max(
                    1,
                    tensor_block_size // self.hash_block_size,
                ),
                tail_blocks=self.group_tail_blocks[group_id],
                window_spans=self.group_window_spans[group_id],
            )
```

Remove the empty `get_group_cache_meta()` method.

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_chunk_prefill.py::test_layout_extracts_segment_addresses_batch \
  test/test_hma_connector_chunk_prefill.py::test_group_meta_uses_integer_ratios_and_zero_tail \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add ucm/integration/vllm/hma_connector.py test/test_hma_connector_chunk_prefill.py
git commit -m "refactor: add FAWA range metadata primitives"
```

## Task 2: Scheduler Dispatch Metadata And Candidate Slicing

**Files:**
- Modify: `ucm/integration/vllm/hma_connector.py`
- Test: `test/test_hma_connector_chunk_prefill.py`

- [ ] **Step 1: Replace old row-based tests with flat metadata tests**

In `test/test_hma_connector_chunk_prefill.py`, update `make_connector()` so it initializes metas:

```python
def make_connector() -> UCMFAWAConnector:
    connector = UCMFAWAConnector.__new__(UCMFAWAConnector)
    connector.hash_block_size = 256
    connector.fa_group_ids = (0,)
    connector.window_group_ids = (1,)
    connector.group_token_block_sizes = (256, 64)
    connector.group_tensor_block_sizes = connector.group_token_block_sizes
    connector.group_tensor_block_ratios = (1, 1)
    connector.group_tail_blocks = (None, 1)
    connector.group_window_spans = ((256,), (64,))
    connector.block_span_layout = None
    connector.requests_meta = {}
    connector._init_group_metas()
    return connector
```

Replace the old chunk prefill row tests with:

```python
def test_dispatch_meta_accumulates_cached_blocks_and_slices_wa_tails():
    connector = make_connector()
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a", b"b"],
        num_token_ids=512,
        token_processed=0,
    )
    connector.requests_meta["req-0"] = req_meta

    first_step = FakeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=FakeCachedRequestData(
            req_ids=["req-0"],
            resumed_req_ids=set(),
            new_block_ids=[([10], [100, 101, 102, 103])],
        ),
        num_scheduled_tokens={"req-0": 256},
        finished_req_ids=set(),
    )
    first_meta = connector.build_connector_meta(first_step)
    first_req = first_meta.request_meta["req-0"]

    assert first_req.dump_keys == [b"a"]
    assert first_req.dump_hash_start == 0
    assert first_req.dump_hash_end == 1
    assert first_req.dump_vllm_block_ids == ([10], [103])

    second_step = FakeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=FakeCachedRequestData(
            req_ids=["req-0"],
            resumed_req_ids=set(),
            new_block_ids=[([11], [104, 105, 106, 107])],
        ),
        num_scheduled_tokens={"req-0": 256},
        finished_req_ids=set(),
    )
    second_meta = connector.build_connector_meta(second_step)
    second_req = second_meta.request_meta["req-0"]

    assert second_req.dump_keys == [b"b"]
    assert second_req.dump_hash_start == 1
    assert second_req.dump_hash_end == 2
    assert second_req.dump_vllm_block_ids == ([11], [107])
    assert req_meta.vllm_block_ids == (
        [10, 11],
        [100, 101, 102, 103, 104, 105, 106, 107],
    )
    assert req_meta.token_processed == 512
```

Add WA load and zero-tail tests:

```python
def test_load_metadata_slices_wa_to_final_boundary():
    connector = make_connector()
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a", b"b", b"c"],
        hbm_hit_block_num=0,
        total_hit_block_num=2,
        num_token_ids=768,
        token_processed=512,
    )
    connector.requests_meta["req-load"] = req_meta

    request = type(
        "ReqData",
        (),
        {
            "req_id": "req-load",
            "block_ids": (
                [10, 11, 12],
                [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            ),
        },
    )()
    metadata = connector.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[request],
            scheduled_cached_reqs=FakeCachedRequestData([], set(), []),
            num_scheduled_tokens={"req-load": 256},
            finished_req_ids=set(),
        )
    )

    dispatch = metadata.request_meta["req-load"]
    assert dispatch.load_keys == [b"a", b"b"]
    assert dispatch.load_hash_start == 0
    assert dispatch.load_hash_end == 2
    assert dispatch.load_vllm_block_ids == ([10, 11], [107])
```

```python
def test_zero_tail_window_group_uses_empty_candidate_list():
    connector = make_connector()
    connector.group_token_block_sizes = (256, 64, 64)
    connector.group_tensor_block_sizes = connector.group_token_block_sizes
    connector.group_tensor_block_ratios = (1, 1, 1)
    connector.group_tail_blocks = (None, 1, 0)
    connector.group_window_spans = ((256,), (64,), ())
    connector.window_group_ids = (1, 2)
    connector._init_group_metas()

    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a"],
        num_token_ids=256,
        token_processed=0,
    )
    connector.requests_meta["req-zero"] = req_meta

    metadata = connector.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=FakeCachedRequestData(
                req_ids=["req-zero"],
                resumed_req_ids=set(),
                new_block_ids=[([10], [100, 101, 102, 103], [])],
            ),
            num_scheduled_tokens={"req-zero": 256},
            finished_req_ids=set(),
        )
    )

    dispatch = metadata.request_meta["req-zero"]
    assert dispatch.dump_vllm_block_ids == ([10], [103], [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_chunk_prefill.py::test_dispatch_meta_accumulates_cached_blocks_and_slices_wa_tails \
  test/test_hma_connector_chunk_prefill.py::test_load_metadata_slices_wa_to_final_boundary \
  test/test_hma_connector_chunk_prefill.py::test_zero_tail_window_group_uses_empty_candidate_list \
  -q
```

Expected: FAIL because dispatch metadata still uses old fields or incorrect candidate slicing.

- [ ] **Step 3: Implement range candidate slicing and flat dispatch**

In `UCMFAWAConnector`, keep one slicing helper:

```python
    def _slice_group_block_ids(
        self,
        group_id: int,
        group_block_ids: list[int],
        hash_start: int,
        hash_end: int,
        *,
        window_tail_only: bool,
    ) -> list[int]:
        if hash_end <= hash_start:
            return []
        meta = self.group_metas[group_id]
        if window_tail_only:
            if not meta.tail_blocks:
                return []
            selected: list[int] = []
            for hash_idx in range(hash_start, hash_end):
                logical_end = (hash_idx + 1) * meta.logical_blocks_per_hash_block
                logical_start = max(
                    hash_idx * meta.logical_blocks_per_hash_block,
                    logical_end - meta.tail_blocks,
                )
                alloc_start = logical_start // meta.hash_blocks_per_tensor_block
                alloc_end = (
                    (logical_end - 1) // meta.hash_blocks_per_tensor_block
                ) + 1
                selected.extend(group_block_ids[alloc_start:alloc_end])
            return selected

        alloc_start = (
            hash_start * meta.logical_blocks_per_hash_block
        ) // meta.hash_blocks_per_tensor_block
        logical_end = hash_end * meta.logical_blocks_per_hash_block
        alloc_end = ((logical_end - 1) // meta.hash_blocks_per_tensor_block) + 1
        return group_block_ids[alloc_start:alloc_end]
```

Replace `_generate_dispatch_meta()` with:

```python
    def _generate_dispatch_meta(
        self,
        req_meta: FAWARequestMeta,
        new_tokens: int,
        new_vllm_block_ids: tuple[list[int], ...],
        need_load: bool = True,
    ) -> FAWARequestDispatchMeta:
        if not req_meta.vllm_block_ids:
            req_meta.vllm_block_ids = tuple(
                [] for _ in range(len(self.group_token_block_sizes))
            )
        if len(new_vllm_block_ids) != len(req_meta.vllm_block_ids):
            raise RuntimeError(
                "FAWA cached allocation update has mismatched group count: "
                f"current={len(req_meta.vllm_block_ids)}, "
                f"new={len(new_vllm_block_ids)}."
            )
        for group_id, block_ids in enumerate(new_vllm_block_ids):
            req_meta.vllm_block_ids[group_id].extend(block_ids)

        load_keys: list[bytes] = []
        load_hash_start = 0
        load_hash_end = 0
        load_vllm_block_ids = tuple(
            [] for _ in range(len(self.group_token_block_sizes))
        )
        if need_load and req_meta.total_hit_block_num > req_meta.hbm_hit_block_num:
            load_hash_start = req_meta.hbm_hit_block_num
            load_hash_end = req_meta.total_hit_block_num
            load_keys = req_meta.ucm_block_ids[load_hash_start:load_hash_end]
            load_groups: list[list[int]] = []
            for group_id, group_block_ids in enumerate(req_meta.vllm_block_ids):
                if group_id in self.window_group_ids:
                    load_groups.append(
                        self._slice_group_block_ids(
                            group_id,
                            group_block_ids,
                            load_hash_end - 1,
                            load_hash_end,
                            window_tail_only=True,
                        )
                    )
                else:
                    load_groups.append(
                        self._slice_group_block_ids(
                            group_id,
                            group_block_ids,
                            load_hash_start,
                            load_hash_end,
                            window_tail_only=False,
                        )
                    )
            load_vllm_block_ids = tuple(load_groups)

        dump_keys: list[bytes] = []
        dump_hash_start = 0
        dump_hash_end = 0
        dump_vllm_block_ids = tuple(
            [] for _ in range(len(self.group_token_block_sizes))
        )
        computed_end_token = min(
            req_meta.num_token_ids,
            req_meta.token_processed + new_tokens,
        )
        if req_meta.token_processed < req_meta.num_token_ids:
            dump_hash_start = req_meta.token_processed // self.hash_block_size
            dump_hash_end = computed_end_token // self.hash_block_size
            if dump_hash_end > dump_hash_start:
                dump_keys = req_meta.ucm_block_ids[dump_hash_start:dump_hash_end]
                dump_groups = []
                for group_id, group_block_ids in enumerate(req_meta.vllm_block_ids):
                    dump_groups.append(
                        self._slice_group_block_ids(
                            group_id,
                            group_block_ids,
                            dump_hash_start,
                            dump_hash_end,
                            window_tail_only=group_id in self.window_group_ids,
                        )
                    )
                dump_vllm_block_ids = tuple(dump_groups)
            req_meta.token_processed = computed_end_token

        return FAWARequestDispatchMeta(
            load_keys=load_keys,
            load_hash_start=load_hash_start,
            load_hash_end=load_hash_end,
            load_vllm_block_ids=load_vllm_block_ids,
            dump_keys=dump_keys,
            dump_hash_start=dump_hash_start,
            dump_hash_end=dump_hash_end,
            dump_vllm_block_ids=dump_vllm_block_ids,
        )
```

Update `build_connector_meta()` so new requests pass `request.block_ids` as a tuple, cached requests pass `scheduled_cached_reqs.new_block_ids[i]` or empty per-group lists, and resumed requests reset `req_meta.vllm_block_ids` before appending:

```python
                if resumed_from_preemption:
                    req_meta.vllm_block_ids = tuple(
                        [] for _ in range(len(self.group_token_block_sizes))
                    )
```

Keep `update_state_after_alloc()` as:

```python
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        pass
```

- [ ] **Step 4: Run Task 2 tests**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_chunk_prefill.py::test_dispatch_meta_accumulates_cached_blocks_and_slices_wa_tails \
  test/test_hma_connector_chunk_prefill.py::test_load_metadata_slices_wa_to_final_boundary \
  test/test_hma_connector_chunk_prefill.py::test_zero_tail_window_group_uses_empty_candidate_list \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add ucm/integration/vllm/hma_connector.py test/test_hma_connector_chunk_prefill.py
git commit -m "refactor: emit FAWA range dispatch metadata"
```

## Task 3: Worker FA/WA Batch Pointer Extraction

**Files:**
- Modify: `ucm/integration/vllm/hma_connector.py`
- Test: `test/test_hma_connector_chunk_prefill.py`

- [ ] **Step 1: Write pointer extraction tests**

Add these tests:

```python
def test_extract_fa_ptr_batches_hash_offsets_for_shared_tensor_block():
    connector = make_connector()
    connector.hash_block_size = 512
    connector.fa_group_ids = (0,)
    connector.window_group_ids = ()
    connector.group_token_block_sizes = (512,)
    connector.group_tensor_block_sizes = (4096,)
    connector.group_tensor_block_ratios = (8,)
    connector.group_tail_blocks = (None,)
    connector.group_window_spans = ((512,),)
    connector._init_group_metas()
    tensor = torch.empty((2, 4096, 1), dtype=torch.float32)
    connector.group_layouts = {0: KVCacheGroupLayout({"layer.0": tensor})}

    ptrs = connector._extract_fa_ptr(
        [b"a", b"b"],
        7,
        9,
        ([3, 4],),
    )

    assert ptrs.shape == (2, 1)
    assert ptrs[0, 0] == np.uint64(tensor[3, 3584].data_ptr())
    assert ptrs[1, 0] == np.uint64(tensor[4, 0].data_ptr())
```

```python
def test_extract_wa_ptr_uses_tail_candidates_and_zero_tail_group():
    connector = make_connector()
    connector.group_token_block_sizes = (256, 64, 64)
    connector.group_tensor_block_sizes = connector.group_token_block_sizes
    connector.group_tensor_block_ratios = (1, 1, 1)
    connector.fa_group_ids = (0,)
    connector.window_group_ids = (1, 2)
    connector.group_tail_blocks = (None, 1, 0)
    connector.group_window_spans = ((256,), (64,), ())
    connector._init_group_metas()
    window_tensor = torch.empty((16, 64, 1), dtype=torch.float32)
    zero_tail_tensor = torch.empty((16, 64, 1), dtype=torch.float32)
    connector.group_layouts = {
        1: KVCacheGroupLayout({"layer.0.wa": window_tensor}),
        2: KVCacheGroupLayout({"layer.0.zero": zero_tail_tensor}),
    }

    ptrs = connector._extract_wa_ptr(
        [b"a", b"b"],
        0,
        2,
        ([], [3, 7], []),
    )

    assert ptrs.shape == (2, 1)
    assert ptrs[0, 0] == np.uint64(window_tensor[3, 0].data_ptr())
    assert ptrs[1, 0] == np.uint64(window_tensor[7, 0].data_ptr())
```

- [ ] **Step 2: Run pointer tests to verify they fail**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_chunk_prefill.py::test_extract_fa_ptr_batches_hash_offsets_for_shared_tensor_block \
  test/test_hma_connector_chunk_prefill.py::test_extract_wa_ptr_uses_tail_candidates_and_zero_tail_group \
  -q
```

Expected: FAIL because `_extract_fa_ptr()` and `_extract_wa_ptr()` are not implemented.

- [ ] **Step 3: Implement compact batch extraction**

In `UCMFAWAConnector`, implement:

```python
    def _extract_fa_ptr(
        self,
        store_keys: list[bytes],
        hash_start: int,
        hash_end: int,
        candidate_vllm_ids: tuple[list[int], ...],
    ) -> np.ndarray:
        if not store_keys:
            return np.empty((0, 0), dtype=np.uint64)
        rows: list[list[np.ndarray]] = [[] for _ in store_keys]
        for group_id in self.fa_group_ids:
            layout = self.group_layouts.get(group_id)
            if layout is None:
                continue
            meta = self.group_metas[group_id]
            row_ids: list[int] = []
            block_ids: list[int] = []
            offsets: list[int] = []
            for row_id, hash_idx in enumerate(range(hash_start, hash_end)):
                rel_hash = hash_idx - hash_start
                if meta.hash_blocks_per_tensor_block > 1:
                    candidate_idx = (
                        hash_idx // meta.hash_blocks_per_tensor_block
                        - hash_start // meta.hash_blocks_per_tensor_block
                    )
                    offset = (
                        hash_idx % meta.hash_blocks_per_tensor_block
                    ) * self.hash_block_size
                else:
                    candidate_idx = rel_hash * meta.logical_blocks_per_hash_block
                    offset = 0
                if candidate_idx >= len(candidate_vllm_ids[group_id]):
                    raise RuntimeError(
                        f"FAWA FA group {group_id} candidate block ids do not "
                        f"cover hash block {hash_idx}."
                    )
                row_ids.append(row_id)
                block_ids.append(candidate_vllm_ids[group_id][candidate_idx])
                offsets.append(offset)
            group_ptrs = layout.extract_segment_addrs_batch(
                np.asarray(block_ids, dtype=np.int64),
                np.asarray(offsets, dtype=np.int64),
                meta.tensor_block_size,
            )
            for row_id, ptr_row in zip(row_ids, group_ptrs):
                rows[row_id].append(ptr_row)
        if any(not row for row in rows):
            raise ValueError("FA KV cache pointer row is empty.")
        return np.vstack(
            [np.concatenate(row).astype(np.uint64, copy=False) for row in rows]
        )
```

Implement `_extract_wa_ptr()` with the same style:

```python
    def _extract_wa_ptr(
        self,
        store_keys: list[bytes],
        hash_start: int,
        hash_end: int,
        candidate_vllm_ids: tuple[list[int], ...],
    ) -> np.ndarray:
        if not store_keys:
            return np.empty((0, 0), dtype=np.uint64)
        rows: list[list[np.ndarray]] = [[] for _ in store_keys]
        for group_id in self.window_group_ids:
            layout = self.group_layouts.get(group_id)
            if layout is None:
                continue
            meta = self.group_metas[group_id]
            if not meta.tail_blocks:
                continue
            row_ids: list[int] = []
            block_ids: list[int] = []
            offsets: list[int] = []
            cursor = 0
            for row_id, hash_idx in enumerate(range(hash_start, hash_end)):
                for span_tokens in meta.window_spans:
                    if cursor >= len(candidate_vllm_ids[group_id]):
                        raise RuntimeError(
                            f"FAWA WA group {group_id} candidate block ids do not "
                            f"cover hash block {hash_idx}."
                        )
                    block_id = candidate_vllm_ids[group_id][cursor]
                    cursor += 1
                    logical_blocks_per_tensor = meta.hash_blocks_per_tensor_block
                    if logical_blocks_per_tensor > 1:
                        offset = (
                            hash_idx % logical_blocks_per_tensor
                        ) * self.hash_block_size
                    else:
                        # Trimmed Ascend state spans begin at the end of the
                        # physical tensor block. Normal WA spans have
                        # span_tokens == token_block_size, so this is zero.
                        offset = max(0, meta.token_block_size - span_tokens)
                    row_ids.append(row_id)
                    block_ids.append(block_id)
                    offsets.append(offset)
            if not block_ids:
                continue
            group_ptrs = layout.extract_segment_addrs_batch(
                np.asarray(block_ids, dtype=np.int64),
                np.asarray(offsets, dtype=np.int64),
                meta.tensor_block_size,
            )
            for row_id, ptr_row in zip(row_ids, group_ptrs):
                rows[row_id].append(ptr_row)
        if any(not row for row in rows):
            raise ValueError("WA KV cache pointer row is empty.")
        return np.vstack(
            [np.concatenate(row).astype(np.uint64, copy=False) for row in rows]
        )
```

- [ ] **Step 4: Run pointer tests**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_chunk_prefill.py::test_extract_fa_ptr_batches_hash_offsets_for_shared_tensor_block \
  test/test_hma_connector_chunk_prefill.py::test_extract_wa_ptr_uses_tail_candidates_and_zero_tail_group \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add ucm/integration/vllm/hma_connector.py test/test_hma_connector_chunk_prefill.py
git commit -m "refactor: batch FAWA pointer extraction"
```

## Task 4: Worker Load/Save Lifecycle And Load Error Anchors

**Files:**
- Modify: `ucm/integration/vllm/hma_connector.py`
- Test: `test/test_hma_connector_load_errors.py`
- Test: `test/test_hma_connector_chunk_prefill.py`

- [ ] **Step 1: Update load-error tests**

Replace `test_row_anchor_vllm_block_ids_uses_first_group_only()` in `test/test_hma_connector_load_errors.py` with:

```python
def test_first_group_anchor_ids_ignore_negative_values():
    connector = object.__new__(UCMFAWAConnector)

    assert connector._first_group_anchor_ids(([3, -1, 7], [5], [])) == {3, 7}
    assert connector._first_group_anchor_ids(([], [5], [])) == set()
```

Keep `test_wait_load_task_reports_vllm_block_ids_once()` unchanged.

- [ ] **Step 2: Run load-error test to verify it fails**

Run:

```bash
python3 -m pytest test/test_hma_connector_load_errors.py -q
```

Expected: FAIL because `_first_group_anchor_ids()` does not exist and old row-anchor helper still exists or is removed.

- [ ] **Step 3: Implement lifecycle changes**

Add this compact helper:

```python
    @staticmethod
    def _first_group_anchor_ids(
        candidate_vllm_ids: tuple[list[int], ...],
    ) -> set[int]:
        if not candidate_vllm_ids:
            return set()
        return {block_id for block_id in candidate_vllm_ids[0] if block_id >= 0}
```

Update `start_load_kv()` to consume flat metadata:

```python
        tasks: list[FAWALoadTask] = []
        for request_id, request in metadata.request_meta.items():
            if not request.load_keys:
                continue
            try:
                if self.fa_store is None:
                    raise RuntimeError("FA store is not initialized.")
                fa_ptrs = self._extract_fa_ptr(
                    request.load_keys,
                    request.load_hash_start,
                    request.load_hash_end,
                    request.load_vllm_block_ids,
                )
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "FA",
                        self.fa_store,
                        request.load_keys,
                        fa_ptrs,
                        self._first_group_anchor_ids(request.load_vllm_block_ids),
                    )
                )

                if self.wa_store is None:
                    raise RuntimeError("WA store is not initialized.")
                window_keys = request.load_keys[-1:]
                window_ptrs = self._extract_wa_ptr(
                    window_keys,
                    request.load_hash_end - 1,
                    request.load_hash_end,
                    request.load_vllm_block_ids,
                )
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "WA",
                        self.wa_store,
                        window_keys,
                        window_ptrs,
                        self._first_group_anchor_ids(request.load_vllm_block_ids),
                    )
                )
            except Exception as e:
                logger.error(
                    f"request {request_id} submit FAWA load task "
                    f"error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    self._first_group_anchor_ids(request.load_vllm_block_ids)
                )
```

Update `wait_for_save()` to consume flat metadata and batch all request dump rows:

```python
            total_keys: list[bytes] = []
            fa_ptr_rows: list[np.ndarray] = []
            wa_ptr_rows: list[np.ndarray] = []
            for request in metadata.request_meta.values():
                if not request.dump_keys:
                    continue
                total_keys.extend(request.dump_keys)
                fa_ptr_rows.append(
                    self._extract_fa_ptr(
                        request.dump_keys,
                        request.dump_hash_start,
                        request.dump_hash_end,
                        request.dump_vllm_block_ids,
                    )
                )
                wa_ptr_rows.append(
                    self._extract_wa_ptr(
                        request.dump_keys,
                        request.dump_hash_start,
                        request.dump_hash_end,
                        request.dump_vllm_block_ids,
                    )
                )

            if not total_keys:
                return

            fa_ptrs = np.vstack(fa_ptr_rows)
            window_ptrs = np.vstack(wa_ptr_rows)
```

Remove `_row_anchor_vllm_block_ids()`, `_select_rows()`, `_extract_group_addrs()`, `_try_select_group_block_ids()`, `_record_allocated_group_block_ids()`, `_replace_allocated_blocks()`, `_record_ready_group_block_ids()`, `_group_rows_for_indices()`, and other dead row-recording helpers.

- [ ] **Step 4: Run lifecycle tests**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_load_errors.py \
  test/test_hma_connector_chunk_prefill.py \
  -q
```

Expected: PASS for all migrated focused tests.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add ucm/integration/vllm/hma_connector.py test/test_hma_connector_load_errors.py test/test_hma_connector_chunk_prefill.py
git commit -m "refactor: use flat FAWA metadata in worker lifecycle"
```

## Task 5: E2E Test Migration And Full Focused Verification

**Files:**
- Modify: `test/test_hma_connector_gpu_tp4_e2e.py`
- Modify: `test/test_hma_connector_ascend_tp4_e2e.py`
- Modify: `ucm/integration/vllm/hma_connector.py` only for fixes discovered by E2E test migration.

- [ ] **Step 1: Update GPU E2E metadata assertions**

In `test/test_hma_connector_gpu_tp4_e2e.py`, replace assertions like:

```python
assert producer_dispatch.load_block_ids == ([], [])
assert producer_dispatch.dump_block_ids[0] == prefix_keys
producer_rows = producer_dispatch.dump_block_ids[1]
```

with flat-field assertions:

```python
assert producer_dispatch.load_keys == []
assert producer_dispatch.load_hash_start == 0
assert producer_dispatch.load_hash_end == 0
assert producer_dispatch.dump_keys == prefix_keys
assert producer_dispatch.dump_hash_end > producer_dispatch.dump_hash_start
producer_candidate_ids = producer_dispatch.dump_vllm_block_ids
assert len(producer_candidate_ids) == len(producer_scheduler.group_token_block_sizes)
```

Replace load assertions like:

```python
assert dispatch.load_block_ids[0] == prefix_keys
assert dispatch.dump_block_ids == ([], [])
```

with:

```python
assert dispatch.load_keys == prefix_keys
assert dispatch.load_hash_end - dispatch.load_hash_start == len(prefix_keys)
assert dispatch.dump_keys == []
assert dispatch.dump_hash_start == 0
assert dispatch.dump_hash_end == 0
```

Where the test previously inspected row contents, assert candidate slicing instead:

```python
for group_id in worker.window_group_ids:
    tail_blocks = worker.group_metas[group_id].tail_blocks
    if tail_blocks == 0:
        assert dispatch.load_vllm_block_ids[group_id] == []
    else:
        assert len(dispatch.load_vllm_block_ids[group_id]) == tail_blocks
```

- [ ] **Step 2: Update Ascend E2E metadata assertions**

In `test/test_hma_connector_ascend_tp4_e2e.py`, replace `load_block_ids` and `dump_block_ids` checks with flat fields:

```python
for request_meta in metadata.request_meta.values():
    assert request_meta.load_keys == prefix_keys
    assert request_meta.load_hash_end - request_meta.load_hash_start == len(prefix_keys)
    for group_id in worker.window_group_ids:
        if worker.group_metas[group_id].tail_blocks == 0:
            assert request_meta.load_vllm_block_ids[group_id] == []
```

For dump checks:

```python
for request_meta in metadata.request_meta.values():
    assert request_meta.dump_hash_end >= request_meta.dump_hash_start
    for group_id in worker.window_group_ids:
        tail_blocks = worker.group_metas[group_id].tail_blocks
        expected = (
            0
            if tail_blocks == 0
            else tail_blocks * len(request_meta.dump_keys)
        )
        assert len(request_meta.dump_vllm_block_ids[group_id]) == expected
```

- [ ] **Step 3: Run E2E simulations**

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_gpu_tp4_e2e.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  -q
```

Expected: PASS, or SKIP for accelerator-specific tests when the required accelerator is unavailable.

- [ ] **Step 4: Run compile and focused pytest suite**

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

Expected: no output and exit code 0.

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_load_errors.py \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py
```

Expected: all focused tests pass, with accelerator tests skipped only when hardware is unavailable.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add \
  ucm/integration/vllm/hma_connector.py \
  test/test_hma_connector_gpu_tp4_e2e.py \
  test/test_hma_connector_ascend_tp4_e2e.py
git commit -m "test: migrate FAWA range metadata e2e coverage"
```

## Task 6: Cleanup Review And Final Verification

**Files:**
- Modify: `ucm/integration/vllm/hma_connector.py`
- Modify tests only if cleanup changes require assertion updates.

- [ ] **Step 1: Scan for removed concepts**

Run:

```bash
rg -n "KVCacheSegment|KVCacheGroupRow|KVCacheGroupRows|KVCacheGroupAllocation|load_block_ids|dump_block_ids|record_block_cursor|store_block_cursor|allocated_group_block_ids|_row_anchor_vllm_block_ids|_group_rows_for_indices" \
  ucm/integration/vllm/hma_connector.py \
  test/test_hma_connector_load_errors.py \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py
```

Expected: no matches, except `load_block_ids`/`dump_block_ids` may appear in unrelated comments only if those comments explicitly describe removed old metadata. Remove such comments to keep the codebase unambiguous.

- [ ] **Step 2: Scan helper count and style**

Run:

```bash
rg -n "^    def _" ucm/integration/vllm/hma_connector.py
```

Expected: private helpers remain focused on existing store/config helpers plus the new small set: `_init_group_metas`, `_slice_group_block_ids`, `_extract_fa_ptr`, `_extract_wa_ptr`, and `_first_group_anchor_ids`. Do not split simple arithmetic into additional private helpers.

- [ ] **Step 3: Run final verification**

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

Expected: no output and exit code 0.

Run:

```bash
python3 -m pytest \
  test/test_hma_connector_load_errors.py \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py
```

Expected: all focused tests pass, with accelerator tests skipped only when hardware is unavailable.

- [ ] **Step 4: Commit cleanup**

Run:

```bash
git add \
  ucm/integration/vllm/hma_connector.py \
  test/test_hma_connector_load_errors.py \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py
git commit -m "refactor: remove old FAWA row metadata path"
```

## Notes For Execution

- The working tree currently has user edits in `ucm/integration/vllm/hma_connector.py`. Treat them as the starting point and do not revert them.
- The design intentionally overrides older FAWA skill notes that require `update_state_after_alloc()` to record allocation rows. For this refactor, `update_state_after_alloc()` is a no-op by user-approved design.
- Keep field names explicit. Do not reintroduce tuple-packed `(keys, rows)` metadata.
- Keep WA candidate slicing in scheduler metadata. Worker extraction should not receive all WA blocks for a hash block.
- Keep `KVCacheSegment` fully removed from implementation and tests.
