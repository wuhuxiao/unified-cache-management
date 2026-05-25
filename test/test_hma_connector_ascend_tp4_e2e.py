from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from ucm.integration.vllm.hma_connector import (
    FAWARequestMeta,
    KVCacheGroupLayout,
    KVCacheGroupMeta,
    UCMFAWAConnector,
    UCMFAWAConnectorMetadata,
)


@dataclass
class FakeBlock:
    block_id: int
    is_null: bool = False


@dataclass
class FakeKVCacheBlocks:
    blocks: tuple[list[FakeBlock], ...]


@dataclass
class FakeRequest:
    request_id: str
    all_token_ids: list[int] | None = None
    num_tokens: int = 0
    block_hashes: list[bytes] | None = None


@dataclass
class FakeCachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str]
    new_block_ids: list[tuple[list[int], ...] | None]


@dataclass
class FakeSchedulerOutput:
    scheduled_new_reqs: list
    scheduled_cached_reqs: FakeCachedRequestData
    num_scheduled_tokens: dict[str, int]
    finished_req_ids: set[str]


class AscendKVCacheGroupSpec:
    def __init__(self, layer_names, kv_cache_spec):
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec


@dataclass
class FakeKVCacheConfig:
    kv_cache_groups: list[AscendKVCacheGroupSpec]


def make_spec(name: str, **attrs):
    spec = type(name, (), {})()
    for key, value in attrs.items():
        setattr(spec, key, value)
    return spec


def make_ascend_connector() -> UCMFAWAConnector:
    c4_layers = [f"layer.{i}.c4" for i in range(21)]
    c128_layers = [f"layer.{i}.c128" for i in range(20)]
    swa_a_layers = [*c4_layers, "layer.extra.swa_a"]
    swa_b_layers = [*c128_layers, "layer.extra.swa_b", "mtp.extra.swa_b"]
    connector = UCMFAWAConnector.__new__(UCMFAWAConnector)
    connector._vllm_config = type(
        "FakeVllmConfig",
        (),
        {
            "model_config": type(
                "FakeModelConfig",
                (),
                {
                    "hf_config": type(
                        "FakeHFConfig",
                        (),
                        {"compress_ratios": [4] * 128},
                    )()
                },
            )()
        },
    )()
    connector._kv_cache_config = FakeKVCacheConfig(
        [
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec(
                    "Compress4AttentionSpec",
                    block_size=128,
                    compress_ratio=4,
                ),
            ),
            AscendKVCacheGroupSpec(
                swa_a_layers,
                make_spec(
                    "SWAAttentionSpec",
                    block_size=128,
                    sliding_window=128,
                ),
            ),
            AscendKVCacheGroupSpec(
                swa_b_layers,
                make_spec(
                    "SWAAttentionSpec",
                    block_size=128,
                    sliding_window=128,
                ),
            ),
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec(
                    "C4IndexerSpec",
                    block_size=1024,
                    compress_ratio=4,
                ),
            ),
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec(
                    "C4AttnKVStateSpec",
                    block_size=32,
                    sliding_window=8,
                ),
            ),
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec(
                    "C4AttnScoreStateSpec",
                    block_size=32,
                    sliding_window=8,
                ),
            ),
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec(
                    "C4IndexerKVStateSpec",
                    block_size=128,
                    sliding_window=8,
                ),
            ),
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec(
                    "C4IndexerScoreStateSpec",
                    block_size=128,
                    sliding_window=8,
                ),
            ),
            AscendKVCacheGroupSpec(
                c128_layers,
                make_spec(
                    "Compress128AttentionSpec",
                    block_size=128,
                    compress_ratio=128,
                ),
            ),
            AscendKVCacheGroupSpec(
                c128_layers,
                make_spec(
                    "C128AttnKVStateSpec",
                    block_size=64,
                    sliding_window=128,
                ),
            ),
            AscendKVCacheGroupSpec(
                c128_layers,
                make_spec(
                    "C128AttnScoreStateSpec",
                    block_size=64,
                    sliding_window=128,
                ),
            ),
        ]
    )
    connector.hash_block_size = UCMFAWAConnector.DEFAULT_HASH_BLOCK_SIZE
    connector.is_ascend_layout = False
    connector.fa_group_ids = []
    connector.window_group_ids = []
    connector.group_metas = {}
    connector._init_group_metas()
    connector.requests_meta = {}
    return connector


def hbm_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda")
        except RuntimeError as exc:
            pytest.skip(f"CUDA device is not usable: {exc}")
        return torch.device("cuda:0")
    pytest.skip("TP4 FAWA end-to-end tests require an NPU or CUDA device.")


class TensorKVStore:
    def __init__(self, tensor_size_list: list[int], name: str):
        self.tensor_size_list = [int(size) for size in tensor_size_list]
        self.shard_size = sum(self.tensor_size_list)
        self.name = name
        self.data: dict[bytes, list[torch.Tensor]] = {}
        self.ptr_registry: dict[int, torch.Tensor] = {}
        self.lookup_history: list[list[bytes]] = []
        self.load_history: list[list[bytes]] = []
        self.dump_history: list[list[bytes]] = []

    def lookup_on_prefix(self, keys):
        keys = list(keys)
        self.lookup_history.append(keys)
        hit_index = -1
        for i, key in enumerate(keys):
            if key not in self.data:
                break
            hit_index = i
        return hit_index

    def lookup(self, keys):
        keys = list(keys)
        self.lookup_history.append(keys)
        return [key in self.data for key in keys]

    def load_data(self, keys, shard_indexs, ptrs):
        keys = list(keys)
        ptrs = np.asarray(ptrs, dtype=np.uint64)
        self.load_history.append(keys)
        for row_idx, key in enumerate(keys):
            for col_idx, ptr in enumerate(ptrs[row_idx]):
                view = self._view_for_ptr(int(ptr), self.tensor_size_list[col_idx])
                view.copy_(self.data[key][col_idx].to(view.device).reshape(view.shape))
        return ("load", self.name, tuple(keys))

    def dump_data(self, keys, shard_indexs, ptrs, event_handle):
        keys = list(keys)
        ptrs = np.asarray(ptrs, dtype=np.uint64)
        self.dump_history.append(keys)
        for row_idx, key in enumerate(keys):
            chunks = []
            for ptr in ptrs[row_idx]:
                col_idx = len(chunks)
                chunks.append(
                    self._view_for_ptr(int(ptr), self.tensor_size_list[col_idx])
                    .detach()
                    .cpu()
                )
            self.data[key] = chunks
        return ("dump", self.name, tuple(keys))

    def wait(self, task):
        return None

    def seed(self, key: bytes, value: int) -> None:
        self.data[key] = [
            torch.full((size,), value, dtype=torch.uint8)
            for size in self.tensor_size_list
        ]

    def register_layouts(self, layouts: dict[int, KVCacheGroupLayout]) -> None:
        for layout in layouts.values():
            for tensor_or_tuple in layout.kvcaches.values():
                tensors = (
                    tensor_or_tuple
                    if isinstance(tensor_or_tuple, tuple)
                    else (tensor_or_tuple,)
                )
                for tensor in tensors:
                    self.ptr_registry[int(tensor.data_ptr())] = tensor

    def _view_for_ptr(self, ptr: int, size: int) -> torch.Tensor:
        for base_ptr, tensor in self.ptr_registry.items():
            tensor_bytes = tensor.numel() * tensor.element_size()
            if base_ptr <= ptr and ptr + size <= base_ptr + tensor_bytes:
                offset = ptr - base_ptr
                assert offset % tensor.element_size() == 0
                return tensor.view(torch.uint8).reshape(-1).narrow(0, offset, size)
        raise KeyError(ptr)

    def ptrs_bytes(self, ptr_row, size_list: list[int]) -> bytes:
        return b"".join(
            self._view_for_ptr(int(ptr), int(size)).detach().cpu().numpy().tobytes()
            for ptr, size in zip(ptr_row, size_list)
        )

    def fill_ptrs(self, ptrs: np.ndarray, size_list: list[int], value: int) -> None:
        for ptr_row in np.asarray(ptrs, dtype=np.uint64):
            for ptr, size in zip(ptr_row, size_list):
                self._view_for_ptr(int(ptr), int(size)).fill_(value)

    def stored_bytes(self, key: bytes) -> bytes:
        return b"".join(chunk.numpy().tobytes() for chunk in self.data[key])


def make_token_ids(block_size: int, block_values: list[int]) -> list[int]:
    return [
        token for block_value in block_values for token in [block_value] * block_size
    ]


def generated_hashes(block_size: int, token_ids: list[int], seed) -> list[bytes]:
    return [
        hashlib.md5(f"h{token_ids[start]}".encode()).digest()
        for start in range(0, len(token_ids), block_size)
        if len(token_ids[start : start + block_size]) == block_size
    ]


def req_data(request: FakeRequest, block_ids: tuple[list[int], ...]):
    return type(
        "ReqData",
        (),
        {"req_id": request.request_id, "block_ids": block_ids},
    )()


def build_allocation(
    connector: UCMFAWAConnector,
    canonical_blocks: int,
    base_block_id: int,
) -> tuple[list[int], ...]:
    allocation: list[list[int]] = []
    for group_id in sorted(connector.group_metas):
        max_tensor_idx = -1
        for canonical_idx in range(canonical_blocks):
            computed_end = (canonical_idx + 1) * connector.hash_block_size
            for group_block_idx in group_block_range(connector, group_id, computed_end):
                max_tensor_idx = max(max_tensor_idx, group_block_idx)
        allocation.append([base_block_id + idx for idx in range(max_tensor_idx + 1)])
    return tuple(allocation)


def group_block_range(
    connector: UCMFAWAConnector,
    group_id: int,
    computed_end_token: int,
) -> range:
    meta = connector.group_metas[group_id]
    group_token_block_size = meta.token_block_size
    end_block = math.ceil(computed_end_token / group_token_block_size)
    tail_blocks = meta.tail_blocks
    if tail_blocks is None:
        start_token = max(0, computed_end_token - connector.hash_block_size)
        start_block = start_token // group_token_block_size
    elif tail_blocks == 0:
        return range(0, 0)
    else:
        start_block = max(0, end_block - tail_blocks)
    return range(start_block, end_block)


def dump_candidate_count(
    connector: UCMFAWAConnector,
    group_id: int,
    hash_start: int,
    hash_end: int,
) -> int:
    meta = connector.group_metas[group_id]
    if group_id in connector.window_group_ids:
        if not meta.tail_tokens:
            return 0
        return (hash_end - hash_start) * meta.tail_blocks

    boundary_tokens = np.arange(hash_start, hash_end) * connector.hash_block_size - 1
    selected = boundary_tokens // meta.token_block_size
    return len(set(selected.tolist()))


def allocation_delta(
    current: tuple[list[int], ...],
    target: tuple[list[int], ...],
) -> tuple[list[int], ...]:
    return tuple(
        list(target_group[len(current_group) :])
        for current_group, target_group in zip(current, target)
    )


def make_layouts(
    device: torch.device, num_blocks: int
) -> dict[int, KVCacheGroupLayout]:
    def tensor(block_tokens: int, head_dim: int = 1) -> torch.Tensor:
        return torch.empty(
            (num_blocks, block_tokens, 1, head_dim), dtype=torch.uint8, device=device
        )

    return {
        0: KVCacheGroupLayout({"layer.0.c4": tensor(128)}),
        1: KVCacheGroupLayout({"layer.0.swa_a": tensor(128)}),
        2: KVCacheGroupLayout({"layer.0.swa_b": tensor(128)}),
        3: KVCacheGroupLayout(
            {"layer.0.c4_indexer": tuple(tensor(128) for _ in range(8))}
        ),
        4: KVCacheGroupLayout({"layer.0.c4_kv_state": tensor(32)}),
        5: KVCacheGroupLayout({"layer.0.c4_score_state": tensor(32)}),
        6: KVCacheGroupLayout({"layer.0.c4_index_kv_state": tensor(128)}),
        7: KVCacheGroupLayout({"layer.0.c4_index_score_state": tensor(128)}),
        8: KVCacheGroupLayout({"layer.0.c128": tuple(tensor(128) for _ in range(4))}),
        9: KVCacheGroupLayout({"layer.0.c128_kv_state": tensor(64)}),
        10: KVCacheGroupLayout({"layer.0.c128_score_state": tensor(64)}),
    }


def make_worker(
    scheduler: UCMFAWAConnector,
    group_layouts: dict[int, KVCacheGroupLayout],
    fa_store: TensorKVStore,
    wa_store: TensorKVStore,
    tp_rank: int,
) -> UCMFAWAConnector:
    worker = scheduler.__class__.__new__(scheduler.__class__)
    worker.hash_block_size = scheduler.hash_block_size
    worker.fa_group_ids = scheduler.fa_group_ids
    worker.window_group_ids = scheduler.window_group_ids
    worker.group_metas = dict(scheduler.group_metas)
    worker.is_ascend_layout = scheduler.is_ascend_layout
    worker.group_layouts = group_layouts
    worker.fa_store = fa_store
    worker.wa_store = wa_store
    worker.tp_rank = tp_rank
    worker.tp_size = 4
    worker._connector_metadata = None
    worker._invalid_block_ids = set()
    worker._get_dump_event_handle = lambda: 0
    fa_store.register_layouts(group_layouts)
    wa_store.register_layouts(group_layouts)
    return worker


def ptr_size_list(
    worker: UCMFAWAConnector,
    group_ids: tuple[int, ...],
) -> list[int]:
    return worker._store_tensor_size_list(worker.group_layouts, group_ids)


def fill_ptrs(
    worker: UCMFAWAConnector,
    group_ids: tuple[int, ...],
    ptrs: np.ndarray,
    value: int,
) -> None:
    store = worker.fa_store if group_ids == worker.fa_group_ids else worker.wa_store
    store.fill_ptrs(ptrs, ptr_size_list(worker, group_ids), value)


def ptr_row_bytes(
    worker: UCMFAWAConnector,
    store: TensorKVStore,
    group_ids: tuple[int, ...],
    ptrs: np.ndarray,
) -> list[bytes]:
    sizes = ptr_size_list(worker, group_ids)
    return [store.ptrs_bytes(row, sizes) for row in ptrs]


def bind_and_register(
    worker: UCMFAWAConnector, metadata: UCMFAWAConnectorMetadata
) -> None:
    worker.bind_connector_metadata(metadata)


def test_ascend_tp4_end_to_end_partial_external_hit_multi_request_chunk_prefill():
    device = hbm_device()
    scheduler = make_ascend_connector()
    scheduler.persist_token_threshold = 0
    scheduler.generate_hash = generated_hashes
    scheduler._seed = b"seed"

    template_layouts = make_layouts(device, num_blocks=128)
    fa_size_list = scheduler._store_tensor_size_list(
        template_layouts, scheduler.fa_group_ids
    )
    wa_size_list = scheduler._store_tensor_size_list(
        template_layouts, scheduler.window_group_ids
    )
    fa_store = TensorKVStore(fa_size_list, "fa")
    wa_store = TensorKVStore(wa_size_list, "wa")
    scheduler.fa_store = fa_store
    scheduler.wa_store = wa_store

    requests = [
        FakeRequest(
            "req-a", make_token_ids(scheduler.hash_block_size, [1, 2, 3, 4]), 2048
        ),
        FakeRequest(
            "req-b", make_token_ids(scheduler.hash_block_size, [1, 2, 5, 6]), 2048
        ),
    ]
    prefix_keys = generated_hashes(
        scheduler.hash_block_size, requests[0].all_token_ids, b"seed"
    )[:2]
    for i, key in enumerate(prefix_keys):
        fa_store.seed(key, 0x30 + i)
        wa_store.seed(key, 0x70 + i)

    workers = [
        make_worker(
            scheduler,
            make_layouts(device, num_blocks=128),
            fa_store,
            wa_store,
            tp_rank,
        )
        for tp_rank in range(4)
    ]

    initial_allocs = {
        request.request_id: build_allocation(scheduler, 2, 1 + idx * 20)
        for idx, request in enumerate(requests)
    }
    target_allocs = {
        request.request_id: build_allocation(scheduler, 4, 1 + idx * 20)
        for idx, request in enumerate(requests)
    }

    for request in requests:
        hit_tokens, is_async = scheduler.get_num_new_matched_tokens(request, 0)
        assert hit_tokens == 2 * scheduler.hash_block_size
        assert is_async is False
        scheduler.update_state_after_alloc(
            request,
            FakeKVCacheBlocks(
                tuple(
                    [FakeBlock(block_id) for block_id in group]
                    for group in initial_allocs[request.request_id]
                )
            ),
            hit_tokens,
        )

    first_metadata = scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[
                req_data(request, initial_allocs[request.request_id])
                for request in requests
            ],
            scheduled_cached_reqs=FakeCachedRequestData([], set(), []),
            num_scheduled_tokens={request.request_id: 0 for request in requests},
            finished_req_ids=set(),
        )
    )
    assert isinstance(first_metadata, UCMFAWAConnectorMetadata)
    for request in requests:
        request_meta = first_metadata.request_meta[request.request_id]
        assert request_meta.load_keys == prefix_keys
        assert request_meta.load_hash_end - request_meta.load_hash_start == len(
            prefix_keys
        )
        assert request_meta.dump_keys == []
        assert request_meta.dump_hash_end == request_meta.dump_hash_start
        for group_id in scheduler.window_group_ids:
            tail_blocks = scheduler.group_metas[group_id].tail_blocks
            if tail_blocks == 0:
                assert request_meta.load_vllm_block_ids[group_id] == []
            else:
                assert len(request_meta.load_vllm_block_ids[group_id]) == tail_blocks

    for worker in workers:
        bind_and_register(worker, first_metadata)
        worker.start_load_kv(None)
        for request in requests:
            request_meta = first_metadata.request_meta[request.request_id]
            fa_ptrs = worker._extract_fa_ptr(
                request_meta.load_keys,
                request_meta.load_hash_start,
                request_meta.load_hash_end,
                request_meta.load_vllm_block_ids,
            )
            wa_ptrs = worker._extract_wa_ptr(
                request_meta.load_keys[-1:],
                request_meta.load_vllm_block_ids,
            )
            assert ptr_row_bytes(worker, fa_store, worker.fa_group_ids, fa_ptrs) == [
                fa_store.stored_bytes(key) for key in prefix_keys
            ]
            expected_wa = (
                [] if wa_ptrs.size == 0 else [wa_store.stored_bytes(prefix_keys[-1])]
            )
            assert (
                ptr_row_bytes(worker, wa_store, worker.window_group_ids, wa_ptrs)
                == expected_wa
            )

    third_block_allocs = {
        request.request_id: build_allocation(
            scheduler,
            3,
            initial_allocs[request.request_id][0][0],
        )
        for request in requests
    }
    first_deltas = {}
    second_deltas = {}
    for request in requests:
        first_deltas[request.request_id] = allocation_delta(
            initial_allocs[request.request_id],
            third_block_allocs[request.request_id],
        )
        second_deltas[request.request_id] = allocation_delta(
            third_block_allocs[request.request_id],
            target_allocs[request.request_id],
        )

    partial_metadata = scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=FakeCachedRequestData(
                [request.request_id for request in requests],
                set(),
                [first_deltas[request.request_id] for request in requests],
            ),
            num_scheduled_tokens={
                request.request_id: scheduler.hash_block_size for request in requests
            },
            finished_req_ids=set(),
        )
    )
    expected_partial_dump_keys = {
        request.request_id: [
            generated_hashes(scheduler.hash_block_size, request.all_token_ids, b"seed")[
                2
            ]
        ]
        for request in requests
    }
    for request in requests:
        request_meta = partial_metadata.request_meta[request.request_id]
        assert request_meta.dump_keys == expected_partial_dump_keys[request.request_id]
        assert request_meta.dump_hash_end - request_meta.dump_hash_start == 1

    final_metadata = scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=FakeCachedRequestData(
                [request.request_id for request in requests],
                set(),
                [second_deltas[request.request_id] for request in requests],
            ),
            num_scheduled_tokens={
                request.request_id: scheduler.hash_block_size for request in requests
            },
            finished_req_ids=set(),
        )
    )
    expected_dump_keys = {
        request.request_id: [
            block_hash
            for block_hash in generated_hashes(
                scheduler.hash_block_size, request.all_token_ids, b"seed"
            )[3:4]
        ]
        for request in requests
    }
    for request in requests:
        request_meta = final_metadata.request_meta[request.request_id]
        assert request_meta.dump_keys == expected_dump_keys[request.request_id]
        assert request_meta.dump_hash_end - request_meta.dump_hash_start == len(
            request_meta.dump_keys
        )
        for group_id in scheduler.window_group_ids:
            expected = dump_candidate_count(
                scheduler,
                group_id,
                request_meta.dump_hash_start,
                request_meta.dump_hash_end,
            )
            assert len(request_meta.dump_vllm_block_ids[group_id]) == expected

    expected_fa_dump_keys: list[bytes] = []
    expected_wa_dump_keys: list[bytes] = []
    for worker in workers:
        bind_and_register(worker, final_metadata)
        wa_dump_ring_idx = 0
        for request_meta in final_metadata.request_meta.values():
            num_keys = len(request_meta.dump_keys)
            tp_block_start = num_keys * worker.tp_rank // worker.tp_size
            tp_block_end = num_keys * (worker.tp_rank + 1) // worker.tp_size
            tp_dump_keys = request_meta.dump_keys[tp_block_start:tp_block_end]
            if tp_dump_keys:
                tp_dump_vllm_block_ids = tuple(
                    group_block_ids[tp_block_start:tp_block_end]
                    for group_block_ids in request_meta.dump_vllm_block_ids
                )
                fa_ptrs = worker._extract_fa_ptr(
                    tp_dump_keys,
                    request_meta.dump_hash_start + tp_block_start,
                    request_meta.dump_hash_start + tp_block_end,
                    tp_dump_vllm_block_ids,
                )
                fill_ptrs(worker, worker.fa_group_ids, fa_ptrs, 0x91)
                expected_fa_dump_keys.extend(tp_dump_keys)
            if wa_dump_ring_idx % worker.tp_size == worker.tp_rank:
                wa_ptrs = worker._extract_wa_ptr(
                    request_meta.dump_keys[-1:],
                    request_meta.dump_vllm_block_ids,
                )
                fill_ptrs(worker, worker.window_group_ids, wa_ptrs, 0xE1)
                expected_wa_dump_keys.extend(request_meta.dump_keys[-1:])
                wa_dump_ring_idx += 1
    before_fa_dump_count = len(fa_store.dump_history)
    before_wa_dump_count = len(wa_store.dump_history)
    for worker in workers:
        worker.wait_for_save()

    assert [
        key
        for history in fa_store.dump_history[before_fa_dump_count:]
        for key in history
    ] == expected_fa_dump_keys
    assert [
        key
        for history in wa_store.dump_history[before_wa_dump_count:]
        for key in history
    ] == expected_wa_dump_keys
    for key in expected_wa_dump_keys:
        assert fa_store.stored_bytes(key) != wa_store.stored_bytes(key)
        assert set(fa_store.stored_bytes(key)) == {0x91}
        assert set(wa_store.stored_bytes(key)) == {0xE1}

    req_a_meta = final_metadata.request_meta["req-a"]
    for group_id in scheduler.fa_group_ids:
        meta = scheduler.group_metas[group_id]
        boundary_tokens = (
            np.arange(req_a_meta.dump_hash_start, req_a_meta.dump_hash_end)
            * scheduler.hash_block_size
            - 1
        )
        expected = len(set((boundary_tokens // meta.token_block_size).tolist()))
        assert len(req_a_meta.dump_vllm_block_ids[group_id]) == expected
    for group_id in scheduler.window_group_ids:
        if scheduler.group_metas[group_id].tail_blocks == 0:
            assert req_a_meta.dump_vllm_block_ids[group_id] == []
