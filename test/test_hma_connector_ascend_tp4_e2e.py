from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest
import torch

from test_hma_connector_chunk_prefill import (
    FakeBlock,
    FakeCachedRequestData,
    FakeKVCacheBlocks,
    FakeRequest,
    FakeSchedulerOutput,
    make_ascend_connector,
)
from ucm.integration.vllm.hma_connector import (
    KVCacheGroupLayout,
    UCMFAWAConnector,
    UCMFAWAConnectorMetadata,
)


def hbm_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
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

    def load_data(self, keys, shard_indexs, ptrs):
        keys = list(keys)
        ptrs = np.asarray(ptrs, dtype=np.uint64)
        self.load_history.append(keys)
        for row_idx, key in enumerate(keys):
            for col_idx, ptr in enumerate(ptrs[row_idx]):
                view = self.ptr_registry[int(ptr)]
                view.copy_(self.data[key][col_idx].to(view.device).reshape(view.shape))
        return ("load", self.name, tuple(keys))

    def dump_data(self, keys, shard_indexs, ptrs, event_handle):
        keys = list(keys)
        ptrs = np.asarray(ptrs, dtype=np.uint64)
        self.dump_history.append(keys)
        for row_idx, key in enumerate(keys):
            chunks = []
            for ptr in ptrs[row_idx]:
                chunks.append(self.ptr_registry[int(ptr)].detach().cpu().reshape(-1))
            self.data[key] = chunks
        return ("dump", self.name, tuple(keys))

    def wait(self, task):
        return None

    def seed(self, key: bytes, value: int) -> None:
        self.data[key] = [
            torch.full((size,), value, dtype=torch.uint8)
            for size in self.tensor_size_list
        ]

    def register_views(self, views: Iterable[torch.Tensor]) -> None:
        for view in views:
            flat_view = view.reshape(-1)
            self.ptr_registry[int(view.data_ptr())] = flat_view

    def row_bytes(self, ptr_row) -> bytes:
        chunks = []
        for ptr in ptr_row:
            chunks.append(
                self.ptr_registry[int(ptr)].detach().cpu().reshape(-1).numpy().tobytes()
            )
        return b"".join(chunks)

    def stored_bytes(self, key: bytes) -> bytes:
        return b"".join(chunk.numpy().tobytes() for chunk in self.data[key])


def make_token_ids(block_size: int, block_values: list[int]) -> list[int]:
    return [
        token
        for block_value in block_values
        for token in [block_value] * block_size
    ]


def generated_hashes(block_size: int, token_ids: list[int], seed) -> list[bytes]:
    return [
        f"h{token_ids[start]}".encode()
        for start in range(0, len(token_ids), block_size)
        if len(token_ids[start : start + block_size]) == block_size
    ]


def req_data(request: FakeRequest):
    return type("ReqData", (), {"req_id": request.request_id})()


def build_allocation(
    connector: UCMFAWAConnector,
    canonical_blocks: int,
    base_block_id: int,
) -> tuple[list[int], ...]:
    allocation: list[list[int]] = []
    for group_id in range(len(connector.group_token_block_sizes)):
        max_physical_idx = -1
        for canonical_idx in range(canonical_blocks):
            computed_end = (canonical_idx + 1) * connector.hash_block_size
            for group_block_idx in connector._group_block_range(group_id, computed_end):
                physical_idx = connector.block_span_layout.allocation_index(
                    group_id,
                    group_block_idx,
                )
                max_physical_idx = max(max_physical_idx, physical_idx)
        allocation.append([base_block_id + idx for idx in range(max_physical_idx + 1)])
    return tuple(allocation)


def allocation_delta(
    current: tuple[list[int], ...],
    target: tuple[list[int], ...],
) -> tuple[list[int], ...]:
    return tuple(
        list(target_group[len(current_group) :])
        for current_group, target_group in zip(current, target)
    )


def split_delta_first_group(
    delta: tuple[list[int], ...],
) -> tuple[tuple[list[int], ...], tuple[list[int], ...]]:
    first = tuple(list(group) if group_id == 0 else [] for group_id, group in enumerate(delta))
    second = tuple([] if group_id == 0 else list(group) for group_id, group in enumerate(delta))
    return first, second


def make_layouts(device: torch.device, num_blocks: int) -> dict[int, KVCacheGroupLayout]:
    def tensor(block_tokens: int, head_dim: int = 1) -> torch.Tensor:
        return torch.empty((num_blocks, block_tokens, 1, head_dim), dtype=torch.uint8, device=device)

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
        8: KVCacheGroupLayout(
            {"layer.0.c128": tuple(tensor(128) for _ in range(4))}
        ),
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
    worker.group_token_block_sizes = scheduler.group_token_block_sizes
    worker.group_physical_token_block_sizes = scheduler.group_physical_token_block_sizes
    worker.group_tail_blocks = scheduler.group_tail_blocks
    worker.group_window_spans = scheduler.group_window_spans
    worker.block_span_layout = scheduler.block_span_layout
    worker._ascend_layout = scheduler._ascend_layout
    worker.group_layouts = group_layouts
    worker._window_scratch_views = {}
    worker.fa_store = fa_store
    worker.wa_store = wa_store
    worker.tp_rank = tp_rank
    worker._connector_metadata = None
    worker._get_dump_event_handle = lambda: 0
    return worker


def register_rows(
    worker: UCMFAWAConnector,
    store: TensorKVStore,
    group_rows,
    group_ids: tuple[int, ...],
) -> None:
    for selected_row in worker._select_rows(group_rows, group_ids):
        for selected_group, group_id in zip(selected_row, group_ids):
            layout = worker.group_layouts.get(group_id)
            if layout is None or not selected_group:
                continue
            store.register_views(
                layout.extract_segment_tensor_views(
                    selected_group,
                    worker.group_physical_token_block_sizes[group_id],
                )
            )


def fill_rows(
    worker: UCMFAWAConnector,
    group_rows,
    group_ids: tuple[int, ...],
    value: int,
) -> None:
    for selected_row in worker._select_rows(group_rows, group_ids):
        for selected_group, group_id in zip(selected_row, group_ids):
            layout = worker.group_layouts.get(group_id)
            if layout is None or not selected_group:
                continue
            for view in layout.extract_segment_tensor_views(
                selected_group,
                worker.group_physical_token_block_sizes[group_id],
            ):
                view.fill_(value)


def row_bytes(
    worker: UCMFAWAConnector,
    store: TensorKVStore,
    group_rows,
    group_ids: tuple[int, ...],
) -> list[bytes]:
    ptrs = worker._extract_group_addrs(worker._select_rows(group_rows, group_ids), group_ids)
    return [store.row_bytes(row) for row in ptrs]


def bind_and_register(worker: UCMFAWAConnector, metadata: UCMFAWAConnectorMetadata) -> None:
    worker.bind_connector_metadata(metadata)
    for request_meta in metadata.request_meta.values():
        load_keys, load_rows = request_meta.load_block_ids
        if load_keys:
            register_rows(worker, worker.fa_store, load_rows, worker.fa_group_ids)
            register_rows(worker, worker.wa_store, load_rows[-1:], worker.window_group_ids)
        dump_keys, dump_rows = request_meta.dump_block_ids
        if dump_keys:
            register_rows(worker, worker.fa_store, dump_rows, worker.fa_group_ids)
            register_rows(worker, worker.wa_store, dump_rows, worker.window_group_ids)


def test_ascend_tp4_end_to_end_partial_external_hit_multi_request_chunk_prefill():
    device = hbm_device()
    scheduler = make_ascend_connector()
    scheduler.persist_token_threshold = 0
    scheduler.generate_hash = generated_hashes
    scheduler._seed = b"seed"

    template_layouts = make_layouts(device, num_blocks=128)
    fa_size_list = scheduler._store_tensor_size_list(template_layouts, scheduler.fa_group_ids)
    wa_size_list = scheduler._store_tensor_size_list(template_layouts, scheduler.window_group_ids)
    fa_store = TensorKVStore(fa_size_list, "fa")
    wa_store = TensorKVStore(wa_size_list, "wa")
    scheduler.fa_store = fa_store
    scheduler.wa_store = wa_store

    requests = [
        FakeRequest("req-a", make_token_ids(scheduler.hash_block_size, [1, 2, 3, 4]), 2048),
        FakeRequest("req-b", make_token_ids(scheduler.hash_block_size, [1, 2, 5, 6]), 2048),
    ]
    prefix_keys = [
        scheduler._block_key(block_hash)
        for block_hash in generated_hashes(scheduler.hash_block_size, requests[0].all_token_ids, b"seed")[:2]
    ]
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
            FakeKVCacheBlocks(tuple([FakeBlock(block_id) for block_id in group] for group in initial_allocs[request.request_id])),
            hit_tokens,
        )

    first_metadata = scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[req_data(request) for request in requests],
            scheduled_cached_reqs=FakeCachedRequestData([], set(), []),
            num_scheduled_tokens={request.request_id: 512 for request in requests},
            finished_req_ids=set(),
        )
    )
    assert isinstance(first_metadata, UCMFAWAConnectorMetadata)
    for request in requests:
        request_meta = first_metadata.request_meta[request.request_id]
        assert request_meta.load_block_ids[0] == prefix_keys
        assert request_meta.dump_block_ids == ([], [])

    for worker in workers:
        bind_and_register(worker, first_metadata)
        worker.start_load_kv(None)
        for request in requests:
            rows = first_metadata.request_meta[request.request_id].load_block_ids[1]
            assert row_bytes(worker, fa_store, rows, worker.fa_group_ids) == [
                fa_store.stored_bytes(key) for key in prefix_keys
            ]
            assert row_bytes(worker, wa_store, rows[-1:], worker.window_group_ids) == [
                wa_store.stored_bytes(prefix_keys[-1])
            ]

    first_deltas = {}
    second_deltas = {}
    for request in requests:
        first_deltas[request.request_id], second_deltas[request.request_id] = split_delta_first_group(
            allocation_delta(initial_allocs[request.request_id], target_allocs[request.request_id])
        )

    partial_metadata = scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=FakeCachedRequestData(
                [request.request_id for request in requests],
                set(),
                [first_deltas[request.request_id] for request in requests],
            ),
            num_scheduled_tokens={request.request_id: 512 for request in requests},
            finished_req_ids=set(),
        )
    )
    for request in requests:
        assert partial_metadata.request_meta[request.request_id].dump_block_ids == ([], [])

    final_metadata = scheduler.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=FakeCachedRequestData(
                [request.request_id for request in requests],
                set(),
                [second_deltas[request.request_id] for request in requests],
            ),
            num_scheduled_tokens={request.request_id: 512 for request in requests},
            finished_req_ids=set(),
        )
    )
    expected_dump_keys = {
        request.request_id: [
            scheduler._block_key(block_hash)
            for block_hash in generated_hashes(scheduler.hash_block_size, request.all_token_ids, b"seed")[2:4]
        ]
        for request in requests
    }
    for request in requests:
        assert final_metadata.request_meta[request.request_id].dump_block_ids[0] == expected_dump_keys[request.request_id]

    rank0 = workers[0]
    bind_and_register(rank0, final_metadata)
    for request_meta in final_metadata.request_meta.values():
        dump_rows = request_meta.dump_block_ids[1]
        fill_rows(rank0, dump_rows, rank0.fa_group_ids, 0x91)
        fill_rows(rank0, dump_rows, rank0.window_group_ids, 0xE1)
    before_dump_count = len(fa_store.dump_history)
    for worker in workers[1:]:
        bind_and_register(worker, final_metadata)
        worker.wait_for_save()
    assert len(fa_store.dump_history) == before_dump_count

    rank0.wait_for_save()
    assert fa_store.dump_history[-1] == expected_dump_keys["req-a"] + expected_dump_keys["req-b"]
    assert wa_store.dump_history[-1] == expected_dump_keys["req-a"] + expected_dump_keys["req-b"]
    for key in fa_store.dump_history[-1]:
        assert fa_store.stored_bytes(key) != wa_store.stored_bytes(key)
        assert set(fa_store.stored_bytes(key)) == {0x91}
        assert set(wa_store.stored_bytes(key)) == {0xE1}

    req_a_rows = scheduler.requests_meta["req-a"].group_block_ids
    assert req_a_rows[0][3][0].block_id == req_a_rows[3][3][0].block_id
    assert [req_a_rows[i][3][0].offset for i in range(4)] == [0, 512, 1024, 1536]
    assert req_a_rows[0][8][0].block_id == req_a_rows[3][8][0].block_id
    assert [req_a_rows[i][8][0].offset for i in range(4)] == [0, 512, 1024, 1536]
