from __future__ import annotations

import gc
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from ucm.integration.vllm.hma_connector import UCMFAWAConnector
from ucm.integration.vllm.hma_connector import FAWABlockSpanLayout
from ucm.integration.vllm.hma_connector import UCMAscendFAWAConnector


REQUEST_COUNT = 128
INPUT_TOKENS = 1024 * 1024
CHUNK_PREFILL_TOKENS = 4 * 1024
HASH_BLOCK_SIZE = 256
WINDOW_BLOCK_SIZE = 64
CANONICAL_BLOCKS = INPUT_TOKENS // HASH_BLOCK_SIZE
WINDOW_BLOCKS = INPUT_TOKENS // WINDOW_BLOCK_SIZE
ASCEND_HASH_BLOCK_SIZE = 512
ASCEND_CANONICAL_BLOCKS = INPUT_TOKENS // ASCEND_HASH_BLOCK_SIZE


@dataclass
class PerfRequest:
    request_id: str
    all_token_ids: "LazyTokenIds"
    num_tokens: int


class LazyTokenIds:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length


class VirtualBlock:
    __slots__ = ("block_id",)

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id


class VirtualBlockGroup:
    def __init__(self, start: int, length: int) -> None:
        self.start = start
        self.length = length

    def __iter__(self) -> Iterator[VirtualBlock]:
        for offset in range(self.length):
            yield VirtualBlock(self.start + offset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                VirtualBlock(self.start + offset)
                for offset in range(*index.indices(self.length))
            ]
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        return VirtualBlock(self.start + index)


@dataclass
class PerfKVCacheBlocks:
    blocks: tuple[VirtualBlockGroup, ...]


@dataclass
class PerfNewRequestData:
    req_id: str


@dataclass
class PerfCachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str]
    new_block_ids: list[tuple[list[int], ...] | None]


@dataclass
class PerfSchedulerOutput:
    scheduled_new_reqs: list[PerfNewRequestData]
    scheduled_cached_reqs: PerfCachedRequestData
    num_scheduled_tokens: dict[str, int]
    finished_req_ids: set[str]


class AscendKVCacheGroupSpec:
    def __init__(self, layer_names, kv_cache_spec):
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec


@dataclass
class FakeKVCacheConfig:
    kv_cache_groups: list[AscendKVCacheGroupSpec]


class StubStore:
    def __init__(self, hit_blocks: int) -> None:
        self.hit_blocks = hit_blocks
        self.lookup_on_prefix_calls = 0
        self.lookup_calls = 0
        self.load_calls = 0
        self.dump_calls = 0
        self.wait_calls = 0
        self.lookup_key_count = 0
        self.load_key_count = 0
        self.dump_key_count = 0
        self.load_ptr_shapes: list[tuple[int, ...]] = []
        self.dump_ptr_shapes: list[tuple[int, ...]] = []

    def lookup_on_prefix(self, keys: list[bytes]) -> int:
        self.lookup_on_prefix_calls += 1
        self.lookup_key_count += len(keys)
        return min(self.hit_blocks, len(keys)) - 1

    def lookup(self, keys: list[bytes]) -> list[bool]:
        self.lookup_calls += 1
        self.lookup_key_count += len(keys)
        return [idx < self.hit_blocks for idx, _ in enumerate(keys)]

    def load_data(self, block_ids, shard_indexs, dst_addr):
        del shard_indexs
        self.load_calls += 1
        self.load_key_count += len(block_ids)
        self.load_ptr_shapes.append(tuple(dst_addr.shape))
        return object()

    def dump_data(self, block_ids, shard_indexs, src_addr, prerequisite_handle=0):
        del shard_indexs, prerequisite_handle
        self.dump_calls += 1
        self.dump_key_count += len(block_ids)
        self.dump_ptr_shapes.append(tuple(src_addr.shape))
        return object()

    def wait(self, task) -> None:
        del task
        self.wait_calls += 1


class StubLayout:
    def __init__(self, base_ptr: int, block_stride: int = 4096) -> None:
        self.base_ptr = np.uint64(base_ptr)
        self.block_stride = np.uint64(block_stride)
        self.base_ptrs = np.asarray([self.base_ptr], dtype=np.uint64)
        self.block_strides = np.asarray([self.block_stride], dtype=np.uint64)

    def extract_segment_addrs_flat(self, segment, group_tensor_block_size: int):
        del group_tensor_block_size
        return np.asarray(
            [self.base_ptr + np.uint64(segment.block_id * self.block_stride)],
            dtype=np.uint64,
        )

    def extract_segment_addrs_flat_batch(
        self,
        block_ids: np.ndarray,
        offsets: np.ndarray,
        group_tensor_block_size: int,
    ):
        del offsets, group_tensor_block_size
        return (
            np.asarray(block_ids, dtype=np.uint64)[:, None] * self.block_strides[None, :]
            + self.base_ptrs[None, :]
        )


def generated_hashes(block_size: int, token_ids: LazyTokenIds, seed) -> list[bytes]:
    del seed
    return [
        f"h{block_idx}".encode()
        for block_idx in range(len(token_ids) // block_size)
    ]


def make_spec(name: str, **attrs):
    spec = type(name, (), {})()
    for key, value in attrs.items():
        setattr(spec, key, value)
    return spec


def make_connector(hit_blocks: int) -> UCMFAWAConnector:
    connector = UCMFAWAConnector.__new__(UCMFAWAConnector)
    connector.hash_block_size = HASH_BLOCK_SIZE
    connector.block_size = HASH_BLOCK_SIZE
    connector.fa_group_ids = (0,)
    connector.window_group_ids = (1,)
    connector.group_token_block_sizes = (HASH_BLOCK_SIZE, WINDOW_BLOCK_SIZE)
    connector.group_tensor_block_sizes = connector.group_token_block_sizes
    connector.group_tensor_block_ratios = (1, 1)
    connector.group_tail_blocks = (None, 1)
    connector.group_window_spans = ((HASH_BLOCK_SIZE,), (WINDOW_BLOCK_SIZE,))
    connector.block_span_layout = None
    connector.requests_meta = {}
    connector.request_hasher = lambda value: b"key:" + value[1]
    connector.persist_token_threshold = 0
    connector.generate_hash = generated_hashes
    connector._seed = b"seed"
    connector.fa_store = StubStore(hit_blocks)
    connector.wa_store = StubStore(hit_blocks)
    connector.store = connector.fa_store
    connector.group_layouts = {
        0: StubLayout(0x1000_0000),
        1: StubLayout(0x2000_0000),
    }
    connector._invalid_block_ids = set()
    connector.tp_rank = 0
    connector._connector_metadata = None
    connector._get_dump_event_handle = lambda: 0
    return connector


def make_ascend_connector(hit_blocks: int) -> UCMAscendFAWAConnector:
    c4_layers = [f"layer.{i}.c4" for i in range(21)]
    c128_layers = [f"layer.{i}.c128" for i in range(20)]
    swa_a_layers = [*c4_layers, "layer.extra.swa_a"]
    swa_b_layers = [*c128_layers, "layer.extra.swa_b", "mtp.extra.swa_b"]
    connector = UCMAscendFAWAConnector.__new__(UCMAscendFAWAConnector)
    connector._kv_cache_config = FakeKVCacheConfig(
        [
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec("Compress4AttentionSpec", block_size=128, compress_ratio=4),
            ),
            AscendKVCacheGroupSpec(
                swa_a_layers,
                make_spec("SWAAttentionSpec", block_size=128, sliding_window=128),
            ),
            AscendKVCacheGroupSpec(
                swa_b_layers,
                make_spec("SWAAttentionSpec", block_size=128, sliding_window=128),
            ),
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec("C4IndexerSpec", block_size=1024, compress_ratio=4),
            ),
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec("C4AttnKVStateSpec", block_size=32, sliding_window=8),
            ),
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec("C4AttnScoreStateSpec", block_size=32, sliding_window=8),
            ),
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec("C4IndexerKVStateSpec", block_size=128, sliding_window=8),
            ),
            AscendKVCacheGroupSpec(
                c4_layers,
                make_spec("C4IndexerScoreStateSpec", block_size=128, sliding_window=8),
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
                make_spec("C128AttnKVStateSpec", block_size=64, sliding_window=128),
            ),
            AscendKVCacheGroupSpec(
                c128_layers,
                make_spec("C128AttnScoreStateSpec", block_size=64, sliding_window=128),
            ),
        ]
    )
    connector.fa_group_ids = (0, 3, 8)
    connector.window_group_ids = (1, 2, 4, 5, 6, 7, 9, 10)
    connector.block_span_layout = FAWABlockSpanLayout(
        connector._kv_cache_config,
        connector.fa_group_ids,
    )
    connector._ascend_layout = connector.block_span_layout.is_ascend
    connector.hash_block_size = connector.block_span_layout.hash_block_size
    connector.block_size = connector.hash_block_size
    connector.group_token_block_sizes = (
        connector.block_span_layout.group_token_block_sizes
    )
    connector.group_tensor_block_sizes = (
        connector.block_span_layout.group_tensor_block_sizes
    )
    connector.group_tensor_block_ratios = (
        connector.block_span_layout.group_tensor_block_ratios
    )
    connector.group_tail_blocks = connector._get_group_tail_blocks()
    connector.group_window_spans = connector._get_group_window_spans()
    connector.requests_meta = {}
    connector.request_hasher = lambda value: b"key:" + value[1]
    connector.persist_token_threshold = 0
    connector.generate_hash = generated_hashes
    connector._seed = b"seed"
    connector.fa_store = StubStore(hit_blocks)
    connector.wa_store = StubStore(hit_blocks)
    connector.store = connector.fa_store
    connector.group_layouts = {
        group_id: StubLayout(0x3000_0000 + group_id * 0x0100_0000)
        for group_id in range(len(connector.group_token_block_sizes))
    }
    connector._invalid_block_ids = set()
    connector.tp_rank = 0
    connector._connector_metadata = None
    connector._get_dump_event_handle = lambda: 0
    return connector


def make_requests(request_count: int) -> list[PerfRequest]:
    return [
        PerfRequest(
            request_id=f"req-{idx}",
            all_token_ids=LazyTokenIds(INPUT_TOKENS),
            num_tokens=INPUT_TOKENS,
        )
        for idx in range(request_count)
    ]


def make_blocks(request_index: int) -> PerfKVCacheBlocks:
    base = request_index * (WINDOW_BLOCKS + CANONICAL_BLOCKS + 1024)
    return PerfKVCacheBlocks(
        (
            VirtualBlockGroup(base, CANONICAL_BLOCKS),
            VirtualBlockGroup(base + CANONICAL_BLOCKS, WINDOW_BLOCKS),
        )
    )


def make_ascend_blocks(request_index: int) -> PerfKVCacheBlocks:
    base = request_index * 1_000_000
    return PerfKVCacheBlocks(
        (
            VirtualBlockGroup(base + 0, ASCEND_CANONICAL_BLOCKS),
            VirtualBlockGroup(base + 100_000, ASCEND_CANONICAL_BLOCKS * 4),
            VirtualBlockGroup(base + 200_000, ASCEND_CANONICAL_BLOCKS * 4),
            VirtualBlockGroup(base + 300_000, ASCEND_CANONICAL_BLOCKS // 8),
            VirtualBlockGroup(base + 400_000, ASCEND_CANONICAL_BLOCKS * 16),
            VirtualBlockGroup(base + 500_000, ASCEND_CANONICAL_BLOCKS * 16),
            VirtualBlockGroup(base + 600_000, ASCEND_CANONICAL_BLOCKS * 4),
            VirtualBlockGroup(base + 700_000, ASCEND_CANONICAL_BLOCKS * 4),
            VirtualBlockGroup(base + 800_000, ASCEND_CANONICAL_BLOCKS // 32),
            VirtualBlockGroup(base + 900_000, ASCEND_CANONICAL_BLOCKS * 2),
            VirtualBlockGroup(base + 950_000, ASCEND_CANONICAL_BLOCKS * 2),
        )
    )


def new_scheduler_output(
    requests: list[PerfRequest],
    scheduled_tokens: int,
) -> PerfSchedulerOutput:
    return PerfSchedulerOutput(
        scheduled_new_reqs=[
            PerfNewRequestData(request.request_id) for request in requests
        ],
        scheduled_cached_reqs=PerfCachedRequestData([], set(), []),
        num_scheduled_tokens={
            request.request_id: scheduled_tokens for request in requests
        },
        finished_req_ids=set(),
    )


def elapsed_ms(func):
    start = time.perf_counter_ns()
    result = func()
    return result, (time.perf_counter_ns() - start) / 1_000_000


def summarize_store(store: StubStore) -> dict[str, object]:
    return {
        "lookup_on_prefix_calls": store.lookup_on_prefix_calls,
        "lookup_calls": store.lookup_calls,
        "load_calls": store.load_calls,
        "dump_calls": store.dump_calls,
        "wait_calls": store.wait_calls,
        "lookup_key_count": store.lookup_key_count,
        "load_key_count": store.load_key_count,
        "dump_key_count": store.dump_key_count,
        "load_ptr_shapes": store.load_ptr_shapes[:3],
        "dump_ptr_shapes": store.dump_ptr_shapes[:3],
    }


def run_prefix_hit_case(request_count: int) -> dict[str, object]:
    connector = make_connector(hit_blocks=CANONICAL_BLOCKS)
    requests = make_requests(request_count)

    def lookup_all():
        for request in requests:
            hit_tokens, is_async = connector.get_num_new_matched_tokens(request, 0)
            assert hit_tokens == INPUT_TOKENS - 1
            assert is_async is False

    _, get_num_ms = elapsed_ms(lookup_all)

    def update_all():
        for idx, request in enumerate(requests):
            connector.update_state_after_alloc(
                request,
                make_blocks(idx),
                INPUT_TOKENS - 1,
            )

    _, update_ms = elapsed_ms(update_all)

    metadata, build_meta_ms = elapsed_ms(
        lambda: connector.build_connector_meta(
            new_scheduler_output(requests, scheduled_tokens=HASH_BLOCK_SIZE)
        )
    )
    connector.bind_connector_metadata(metadata)
    _, start_load_ms = elapsed_ms(lambda: connector.start_load_kv(None))
    _, wait_save_ms = elapsed_ms(lambda: connector.wait_for_save())

    return {
        "case": "prefix_hit_load",
        "request_count": request_count,
        "input_tokens": INPUT_TOKENS,
        "canonical_blocks_per_request": CANONICAL_BLOCKS,
        "timing_ms": {
            "get_num_new_matched_tokens": get_num_ms,
            "update_state_after_alloc": update_ms,
            "build_connector_meta": build_meta_ms,
            "start_load_kv": start_load_ms,
            "wait_for_save": wait_save_ms,
            "total": (
                get_num_ms
                + update_ms
                + build_meta_ms
                + start_load_ms
                + wait_save_ms
            ),
        },
        "fa_store": summarize_store(connector.fa_store),
        "wa_store": summarize_store(connector.wa_store),
    }


def run_ascend_prefix_hit_case(request_count: int) -> dict[str, object]:
    connector = make_ascend_connector(hit_blocks=ASCEND_CANONICAL_BLOCKS)
    requests = make_requests(request_count)

    def lookup_all():
        for request in requests:
            hit_tokens, is_async = connector.get_num_new_matched_tokens(request, 0)
            assert hit_tokens == INPUT_TOKENS - 1
            assert is_async is False

    _, get_num_ms = elapsed_ms(lookup_all)

    def update_all():
        for idx, request in enumerate(requests):
            connector.update_state_after_alloc(
                request,
                make_ascend_blocks(idx),
                INPUT_TOKENS - 1,
            )

    _, update_ms = elapsed_ms(update_all)

    metadata, build_meta_ms = elapsed_ms(
        lambda: connector.build_connector_meta(
            new_scheduler_output(requests, scheduled_tokens=ASCEND_HASH_BLOCK_SIZE)
        )
    )
    connector.bind_connector_metadata(metadata)
    _, start_load_ms = elapsed_ms(lambda: connector.start_load_kv(None))
    _, wait_save_ms = elapsed_ms(lambda: connector.wait_for_save())

    return {
        "case": "ascend_prefix_hit_load",
        "request_count": request_count,
        "input_tokens": INPUT_TOKENS,
        "canonical_blocks_per_request": ASCEND_CANONICAL_BLOCKS,
        "timing_ms": {
            "get_num_new_matched_tokens": get_num_ms,
            "update_state_after_alloc": update_ms,
            "build_connector_meta": build_meta_ms,
            "start_load_kv": start_load_ms,
            "wait_for_save": wait_save_ms,
            "total": (
                get_num_ms
                + update_ms
                + build_meta_ms
                + start_load_ms
                + wait_save_ms
            ),
        },
        "fa_store": summarize_store(connector.fa_store),
        "wa_store": summarize_store(connector.wa_store),
    }


def run_chunk_prefill_dump_case(request_count: int) -> dict[str, object]:
    connector = make_connector(hit_blocks=0)
    requests = make_requests(request_count)

    def lookup_all():
        for request in requests:
            hit_tokens, is_async = connector.get_num_new_matched_tokens(request, 0)
            assert hit_tokens == 0
            assert is_async is False

    _, get_num_ms = elapsed_ms(lookup_all)

    def update_all():
        for idx, request in enumerate(requests):
            connector.update_state_after_alloc(request, make_blocks(idx), 0)

    _, update_ms = elapsed_ms(update_all)

    metadata, build_meta_ms = elapsed_ms(
        lambda: connector.build_connector_meta(
            new_scheduler_output(requests, scheduled_tokens=CHUNK_PREFILL_TOKENS)
        )
    )
    connector.bind_connector_metadata(metadata)
    _, start_load_ms = elapsed_ms(lambda: connector.start_load_kv(None))
    _, wait_save_ms = elapsed_ms(lambda: connector.wait_for_save())

    return {
        "case": "chunk_prefill_dump",
        "request_count": request_count,
        "input_tokens": INPUT_TOKENS,
        "chunk_prefill_tokens": CHUNK_PREFILL_TOKENS,
        "chunk_count": 1,
        "canonical_blocks_per_request": CANONICAL_BLOCKS,
        "timing_ms": {
            "get_num_new_matched_tokens": get_num_ms,
            "update_state_after_alloc": update_ms,
            "build_connector_meta": build_meta_ms,
            "start_load_kv": start_load_ms,
            "wait_for_save": wait_save_ms,
            "total": (
                get_num_ms
                + update_ms
                + build_meta_ms
                + start_load_ms
                + wait_save_ms
            ),
        },
        "fa_store": summarize_store(connector.fa_store),
        "wa_store": summarize_store(connector.wa_store),
    }


def run_ascend_chunk_prefill_dump_case(request_count: int) -> dict[str, object]:
    connector = make_ascend_connector(hit_blocks=0)
    requests = make_requests(request_count)

    def lookup_all():
        for request in requests:
            hit_tokens, is_async = connector.get_num_new_matched_tokens(request, 0)
            assert hit_tokens == 0
            assert is_async is False

    _, get_num_ms = elapsed_ms(lookup_all)

    def update_all():
        for idx, request in enumerate(requests):
            connector.update_state_after_alloc(request, make_ascend_blocks(idx), 0)

    _, update_ms = elapsed_ms(update_all)

    metadata, build_meta_ms = elapsed_ms(
        lambda: connector.build_connector_meta(
            new_scheduler_output(requests, scheduled_tokens=CHUNK_PREFILL_TOKENS)
        )
    )
    connector.bind_connector_metadata(metadata)
    _, start_load_ms = elapsed_ms(lambda: connector.start_load_kv(None))
    _, wait_save_ms = elapsed_ms(lambda: connector.wait_for_save())

    return {
        "case": "ascend_chunk_prefill_dump",
        "request_count": request_count,
        "input_tokens": INPUT_TOKENS,
        "chunk_prefill_tokens": CHUNK_PREFILL_TOKENS,
        "chunk_count": 1,
        "canonical_blocks_per_request": ASCEND_CANONICAL_BLOCKS,
        "timing_ms": {
            "get_num_new_matched_tokens": get_num_ms,
            "update_state_after_alloc": update_ms,
            "build_connector_meta": build_meta_ms,
            "start_load_kv": start_load_ms,
            "wait_for_save": wait_save_ms,
            "total": (
                get_num_ms
                + update_ms
                + build_meta_ms
                + start_load_ms
                + wait_save_ms
            ),
        },
        "fa_store": summarize_store(connector.fa_store),
        "wa_store": summarize_store(connector.wa_store),
    }


@pytest.mark.stage(2)
@pytest.mark.feature("hma_fawa_runtime_perf")
def test_hma_fawa_connector_runtime_metadata_perf():
    request_count = REQUEST_COUNT
    with (
        patch("ucm.integration.vllm.hma_connector.logger.info", lambda *a, **k: None),
        patch(
            "ucm.integration.vllm.hma_connector.logger.info_once",
            lambda *a, **k: None,
        ),
        ):
            prefix_hit = run_prefix_hit_case(request_count)
            gc.collect()
            chunk_prefill = run_chunk_prefill_dump_case(request_count)
            gc.collect()
            ascend_prefix_hit = run_ascend_prefix_hit_case(request_count)
            gc.collect()
            ascend_chunk_prefill = run_ascend_chunk_prefill_dump_case(request_count)

    expected_rows = request_count * CANONICAL_BLOCKS
    expected_ascend_rows = request_count * ASCEND_CANONICAL_BLOCKS
    assert prefix_hit["fa_store"]["load_key_count"] == expected_rows
    assert prefix_hit["wa_store"]["load_key_count"] == request_count
    expected_chunk_rows = request_count * (CHUNK_PREFILL_TOKENS // HASH_BLOCK_SIZE)
    expected_ascend_chunk_rows = request_count * (
        CHUNK_PREFILL_TOKENS // ASCEND_HASH_BLOCK_SIZE
    )
    assert chunk_prefill["fa_store"]["dump_key_count"] == expected_chunk_rows
    assert chunk_prefill["wa_store"]["dump_key_count"] == expected_chunk_rows
    assert ascend_prefix_hit["fa_store"]["load_key_count"] == expected_ascend_rows
    assert ascend_prefix_hit["wa_store"]["load_key_count"] == request_count
    assert (
        ascend_chunk_prefill["fa_store"]["dump_key_count"]
        == expected_ascend_chunk_rows
    )
    assert (
        ascend_chunk_prefill["wa_store"]["dump_key_count"]
        == expected_ascend_chunk_rows
    )

    print(
        "\nUCMFAWAConnector metadata perf summary:\n"
        + json.dumps(
            {
                "prefix_hit": prefix_hit,
                "chunk_prefill": chunk_prefill,
                "ascend_prefix_hit": ascend_prefix_hit,
                "ascend_chunk_prefill": ascend_chunk_prefill,
            },
            indent=2,
            sort_keys=True,
        )
    )
