from dataclasses import dataclass

import numpy as np
import pytest
import torch

from ucm.integration.vllm.hma_connector import (
    FAWABlockSpanLayout,
    FAWADispatchBlockPlan,
    FAWARequestDispatchMeta,
    FAWARequestMeta,
    KVCacheGroupLayout,
    KVCacheSegment,
    UCMAscendFAWAConnector,
    UCMFAWAConnector,
    UCMFAWAConnectorMetadata,
)


@dataclass
class FakeBlock:
    block_id: int


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


def select_rows(group_rows: list, group_ids: tuple[int, ...]) -> list:
    return [
        tuple(list(group_row[group_id]) for group_id in group_ids)
        for group_row in group_rows
    ]


def materialize_rows(
    connector: UCMFAWAConnector,
    req_meta: FAWARequestMeta,
    indices,
) -> list[tuple[list[KVCacheSegment], ...]]:
    row_indices = np.asarray(tuple(indices), dtype=np.int64)
    if row_indices.size == 0:
        return []
    group_blocks_by_group: list[np.ndarray] = []
    offsets_by_group: list[np.ndarray] = []
    lengths_by_group: list[np.ndarray] = []
    for group_id in range(len(connector.group_token_block_sizes)):
        segment_count = connector._expected_group_row_segments(group_id)
        if segment_count == 0:
            group_blocks_by_group.append(
                np.empty((len(row_indices), 0), dtype=np.int64)
            )
            offsets_by_group.append(np.empty((len(row_indices), 0), dtype=np.int64))
            lengths_by_group.append(np.empty((len(row_indices), 0), dtype=np.int64))
            continue
        group_blocks, offsets, lengths = connector._gather_group_blocks_and_offsets(
            req_meta,
            group_id,
            row_indices,
        )
        group_blocks_by_group.append(group_blocks)
        offsets_by_group.append(offsets)
        lengths_by_group.append(lengths)

    rows: list[tuple[list[KVCacheSegment], ...]] = []
    for row_pos in range(len(row_indices)):
        row: list[list[KVCacheSegment]] = []
        for group_id in range(len(connector.group_token_block_sizes)):
            group_segments: list[KVCacheSegment] = []
            for segment_idx in range(group_blocks_by_group[group_id].shape[1]):
                group_segments.append(
                    KVCacheSegment(
                        int(group_blocks_by_group[group_id][row_pos, segment_idx]),
                        int(offsets_by_group[group_id][row_pos, segment_idx]),
                        int(lengths_by_group[group_id][row_pos, segment_idx]),
                    )
                )
            row.append(group_segments)
        rows.append(tuple(row))
    return rows


def materialize_plan(
    connector: UCMFAWAConnector,
    plan: FAWADispatchBlockPlan | None,
) -> tuple[list[bytes], list[tuple[list[KVCacheSegment], ...]]]:
    if plan is None:
        return [], []
    return plan.keys, materialize_rows(connector, plan.req_meta, plan.indices)


def load_block_ids(
    connector: UCMFAWAConnector,
    dispatch: FAWARequestDispatchMeta,
) -> tuple[list[bytes], list[tuple[list[KVCacheSegment], ...]]]:
    return materialize_plan(connector, dispatch.load_block_plan)


def dump_block_ids(
    connector: UCMFAWAConnector,
    dispatch: FAWARequestDispatchMeta,
) -> tuple[list[bytes], list[tuple[list[KVCacheSegment], ...]]]:
    return materialize_plan(connector, dispatch.dump_block_plan)


def layout_segment_addrs(
    layout: KVCacheGroupLayout,
    segments: list[KVCacheSegment],
    group_tensor_block_size: int,
) -> np.ndarray:
    if not segments:
        return np.empty((0, len(layout.base_ptrs)), dtype=np.uint64)
    block_ids = np.asarray([segment.block_id for segment in segments], dtype=np.int64)
    offsets = np.asarray([segment.offset for segment in segments], dtype=np.int64)
    return layout.extract_segment_addrs_flat_batch(
        block_ids,
        offsets,
        group_tensor_block_size,
    )


def make_connector() -> UCMFAWAConnector:
    connector = UCMFAWAConnector.__new__(UCMFAWAConnector)
    connector.hash_block_size = 256
    connector.fa_group_ids = (0,)
    connector.window_group_ids = (1,)
    connector.group_token_block_sizes = (256, 64)
    connector.group_tensor_block_sizes = connector.group_token_block_sizes
    connector.group_tail_blocks = (None, 1)
    connector.group_window_spans = ((256,), (64,))
    connector.block_span_layout = None
    connector.requests_meta = {}
    return connector


def make_two_tail_connector() -> UCMFAWAConnector:
    connector = make_connector()
    connector.group_tail_blocks = (None, 2)
    connector.group_window_spans = ((256,), (64, 64))
    return connector


def test_cached_chunk_prefill_appends_new_block_ids_for_later_dump():
    connector = make_connector()
    request = FakeRequest("req-0")
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a", b"b"],
        hbm_hit_block_num=0,
        total_hit_block_num=0,
        num_token_ids=512,
        token_processed=0,
    )
    connector.requests_meta[request.request_id] = req_meta

    connector.update_state_after_alloc(
        request,
        FakeKVCacheBlocks(
            (
                [FakeBlock(10)],
                [FakeBlock(100), FakeBlock(101), FakeBlock(102), FakeBlock(103)],
            )
        ),
        0,
    )

    first_step = FakeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=FakeCachedRequestData(
            req_ids=[request.request_id],
            resumed_req_ids=set(),
            new_block_ids=[None],
        ),
        num_scheduled_tokens={request.request_id: 128},
        finished_req_ids=set(),
    )
    first_meta = connector.build_connector_meta(first_step)
    assert isinstance(first_meta, UCMFAWAConnectorMetadata)
    first_dispatch = first_meta.request_meta[request.request_id]
    assert first_dispatch.dump_block_plan is None
    assert req_meta.token_processed == 128
    assert req_meta.record_block_cursor == 1

    second_step = FakeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=FakeCachedRequestData(
            req_ids=[request.request_id],
            resumed_req_ids=set(),
            new_block_ids=[
                (
                    [11],
                    [104, 105, 106, 107],
                )
            ],
        ),
        num_scheduled_tokens={request.request_id: 384},
        finished_req_ids=set(),
    )
    second_meta = connector.build_connector_meta(second_step)

    assert isinstance(second_meta, UCMFAWAConnectorMetadata)
    request_meta = second_meta.request_meta[request.request_id]
    assert request_meta.dump_block_plan is not None
    assert request_meta.dump_block_plan.indices == (0, 1)
    assert dump_block_ids(connector, request_meta) == (
        [b"a", b"b"],
        [
            ([10], [103]),
            ([11], [107]),
        ],
    )
    assert [group.tolist() for group in req_meta.allocated_group_block_ids] == [
        [10, 11],
        [100, 101, 102, 103, 104, 105, 106, 107],
    ]
    assert req_meta.record_block_cursor == 2
    assert req_meta.token_processed == 512


def test_window_rows_require_hash_block_to_cover_tail_window():
    connector = make_two_tail_connector()
    connector.hash_block_size = 64
    connector.group_token_block_sizes = (64, 64)
    connector.group_tensor_block_sizes = connector.group_token_block_sizes
    connector.group_window_spans = ((64,), (64, 64))
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a"],
        hbm_hit_block_num=0,
        total_hit_block_num=0,
        num_token_ids=64,
        token_processed=0,
    )

    with pytest.raises(RuntimeError, match="hash block boundary"):
        connector._replace_allocated_blocks(
            req_meta,
            FakeKVCacheBlocks(
                (
                    [FakeBlock(10)],
                    [FakeBlock(100)],
                )
            ),
        )


def test_cached_chunk_prefill_replaces_block_ids_for_resumed_request():
    connector = make_connector()
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a"],
        num_token_ids=256,
        allocated_group_block_ids=([1], [10, 11, 12, 13]),
    )
    connector.requests_meta["req-1"] = req_meta

    scheduler_output = FakeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=FakeCachedRequestData(
            req_ids=["req-1"],
            resumed_req_ids={"req-1"},
            new_block_ids=[([2], [20, 21, 22, 23])],
        ),
        num_scheduled_tokens={"req-1": 256},
        finished_req_ids=set(),
    )
    metadata = connector.build_connector_meta(scheduler_output)

    request_dispatch = metadata.request_meta["req-1"]
    assert request_dispatch.dump_block_plan is not None
    assert request_dispatch.dump_block_plan.indices == (0,)
    assert dump_block_ids(connector, request_dispatch) == (
        [b"a"],
        [([2], [23])],
    )
    assert [group.tolist() for group in req_meta.allocated_group_block_ids] == [
        [2],
        [20, 21, 22, 23],
    ]


def test_update_state_after_alloc_skips_canonical_block_until_all_groups_allocated():
    connector = make_connector()
    request = FakeRequest("req-2")
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a"],
        num_token_ids=256,
    )
    connector.requests_meta[request.request_id] = req_meta

    connector.update_state_after_alloc(
        request,
        FakeKVCacheBlocks(
            (
                [FakeBlock(10)],
                [FakeBlock(100), FakeBlock(101), FakeBlock(102)],
            )
        ),
        0,
    )

    assert req_meta.record_block_cursor == 0
    assert [list(group) for group in req_meta.allocated_group_block_ids] == [
        [10],
        [100, 101, 102],
    ]

    scheduler_output = FakeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=FakeCachedRequestData(
            req_ids=[request.request_id],
            resumed_req_ids=set(),
            new_block_ids=[
                (
                    [],
                    [103],
                )
            ],
        ),
        num_scheduled_tokens={request.request_id: 256},
        finished_req_ids=set(),
    )
    metadata = connector.build_connector_meta(scheduler_output)

    request_dispatch = metadata.request_meta[request.request_id]
    assert request_dispatch.dump_block_plan is not None
    assert request_dispatch.dump_block_plan.indices == (0,)
    assert dump_block_ids(connector, request_dispatch) == (
        [b"a"],
        [([10], [103])],
    )
    assert req_meta.record_block_cursor == 1
    assert req_meta.token_processed == 256


def test_tail_zero_group_has_empty_row_and_no_block_requirement():
    connector = make_connector()
    connector.group_token_block_sizes = (256, 64, 8)
    connector.group_tensor_block_sizes = connector.group_token_block_sizes
    connector.group_tail_blocks = (None, 1, 0)
    connector.group_window_spans = ((256,), (64,), ())
    connector.window_group_ids = (1, 2)
    request = FakeRequest("req-tail-zero")
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a"],
        num_token_ids=256,
    )
    connector.requests_meta[request.request_id] = req_meta

    connector.update_state_after_alloc(
        request,
        FakeKVCacheBlocks(
            (
                [FakeBlock(10)],
                [FakeBlock(100), FakeBlock(101), FakeBlock(102), FakeBlock(103)],
                [],
            )
        ),
        0,
    )

    assert materialize_rows(connector, req_meta, [0])[0] == ([10], [103], [])


class FakeStore:
    def __init__(self, hit_index: int, lookup_hits: list[bool] | None = None):
        self.hit_index = hit_index
        self.lookup_hits = lookup_hits
        self.lookup_keys = None
        self.lookup_batch_keys = None

    def lookup_on_prefix(self, keys):
        self.lookup_keys = list(keys)
        return self.hit_index

    def lookup(self, keys):
        self.lookup_batch_keys = list(keys)
        if self.lookup_hits is not None:
            return self.lookup_hits[: len(keys)]
        return [idx <= self.hit_index for idx, _ in enumerate(keys)]


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


def make_ascend_connector() -> UCMAscendFAWAConnector:
    c4_layers = [f"layer.{i}.c4" for i in range(21)]
    c128_layers = [f"layer.{i}.c128" for i in range(20)]
    swa_a_layers = [*c4_layers, "layer.extra.swa_a"]
    swa_b_layers = [*c128_layers, "layer.extra.swa_b", "mtp.extra.swa_b"]
    connector = UCMAscendFAWAConnector.__new__(UCMAscendFAWAConnector)
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
    connector.fa_group_ids = (0, 3, 8)
    connector.window_group_ids = (1, 2, 4, 5, 6, 7, 9, 10)
    connector.block_span_layout = FAWABlockSpanLayout(
        connector._kv_cache_config,
        connector.fa_group_ids,
    )
    connector._ascend_layout = connector.block_span_layout.is_ascend
    connector.hash_block_size = connector.block_span_layout.hash_block_size
    connector.group_token_block_sizes = (
        connector.block_span_layout.group_token_block_sizes
    )
    connector.group_tensor_block_sizes = (
        connector.block_span_layout.group_tensor_block_sizes
    )
    connector.group_tail_blocks = connector._get_group_tail_blocks()
    connector.group_window_spans = connector._get_group_window_spans()
    connector.requests_meta = {}
    return connector


def test_get_num_new_matched_tokens_uses_generated_hashes():
    connector = make_connector()
    connector.persist_token_threshold = 0
    connector.fa_store = FakeStore(2)
    connector.wa_store = FakeStore(2)
    connector.generate_hash = lambda *args, **kwargs: [b"g0", b"g1", b"g2"]
    connector._seed = 0

    request = FakeRequest(
        "req-3",
        all_token_ids=[1] * 768,
        num_tokens=768,
        block_hashes=[b"h0", b"h1", b"h2"],
    )
    hit_tokens, is_async = connector.get_num_new_matched_tokens(request, 0)

    assert hit_tokens == 767
    assert is_async is False
    assert connector.fa_store.lookup_keys == [b"g0", b"g1", b"g2"]
    assert connector.wa_store.lookup_batch_keys == [b"g0", b"g1", b"g2"]
    assert connector.requests_meta["req-3"].token_processed == 768
    assert connector.requests_meta["req-3"].store_block_cursor == 3


def test_window_hit_uses_latest_boundary_inside_fa_hit_range():
    connector = make_connector()
    connector.persist_token_threshold = 0
    connector.fa_store = FakeStore(2)
    connector.wa_store = FakeStore(-1, [False, False, True])
    connector.generate_hash = lambda *args, **kwargs: [b"g0", b"g1", b"g2"]
    connector._seed = 0

    request = FakeRequest(
        "req-wa-latest",
        all_token_ids=[1] * 768,
        num_tokens=769,
    )
    hit_tokens, is_async = connector.get_num_new_matched_tokens(request, 0)

    assert hit_tokens == 768
    assert is_async is False
    assert connector.fa_store.lookup_keys == [b"g0", b"g1", b"g2"]
    assert connector.wa_store.lookup_batch_keys == [
        b"g0",
        b"g1",
        b"g2",
    ]
    assert connector.requests_meta["req-wa-latest"].total_hit_block_num == 3


def test_external_hit_records_allocated_blocks_for_load_plan():
    connector = make_connector()
    request = FakeRequest("req-4")
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a", b"b"],
        hbm_hit_block_num=0,
        total_hit_block_num=2,
        num_token_ids=768,
        token_processed=512,
        store_block_cursor=2,
    )
    connector.requests_meta[request.request_id] = req_meta

    connector.update_state_after_alloc(
        request,
        FakeKVCacheBlocks(
            (
                [FakeBlock(10), FakeBlock(11)],
                [
                    FakeBlock(100),
                    FakeBlock(101),
                    FakeBlock(102),
                    FakeBlock(103),
                    FakeBlock(104),
                    FakeBlock(105),
                    FakeBlock(106),
                    FakeBlock(107),
                ],
            )
        ),
        512,
    )

    scheduler_output = FakeSchedulerOutput(
        scheduled_new_reqs=[type("ReqData", (), {"req_id": request.request_id})()],
        scheduled_cached_reqs=FakeCachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_block_ids=[],
        ),
        num_scheduled_tokens={request.request_id: 256},
        finished_req_ids=set(),
    )
    metadata = connector.build_connector_meta(scheduler_output)

    request_dispatch = metadata.request_meta[request.request_id]
    assert request_dispatch.load_block_plan is not None
    assert request_dispatch.load_block_plan.indices == (0, 1)
    assert load_block_ids(connector, request_dispatch) == (
        [b"a", b"b"],
        [
            ([10], [103]),
            ([11], [107]),
        ],
    )


def test_ascend_layout_uses_512_token_hash_blocks_and_mixed_spec_sizes():
    connector = make_ascend_connector()

    assert connector._ascend_layout is True
    assert connector.hash_block_size == 512
    assert connector.group_token_block_sizes == (
        512,
        128,
        128,
        512,
        32,
        32,
        128,
        128,
        512,
        64,
        64,
    )
    assert connector.group_tensor_block_sizes == (
        512,
        128,
        128,
        4096,
        32,
        32,
        128,
        128,
        16384,
        64,
        64,
    )
    assert connector.group_tail_blocks == (
        None,
        1,
        1,
        None,
        1,
        1,
        1,
        1,
        None,
        0,
        0,
    )
    assert connector.group_window_spans[4] == (4,)
    assert connector.group_window_spans[5] == (4,)
    assert connector.group_window_spans[6] == (4,)
    assert connector.group_window_spans[7] == (4,)
    assert connector.group_window_spans[9] == ()
    assert connector.group_window_spans[10] == ()


def test_ascend_split_uses_tensor_indices_from_registered_layer_tuple():
    connector = make_ascend_connector()
    c4_tensors = tuple(torch.empty((1, 1, 1), dtype=torch.float32) for _ in range(8))
    c128_tensors = tuple(torch.empty((1, 1, 1), dtype=torch.float32) for _ in range(4))
    swa_tensor = torch.empty((1, 1, 1), dtype=torch.float32)
    kv_caches = {
        "layer.0.c4": c4_tensors,
        "layer.0.c128": c128_tensors,
        "layer.extra.swa_a": (swa_tensor,),
        "layer.extra.swa_b": (swa_tensor,),
    }

    grouped = connector._split_kv_caches_by_vllm_groups(kv_caches)

    assert grouped[0]["layer.0.c4"] is c4_tensors[0]
    assert grouped[1]["layer.0.c4"] is c4_tensors[1]
    assert grouped[3]["layer.0.c4"] == (c4_tensors[2], c4_tensors[3])
    assert grouped[4]["layer.0.c4"] is c4_tensors[4]
    assert grouped[5]["layer.0.c4"] is c4_tensors[5]
    assert grouped[6]["layer.0.c4"] is c4_tensors[6]
    assert grouped[7]["layer.0.c4"] is c4_tensors[7]
    assert grouped[2]["layer.0.c128"] is c128_tensors[0]
    assert grouped[8]["layer.0.c128"] is c128_tensors[1]
    assert grouped[9]["layer.0.c128"] is c128_tensors[2]
    assert grouped[10]["layer.0.c128"] is c128_tensors[3]
    assert grouped[1]["layer.extra.swa_a"] is swa_tensor
    assert grouped[2]["layer.extra.swa_b"] is swa_tensor


def test_ascend_canonical_blocks_map_to_tensor_block_offsets():
    connector = make_ascend_connector()
    request = FakeRequest("req-ascend-map")
    req_meta = FAWARequestMeta(
        ucm_block_ids=[bytes([i]) for i in range(130)],
        num_token_ids=512 * 130,
    )
    connector.requests_meta[request.request_id] = req_meta

    connector.update_state_after_alloc(
        request,
        FakeKVCacheBlocks(
            (
                [FakeBlock(i) for i in range(130)],
                [FakeBlock(1000 + i) for i in range(520)],
                [FakeBlock(2000 + i) for i in range(520)],
                [FakeBlock(3000 + i) for i in range(17)],
                [FakeBlock(4000 + i) for i in range(2080)],
                [FakeBlock(5000 + i) for i in range(2080)],
                [FakeBlock(6000 + i) for i in range(520)],
                [FakeBlock(7000 + i) for i in range(520)],
                [FakeBlock(8000 + i) for i in range(5)],
                [FakeBlock(9000 + i) for i in range(1040)],
                [FakeBlock(10000 + i) for i in range(1040)],
            )
        ),
        0,
    )

    rows = {
        idx: row for idx, row in zip((0, 7, 8, 31, 32, 128), materialize_rows(
            connector, req_meta, [0, 7, 8, 31, 32, 128]
        ))
    }
    assert rows[0][0] == [KVCacheSegment(0, 0, 512)]
    assert rows[0][1] == [KVCacheSegment(1003, 0, 128)]
    assert rows[0][2] == [KVCacheSegment(2003, 0, 128)]
    assert rows[0][3] == [KVCacheSegment(3000, 0, 512)]
    assert rows[7][3] == [KVCacheSegment(3000, 3584, 512)]
    assert rows[8][3] == [KVCacheSegment(3001, 0, 512)]
    assert rows[0][4] == [KVCacheSegment(4015, 28, 4)]
    assert rows[0][6] == [KVCacheSegment(6003, 124, 4)]
    assert rows[0][8] == [KVCacheSegment(8000, 0, 512)]
    assert rows[31][8] == [KVCacheSegment(8000, 15872, 512)]
    assert rows[32][8] == [KVCacheSegment(8001, 0, 512)]
    assert rows[128][8] == [KVCacheSegment(8004, 0, 512)]
    assert rows[0][9] == []


def test_ascend_load_plan_indices_cover_external_range_and_wa_uses_final_boundary():
    connector = make_ascend_connector()
    req_meta = FAWARequestMeta(
        ucm_block_ids=[bytes([i]) for i in range(12)],
        hbm_hit_block_num=1,
        total_hit_block_num=10,
        num_token_ids=12 * connector.hash_block_size,
        token_processed=10 * connector.hash_block_size,
        store_block_cursor=12,
    )
    req_meta.allocated_group_block_ids = tuple(
        np.arange(1000 * group_id, 1000 * group_id + 512, dtype=np.int64)
        for group_id in range(len(connector.group_token_block_sizes))
    )
    connector._record_ready_group_block_ids(req_meta)
    connector.requests_meta["req-ascend-load-dump"] = req_meta

    metadata = connector._make_dispatch_meta(
        "req-ascend-load-dump",
        req_meta,
        512,
        True,
    )

    assert metadata.load_block_plan is not None
    assert metadata.load_block_plan.keys == [bytes([i]) for i in range(1, 10)]
    assert metadata.load_block_plan.indices == tuple(range(1, 10))
    rows = materialize_rows(
        connector,
        req_meta,
        metadata.load_block_plan.indices[-1:],
    )
    assert select_rows(rows, connector.window_group_ids) == [
        (
            [KVCacheSegment(1039, 0, 128)],
            [KVCacheSegment(2039, 0, 128)],
            [KVCacheSegment(4159, 28, 4)],
            [KVCacheSegment(5159, 28, 4)],
            [KVCacheSegment(6039, 124, 4)],
            [KVCacheSegment(7039, 124, 4)],
            [],
            [],
        )
    ]
    assert metadata.dump_block_plan is None


def test_ascend_window_tensor_sizes_use_trimmed_state_spans():
    connector = make_ascend_connector()
    group_layouts = {
        1: KVCacheGroupLayout({"layer.0.swa": torch.empty((1, 128, 1, 512))}),
        4: KVCacheGroupLayout({"layer.0.c4_kv": torch.empty((1, 32, 1, 1024))}),
        6: KVCacheGroupLayout({"layer.0.c4_indexer_kv": torch.empty((1, 128, 1, 256))}),
        9: KVCacheGroupLayout({"layer.0.c128_kv": torch.empty((1, 64, 1, 512))}),
    }

    assert connector._store_tensor_size_list(group_layouts, (1, 4, 6, 9)) == [
        int(128 * 1 * 512 * torch.empty((), dtype=torch.float32).element_size()),
        int(4 * 1 * 1024 * torch.empty((), dtype=torch.float32).element_size()),
        int(4 * 1 * 256 * torch.empty((), dtype=torch.float32).element_size()),
    ]


def test_ascend_chunk_prefill_waits_for_complete_segment_rows():
    connector = make_ascend_connector()
    request = FakeRequest("req-ascend-chunk")
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a"] * 8,
        num_token_ids=4096,
    )
    connector.requests_meta[request.request_id] = req_meta

    connector.update_state_after_alloc(
        request,
        FakeKVCacheBlocks(
            (
                [FakeBlock(i) for i in range(8)],
                [],
                [FakeBlock(2000 + i) for i in range(32)],
                [FakeBlock(3000)],
                [FakeBlock(4000 + i) for i in range(128)],
                [FakeBlock(5000 + i) for i in range(128)],
                [FakeBlock(6000 + i) for i in range(32)],
                [FakeBlock(7000 + i) for i in range(32)],
                [FakeBlock(8000)],
                [FakeBlock(9000 + i) for i in range(64)],
                [FakeBlock(10000 + i) for i in range(64)],
            )
        ),
        0,
    )

    assert req_meta.record_block_cursor == 0

    scheduler_output = FakeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=FakeCachedRequestData(
            req_ids=[request.request_id],
            resumed_req_ids=set(),
            new_block_ids=[
                (
                    [],
                    [1000 + i for i in range(32)],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
            ],
        ),
        num_scheduled_tokens={request.request_id: 4096},
        finished_req_ids=set(),
    )
    metadata = connector.build_connector_meta(scheduler_output)

    request_dispatch = metadata.request_meta[request.request_id]
    assert request_dispatch.dump_block_plan is not None
    assert len(request_dispatch.dump_block_plan.keys) == 8
    assert materialize_rows(connector, req_meta, [7])[0][3] == [
        KVCacheSegment(3000, 3584, 512)
    ]
    assert req_meta.store_block_cursor == 8


def test_layout_extracts_segment_addresses_and_sizes():
    tensor = torch.empty((2, 128, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    addrs = layout_segment_addrs(
        layout,
        [KVCacheSegment(1, 4096, 512)],
        group_tensor_block_size=16384,
    )

    assert addrs.shape == (1, 1)
    assert addrs[0, 0] == np.uint64(tensor[1, 32].data_ptr())
    assert layout.segment_tensor_size_list(512, 16384) == [
        int(tensor[1, 32:36].numel() * tensor.element_size())
    ]


def test_layout_handles_single_4d_ascend_tensor_shape():
    tensor = torch.empty((2, 128, 1, 512), dtype=torch.bfloat16)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    addrs = layout_segment_addrs(
        layout,
        [KVCacheSegment(1, 512, 512)],
        group_tensor_block_size=16384,
    )
    assert layout.tensor_size_list == [int(tensor[0].numel() * tensor.element_size())]
    assert addrs.shape == (1, 1)
    assert addrs[0, 0] == np.uint64(tensor[1, 4].data_ptr())
    assert layout.segment_tensor_size_list(512, 16384) == [
        int(tensor[1, 4:8].numel() * tensor.element_size())
    ]


def test_layout_handles_gpu_4d_kv_axis_before_tensor_block_size():
    tensor = torch.empty((2, 2, 64, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    addrs = layout_segment_addrs(
        layout,
        [KVCacheSegment(1, 0, 64)],
        group_tensor_block_size=64,
    )
    assert layout.tensor_block_size == 64
    assert layout.tensor_size_list == [
        int(tensor[0, 0].numel() * tensor.element_size()),
        int(tensor[0, 1].numel() * tensor.element_size()),
    ]
    assert addrs.shape == (1, 2)
    assert addrs[0, 0] == np.uint64(tensor[1, 0].data_ptr())
    assert addrs[0, 1] == np.uint64(tensor[1, 1].data_ptr())


def test_layout_handles_mixed_3d_tensor_block_sizes():
    tensor64 = torch.empty((2, 64, 584), dtype=torch.uint8)
    tensor2 = torch.empty((2, 2, 584), dtype=torch.uint8)
    layout = KVCacheGroupLayout(
        {
            "layer.0.attn": tensor64,
            "layer.1.attn": tensor2,
        }
    )

    addrs = layout_segment_addrs(
        layout,
        [KVCacheSegment(1, 128, 128)],
        group_tensor_block_size=256,
    )
    assert layout.tensor_size_list == [
        int(tensor64[0].numel() * tensor64.element_size()),
        int(tensor2[0].numel() * tensor2.element_size()),
    ]
    assert addrs.shape == (1, 2)
    assert addrs[0, 0] == np.uint64(tensor64[1, 32].data_ptr())
    assert addrs[0, 1] == np.uint64(tensor2[1, 1].data_ptr())
    assert layout.segment_tensor_size_list(128, 256) == [
        int(tensor64[1, 32:64].numel() * tensor64.element_size()),
        int(tensor2[1, 1:2].numel() * tensor2.element_size()),
    ]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
