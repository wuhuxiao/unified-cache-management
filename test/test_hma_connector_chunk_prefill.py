from dataclasses import dataclass

import numpy as np
import pytest
import torch

from ucm.integration.vllm.hma_connector import (
    FAWABlockSpanLayout,
    FAWARequestMeta,
    KVCacheGroupMeta,
    KVCacheGroupLayout,
    UCMAscendFAWAConnector,
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


def make_connector() -> UCMFAWAConnector:
    connector = UCMFAWAConnector.__new__(UCMFAWAConnector)
    connector.hash_block_size = 256
    connector.fa_group_ids = (0,)
    connector.window_group_ids = (1,)
    connector.block_span_layout = None
    connector.requests_meta = {}
    set_group_metas(
        connector,
        token_block_sizes=(256, 64),
        tensor_block_sizes=(256, 64),
        tail_blocks=(None, 1),
        window_spans=((256,), (64,)),
    )
    return connector


def set_group_metas(
    connector: UCMFAWAConnector,
    *,
    token_block_sizes: tuple[int, ...],
    tensor_block_sizes: tuple[int, ...],
    tail_blocks: tuple[int | None, ...],
    window_spans: tuple[tuple[int, ...], ...],
) -> None:
    connector.group_metas = {
        group_id: KVCacheGroupMeta(
            group_id=group_id,
            token_block_size=token_block_size,
            tensor_block_size=tensor_block_sizes[group_id],
            logical_blocks_per_hash_block=connector.hash_block_size
            // token_block_size,
            hash_blocks_per_tensor_block=max(
                1,
                tensor_block_sizes[group_id] // connector.hash_block_size,
            ),
            tail_blocks=tail_blocks[group_id],
            window_spans=window_spans[group_id],
        )
        for group_id, token_block_size in enumerate(token_block_sizes)
    }


def make_two_tail_connector() -> UCMFAWAConnector:
    connector = make_connector()
    set_group_metas(
        connector,
        token_block_sizes=(256, 64),
        tensor_block_sizes=(256, 64),
        tail_blocks=(None, 2),
        window_spans=((256,), (64, 64)),
    )
    return connector


def test_group_meta_uses_integer_ratios_and_zero_tail():
    connector = make_connector()
    connector.window_group_ids = (1, 2)
    set_group_metas(
        connector,
        token_block_sizes=(256, 64, 64),
        tensor_block_sizes=(256, 512, 64),
        tail_blocks=(None, 1, 0),
        window_spans=((256,), (64,), ()),
    )

    assert connector.group_metas[0].logical_blocks_per_hash_block == 1
    assert connector.group_metas[0].hash_blocks_per_tensor_block == 1
    assert connector.group_metas[1].logical_blocks_per_hash_block == 4
    assert connector.group_metas[1].hash_blocks_per_tensor_block == 2
    assert connector.group_metas[1].tail_blocks == 1
    assert connector.group_metas[2].tail_blocks == 0
    assert connector.group_metas[2].window_spans == ()


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
            token_block_size=128,
            tensor_block_size=512,
            logical_blocks_per_hash_block=2,
            hash_blocks_per_tensor_block=2,
            tail_blocks=2,
            window_spans=(128, 128),
        ),
    }

    assert connector.group_token_block_sizes == (256, 128)
    assert connector.group_tensor_block_sizes == (256, 512)
    assert connector.group_tensor_block_ratios == (1, 4)
    assert connector.group_tail_blocks == (None, 2)
    assert connector.group_window_spans == ((256,), (128, 128))


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


def test_zero_tail_window_group_uses_empty_candidate_list():
    connector = make_connector()
    connector.window_group_ids = (1, 2)
    set_group_metas(
        connector,
        token_block_sizes=(256, 64, 64),
        tensor_block_sizes=(256, 64, 64),
        tail_blocks=(None, 1, 0),
        window_spans=((256,), (64,), ()),
    )

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


def test_cached_resumed_from_preemption_list_resets_only_matching_request():
    connector = make_connector()
    first_meta = FAWARequestMeta(
        ucm_block_ids=[b"a", b"b"],
        num_token_ids=512,
        vllm_block_ids=([10], [100, 101, 102, 103]),
        token_processed=256,
    )
    second_meta = FAWARequestMeta(
        ucm_block_ids=[b"c", b"d"],
        num_token_ids=512,
        vllm_block_ids=([20], [200, 201, 202, 203]),
        token_processed=0,
    )
    connector.requests_meta["req-append"] = first_meta
    connector.requests_meta["req-reset"] = second_meta

    cached_reqs = type(
        "CachedReqs",
        (),
        {
            "req_ids": ["req-append", "req-reset"],
            "resumed_req_ids": set(),
            "resumed_from_preemption": [False, True],
            "new_block_ids": [
                ([11], [104, 105, 106, 107]),
                ([21], [204, 205, 206, 207]),
            ],
        },
    )()
    metadata = connector.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_reqs,
            num_scheduled_tokens={"req-append": 256, "req-reset": 256},
            finished_req_ids=set(),
        )
    )

    assert metadata.request_meta["req-append"].dump_vllm_block_ids == ([11], [107])
    assert first_meta.vllm_block_ids == (
        [10, 11],
        [100, 101, 102, 103, 104, 105, 106, 107],
    )
    assert metadata.request_meta["req-reset"].dump_vllm_block_ids == ([21], [207])
    assert second_meta.vllm_block_ids == ([21], [204, 205, 206, 207])


def test_slice_group_block_ids_uses_tensor_block_ratio_for_large_blocks():
    connector = make_connector()
    connector.hash_block_size = 256
    connector.fa_group_ids = (0,)
    connector.window_group_ids = (1,)
    set_group_metas(
        connector,
        token_block_sizes=(64, 64),
        tensor_block_sizes=(512, 512),
        tail_blocks=(None, 1),
        window_spans=((256,), (64,)),
    )

    assert connector._slice_group_block_ids(
        0,
        [100, 101],
        0,
        1,
        window_tail_only=False,
    ) == [100]
    assert connector._slice_group_block_ids(
        0,
        [100, 101],
        0,
        3,
        window_tail_only=False,
    ) == [100, 101]
    assert connector._slice_group_block_ids(
        1,
        [100, 101],
        0,
        1,
        window_tail_only=True,
    ) == [100]
    assert connector._slice_group_block_ids(
        1,
        [100, 101],
        2,
        3,
        window_tail_only=True,
    ) == [101]


def test_extract_fa_ptr_batches_hash_offsets_for_shared_tensor_block():
    connector = make_connector()
    connector.hash_block_size = 512
    connector.fa_group_ids = (0,)
    connector.window_group_ids = ()
    set_group_metas(
        connector,
        token_block_sizes=(512,),
        tensor_block_sizes=(4096,),
        tail_blocks=(None,),
        window_spans=((512,),),
    )
    tensor = torch.empty((5, 4096, 1), dtype=torch.float32)
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


def test_extract_wa_ptr_uses_tail_candidates_and_zero_tail_group():
    connector = make_connector()
    connector.fa_group_ids = (0,)
    connector.window_group_ids = (1, 2)
    set_group_metas(
        connector,
        token_block_sizes=(256, 64, 64),
        tensor_block_sizes=(256, 64, 64),
        tail_blocks=(None, 1, 0),
        window_spans=((256,), (64,), ()),
    )
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


def test_extract_wa_ptr_offsets_trimmed_window_span_to_tail_start():
    connector = make_connector()
    connector.hash_block_size = 32
    connector.fa_group_ids = (0,)
    connector.window_group_ids = (1,)
    set_group_metas(
        connector,
        token_block_sizes=(32, 32),
        tensor_block_sizes=(32, 32),
        tail_blocks=(None, 1),
        window_spans=((32,), (4,)),
    )
    tensor = torch.empty((8, 32, 1), dtype=torch.float32)
    connector.group_layouts = {1: KVCacheGroupLayout({"layer.0.wa": tensor})}

    ptrs = connector._extract_wa_ptr(
        [b"a"],
        0,
        1,
        ([], [5]),
    )

    assert ptrs.shape == (1, 1)
    assert ptrs[0, 0] == np.uint64(tensor[5, 28].data_ptr())


def test_extract_wa_ptr_rejects_mismatched_key_range_length():
    connector = make_connector()
    connector.group_layouts = {
        1: KVCacheGroupLayout(
            {"layer.0.wa": torch.empty((16, 64, 1), dtype=torch.float32)}
        )
    }

    with pytest.raises(ValueError, match="store key count"):
        connector._extract_wa_ptr(
            [b"a", b"b"],
            1,
            2,
            ([], [7]),
        )


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
    set_group_metas(
        connector,
        token_block_sizes=connector.block_span_layout.group_token_block_sizes,
        tensor_block_sizes=connector.block_span_layout.group_tensor_block_sizes,
        tail_blocks=(
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
        ),
        window_spans=(
            (512,),
            (128,),
            (128,),
            (512,),
            (4,),
            (4,),
            (4,),
            (4,),
            (512,),
            (),
            (),
        ),
    )
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
    )
    connector.requests_meta[request.request_id] = req_meta

    scheduler_output = FakeSchedulerOutput(
        scheduled_new_reqs=[
            type(
                "ReqData",
                (),
                {
                    "req_id": request.request_id,
                    "block_ids": (
                        [10, 11],
                        [100, 101, 102, 103, 104, 105, 106, 107],
                    ),
                },
            )()
        ],
        scheduled_cached_reqs=FakeCachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_block_ids=[],
        ),
        num_scheduled_tokens={request.request_id: 256},
        finished_req_ids=set(),
    )
    metadata = connector.build_connector_meta(scheduler_output)
    dispatch = metadata.request_meta[request.request_id]

    assert dispatch.load_keys == [b"a", b"b"]
    assert dispatch.load_hash_start == 0
    assert dispatch.load_hash_end == 2
    assert dispatch.load_vllm_block_ids == ([10, 11], [107])


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


def test_ascend_candidate_slicing_maps_to_tensor_block_offsets():
    connector = make_ascend_connector()
    group_blocks = tuple(
        [group_id * 1000 + idx for idx in range(3000)]
        for group_id in range(len(connector.group_token_block_sizes))
    )

    assert connector._slice_group_block_ids(
        0, group_blocks[0], 0, 1, window_tail_only=False
    ) == [0]
    assert connector._slice_group_block_ids(
        3, group_blocks[3], 7, 8, window_tail_only=False
    ) == [3000]
    assert connector._slice_group_block_ids(
        3, group_blocks[3], 8, 9, window_tail_only=False
    ) == [3001]
    assert connector._slice_group_block_ids(
        8, group_blocks[8], 31, 32, window_tail_only=False
    ) == [8000]
    assert connector._slice_group_block_ids(
        8, group_blocks[8], 32, 33, window_tail_only=False
    ) == [8001]
    assert connector._slice_group_block_ids(
        4, group_blocks[4], 0, 1, window_tail_only=True
    ) == [4015]
    assert connector._slice_group_block_ids(
        6, group_blocks[6], 0, 1, window_tail_only=True
    ) == [6003]
    assert connector._slice_group_block_ids(
        9, group_blocks[9], 0, 1, window_tail_only=True
    ) == []


def test_ascend_wa_store_loads_only_final_external_boundary_and_dumps_each_boundary():
    connector = make_ascend_connector()

    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a", b"b", b"c"],
        hbm_hit_block_num=0,
        total_hit_block_num=2,
        num_token_ids=1536,
        token_processed=1024,
        vllm_block_ids=(
            [0, 1, 2],
            [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14],
            [10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24],
            [30],
            [0] * 31 + [27] + [0] * 15 + [31] + [0] * 15 + [35],
            [0] * 31 + [37] + [0] * 15 + [41] + [0] * 15 + [45],
            [0, 1, 2, 3, 4, 5, 6, 47, 52, 53, 54, 55],
            [0, 1, 2, 3, 4, 5, 6, 57, 62, 63, 64, 65],
            [80],
            [],
            [],
        ),
    )
    connector.requests_meta["req-ascend-load-dump"] = req_meta

    metadata = connector._generate_dispatch_meta(
        req_meta,
        512,
        tuple([] for _ in connector.group_metas),
        True,
    )

    assert metadata.load_keys == [b"a", b"b"]
    assert metadata.load_hash_start == 0
    assert metadata.load_hash_end == 2
    assert metadata.load_vllm_block_ids[0] == [0, 1]
    assert metadata.load_vllm_block_ids[1] == [7]
    assert metadata.load_vllm_block_ids[2] == [17]
    assert metadata.load_vllm_block_ids[4] == [27]
    assert metadata.load_vllm_block_ids[5] == [37]
    assert metadata.load_vllm_block_ids[6] == [47]
    assert metadata.load_vllm_block_ids[7] == [57]
    assert metadata.load_vllm_block_ids[9] == []
    assert metadata.load_vllm_block_ids[10] == []
    assert metadata.dump_keys == [b"c"]
    assert metadata.dump_hash_start == 2
    assert metadata.dump_hash_end == 3
    assert metadata.dump_vllm_block_ids[0] == [2]
    assert metadata.dump_vllm_block_ids[1] == [14]
    assert metadata.dump_vllm_block_ids[4] == [31]


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


def test_ascend_chunk_prefill_emits_incremental_range_metadata():
    connector = make_ascend_connector()
    request = FakeRequest("req-ascend-chunk")
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a"] * 8,
        num_token_ids=4096,
    )
    connector.requests_meta[request.request_id] = req_meta

    scheduler_output = FakeSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=FakeCachedRequestData(
            req_ids=[request.request_id],
            resumed_req_ids=set(),
            new_block_ids=[
                (
                    [0],
                    [1000 + i for i in range(32)],
                    [2000 + i for i in range(32)],
                    [3000],
                    [4000 + i for i in range(128)],
                    [5000 + i for i in range(128)],
                    [6000 + i for i in range(32)],
                    [7000 + i for i in range(32)],
                    [8000],
                    [],
                    [],
                )
            ],
        ),
        num_scheduled_tokens={request.request_id: 512},
        finished_req_ids=set(),
    )
    metadata = connector.build_connector_meta(scheduler_output)
    dispatch = metadata.request_meta[request.request_id]

    assert dispatch.dump_keys == [b"a"]
    assert dispatch.dump_hash_start == 0
    assert dispatch.dump_hash_end == 1
    assert dispatch.dump_vllm_block_ids[0] == [0]
    assert dispatch.dump_vllm_block_ids[1] == [1003]
    assert dispatch.dump_vllm_block_ids[3] == [3000]
    assert req_meta.token_processed == 512


def test_layout_extracts_batch_addresses_and_sizes():
    tensor = torch.empty((2, 128, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    addrs = layout.extract_segment_addrs_batch(
        np.asarray([1], dtype=np.int64),
        np.asarray([4096], dtype=np.int64),
        group_tensor_block_size=16384,
    )

    assert addrs.shape == (1, 1)
    assert addrs[0, 0] == np.uint64(tensor[1, 32].data_ptr())
    assert layout.segment_tensor_size_list(512, 16384) == [
        int(tensor[1, 32:36].numel() * tensor.element_size())
    ]


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


def test_layout_rejects_batch_address_mismatched_lengths():
    tensor = torch.empty((4, 128, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    with pytest.raises(ValueError, match="same length"):
        layout.extract_segment_addrs_batch(
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            group_tensor_block_size=16384,
        )


def test_layout_rejects_batch_address_negative_offsets():
    tensor = torch.empty((4, 128, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    with pytest.raises(ValueError, match="Negative KV cache logical offset"):
        layout.extract_segment_addrs_batch(
            np.asarray([1], dtype=np.int64),
            np.asarray([-1], dtype=np.int64),
            group_tensor_block_size=16384,
        )


def test_layout_rejects_batch_address_negative_block_ids():
    tensor = torch.empty((4, 128, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    with pytest.raises(ValueError, match="Negative KV cache block id"):
        layout.extract_segment_addrs_batch(
            np.asarray([-1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            group_tensor_block_size=16384,
        )


def test_layout_rejects_batch_address_non_1d_inputs():
    tensor = torch.empty((4, 128, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    with pytest.raises(ValueError, match="block ids.*1-D"):
        layout.extract_segment_addrs_batch(
            np.asarray([[1]], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            group_tensor_block_size=16384,
        )

    with pytest.raises(ValueError, match="logical offsets.*1-D"):
        layout.extract_segment_addrs_batch(
            np.asarray([1], dtype=np.int64),
            np.asarray([[0]], dtype=np.int64),
            group_tensor_block_size=16384,
        )


def test_layout_handles_single_4d_ascend_tensor_shape():
    tensor = torch.empty((2, 128, 1, 512), dtype=torch.bfloat16)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    addrs = layout.extract_segment_addrs_batch(
        np.asarray([1], dtype=np.int64),
        np.asarray([512], dtype=np.int64),
        group_tensor_block_size=16384,
    )
    block_views = layout.extract_block_tensor_views([1])

    assert layout.tensor_size_list == [int(tensor[0].numel() * tensor.element_size())]
    assert addrs.shape == (1, 1)
    assert addrs[0, 0] == np.uint64(tensor[1, 4].data_ptr())
    assert layout.segment_tensor_size_list(512, 16384) == [
        int(tensor[1, 4:8].numel() * tensor.element_size())
    ]
    assert len(block_views) == 1
    assert block_views[0].shape == tensor[1].shape
    assert block_views[0].data_ptr() == tensor[1].data_ptr()


def test_layout_handles_gpu_4d_kv_axis_before_tensor_block_size():
    tensor = torch.empty((2, 2, 64, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    addrs = layout.extract_segment_addrs_batch(
        np.asarray([1], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        group_tensor_block_size=64,
    )
    block_views = layout.extract_block_tensor_views([1])

    assert layout.tensor_block_size == 64
    assert layout.tensor_size_list == [
        int(tensor[0, 0].numel() * tensor.element_size()),
        int(tensor[0, 1].numel() * tensor.element_size()),
    ]
    assert addrs.shape == (1, 2)
    assert addrs[0, 0] == np.uint64(tensor[1, 0].data_ptr())
    assert addrs[0, 1] == np.uint64(tensor[1, 1].data_ptr())
    assert len(block_views) == 2
    assert block_views[0].data_ptr() == tensor[1, 0].data_ptr()
    assert block_views[1].data_ptr() == tensor[1, 1].data_ptr()


def test_layout_handles_mixed_3d_tensor_block_sizes():
    tensor64 = torch.empty((2, 64, 584), dtype=torch.uint8)
    tensor2 = torch.empty((2, 2, 584), dtype=torch.uint8)
    layout = KVCacheGroupLayout(
        {
            "layer.0.attn": tensor64,
            "layer.1.attn": tensor2,
        }
    )

    addrs = layout.extract_segment_addrs_batch(
        np.asarray([1], dtype=np.int64),
        np.asarray([128], dtype=np.int64),
        group_tensor_block_size=256,
    )
    block_views = layout.extract_block_tensor_views([1])

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
    assert len(block_views) == 2
    assert block_views[0].data_ptr() == tensor64[1].data_ptr()
    assert block_views[1].data_ptr() == tensor2[1].data_ptr()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
