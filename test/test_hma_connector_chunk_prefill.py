from dataclasses import dataclass

import numpy as np
import torch

from ucm.integration.vllm.hma_connector import (
    FAWARequestMeta,
    KVCacheGroupLayout,
    KVCacheGroupMeta,
    UCMFAWAConnector,
)


@dataclass
class FakeRequest:
    request_id: str
    all_token_ids: list[int] | None = None
    num_tokens: int = 0


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
    connector.requests_meta = {}
    set_group_metas(
        connector,
        token_block_sizes=(256, 64),
        tail_blocks=(1, 1),
        tail_tokens=(256, 64),
    )
    return connector


def set_group_metas(
    connector: UCMFAWAConnector,
    *,
    token_block_sizes: tuple[int, ...],
    tail_blocks: tuple[int, ...],
    tail_tokens: tuple[int, ...],
) -> None:
    connector.group_metas = {
        group_id: KVCacheGroupMeta(
            group_id=group_id,
            token_block_size=token_block_size,
            tail_blocks=tail_blocks[group_id],
            tail_tokens=tail_tokens[group_id],
        )
        for group_id, token_block_size in enumerate(token_block_sizes)
    }


def req_data(request_id: str, block_ids: tuple[list[int], ...]):
    return type("ReqData", (), {"req_id": request_id, "block_ids": block_ids})()


def test_dispatch_meta_accumulates_cached_blocks_and_slices_wa_tails():
    connector = make_connector()
    req_meta = FAWARequestMeta(
        ucm_block_ids=[b"a", b"b"],
        num_token_ids=512,
        token_processed=0,
    )
    connector.requests_meta["req-0"] = req_meta

    first_meta = connector.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=FakeCachedRequestData(
                req_ids=["req-0"],
                resumed_req_ids=set(),
                new_block_ids=[([10], [100, 101, 102, 103])],
            ),
            num_scheduled_tokens={"req-0": 256},
            finished_req_ids=set(),
        )
    )
    first_req = first_meta.request_meta["req-0"]

    assert first_req.dump_keys == [b"a"]
    assert first_req.dump_hash_start == 0
    assert first_req.dump_hash_end == 1
    assert first_req.dump_vllm_block_ids == ([10], [])

    second_meta = connector.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=FakeCachedRequestData(
                req_ids=["req-0"],
                resumed_req_ids=set(),
                new_block_ids=[([11], [104, 105, 106, 107])],
            ),
            num_scheduled_tokens={"req-0": 256},
            finished_req_ids=set(),
        )
    )
    second_req = second_meta.request_meta["req-0"]

    assert second_req.dump_keys == [b"b"]
    assert second_req.dump_hash_start == 1
    assert second_req.dump_hash_end == 2
    assert second_req.dump_vllm_block_ids == ([10], [103])
    assert req_meta.vllm_block_ids == (
        [10, 11],
        [100, 101, 102, 103, 104, 105, 106, 107],
    )
    assert req_meta.token_processed == 512


def test_load_metadata_slices_wa_to_final_boundary():
    connector = make_connector()
    connector.requests_meta["req-load"] = FAWARequestMeta(
        ucm_block_ids=[b"a", b"b", b"c"],
        hbm_hit_block_num=0,
        total_hit_block_num=2,
        num_token_ids=768,
        token_processed=512,
    )

    metadata = connector.build_connector_meta(
        FakeSchedulerOutput(
            scheduled_new_reqs=[
                req_data(
                    "req-load",
                    (
                        [10, 11, 12],
                        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                    ),
                )
            ],
            scheduled_cached_reqs=FakeCachedRequestData([], set(), []),
            num_scheduled_tokens={"req-load": 256},
            finished_req_ids=set(),
        )
    )

    dispatch = metadata.request_meta["req-load"]
    assert dispatch.load_keys == [b"a", b"b"]
    assert dispatch.load_hash_start == 0
    assert dispatch.load_hash_end == 2
    assert dispatch.load_vllm_block_ids == ([12, 10], [103])


def test_zero_tail_window_group_uses_empty_candidate_list():
    connector = make_connector()
    connector.window_group_ids = (1, 2)
    set_group_metas(
        connector,
        token_block_sizes=(256, 64, 64),
        tail_blocks=(1, 1, 1),
        tail_tokens=(256, 64, 0),
    )
    connector.requests_meta["req-zero"] = FAWARequestMeta(
        ucm_block_ids=[b"a"],
        num_token_ids=256,
        token_processed=0,
    )

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

    assert metadata.request_meta["req-zero"].dump_vllm_block_ids == ([10], [], [])


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

    assert metadata.request_meta["req-append"].dump_vllm_block_ids == ([10], [103])
    assert first_meta.vllm_block_ids == (
        [10, 11],
        [100, 101, 102, 103, 104, 105, 106, 107],
    )
    assert metadata.request_meta["req-reset"].dump_vllm_block_ids == ([21], [])
    assert second_meta.vllm_block_ids == ([21], [204, 205, 206, 207])


def test_extract_fa_ptr_uses_current_token_block_offsets():
    connector = make_connector()
    connector.hash_block_size = 256
    connector.fa_group_ids = (0,)
    connector.window_group_ids = ()
    set_group_metas(
        connector,
        token_block_sizes=(512,),
        tail_blocks=(1,),
        tail_tokens=(256,),
    )
    tensor = torch.empty((5, 128, 1), dtype=torch.float32)
    connector.group_layouts = {0: KVCacheGroupLayout({"layer.0": tensor})}

    ptrs = connector._extract_fa_ptr(
        [b"a", b"b"],
        1,
        3,
        ([3, 4],),
    )

    assert ptrs.shape == (2, 1)
    assert ptrs[0, 0] == np.uint64(tensor[3, 64].data_ptr())
    assert ptrs[1, 0] == np.uint64(tensor[4, 0].data_ptr())


def test_extract_wa_ptr_offsets_trimmed_tail_to_tail_start():
    connector = make_connector()
    connector.hash_block_size = 32
    connector.fa_group_ids = ()
    connector.window_group_ids = (0,)
    set_group_metas(
        connector,
        token_block_sizes=(32,),
        tail_blocks=(1,),
        tail_tokens=(4,),
    )
    tensor = torch.empty((8, 32, 1), dtype=torch.float32)
    connector.group_layouts = {0: KVCacheGroupLayout({"layer.0.wa": tensor})}

    ptrs = connector._extract_wa_ptr(
        [b"a"],
        ([5],),
    )

    assert ptrs.shape == (1, 1)
    assert ptrs[0, 0] == np.uint64(tensor[5, 28].data_ptr())


def test_extract_wa_ptr_uses_block_ids_without_hash_range():
    connector = make_connector()
    connector.group_layouts = {
        1: KVCacheGroupLayout(
            {"layer.0.wa": torch.empty((16, 64, 1), dtype=torch.float32)}
        )
    }

    ptrs = connector._extract_wa_ptr([b"a"], ([], [7]))

    assert ptrs.shape == (1, 1)


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
    assert connector.wa_store.lookup_batch_keys == [b"g0", b"g1", b"g2"]
    assert connector.requests_meta["req-wa-latest"].total_hit_block_num == 3


def test_layout_extracts_batch_addresses_and_sizes():
    tensor = torch.empty((2, 128, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    addrs = layout.extract_addrs_with_offsets(
        np.asarray([1], dtype=np.uint64),
        16384,
        np.asarray([4096], dtype=np.uint64),
    )

    assert addrs.shape == (1, 1)
    assert addrs[0, 0] == np.uint64(tensor[1, 32].data_ptr())
    assert layout.segment_tensor_size_list(512, 16384) == [
        int(tensor[1, 32:36].numel() * tensor.element_size())
    ]


def test_layout_handles_gpu_4d_kv_axis_before_tensor_block_size():
    tensor = torch.empty((2, 2, 64, 3), dtype=torch.float32)
    layout = KVCacheGroupLayout({"layer.0": tensor})

    addrs = layout.extract_addrs(
        np.asarray([1], dtype=np.uint64),
    )

    assert layout.tensor_block_size == 64
    assert layout.tensor_block_sizes.tolist() == [64, 64]
    assert layout.segment_tensor_size_list(64, 64) == [
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

    addrs = layout.extract_addrs_with_offsets(
        np.asarray([1], dtype=np.uint64),
        256,
        np.asarray([128], dtype=np.uint64),
    )

    assert addrs.shape == (1, 2)
    assert addrs[0, 0] == np.uint64(tensor64[1, 32].data_ptr())
    assert addrs[0, 1] == np.uint64(tensor2[1, 1].data_ptr())
    assert layout.segment_tensor_size_list(128, 256) == [
        int(tensor64[1, 32:64].numel() * tensor64.element_size()),
        int(tensor2[1, 1:2].numel() * tensor2.element_size()),
    ]
