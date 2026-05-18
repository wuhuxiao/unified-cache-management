from __future__ import annotations

import os
import time
import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.model_executor.layers.deepseek_compressor import CompressorStateCache
from vllm.model_executor.layers.deepseek_v4_attention import (
    DeepseekV4IndexerCache,
    DeepseekV4MLAAttention,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekV4SWACache,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock, get_kv_cache_configs
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

from ucm.integration.vllm.hma_connector import (
    FAWARequestDispatchMeta,
    KVCacheGroupLayout,
    UCMFAWAConnector,
    UCMFAWAConnectorMetadata,
)
from ucm.integration.vllm.ucm_connector import UCMConnector
from ucm.store.pipeline.connector import UcmPipelineStore


MODEL_PATH_ENV = "DEEPSEEK_V4_FLASH_MODEL_PATH"
DEFAULT_MODEL_PATH = "/home/models/DeepSeek-V4-Flash"
HASH_BLOCK_VALUES = {
    "req-gpu-a": [1, 2, 3, 4],
    "req-gpu-b": [1, 2, 5, 6],
}
PRODUCER_BLOCK_VALUES = [1, 2]
GPU_KV_BLOCKS_OVERRIDE = 640
MAX_TEST_KV_CACHE_BYTES = 32 * 1024**3


def hbm_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("GPU TP4 FAWA E2E requires CUDA.")
    return torch.device("cuda:0")


def model_path() -> str:
    path = os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
    if not Path(path).is_dir():
        pytest.skip(f"DeepSeek V4 Flash model directory is missing: {path}")
    return path


def make_token_ids(block_size: int, block_values: list[int]) -> list[int]:
    return [
        token for block_value in block_values for token in [block_value] * block_size
    ]


def generated_hashes(
    block_size: int,
    token_ids: Iterable[int],
    seed: bytes,
) -> list[bytes]:
    del seed
    tokens = list(token_ids)
    return [
        hashlib.md5(f"h{tokens[start]}".encode()).digest()
        for start in range(0, len(tokens), block_size)
        if len(tokens[start : start + block_size]) == block_size
    ]


def make_request(
    request_id: str,
    block_size: int,
    block_values: list[int],
) -> Request:
    request = Request(
        request_id=request_id,
        prompt_token_ids=make_token_ids(block_size, block_values),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    # vLLM pooling requests expose max_tokens=1 and num_tokens as the prompt
    # length, which is enough for connector lookup and persistence decisions.
    return request


@dataclass
class LocalWorldGroup:
    local_rank: int = 0
    rank: int = 0


@dataclass
class CudaCapability:
    major: int = 9
    minor: int = 0


class CapturingStore:
    def __init__(
        self,
        store: UcmPipelineStore,
        registry: "TensorRegistry",
        group_layouts: dict[int, KVCacheGroupLayout],
        group_ids: tuple[int, ...],
    ) -> None:
        self.store = store
        self.registry = registry
        self.group_layouts = group_layouts
        self.group_ids = group_ids
        self.dump_history: list[list[bytes]] = []
        self.load_history: list[list[bytes]] = []
        self.dump_rows: dict[bytes, bytes] = {}
        self.load_ptrs: list[np.ndarray] = []
        self.dump_ptrs: list[np.ndarray] = []

    def __getattr__(self, name: str):
        return getattr(self.store, name)

    def lookup_on_prefix(self, block_ids: list[bytes]) -> int:
        return self.store.lookup_on_prefix(block_ids)

    def load_data(self, block_ids, shard_indexs, dst_addr):
        self.load_history.append(list(block_ids))
        self.load_ptrs.append(np.asarray(dst_addr, dtype=np.uint64).copy())
        return self.store.load_data(block_ids, shard_indexs, dst_addr)

    def dump_data(self, block_ids, shard_indexs, src_addr, prerequisite_handle=0):
        ptrs = np.asarray(src_addr, dtype=np.uint64)
        self.dump_history.append(list(block_ids))
        self.dump_ptrs.append(ptrs.copy())
        for key, ptr_row in zip(block_ids, ptrs):
            try:
                self.dump_rows[key] = self.registry.row_bytes(ptr_row)
            except KeyError:
                pass
        return self.store.dump_data(
            block_ids,
            shard_indexs,
            src_addr,
            prerequisite_handle,
        )

    def wait(self, task) -> None:
        return self.store.wait(task)

class TensorRegistry:
    def __init__(self) -> None:
        self.by_ptr: dict[int, torch.Tensor] = {}

    def register_layouts(self, layouts: dict[int, KVCacheGroupLayout]) -> None:
        for layout in layouts.values():
            for tensor_or_tuple in layout.kvcaches.values():
                tensors = (
                    tensor_or_tuple
                    if isinstance(tensor_or_tuple, tuple)
                    else (tensor_or_tuple,)
                )
                for tensor in tensors:
                    self.by_ptr[int(tensor.data_ptr())] = tensor

    def _view_for_ptr(self, ptr: int, size: int) -> torch.Tensor:
        for base_ptr, tensor in self.by_ptr.items():
            storage = tensor.untyped_storage()
            storage_ptr = int(storage.data_ptr())
            if storage_ptr <= ptr and ptr + size <= storage_ptr + storage.nbytes():
                view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                return view.set_(storage, ptr - storage_ptr, (size,), (1,))
            tensor_bytes = tensor.numel() * tensor.element_size()
            if tensor.is_contiguous() and base_ptr <= ptr and ptr + size <= (
                base_ptr + tensor_bytes
            ):
                offset = ptr - base_ptr
                return tensor.view(torch.uint8).narrow(0, offset, size)
        raise KeyError(ptr)

    def row_bytes(self, ptr_row: Iterable[int]) -> bytes:
        return b"".join(
            self.by_ptr[int(ptr)].detach().cpu().reshape(-1).numpy().tobytes()
            for ptr in ptr_row
        )

    def ptrs_bytes(self, ptr_row: Iterable[int], size_list: list[int]) -> bytes:
        return b"".join(
            self._view_for_ptr(int(ptr), int(size)).detach().cpu().numpy().tobytes()
            for ptr, size in zip(ptr_row, size_list)
        )

    def fill_ptrs(self, ptrs: np.ndarray, size_list: list[int], value: int) -> None:
        for ptr_row in np.asarray(ptrs, dtype=np.uint64):
            for ptr, size in zip(ptr_row, size_list):
                self._view_for_ptr(int(ptr), int(size)).fill_(value)


def make_vllm_config(
    *,
    model: str,
    storage_backend: str,
    engine_id: str,
    tp_rank: int = 0,
) -> VllmConfig:
    model_config = ModelConfig(
        model=model,
        tokenizer=None,
        tokenizer_mode="deepseek_v4",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        max_model_len=1024,
        disable_sliding_window=False,
        skip_tokenizer_init=True,
        config_format="auto",
        limit_mm_per_prompt={},
    )
    cache_config = CacheConfig(
        block_size=256,
        gpu_memory_utilization=0.9,
        cache_dtype="fp8",
        num_gpu_blocks_override=GPU_KV_BLOCKS_OVERRIDE,
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=4,
        rank=tp_rank,
    )
    scheduler_config = SchedulerConfig(
        max_model_len=1024,
        is_encoder_decoder=False,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        disable_hybrid_kv_cache_manager=False,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="UCMConnector",
        kv_role="kv_both",
        kv_connector_module_path="ucm.integration.vllm.ucm_connector",
        engine_id=engine_id,
        kv_connector_extra_config={
            "enable_event_sync": False,
            "persist_token_threshold": 0,
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmPipelineStore",
                    "ucm_connector_config": {
                        "store_pipeline": "Cache|Posix",
                        "storage_backends": storage_backend,
                        "io_direct": False,
                        "waiting_queue_depth": 16,
                        "running_queue_depth": 64,
                        "share_buffer_enable": False,
                        "cache_buffer_capacity_gb": 1,
                    },
                }
            ],
        },
    )
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(device="cuda"),
        kv_transfer_config=kv_transfer_config,
    )


def register_deepseek_v4_flash_cache_layers(vllm_config: VllmConfig) -> None:
    config = vllm_config.model_config.hf_config
    with set_current_vllm_config(vllm_config):
        with patch(
            "vllm.platforms.current_platform.get_device_capability",
            return_value=CudaCapability(),
        ):
            for layer_id in range(config.num_hidden_layers):
                # Mirror DeepseekV4Attention cache module registration for every
                # decoder layer while avoiding heavyweight linear/MoE weights.
                compress_ratio = max(1, int(config.compress_ratios[layer_id]))
                prefix = f"model.layers.{layer_id}.attn"
                swa_cache = DeepseekV4SWACache(
                    head_dim=config.head_dim,
                    window_size=config.sliding_window,
                    dtype=torch.uint8,
                    prefix=f"{prefix}.swa_cache",
                    cache_config=vllm_config.cache_config,
                )
                if compress_ratio <= 1:
                    continue

                DeepseekV4MLAAttention(
                    num_heads=config.num_attention_heads
                    // vllm_config.parallel_config.tensor_parallel_size,
                    head_dim=config.head_dim,
                    scale=config.head_dim**-0.5,
                    qk_nope_head_dim=config.head_dim - config.qk_rope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.head_dim,
                    compress_ratio=compress_ratio,
                    window_size=config.sliding_window,
                    head_bytes=584,
                    swa_cache_layer=swa_cache,
                    attn_sink=torch.empty(64, dtype=torch.float32, device="cuda"),
                    cache_config=vllm_config.cache_config,
                    prefix=prefix,
                )
                state_width_multiplier = 1 + (compress_ratio == 4)
                CompressorStateCache(
                    state_dim=2 * state_width_multiplier * config.head_dim,
                    dtype=torch.float32,
                    compress_ratio=compress_ratio,
                    prefix=f"{prefix}.compressor.state_cache",
                )

                if compress_ratio != 4:
                    continue
                indexer_prefix = f"{prefix}.indexer"
                DeepseekV4IndexerCache(
                    head_dim=config.index_head_dim + config.index_head_dim // 128 * 4,
                    dtype=torch.uint8,
                    prefix=f"{indexer_prefix}.k_cache",
                    cache_config=vllm_config.cache_config,
                    compress_ratio=compress_ratio,
                )
                CompressorStateCache(
                    state_dim=2 * 2 * config.index_head_dim,
                    dtype=torch.float32,
                    compress_ratio=compress_ratio,
                    prefix=f"{indexer_prefix}.compressor.state_cache",
                )


def make_kv_cache_config(vllm_config: VllmConfig) -> KVCacheConfig:
    register_deepseek_v4_flash_cache_layers(vllm_config)
    spec = get_kv_cache_spec(vllm_config)
    assert spec
    kv_cache_config = get_kv_cache_configs(
        vllm_config,
        [spec],
        [8 * 1024**3],
    )[0]
    assert UCMFAWAConnector.can_handle_kv_cache_config(kv_cache_config)
    assert len(spec) > 100
    assert len(kv_cache_config.kv_cache_groups) == 5
    assert kv_cache_config.num_blocks == GPU_KV_BLOCKS_OVERRIDE
    assert (
        sum(kv_cache_tensor.size for kv_cache_tensor in kv_cache_config.kv_cache_tensors)
        <= MAX_TEST_KV_CACHE_BYTES
    )
    return kv_cache_config


def create_scheduler_connector(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> UCMConnector:
    wrapper = cast(
        UCMConnector,
        KVConnectorFactory.create_connector(
            config=vllm_config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=kv_cache_config,
        ),
    )
    assert isinstance(wrapper.connector, UCMFAWAConnector)
    wrapper.connector.generate_hash = generated_hashes
    wrapper.connector._seed = b"seed"
    return wrapper


def create_worker_connector(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor],
) -> UCMConnector:
    with patch(
        "ucm.integration.vllm.ucm_connector.get_world_group",
        return_value=LocalWorldGroup(),
    ):
        with patch(
            "ucm.integration.vllm.device.CudaDevice.get_cpu_affinity",
            return_value=None,
        ):
            wrapper = cast(
                UCMConnector,
                KVConnectorFactory.create_connector(
                    config=vllm_config,
                    role=KVConnectorRole.WORKER,
                    kv_cache_config=kv_cache_config,
                ),
            )
            wrapper.register_kv_caches(kv_caches)
    assert isinstance(wrapper.connector, UCMFAWAConnector)
    return wrapper


def allocate_kv_caches(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    raw_tensors: dict[str, torch.Tensor] = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        raw = torch.empty(kv_cache_tensor.size, dtype=torch.uint8, device=device)
        for layer_name in kv_cache_tensor.shared_by:
            raw_tensors[layer_name] = raw

    kv_caches: dict[str, torch.Tensor] = {}
    for group in kv_cache_config.kv_cache_groups:
        nested_specs = getattr(group.kv_cache_spec, "kv_cache_specs", {})
        for layer_name in group.layer_names:
            spec = cast(AttentionSpec, nested_specs[layer_name])
            raw = raw_tensors[layer_name]
            num_blocks = raw.numel() // spec.page_size_bytes
            dtype_size = torch.empty((), dtype=spec.dtype).element_size()
            if layer_name.endswith("state_cache"):
                shape = (num_blocks, spec.storage_block_size, spec.head_size)
                kv_caches[layer_name] = torch.as_strided(
                    raw.view(spec.dtype),
                    size=shape,
                    stride=(spec.page_size_bytes // dtype_size, shape[2], 1),
                )
                continue
            backend = (
                DeepseekSparseSWABackend
                if layer_name.endswith("swa_cache")
                else DeepseekV4IndexerBackend
                if ".indexer.k_cache" in layer_name
                else DeepseekV4FlashMLASparseBackend
            )
            shape = backend.get_kv_cache_shape(
                num_blocks,
                spec.storage_block_size,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=vllm_config_cache_dtype(spec),
            )
            if spec.page_size_padded is not None:
                kv_caches[layer_name] = torch.as_strided(
                    raw.view(spec.dtype),
                    size=shape,
                    stride=(
                        spec.page_size_bytes // dtype_size,
                        shape[2],
                        1,
                    ),
                )
            else:
                kv_caches[layer_name] = raw.view(spec.dtype).view(shape)
    return kv_caches


def vllm_config_cache_dtype(spec: AttentionSpec) -> str:
    return getattr(spec, "cache_dtype_str", None) or "fp8_ds_mla"


def wait_lookup_on_prefix(store, keys: list[bytes], expected: int) -> None:
    deadline = time.monotonic() + 5.0
    last = -1
    while time.monotonic() < deadline:
        last = store.lookup_on_prefix(keys)
        if last == expected:
            return
        time.sleep(0.05)
    assert last == expected


def build_allocation(
    connector: UCMFAWAConnector,
    canonical_blocks: int,
    base_block_id: int,
) -> tuple[list[int], ...]:
    allocation: list[list[int]] = []
    for group_id, meta in sorted(connector.group_metas.items()):
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


def shifted_allocation(
    allocation: tuple[list[int], ...],
    offset: int,
) -> tuple[list[int], ...]:
    return tuple([block_id + offset for block_id in group] for group in allocation)


def kv_cache_blocks(group_block_ids: tuple[list[int], ...]) -> KVCacheBlocks:
    return KVCacheBlocks(
        tuple(
            [KVCacheBlock(block_id=block_id) for block_id in group]
            for group in group_block_ids
        )
    )


def scheduler_output(
    *,
    new_req_ids: list[str] | None = None,
    new_req_block_ids: dict[str, tuple[list[int], ...]] | None = None,
    cached_req_ids: list[str] | None = None,
    resumed_req_ids: set[str] | None = None,
    new_block_ids: list[tuple[list[int], ...] | None] | None = None,
    num_scheduled_tokens: dict[str, int] | None = None,
    finished_req_ids: set[str] | None = None,
) -> SchedulerOutput:
    new_req_ids = new_req_ids or []
    new_req_block_ids = new_req_block_ids or {}
    cached_req_ids = cached_req_ids or []
    resumed_req_ids = resumed_req_ids or set()
    new_block_ids = new_block_ids or []
    num_scheduled_tokens = num_scheduled_tokens or {}
    finished_req_ids = finished_req_ids or set()
    group_count = next(
        (len(block_ids) for block_ids in new_req_block_ids.values()),
        0,
    )
    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=None,
                mm_features=[],
                sampling_params=None,
                pooling_params=None,
                block_ids=new_req_block_ids.get(
                    req_id,
                    tuple([] for _ in range(group_count)),
                ),
                num_computed_tokens=0,
                lora_request=None,
            )
            for req_id in new_req_ids
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=cached_req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=[[] for _ in cached_req_ids],
            all_token_ids={},
            new_block_ids=new_block_ids,
            num_computed_tokens=[0 for _ in cached_req_ids],
            num_output_tokens=[0 for _ in cached_req_ids],
        ),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids,
        free_encoder_mm_hashes=[],
    )


def ptr_size_list(
    worker: UCMFAWAConnector,
    group_ids: tuple[int, ...],
) -> list[int]:
    return worker._store_tensor_size_list(worker.group_layouts, group_ids)


def fill_ptrs(
    worker: UCMFAWAConnector,
    registry: TensorRegistry,
    group_ids: tuple[int, ...],
    ptrs: np.ndarray,
    value: int,
) -> None:
    registry.fill_ptrs(ptrs, ptr_size_list(worker, group_ids), value)


def ptr_row_bytes(
    worker: UCMFAWAConnector,
    registry: TensorRegistry,
    group_ids: tuple[int, ...],
    ptrs: np.ndarray,
) -> list[bytes]:
    sizes = ptr_size_list(worker, group_ids)
    return [registry.ptrs_bytes(ptr_row, sizes) for ptr_row in ptrs]


def wrap_store(
    connector: UCMFAWAConnector,
    registry: TensorRegistry,
) -> None:
    assert isinstance(connector.fa_store, UcmPipelineStore)
    assert isinstance(connector.wa_store, UcmPipelineStore)
    connector.fa_store = CapturingStore(
        connector.fa_store,
        registry,
        connector.group_layouts,
        connector.fa_group_ids,
    )
    connector.wa_store = CapturingStore(
        connector.wa_store,
        registry,
        connector.group_layouts,
        connector.window_group_ids,
    )
    connector.store = connector.fa_store


def assert_real_fawa_wrapper(wrapper: UCMConnector) -> UCMFAWAConnector:
    assert isinstance(wrapper.connector, UCMFAWAConnector)
    assert isinstance(wrapper.connector.fa_store, UcmPipelineStore)
    assert isinstance(wrapper.connector.wa_store, UcmPipelineStore)
    return wrapper.connector


def test_gpu_tp4_deepseek_v4_flash_hma_e2e(tmp_path):
    device = hbm_device()
    engine_id = f"gpu-tp4-e2e-{os.getpid()}"
    storage_backend = str(tmp_path / "ucm")
    base_vllm_config = make_vllm_config(
        model=model_path(),
        storage_backend=storage_backend,
        engine_id=engine_id,
    )
    kv_cache_config = make_kv_cache_config(base_vllm_config)

    scheduler_wrapper = create_scheduler_connector(base_vllm_config, kv_cache_config)
    scheduler = assert_real_fawa_wrapper(scheduler_wrapper)
    assert scheduler.hash_block_size == 256
    assert scheduler.fa_group_ids
    assert scheduler.window_group_ids

    worker_wrappers: list[UCMConnector] = []
    worker_connectors: list[UCMFAWAConnector] = []
    registries: list[TensorRegistry] = []
    for tp_rank in range(4):
        worker_config = make_vllm_config(
            model=model_path(),
            storage_backend=storage_backend,
            engine_id=engine_id,
            tp_rank=tp_rank,
        )
        kv_caches = allocate_kv_caches(kv_cache_config, device)
        worker_wrapper = create_worker_connector(
            worker_config,
            kv_cache_config,
            kv_caches,
        )
        worker = assert_real_fawa_wrapper(worker_wrapper)
        worker.tp_rank = tp_rank
        registry = TensorRegistry()
        registry.register_layouts(worker.group_layouts)
        wrap_store(worker, registry)
        worker_wrappers.append(worker_wrapper)
        worker_connectors.append(worker)
        registries.append(registry)

    rank0_wrapper = worker_wrappers[0]
    rank0 = worker_connectors[0]

    # Stage 1: first batch computes and persists a reusable prefix through the
    # normal scheduler/worker dump path. This models the batch that warms the
    # external UCM cache; no FakeStore or direct store writes are used.
    producer = make_request(
        "req-gpu-producer",
        scheduler.hash_block_size,
        PRODUCER_BLOCK_VALUES,
    )
    prefix_keys = generated_hashes(
        scheduler.hash_block_size,
        producer.all_token_ids,
        b"seed",
    )
    producer_alloc = build_allocation(scheduler, 2, 1)
    hit_tokens, is_async = scheduler_wrapper.get_num_new_matched_tokens(producer, 0)
    assert hit_tokens == 0
    assert is_async is False
    scheduler_wrapper.update_state_after_alloc(
        producer,
        kv_cache_blocks(producer_alloc),
        hit_tokens,
    )
    producer_metadata = scheduler_wrapper.build_connector_meta(
        scheduler_output(
            new_req_ids=[producer.request_id],
            new_req_block_ids={producer.request_id: producer_alloc},
            num_scheduled_tokens={
                producer.request_id: len(producer.all_token_ids),
            },
        )
    )
    assert isinstance(producer_metadata, UCMFAWAConnectorMetadata)
    producer_dispatch = producer_metadata.request_meta[producer.request_id]
    assert producer_dispatch.load_keys == []
    assert producer_dispatch.load_hash_start == 0
    assert producer_dispatch.load_hash_end == 0
    assert producer_dispatch.dump_keys == prefix_keys
    assert producer_dispatch.dump_hash_end > producer_dispatch.dump_hash_start
    assert len(producer_dispatch.dump_vllm_block_ids) == len(
        scheduler.group_metas
    )

    seeded_fa_bytes: dict[bytes, bytes] = {}
    seeded_wa_bytes: dict[bytes, bytes] = {}
    producer_fa_ptrs = rank0._extract_fa_ptr(
        producer_dispatch.dump_keys,
        producer_dispatch.dump_hash_start,
        producer_dispatch.dump_hash_end,
        producer_dispatch.dump_vllm_block_ids,
    )
    producer_wa_ptrs = rank0._extract_wa_ptr(
        producer_dispatch.dump_keys,
        producer_dispatch.dump_hash_start,
        producer_dispatch.dump_hash_end,
        producer_dispatch.dump_vllm_block_ids,
    )
    for idx, key in enumerate(producer_dispatch.dump_keys):
        fill_ptrs(
            rank0,
            registries[0],
            rank0.fa_group_ids,
            producer_fa_ptrs[idx : idx + 1],
            0x21 + idx,
        )
        fill_ptrs(
            rank0,
            registries[0],
            rank0.window_group_ids,
            producer_wa_ptrs[idx : idx + 1],
            0x61 + idx,
        )
        seeded_fa_bytes[key] = ptr_row_bytes(
            rank0,
            registries[0],
            rank0.fa_group_ids,
            producer_fa_ptrs[idx : idx + 1],
        )[0]
        seeded_wa_bytes[key] = ptr_row_bytes(
            rank0,
            registries[0],
            rank0.window_group_ids,
            producer_wa_ptrs[idx : idx + 1],
        )[0]

    before_rank0_dump_count = len(cast(CapturingStore, rank0.fa_store).dump_history)
    for wrapper in worker_wrappers[1:]:
        wrapper.bind_connector_metadata(producer_metadata)
        assert wrapper.has_connector_metadata()
        wrapper.save_kv_layer("unused", torch.empty(0, device=device), None)
        wrapper.wait_for_save()
        wrapper.clear_connector_metadata()
        assert not wrapper.has_connector_metadata()
    assert len(cast(CapturingStore, rank0.fa_store).dump_history) == before_rank0_dump_count

    rank0_wrapper.bind_connector_metadata(producer_metadata)
    assert rank0_wrapper.has_connector_metadata()
    rank0_wrapper.save_kv_layer("unused", torch.empty(0, device=device), None)
    rank0_wrapper.wait_for_save()
    assert rank0_wrapper.get_block_ids_with_load_errors() == set()
    assert rank0_wrapper.get_finished({producer.request_id}) == (None, None)
    assert rank0_wrapper.build_connector_worker_meta() is None
    rank0_wrapper.clear_connector_metadata()
    assert not rank0_wrapper.has_connector_metadata()
    wait_lookup_on_prefix(scheduler.fa_store, prefix_keys, 1)
    wait_lookup_on_prefix(scheduler.wa_store, prefix_keys, 1)
    assert scheduler_wrapper.build_connector_meta(
        scheduler_output(finished_req_ids={producer.request_id})
    ).request_meta == {}
    assert scheduler_wrapper.request_finished_all_groups(
        producer,
        producer_alloc,
    ) == (False, None)

    # Stage 2: second batch arrives later with the same two-block prefix.
    # Scheduler lookup should return external hits from the first batch's store
    # output, and every TP worker should load real FA/WA rows into fresh HBM
    # allocations.
    requests = [
        make_request(req_id, scheduler.hash_block_size, block_values)
        for req_id, block_values in HASH_BLOCK_VALUES.items()
    ]
    initial_allocs = {
        request.request_id: build_allocation(scheduler, 2, 1 + idx * 300)
        for idx, request in enumerate(requests)
    }
    target_allocs = {
        request.request_id: build_allocation(scheduler, 4, 1 + idx * 300)
        for idx, request in enumerate(requests)
    }

    for request in requests:
        hit_tokens, is_async = scheduler_wrapper.get_num_new_matched_tokens(request, 0)
        assert hit_tokens == 2 * scheduler.hash_block_size
        assert is_async is False
        scheduler_wrapper.update_state_after_alloc(
            request,
            kv_cache_blocks(initial_allocs[request.request_id]),
            hit_tokens,
        )

    first_metadata = scheduler_wrapper.build_connector_meta(
        scheduler_output(
            new_req_ids=[request.request_id for request in requests],
            new_req_block_ids={
                request.request_id: initial_allocs[request.request_id]
                for request in requests
            },
            num_scheduled_tokens={
                request.request_id: 0 for request in requests
            },
        )
    )
    assert isinstance(first_metadata, UCMFAWAConnectorMetadata)
    for request in requests:
        dispatch = first_metadata.request_meta[request.request_id]
        assert dispatch.load_keys == prefix_keys
        assert dispatch.load_hash_end - dispatch.load_hash_start == len(prefix_keys)
        assert dispatch.dump_keys == []
        assert dispatch.dump_hash_end == dispatch.dump_hash_start
        for group_id in scheduler.window_group_ids:
            tail_blocks = scheduler.group_metas[group_id].tail_blocks
            if tail_blocks == 0:
                assert dispatch.load_vllm_block_ids[group_id] == []
            else:
                assert len(dispatch.load_vllm_block_ids[group_id]) == tail_blocks

    for wrapper, worker, registry in zip(worker_wrappers, worker_connectors, registries):
        wrapper.bind_connector_metadata(first_metadata)
        assert wrapper.has_connector_metadata()
        wrapper.start_load_kv(None)
        wrapper.wait_for_layer_load("unused")
        fa_store = cast(CapturingStore, worker.fa_store)
        wa_store = cast(CapturingStore, worker.wa_store)
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
                request_meta.load_hash_end - 1,
                request_meta.load_hash_end,
                request_meta.load_vllm_block_ids,
            )
            assert fa_store.load_history[-2:] or fa_store.load_history
            assert wa_store.load_history[-2:] or wa_store.load_history
            assert ptr_row_bytes(worker, registry, worker.fa_group_ids, fa_ptrs) == [
                seeded_fa_bytes[key] for key in prefix_keys
            ]
            assert ptr_row_bytes(
                worker,
                registry,
                worker.window_group_ids,
                wa_ptrs,
            ) == [
                seeded_wa_bytes[prefix_keys[-1]]
            ]
        assert all(history == prefix_keys for history in fa_store.load_history[-2:])
        assert all(history == prefix_keys[-1:] for history in wa_store.load_history[-2:])
        assert wrapper.get_block_ids_with_load_errors() == set()
        wrapper.clear_connector_metadata()
        assert not wrapper.has_connector_metadata()

    # Stage 3: vLLM can preempt a request and later resume it with a replacement
    # allocation snapshot in scheduled_cached_reqs. The connector must discard
    # the old HBM rows, rebuild rows from the resumed allocation, and load the
    # same external prefix into the new blocks.
    resumed_allocs = {
        request_id: shifted_allocation(allocation, 40)
        for request_id, allocation in initial_allocs.items()
    }
    resumed_targets = {
        request_id: shifted_allocation(allocation, 40)
        for request_id, allocation in target_allocs.items()
    }
    resumed_metadata = scheduler_wrapper.build_connector_meta(
        scheduler_output(
            cached_req_ids=[request.request_id for request in requests],
            resumed_req_ids={request.request_id for request in requests},
            new_block_ids=[resumed_allocs[request.request_id] for request in requests],
            num_scheduled_tokens={request.request_id: 0 for request in requests},
        )
    )
    assert isinstance(resumed_metadata, UCMFAWAConnectorMetadata)
    for request in requests:
        dispatch = resumed_metadata.request_meta[request.request_id]
        assert dispatch.load_keys == prefix_keys
        assert dispatch.load_hash_end - dispatch.load_hash_start == len(prefix_keys)
        assert dispatch.dump_keys == []
        assert dispatch.dump_hash_end == dispatch.dump_hash_start
        for group_id in scheduler.window_group_ids:
            tail_blocks = scheduler.group_metas[group_id].tail_blocks
            if tail_blocks == 0:
                assert dispatch.load_vllm_block_ids[group_id] == []
            else:
                assert len(dispatch.load_vllm_block_ids[group_id]) == tail_blocks

    for wrapper, worker, registry in zip(worker_wrappers, worker_connectors, registries):
        wrapper.handle_preemptions(resumed_metadata)
        wrapper.bind_connector_metadata(resumed_metadata)
        wrapper.start_load_kv(None)
        fa_store = cast(CapturingStore, worker.fa_store)
        wa_store = cast(CapturingStore, worker.wa_store)
        for request in requests:
            request_meta = resumed_metadata.request_meta[request.request_id]
            fa_ptrs = worker._extract_fa_ptr(
                request_meta.load_keys,
                request_meta.load_hash_start,
                request_meta.load_hash_end,
                request_meta.load_vllm_block_ids,
            )
            wa_ptrs = worker._extract_wa_ptr(
                request_meta.load_keys[-1:],
                request_meta.load_hash_end - 1,
                request_meta.load_hash_end,
                request_meta.load_vllm_block_ids,
            )
            assert ptr_row_bytes(worker, registry, worker.fa_group_ids, fa_ptrs) == [
                seeded_fa_bytes[key] for key in prefix_keys
            ]
            assert ptr_row_bytes(
                worker,
                registry,
                worker.window_group_ids,
                wa_ptrs,
            ) == [
                seeded_wa_bytes[prefix_keys[-1]]
            ]
        assert all(history == prefix_keys for history in fa_store.load_history[-2:])
        assert all(history == prefix_keys[-1:] for history in wa_store.load_history[-2:])
        wrapper.clear_connector_metadata()

    # Stage 4: chunk prefill continues after the resumed prefix load. Each
    # scheduler tick carries a complete allocation delta for one incremental
    # hash boundary and should emit flat candidate metadata immediately.
    third_block_allocs = {
        request_id: build_allocation(scheduler, 3, allocation[0][0])
        for request_id, allocation in resumed_allocs.items()
    }
    first_deltas: dict[str, tuple[list[int], ...]] = {}
    second_deltas: dict[str, tuple[list[int], ...]] = {}
    for request in requests:
        first_deltas[request.request_id] = allocation_delta(
            resumed_allocs[request.request_id],
            third_block_allocs[request.request_id],
        )
        second_deltas[request.request_id] = allocation_delta(
            third_block_allocs[request.request_id],
            resumed_targets[request.request_id],
        )

    partial_metadata = scheduler_wrapper.build_connector_meta(
        scheduler_output(
            cached_req_ids=[request.request_id for request in requests],
            new_block_ids=[first_deltas[request.request_id] for request in requests],
            num_scheduled_tokens={
                request.request_id: scheduler.hash_block_size for request in requests
            },
        )
    )
    assert isinstance(partial_metadata, UCMFAWAConnectorMetadata)
    expected_partial_dump_keys = {
        request.request_id: [
            generated_hashes(
                scheduler.hash_block_size,
                request.all_token_ids,
                b"seed",
            )[2]
        ]
        for request in requests
    }
    for request in requests:
        request_meta = partial_metadata.request_meta[request.request_id]
        assert request_meta.dump_keys == expected_partial_dump_keys[request.request_id]
        assert request_meta.dump_hash_end - request_meta.dump_hash_start == 1
        for group_id in scheduler.window_group_ids:
            expected = dump_candidate_count(
                scheduler,
                group_id,
                request_meta.dump_hash_start,
                request_meta.dump_hash_end,
            )
            assert len(request_meta.dump_vllm_block_ids[group_id]) == expected

    # Stage 5: the next scheduler tick delivers the remaining group allocations.
    # It reports only another incremental hash block; metadata progress from the
    # previous tick still makes this dump the next boundary.
    final_metadata = scheduler_wrapper.build_connector_meta(
        scheduler_output(
            cached_req_ids=[request.request_id for request in requests],
            new_block_ids=[second_deltas[request.request_id] for request in requests],
            num_scheduled_tokens={
                request.request_id: scheduler.hash_block_size
                for request in requests
            },
        )
    )
    assert isinstance(final_metadata, UCMFAWAConnectorMetadata)
    expected_dump_keys = {
        request.request_id: [
            block_hash
            for block_hash in generated_hashes(
                scheduler.hash_block_size,
                request.all_token_ids,
                b"seed",
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

    rank0 = worker_connectors[0]
    rank0_wrapper.bind_connector_metadata(final_metadata)
    expected_fa_bytes: dict[bytes, bytes] = {}
    expected_wa_bytes: dict[bytes, bytes] = {}
    for request_meta in final_metadata.request_meta.values():
        dump_keys = request_meta.dump_keys
        fa_ptrs = rank0._extract_fa_ptr(
            dump_keys,
            request_meta.dump_hash_start,
            request_meta.dump_hash_end,
            request_meta.dump_vllm_block_ids,
        )
        wa_ptrs = rank0._extract_wa_ptr(
            dump_keys,
            request_meta.dump_hash_start,
            request_meta.dump_hash_end,
            request_meta.dump_vllm_block_ids,
        )
        fill_ptrs(rank0, registries[0], rank0.fa_group_ids, fa_ptrs, 0xA1)
        fill_ptrs(rank0, registries[0], rank0.window_group_ids, wa_ptrs, 0xD1)
        for key, row_bytes_value in zip(
            dump_keys,
            ptr_row_bytes(rank0, registries[0], rank0.fa_group_ids, fa_ptrs),
        ):
            expected_fa_bytes[key] = row_bytes_value
        for key, row_bytes_value in zip(
            dump_keys,
            ptr_row_bytes(
                rank0,
                registries[0],
                rank0.window_group_ids,
                wa_ptrs,
            ),
        ):
            expected_wa_bytes[key] = row_bytes_value

    before_dump_count = len(cast(CapturingStore, rank0.fa_store).dump_history)
    for wrapper, worker in zip(worker_wrappers[1:], worker_connectors[1:]):
        wrapper.bind_connector_metadata(final_metadata)
        wrapper.save_kv_layer("unused", torch.empty(0, device=device), None)
        wrapper.wait_for_save()
        assert not cast(CapturingStore, worker.fa_store).dump_history
        assert not cast(CapturingStore, worker.wa_store).dump_history
        assert wrapper.build_connector_worker_meta() is None
        assert wrapper.get_finished({request.request_id for request in requests}) == (
            None,
            None,
        )
        wrapper.clear_connector_metadata()
    assert len(cast(CapturingStore, rank0.fa_store).dump_history) == before_dump_count

    rank0_wrapper.save_kv_layer("unused", torch.empty(0, device=device), None)
    rank0_wrapper.wait_for_save()
    fa_store = cast(CapturingStore, rank0.fa_store)
    wa_store = cast(CapturingStore, rank0.wa_store)
    expected_flat_keys = expected_dump_keys["req-gpu-a"] + expected_dump_keys["req-gpu-b"]
    assert [
        key
        for history in fa_store.dump_history[before_dump_count:]
        for key in history
    ] == expected_flat_keys
    assert [
        key
        for history in wa_store.dump_history[before_dump_count:]
        for key in history
    ] == expected_flat_keys
    wait_lookup_on_prefix(
        scheduler.fa_store,
        expected_flat_keys,
        len(expected_flat_keys) - 1,
    )
    wait_lookup_on_prefix(
        scheduler.wa_store,
        expected_flat_keys,
        len(expected_flat_keys) - 1,
    )

    for key in expected_flat_keys:
        assert len(expected_fa_bytes[key]) != len(expected_wa_bytes[key])
        assert set(expected_fa_bytes[key]) == {0xA1}
        assert 0xD1 in set(expected_wa_bytes[key])
        assert 0xA1 not in set(expected_wa_bytes[key])
        assert expected_fa_bytes[key] != expected_wa_bytes[key]

    # Stage 6: finish through the remaining scheduler/worker connector APIs.
    # FAWA owns no async block release today, so all completion metadata is empty
    # and the scheduler may free the HMA blocks immediately.
    for request in requests:
        assert scheduler_wrapper.request_finished_all_groups(
            request,
            resumed_targets[request.request_id],
        ) == (False, None)
    scheduler_wrapper.update_connector_output(KVConnectorOutput())
    assert list(scheduler_wrapper.take_events()) == []
    assert UCMConnector.get_required_kvcache_layout(base_vllm_config) is None
    assert not UCMConnector.requires_piecewise_for_cudagraph({})
    assert scheduler_wrapper.prefer_cross_layer_blocks is False
    assert scheduler_wrapper.role == KVConnectorRole.SCHEDULER
    assert scheduler_wrapper.build_connector_meta(
        scheduler_output(finished_req_ids={request.request_id for request in requests})
    ).request_meta == {}

    for wrapper in worker_wrappers:
        assert wrapper.role == KVConnectorRole.WORKER
        assert wrapper.get_handshake_metadata() is None
        assert wrapper.get_kv_connector_stats() is None
        assert wrapper.get_kv_connector_kv_cache_events() is None
        assert wrapper.get_finished_count() is None
        assert wrapper.reset_cache() is None
        wrapper.set_xfer_handshake_metadata({})
        wrapper.set_host_xfer_buffer_ops(lambda *args: None)
        wrapper.shutdown()
