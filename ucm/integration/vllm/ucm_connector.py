import copy
import hashlib
import math
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.integration.vllm.device import create_device
from ucm.logger import init_logger
from ucm.observability import PrometheusStatsLogger
from ucm.shared.metrics import ucmmetrics
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from ucm.sparse.state import has_ucm_sparse

logger = init_logger(__name__)


@dataclass
class RequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    # local_computed_block + external_computed_block
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: list[int] = field(default_factory=list)
    token_processed: int = 0


@dataclass
class RequestDispatchMeta:
    load_block_ids: tuple[
        list[bytes], list[int]
    ]  # [0] mean ucm_block_ids, [1] means vllm_block_ids
    dump_block_ids: tuple[list[bytes], list[int]]


class KVCacheLayout:
    def __init__(
        self, kvcaches, use_layerwise: bool, vllm_config: "VllmConfig"
    ) -> None:
        # each row is a layer, each column is a tensor_size/ptr in the layer (e.g., k, v, rope, k_index)
        self.base_ptrs: np.ndarray  # (n_layers, n_ptrs）
        self.tensor_size_lists: np.ndarray  # (n_layers, n_tensor_sizes)
        self.use_layerwise = use_layerwise
        self.vllm_config = vllm_config
        self.pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        self.num_hidden_layers = getattr(
            self.vllm_config.model_config.hf_text_config, "num_hidden_layers", 0
        )
        self.pp_rank = (
            self.vllm_config.parallel_config.rank
            // self.vllm_config.parallel_config.tensor_parallel_size
        ) % self.vllm_config.parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(self.num_hidden_layers, self.pp_rank, self.pp_size)
        self.local_num_hidden_layers = end - start
        if self.pp_size > 1 and self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be > 0 when pp_size > 1")
        self.layer_name_to_id = {
            name: extract_layer_index(name) for name in kvcaches.keys()
        }
        self.first_layer_id = next(iter(self.layer_name_to_id.values()))
        self._build_layout(kvcaches)

    def _build_layout(self, kvcaches):

        num_rows = len(set(self.layer_name_to_id.values()))
        raw_ptr_rows = [[] for _ in range(num_rows)]
        stride_rows = [[] for _ in range(num_rows)]

        for layer_name, kv_layer in kvcaches.items():
            ptrs = []
            strides = []

            def handle_tensor(t: torch.Tensor, size_dims):
                ptrs.append(t[0].data_ptr())

                stride = math.prod([t.shape[i] for i in size_dims]) * t.element_size()
                strides.append(stride)

            if isinstance(kv_layer, torch.Tensor):
                if kv_layer.dim() == 5:
                    # [2, num_blocks, block_size, num_head, head_dim]
                    handle_tensor(kv_layer[0], (-3, -2, -1))
                    handle_tensor(kv_layer[1], (-3, -2, -1))
                elif kv_layer.dim() == 3:
                    # [num_blocks, block_size, head_dim]
                    handle_tensor(kv_layer, (-2, -1))
                else:
                    raise ValueError(
                        f"Unsupported kv cache tensor shape: {kv_layer.shape}"
                    )
            elif isinstance(kv_layer, Tuple):
                # vllm_ascend >= 0.10.0, ([num_blocks, block_size, num_head, head_dim], ...)
                for tensor in kv_layer:
                    handle_tensor(tensor, (-3, -2, -1))
            else:
                raise TypeError(f"Unsupported kv cache type: {type(kv_layer)}")

            local_layer_id = self.layer_name_to_id[layer_name] - self.first_layer_id
            raw_ptr_rows[local_layer_id].extend(ptrs)
            stride_rows[local_layer_id].extend(strides)

        self.base_ptrs = np.asarray(raw_ptr_rows, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(stride_rows, dtype=np.uint64)

        logger.info(
            f"base_ptrs: {self.base_ptrs.shape}, tensor_size_lists: {self.tensor_size_lists.shape}"
        )

    def extract_block_addrs(
        self, vllm_block_ids: List[int], layer_first: bool = False
    ) -> np.ndarray:
        vllm_block_ids_np = np.array(vllm_block_ids, np.uint64)
        if layer_first:
            # (n_layers, num_blocks, n_ptrs)
            return (
                self.tensor_size_lists[:, None, :] * vllm_block_ids_np[None, :, None]
                + self.base_ptrs[:, None, :]
            )
        return (
            vllm_block_ids_np[:, None, None] * self.tensor_size_lists[None, :, :]
            + self.base_ptrs[None, :, :]
        )  # (num_blocks, n_layers, n_ptrs)

    @property
    def tensor_size_list(self) -> list[int]:
        return (
            self.tensor_size_lists.reshape(-1).tolist()
            if not self.use_layerwise
            else self.tensor_size_lists[0].tolist()
        )

    @property
    def shard_size(self) -> int:
        return int(
            self.tensor_size_lists.sum()
            if not self.use_layerwise
            else self.tensor_size_lists[0].sum()
        )

    @property
    def block_size(self) -> int:
        if self.pp_size > 1:
            return int(self.tensor_size_lists[0].sum() * self.num_hidden_layers)
        return int(self.tensor_size_lists.sum())


class DeepSeekV4GroupKVCacheLayout:
    """Flat pointer layout for one DeepSeek V4 KV cache group.

    DeepSeek V4 registers several cache views per transformer layer. The views
    belonging to one hybrid KV group are not necessarily contiguous by layer id,
    so this layout flattens all registered tensors in a deterministic order.
    """

    def __init__(self, kvcaches: dict[str, torch.Tensor]) -> None:
        self.kvcaches = dict(sorted(kvcaches.items(), key=self._sort_key))
        self.base_ptrs: np.ndarray
        self.block_strides: np.ndarray
        self.tensor_size_lists: np.ndarray
        self._build_layout()

    @staticmethod
    def _sort_key(item: tuple[str, torch.Tensor]) -> tuple[int, str]:
        name, _ = item
        return (extract_layer_index(name), name)

    def _build_layout(self) -> None:
        ptrs: list[int] = []
        strides: list[int] = []
        tensor_sizes: list[int] = []

        def handle_tensor(t: torch.Tensor, size_dims: Sequence[int]) -> None:
            ptrs.append(t[0].data_ptr())
            strides.append(t.stride(0) * t.element_size())
            tensor_size = math.prod([t.shape[i] for i in size_dims]) * t.element_size()
            tensor_sizes.append(tensor_size)

        for layer_name, kv_layer in self.kvcaches.items():
            if isinstance(kv_layer, torch.Tensor):
                if kv_layer.dim() == 5:
                    # [2, num_blocks, block_size, num_head, head_dim]
                    handle_tensor(kv_layer[0], (-3, -2, -1))
                    handle_tensor(kv_layer[1], (-3, -2, -1))
                elif kv_layer.dim() == 3:
                    # [num_blocks, block_size, head_dim]
                    handle_tensor(kv_layer, (-2, -1))
                else:
                    raise ValueError(
                        f"Unsupported DeepSeek V4 kv cache tensor shape for "
                        f"{layer_name}: {kv_layer.shape}"
                    )
            elif isinstance(kv_layer, Tuple):
                for tensor in kv_layer:
                    if tensor.dim() == 4:
                        handle_tensor(tensor, (-3, -2, -1))
                    elif tensor.dim() == 3:
                        handle_tensor(tensor, (-2, -1))
                    else:
                        raise ValueError(
                            f"Unsupported DeepSeek V4 tuple tensor shape for "
                            f"{layer_name}: {tensor.shape}"
                        )
            else:
                raise TypeError(
                    f"Unsupported DeepSeek V4 kv cache type for "
                    f"{layer_name}: {type(kv_layer)}"
                )

        if not ptrs:
            raise ValueError("DeepSeek V4 KV cache group layout is empty.")

        self.base_ptrs = np.asarray(ptrs, dtype=np.uint64)
        self.block_strides = np.asarray(strides, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(tensor_sizes, dtype=np.uint64)
        logger.info(
            f"DeepSeek V4 group layout: views={len(self.kvcaches)}, "
            f"ptrs={len(ptrs)}, block_size={self.block_size}"
        )

    def extract_block_addrs(self, vllm_block_ids: list[int]) -> np.ndarray:
        vllm_block_ids_np = np.array(vllm_block_ids, np.uint64)
        return (
            vllm_block_ids_np[:, None] * self.block_strides[None, :]
            + self.base_ptrs[None, :]
        )

    def extract_block_tensor_views(self, vllm_block_ids: list[int]) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []

        def add_views(tensor: torch.Tensor, block_id: int) -> None:
            tensors.append(tensor[block_id])

        for block_id in vllm_block_ids:
            for layer_name, kv_layer in self.kvcaches.items():
                if isinstance(kv_layer, torch.Tensor):
                    if kv_layer.dim() == 5:
                        add_views(kv_layer[0], block_id)
                        add_views(kv_layer[1], block_id)
                    elif kv_layer.dim() == 3:
                        add_views(kv_layer, block_id)
                    else:
                        raise ValueError(
                            f"Unsupported DeepSeek V4 kv cache tensor shape for "
                            f"{layer_name}: {kv_layer.shape}"
                        )
                elif isinstance(kv_layer, Tuple):
                    for tensor in kv_layer:
                        add_views(tensor, block_id)
                else:
                    raise TypeError(
                        f"Unsupported DeepSeek V4 kv cache type for "
                        f"{layer_name}: {type(kv_layer)}"
                    )
        return tensors

    def extract_block_tensors(self, vllm_block_ids: list[int]) -> list[dict]:
        block_ids = torch.tensor(vllm_block_ids, dtype=torch.long)
        entries: list[dict] = []

        def add_entry(
            layer_name: str,
            view_name: str,
            tensor: torch.Tensor,
        ) -> None:
            selected = tensor.index_select(0, block_ids.to(tensor.device))
            entries.append(
                {
                    "layer_name": layer_name,
                    "view_name": view_name,
                    "source_shape": tuple(tensor.shape),
                    "source_stride": tuple(tensor.stride()),
                    "dtype": str(tensor.dtype),
                    "block_ids": list(vllm_block_ids),
                    "block_stride_bytes": tensor.stride(0) * tensor.element_size(),
                    "block_nbytes": selected[0].numel() * selected.element_size(),
                    "block_contiguous": tensor[0].is_contiguous(),
                    "data": selected.detach().cpu().clone(),
                }
            )

        for layer_name, kv_layer in self.kvcaches.items():
            if isinstance(kv_layer, torch.Tensor):
                if kv_layer.dim() == 5:
                    add_entry(layer_name, "k", kv_layer[0])
                    add_entry(layer_name, "v", kv_layer[1])
                elif kv_layer.dim() == 3:
                    add_entry(layer_name, "state", kv_layer)
                else:
                    raise ValueError(
                        f"Unsupported DeepSeek V4 kv cache tensor shape for "
                        f"{layer_name}: {kv_layer.shape}"
                    )
            elif isinstance(kv_layer, Tuple):
                for view_idx, tensor in enumerate(kv_layer):
                    add_entry(layer_name, f"tuple_{view_idx}", tensor)
            else:
                raise TypeError(
                    f"Unsupported DeepSeek V4 kv cache type for "
                    f"{layer_name}: {type(kv_layer)}"
                )
        return entries

    @property
    def tensor_size_list(self) -> list[int]:
        return self.tensor_size_lists.tolist()

    @property
    def shard_size(self) -> int:
        return int(self.tensor_size_lists.sum())

    @property
    def block_size(self) -> int:
        return self.shard_size


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, RequestDispatchMeta] = field(default_factory=dict)


class RequestHasher:
    """hash(md5) request to generate ucm block id"""

    def __init__(self, vllm_config, rank_id):
        meta = f"{vllm_config.model_config.model}:{vllm_config.parallel_config.tensor_parallel_size}:{vllm_config.model_config.dtype}:{rank_id}"
        self.meta_bytes = meta.encode("utf-8")

    def __call__(self, input_data) -> bytes:
        if isinstance(input_data, bytes):
            input_bytes = input_data
        else:
            input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)

        h = hashlib.md5(self.meta_bytes + input_bytes)
        return h.digest()


class UCMDirectConnector(KVConnectorBase_V1):
    """
    This connector means synchronize:
    load -> forward -> save
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.use_layerwise = False
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.local_rank = (
            -1 if role == KVConnectorRole.SCHEDULER else get_world_group().local_rank
        )
        self.tp_rank = self._vllm_config.parallel_config.rank
        self.block_size = self._vllm_config.cache_config.block_size
        self.is_mla = self._vllm_config.model_config.is_deepseek_mla
        self.num_layers = self._vllm_config.model_config.get_num_layers(
            self._vllm_config.parallel_config
        )
        self.tp_size = self._vllm_config.parallel_config.tensor_parallel_size
        self.kv_cache_dtype: torch.dtype = None
        self.num_head = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = vllm_config.model_config.get_head_size()
        self.element_size = vllm_config.model_config.dtype.itemsize

        if current_platform.is_cuda_alike():
            logger.info("CUDA device is available.")
            torch_dev = torch
            dev_name = "cuda"
        elif current_platform.device_type == "npu":
            logger.info("NPU device is available.")
            torch_dev = torch.npu
            dev_name = "npu"
        else:
            raise RuntimeError("Unsupported device platform for UCMDirectConnector.")

        if self.local_rank >= 0:
            self.device = torch_dev.device(f"{dev_name}:{self.local_rank}")

        self.store: UcmKVStoreBaseV1
        self.rope_store: Optional[UcmKVStoreBaseV1] = None

        # save block info, avoid hash request twice, and track them until request finished
        self.requests_meta: dict[str, RequestMeta] = {}

        ucm_config = Config(vllm_config.kv_transfer_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.launch_config = ucm_config.get_config()
        self.connector_configs = self.launch_config.get("ucm_connectors", [])
        self.enable_event_sync = self.launch_config.get("enable_event_sync", True)
        self.enable_record_traces = self.launch_config.get(
            "enable_record_traces", False
        )
        assert len(self.connector_configs) > 0, "no storage connector name in config."

        self.chunk_size = self.block_size
        self.blocks_per_chunk = self.chunk_size // self.block_size

        defer_scheduler_store = getattr(self, "_defer_scheduler_store", False)
        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-size connector
            if not defer_scheduler_store:
                self.store = self._create_store(None)
        else:
            self.request_hasher = RequestHasher(
                vllm_config, self.tp_rank % self.tp_size
            )

        self.metrics_config = self.launch_config.get("metrics_config_path", "")
        if self.metrics_config:
            worker_id = (
                f"{self.engine_id}_{get_world_group().rank}"
                if role == KVConnectorRole.WORKER
                else self.engine_id
            )
            self.stats_logger = PrometheusStatsLogger(
                vllm_config.model_config.served_model_name,
                worker_id,
                self.metrics_config,
            )
            logger.info(
                f"metrics_config_path: {self.metrics_config}, set worker_id: {worker_id}"
            )

        self.persist_token_threshold = self.launch_config.get(
            "persist_token_threshold", 0
        )

        # invalid block ids due to load errors
        self._invalid_block_ids: set[int] = set()
        self.cp_world_size = 1
        self.hash_block_size = self.block_size
        self.block_size *= self.cp_world_size

    def generate_hash(
        self, block_size: int, token_ids: List[int], parent_block_hash_value: bytes
    ) -> list[bytes]:
        ret = []
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            # Do not hash the block if it is not full.
            if len(block_token_ids) < block_size:
                break

            block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self.request_hasher(
                (parent_block_hash_value, block_token_ids_tuple)
            )
            parent_block_hash_value = hash_value
            ret.append(hash_value)

        return ret

    def _create_store(
        self,
        kv_cache_layout: Optional[KVCacheLayout],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        if len(self.connector_configs) != 1:
            raise RuntimeError(
                f"Expected exactly one connector config, "
                f"but got {len(self.connector_configs)}: "
                f"{self.connector_configs}"
            )

        name = self.connector_configs[0]["ucm_connector_name"]
        module_path = self.connector_configs[0].get("ucm_connector_module_path", None)
        config = copy.deepcopy(self.connector_configs[0]["ucm_connector_config"])
        config.setdefault("share_buffer_enable", self.is_mla)
        if "storage_backends" in config:
            backends = [path for path in config["storage_backends"].split(":")]
            config["storage_backends"] = backends
        config["unique_id"] = f"{self.engine_id}"
        if self._role == KVConnectorRole.WORKER:
            config["device_id"] = self.local_rank
            config["tensor_size_list"] = (
                kv_cache_layout.tensor_size_list * self.blocks_per_chunk
            )
            config["shard_size"] = kv_cache_layout.shard_size * self.blocks_per_chunk
            config["block_size"] = kv_cache_layout.block_size * self.blocks_per_chunk
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
        else:
            config_base = self.block_size * self.element_size * self.head_size
            config["block_size"] = (
                config_base
                * self.num_layers
                * (1 if self.is_mla else self.num_head * 2)
                * self.blocks_per_chunk
            )
        dp_rank = self._vllm_config.parallel_config.rank
        config["posix_gc_enable"] = (
            self._role != KVConnectorRole.WORKER and dp_rank == 0
        )

        logger.info(f"create {name} with config: {config}")
        return UcmConnectorFactoryV1.create_connector(name, config, module_path)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if has_ucm_sparse() and os.getenv("VLLM_HASH_ATTENTION") == "1":
            for layer_name, value in kv_caches.items():
                kv_cache, k_hash = value
                self.kv_caches[layer_name] = kv_cache
        else:
            self.kv_caches = kv_caches
        sample_kv_layer = next(iter(self.kv_caches.values()))
        if self.kv_cache_dtype is None:
            self.kv_cache_dtype = sample_kv_layer[0].dtype
        if isinstance(sample_kv_layer, torch.Tensor):
            logger.info(f"kv cache shape {sample_kv_layer.shape}")
        elif isinstance(sample_kv_layer, Tuple):
            # vllm_ascend >= 0.10.0 uses Tuple for kvcaches
            for i, tensor in enumerate(sample_kv_layer):
                logger.info(f"kv cache shape {i}: {tensor.shape}")
        self.kv_cache_layout = KVCacheLayout(
            self.kv_caches, self.use_layerwise, self._vllm_config
        )
        self.block_data_size = self.kv_cache_layout.block_size
        self.layer_name_to_id = self.kv_cache_layout.layer_name_to_id
        self.layer_ids = sorted(set(self.layer_name_to_id.values()))
        self.first_layer_id = self.layer_ids[0]

        self.device = create_device()

        enable_affinity = os.getenv("VLLM_CPU_AFFINITY") == "1"
        worker_cores, store_cores = (
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        self.store = self._create_store(self.kv_cache_layout, store_cores)

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

        if self.device is None:
            raise RuntimeError(f"Unsupported device platform for UCMDirectConnector.")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert num_computed_tokens % self.block_size == 0
        hbm_hit_block_num = num_computed_tokens // self.block_size

        ucm_block_ids = self.generate_hash(
            self.hash_block_size, request.all_token_ids, self._seed
        )

        if (
            self.enable_record_traces
            and request.request_id not in self.requests_meta
            and len(ucm_block_ids) > 0
        ):
            hex_ucm_block_ids = [id.hex() for id in ucm_block_ids]
            logger.info_once(
                f"timestamp: {time.perf_counter()}, "
                f"input_length: {request.num_tokens}, "
                f"output_length: {request.max_tokens}, "
                f"ucm_block_ids: {hex_ucm_block_ids}"
            )

        # Skip persistence if token count is below the threshold
        if self.persist_token_threshold > request.num_tokens:
            logger.info_once(
                f"Skip persistence: req {request.request_id}, "
                f"input tokens ({request.num_tokens}) < threshold ({self.persist_token_threshold})."
            )
            return 0, False

        external_block_ids = ucm_block_ids[hbm_hit_block_num * self.cp_world_size :]
        if not external_block_ids:
            return 0, False
        try:
            external_hit_blocks = self.store.lookup_on_prefix(external_block_ids) + 1
            external_hit_blocks //= self.cp_world_size
        except RuntimeError as e:
            external_hit_blocks = 0
            logger.error(f"request {request.request_id} look up error. {e}")

        logger.info_once(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(ucm_block_ids)}, "
            f"hit hbm: {hbm_hit_block_num * self.cp_world_size}, "
            f"hit external: {external_hit_blocks * self.cp_world_size}"
        )
        if self.metrics_config:
            ucmmetrics.update_stats(
                {
                    "interval_lookup_hit_rates": external_hit_blocks
                    * self.cp_world_size
                    / len(ucm_block_ids)
                },
            )

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks

        external_hit_tokens = external_hit_blocks * self.block_size

        # When all the tokens are cached in ssd or hbm,
        # we need to recompute the last token. This if condition will be removed
        # once vLLM scheduler provides a better solution in the future.
        num_total_hit_tokens = total_hit_block_num * self.block_size
        if num_total_hit_tokens == request.num_tokens:
            external_hit_tokens -= 1

        self.requests_meta[request.request_id] = RequestMeta(
            ucm_block_ids=ucm_block_ids,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
        )

        return external_hit_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def _generate_dispatch_meta(
        self,
        req_meta: RequestMeta,
        new_tokens: int,
        vllm_block_ids: list[int],
        need_load: bool = True,
    ) -> RequestDispatchMeta:
        """
        Request Blocks layout:
        ----------------------------------------------------------------------------------------------------
        | local_computed_block(HBM hit) | external_computed_block(external hit) | new_block(need to dump)  |
        ----------------------------------------------------------------------------------------------------
        |      hbm_hit_block_num        |                 LOAD                  |     new_blocks_num       |
        ----------------------------------------------------------------------------------------------------
        |                              total_hit_block_num                      |
        ----------------------------------------------------------------------------------------------------
        |                                         scheduled_block_num                                      |
        """

        hbm_hit_block_num = req_meta.hbm_hit_block_num
        total_hit_block_num = req_meta.total_hit_block_num
        ucm_block_ids = req_meta.ucm_block_ids
        req_meta.vllm_block_ids.extend(vllm_block_ids)

        load_ucm_block_ids, load_vllm_block_ids = [], []
        dump_ucm_block_ids, dump_vllm_block_ids = [], []
        if need_load:
            load_ucm_block_ids = ucm_block_ids[
                hbm_hit_block_num
                * self.cp_world_size : total_hit_block_num
                * self.cp_world_size
            ]
            load_vllm_block_ids = vllm_block_ids[hbm_hit_block_num:total_hit_block_num]

        if req_meta.token_processed < req_meta.num_token_ids:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            dump_ucm_block_ids = ucm_block_ids[
                start_idx * self.cp_world_size : end_idx * self.cp_world_size
            ]
            dump_vllm_block_ids = req_meta.vllm_block_ids[start_idx:end_idx]
            req_meta.token_processed += new_tokens

        return RequestDispatchMeta(
            (load_ucm_block_ids, load_vllm_block_ids),
            (dump_ucm_block_ids, dump_vllm_block_ids),
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta = {}
        # for new request, we need to load and dump
        for request in scheduler_output.scheduled_new_reqs:
            request_id, vllm_block_ids = request.req_id, request.block_ids[0]
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    vllm_block_ids,
                )

        # for cached request, there are 3 situation:
        # 1. chunked prefill: we only need dump
        # 2. resumed: we need to handle like new request
        # 3. TODO decode stage: nothing happened
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        if not isinstance(scheduled_cached_reqs, list):
            # >= 0.9.2
            for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    new_block_ids = []
                    if scheduled_cached_reqs.new_block_ids[i] != None:
                        new_block_ids = scheduled_cached_reqs.new_block_ids[i][0]
                    if hasattr(scheduled_cached_reqs, "resumed_from_preemption"):
                        resumed_from_preemption = (
                            scheduled_cached_reqs.resumed_from_preemption[i]
                        )
                    else:
                        resumed_from_preemption = (
                            request_id in scheduled_cached_reqs.resumed_req_ids
                        )
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        new_block_ids,
                        resumed_from_preemption,
                    )
        else:
            for request in scheduled_cached_reqs:
                request_id = request.req_id
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        request.new_block_ids[0],
                        request.resumed_from_preemption,
                    )

        # clear finished request
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(requests_dispatch_meta)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        request_to_task: dict[str, Task] = {}
        is_load = False
        num_loaded_block = 0
        num_loaded_request = 0
        load_start_time = time.perf_counter() * 1000
        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue
            is_load = True
            num_loaded_block += len(request.load_block_ids[0])
            num_loaded_request += 1

            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self.tp_rank != 0 and not self.is_mla:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            total_ptrs = self.kv_cache_layout.extract_block_addrs(vllm_block_ids)
            total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
            shard_indexs = [0] * len(ucm_block_ids)
            try:
                task = self.store.load_data(ucm_block_ids, shard_indexs, total_ptrs)
                request_to_task[request_id] = task
            except RuntimeError as e:
                logger.error(f"request {request_id} submit load task error. {e}")
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                num_loaded_block -= len(request.load_block_ids[0])

        for request_id, task in request_to_task.items():
            try:
                self.store.wait(task)
            except RuntimeError as e:
                logger.error(f"request {request_id} wait load task error. {e}")
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                num_loaded_block -= len(
                    metadata.request_meta[request_id].load_block_ids[0]
                )

        load_end_time = time.perf_counter() * 1000
        load_speed = (
            num_loaded_block
            * self.block_data_size
            / (load_end_time - load_start_time)
            / 1024
            / 1024
        )  # GB/s
        if self.metrics_config and is_load:
            ucmmetrics.update_stats(
                {
                    "load_requests_num": num_loaded_request,
                    "load_blocks_num": num_loaded_block,
                    "load_duration": load_end_time - load_start_time,
                    "load_speed": load_speed,
                }
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def _get_dump_event_handle(self) -> int:
        if not self.enable_event_sync:
            self.device.synchronize()
            return 0

        event_handle = self.device.get_event_handle()
        if event_handle == 0:
            self.device.synchronize()
        return event_handle

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        # TODO support PP
        if self.is_mla and self.tp_rank != 0:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        dump_tasks: List[Task] = []
        is_save = False
        num_saved_block = 0
        num_saved_request = 0
        total_ucm_block_ids, total_vllm_block_ids = [], []
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue
            is_save = True
            num_saved_block += len(request.dump_block_ids[0])
            num_saved_request += 1

            ucm_block_ids, vllm_block_ids = request.dump_block_ids
            if self.tp_rank != 0:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            total_ucm_block_ids.extend(ucm_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)

        if is_save:
            total_ptrs = self.kv_cache_layout.extract_block_addrs(total_vllm_block_ids)
            total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
            shard_indexs = [0] * len(total_ucm_block_ids)
            try:
                event_handle = self._get_dump_event_handle()
                save_start_time = time.perf_counter() * 1000
                task = self.store.dump_data(
                    total_ucm_block_ids, shard_indexs, total_ptrs, event_handle
                )
                dump_tasks.append(task)
            except RuntimeError as e:
                logger.error(f"dump kv cache failed. {e}")
                return

            try:
                for task in dump_tasks:
                    self.store.wait(task)
                save_end_time = time.perf_counter() * 1000
            except RuntimeError as e:
                logger.error(f"wait for dump kv cache failed.{e}")
                return

            save_speed = (
                num_saved_block
                * self.block_data_size
                / (save_end_time - save_start_time)
                / 1024
                / 1024
            )  # GB/s
            if self.metrics_config:
                ucmmetrics.update_stats(
                    {
                        "save_requests_num": num_saved_request,
                        "save_blocks_num": num_saved_block,
                        "save_duration": save_end_time - save_start_time,
                        "save_speed": save_speed,
                    },
                )

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        res = self._invalid_block_ids
        self._invalid_block_ids = set()
        return res


class UCMLayerWiseConnector(UCMDirectConnector):
    """
    This Connector means overlap:
    load l0 -> forward l0 -> save l0
               load l1    -> forward l1 -> save l1
                             load l2    -> forward l2 -> save l2
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)
        # {layer_id: {request_id: Task}}
        self.load_tasks: dict[int, dict[str, Task]] = defaultdict(dict)
        self.dump_tasks: dict[str, Task] = {}
        self.use_layerwise = True
        self.is_save = False
        self.need_load = False
        self.dump_total_ptrs: np.ndarray | None = None
        self.request_data: list[tuple[str, list, np.ndarray]] = []
        self._failure_req_ids: set[str] = set()
        logger.info("Init UCMLayerWiseConnector.")

    def _submit_request_load_tasks_for_layer(
        self,
        layer_id: int,
        local_row: int,
        metadata: "UCMConnectorMetadata",
    ) -> None:
        for request_id, ucm_block_ids, total_ptrs in self.request_data:
            if request_id in self._failure_req_ids:
                continue
            try:
                shard_indexs = [layer_id] * len(ucm_block_ids)
                layer_ptrs = total_ptrs[local_row]
                task = self.store.load_data(ucm_block_ids, shard_indexs, layer_ptrs)
                self.load_tasks[layer_id][request_id] = task
            except RuntimeError as e:
                logger.error(f"request {request_id} submit load task error. {e}")
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                self._failure_req_ids.add(request_id)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        self.load_tasks.clear()
        self.request_data.clear()
        self._failure_req_ids.clear()
        self.need_load = False

        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue

            self.need_load = True
            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self.tp_rank % self.tp_size != 0 and not self.is_mla:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            total_ptrs = self.kv_cache_layout.extract_block_addrs(
                vllm_block_ids, layer_first=True
            )
            self.request_data.append((request_id, ucm_block_ids, total_ptrs))

        if self.need_load:
            self._submit_request_load_tasks_for_layer(self.first_layer_id, 0, metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._connector_metadata:
            return
        if not self.need_load:
            return
        metadata = self._get_connector_metadata()
        current_layer_id = self.layer_name_to_id[layer_name]

        for request_id, task in self.load_tasks.get(current_layer_id, {}).items():
            try:
                self.store.wait(task)
            except RuntimeError as e:
                logger.error(f"request {request_id} wait {layer_name} load failed. {e}")
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                self._failure_req_ids.add(request_id)

        next_layer_id = current_layer_id + 1
        if next_layer_id not in self.layer_ids:
            return
        next_local_row = next_layer_id - self.first_layer_id

        self._submit_request_load_tasks_for_layer(
            next_layer_id, next_local_row, metadata
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        if not self._connector_metadata:
            return
        if self.is_mla and self.tp_rank % self.tp_size != 0:
            return

        metadata = self._get_connector_metadata()

        total_ucm_block_ids, total_vllm_block_ids = [], []
        layer_id = self.layer_name_to_id[layer_name]
        local_layer_id = layer_id - self.first_layer_id
        for _, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue

            self.is_save = True
            ucm_block_ids, vllm_block_ids = request.dump_block_ids
            if self.tp_rank % self.tp_size != 0 and local_layer_id == 0:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            total_ucm_block_ids.extend(ucm_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)

        if self.is_save:
            if self.dump_total_ptrs is None:
                self.dump_total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    total_vllm_block_ids, layer_first=True
                )
            shard_indexs = [layer_id] * len(total_ucm_block_ids)
            try:
                layer_ptrs = np.ascontiguousarray(self.dump_total_ptrs[local_layer_id])
                event_handle = self._get_dump_event_handle()
                task = self.store.dump_data(
                    total_ucm_block_ids, shard_indexs, layer_ptrs, event_handle
                )
                self.dump_tasks[layer_name] = task
            except RuntimeError as e:
                logger.error(f"submit dump task failed. {e}")

    def wait_for_save(self) -> None:
        if not self.is_save:
            return
        try:
            for layer_name in self.kv_caches:
                if layer_name in self.dump_tasks:
                    self.store.wait(self.dump_tasks[layer_name])
        except RuntimeError as e:
            logger.error(f"wait for dump kv cache failed. {e}")
        self.dump_tasks.clear()
        self.is_save = False
        self.dump_total_ptrs = None
        if self.enable_event_sync:
            self.device.destroy_event_handles()


class UCMCPConnector(UCMLayerWiseConnector):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)
        self.use_layerwise = self.launch_config.get("use_layerwise", False)

        try:
            from vllm.distributed import get_dcp_group, get_pcp_group
        except ImportError as e:
            raise ImportError(
                "Please check if the current vLLM version supports DCP and PCP features."
            ) from e

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = (
                get_pcp_group().rank_in_group if self.pcp_world_size > 1 else 0
            )
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.pcp_world_size = 1
            self.pcp_rank = 0
        self.cp_world_size = (
            self._vllm_config.parallel_config.prefill_context_parallel_size
            * self._vllm_config.parallel_config.decode_context_parallel_size
        )
        self.current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank
        old_tp_size = vllm_config.parallel_config.tensor_parallel_size
        logger.info(
            f"pcp_world_size: {self.pcp_world_size}, pcp_rank: {self.pcp_rank}, dcp_world_size: {self.dcp_world_size}, dcp_rank: {self.dcp_rank}"
        )

        self.tp_rank %= self.tp_size
        self.tp_rank //= self.dcp_world_size
        if not self.is_mla:
            vllm_config.parallel_config.tensor_parallel_size //= self.dcp_world_size

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-size connector
            self.store = self._create_store(None)
        else:
            self.request_hasher = RequestHasher(vllm_config, self.tp_rank)
        vllm_config.parallel_config.tensor_parallel_size = old_tp_size
        self.block_size *= self.cp_world_size
        logger.info("Init UCMCPConnector.")

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        # When DCP/PCP features are enabled,
        # the blocks that each device can process are [current_rank :: cp_world_size],
        # where current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank.
        for _, request in connector_metadata.request_meta.items():
            if len(request.load_block_ids[0]) > 0:
                ucm_block_ids, vllm_block_ids = request.load_block_ids
                ucm_block_ids = ucm_block_ids[self.current_rank :: self.cp_world_size]
                request.load_block_ids = (ucm_block_ids, vllm_block_ids)

            if len(request.dump_block_ids[0]) > 0:
                ucm_block_ids, vllm_block_ids = request.dump_block_ids
                ucm_block_ids = ucm_block_ids[self.current_rank :: self.cp_world_size]
                request.dump_block_ids = (ucm_block_ids, vllm_block_ids)
        super().bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context, **kwargs):
        if self.use_layerwise:
            super().start_load_kv(forward_context, **kwargs)
        else:
            super(UCMLayerWiseConnector, self).start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.use_layerwise:
            super().wait_for_layer_load(layer_name)
        else:
            pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if self.use_layerwise:
            super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        else:
            pass

    def wait_for_save(self):
        if self.use_layerwise:
            super().wait_for_save()
        else:
            super(UCMLayerWiseConnector, self).wait_for_save()


class UCMPDConnector(UCMDirectConnector):
    """
    This Connector means overlap (especially for Decode Instance):
    step (req0,1,2) forward -> step (req0,1,2,3) forward
    load req3               -> load req4
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        raise NotImplementedError

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        raise NotImplementedError


class UCMMockConnector(UCMDirectConnector):
    """
    This Connector can control hit ratio, for example: if your hit ratio is 100%,
    you can set "hit_ratio" by config or env_vars, then get_num_new_matched_tokens()
    will reduce hit_tokens under the hit_ratio you set.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)
        self._hit_ratio = float(self.launch_config["hit_ratio"])
        logger.info(f"hit_ratio: {self._hit_ratio}")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        hit_tokens, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        expect_hit_tokens = int(self._hit_ratio * request.num_prompt_tokens)
        if hit_tokens <= expect_hit_tokens:
            return hit_tokens, False
        expect_hit_block_num = expect_hit_tokens // self.block_size
        request_meta = self.requests_meta[request.request_id]
        request_meta.total_hit_block_num = expect_hit_block_num
        request_meta.hbm_hit_block_num = min(
            expect_hit_block_num, request_meta.hbm_hit_block_num
        )

        logger.info(
            "Hijacked By MockConnector,"
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(request_meta.ucm_block_ids)}, "
            f"hit hbm: {request_meta.hbm_hit_block_num}, "
            f"hit external: {request_meta.total_hit_block_num - request_meta.hbm_hit_block_num}"
        )

        return expect_hit_block_num * self.block_size, False


class UCMLiteConnector(UCMDirectConnector):
    def __init__(self, vllm_config, role):
        ucm_config = Config(vllm_config.kv_transfer_config)
        launch_config = ucm_config.get_config()
        enable_record_traces = launch_config.get("enable_record_traces", False)
        persist_token_threshold = launch_config.get("persist_token_threshold", 0)
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmPipelineStore",
                    "ucm_connector_config": {
                        "store_pipeline": "Fake",
                        "share_buffer_enable": True,
                        "buffer_number": 244032232,
                    },
                }
            ],
            "enable_record_traces": enable_record_traces,
            "persist_token_threshold": persist_token_threshold,
            "use_lite": True,
        }
        super().__init__(vllm_config, role)
        self.total_block_nums = 0
        self.total_hit_block_nums = 0
        logger.info("Init UCMLiteConnector.")

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        super().get_num_new_matched_tokens(request, num_computed_tokens)

        external_hit_blocks = 0
        req_blocks_num = len(request.all_token_ids) // self.hash_block_size
        if req_blocks_num < 1:
            return 0, False
        self.total_block_nums += req_blocks_num
        if request.request_id in self.requests_meta:
            request_meta = self.requests_meta[request.request_id]
            external_hit_blocks = (
                request_meta.total_hit_block_num - request_meta.hbm_hit_block_num
            )
            need_dump_blks = request_meta.ucm_block_ids[
                request_meta.total_hit_block_num :
            ]
            shard_indexs = [0] * len(need_dump_blks)
            total_ptrs = [[0]] * len(need_dump_blks)
            try:
                task = self.store.dump_data(need_dump_blks, shard_indexs, total_ptrs)
                self.store.wait(task)
            except RuntimeError as e:
                logger.error(f"request {request.request_id} wait dump task error. {e}")
            self.requests_meta[request.request_id] = RequestMeta()

        self.total_hit_block_nums += external_hit_blocks

        logger.info(
            f"req external hit rate: {(external_hit_blocks / req_blocks_num):.2f}, "
            f"total external hit rate: {(self.total_hit_block_nums / self.total_block_nums):.2f}"
        )
        return 0, False


DeepSeekV4PackedRow = tuple[list[int], ...]
DeepSeekV4PackedRows = list[DeepSeekV4PackedRow]


@dataclass
class DeepSeekV4RequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    token_processed: int = 0
    packed_block_ids: dict[int, DeepSeekV4PackedRow] = field(default_factory=dict)


@dataclass
class DeepSeekV4RequestDispatchMeta:
    load_block_ids: tuple[list[bytes], DeepSeekV4PackedRows]
    dump_block_ids: tuple[list[bytes], DeepSeekV4PackedRows]


@dataclass
class UCMDeepSeekV4ConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, DeepSeekV4RequestDispatchMeta] = field(
        default_factory=dict
    )


@dataclass
class DeepSeekV4LoadTask:
    request_id: str
    label: str
    store: UcmKVStoreBaseV1
    task: Task
    keys: list[bytes]
    packed_group_block_ids: DeepSeekV4PackedRows
    ptrs: np.ndarray
    capture_payload: bool
    byte_count: int


@dataclass
class DeepSeekV4DumpTask:
    label: str
    store: UcmKVStoreBaseV1
    task: Task
    key_count: int
    byte_count: int


class UCMDeepSeekV4Connector(UCMDirectConnector):
    """UCM connector for DeepSeek V4 hybrid KV cache groups.

    DeepSeek V4 uses five KV cache groups with different block sizes. This
    connector stores one 256-token prefix block as one CacheStore block. Each
    stored block packs the full group-0 cache for that prefix block plus the
    group-1/2/3/4 tail blocks needed to reuse the prefix boundary.
    """

    GROUP_BLOCK_SIZES = (256, 64, 64, 4, 8)
    # Conservative HBM-aligned tails for real allocated blocks:
    # - SWA exposes only the previous 128-token window at a 256-token boundary.
    # - C4A state carries the previous 8-token state window.
    # - C128 state carries the previous 128-token state window.
    GROUP_TAIL_BLOCKS = (None, 2, 2, 2, 16)
    HASH_BLOCK_SIZE = 256

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        self._defer_scheduler_store = True
        super().__init__(vllm_config, role)
        self.hash_block_size = self.HASH_BLOCK_SIZE
        self.block_size = self.HASH_BLOCK_SIZE
        self.group_layouts: dict[int, DeepSeekV4GroupKVCacheLayout] = {}
        self._packed_scratch_views: dict[tuple[int, int], list[torch.Tensor]] = {}
        self.group0_store: Optional[UcmKVStoreBaseV1] = None
        self.requests_meta: dict[str, DeepSeekV4RequestMeta] = {}
        if role == KVConnectorRole.SCHEDULER:
            self.store = self._create_packed_store(None)
            self.group0_store = self._create_group0_store(None)
        logger.info("Init UCMDeepSeekV4Connector.")

    def _create_packed_store(
        self,
        group_layouts: Optional[dict[int, DeepSeekV4GroupKVCacheLayout]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        tensor_size_list = None
        if self._role == KVConnectorRole.WORKER:
            if group_layouts is None:
                raise RuntimeError("Worker DeepSeek V4 packed store needs layouts.")
            tensor_size_list = self._packed_tensor_size_list(group_layouts)
        return self._create_deepseek_store(
            "packed",
            "packed",
            tensor_size_list,
            cpu_affinity_cores,
        )

    def _create_group0_store(
        self,
        group_layouts: Optional[dict[int, DeepSeekV4GroupKVCacheLayout]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        tensor_size_list = None
        if self._role == KVConnectorRole.WORKER:
            if group_layouts is None:
                raise RuntimeError("Worker DeepSeek V4 group0 store needs layouts.")
            group0_layout = group_layouts.get(0)
            if group0_layout is None:
                raise RuntimeError("Worker DeepSeek V4 group0 layout is missing.")
            tensor_size_list = group0_layout.tensor_size_list
        return self._create_deepseek_store(
            "group0",
            "group0",
            tensor_size_list,
            cpu_affinity_cores,
        )

    def _base_store_config(
        self,
        store_suffix: str,
    ) -> tuple[str, Optional[str], dict[str, object]]:
        if len(self.connector_configs) != 1:
            raise RuntimeError(
                f"Expected exactly one connector config, "
                f"but got {len(self.connector_configs)}: "
                f"{self.connector_configs}"
            )

        name = self.connector_configs[0]["ucm_connector_name"]
        module_path = self.connector_configs[0].get("ucm_connector_module_path", None)
        config = copy.deepcopy(self.connector_configs[0]["ucm_connector_config"])
        config.setdefault("store_pipeline", "Cache|Empty")
        config.setdefault("share_buffer_enable", True)
        if isinstance(config.get("storage_backends"), str):
            config["storage_backends"] = [
                path for path in config["storage_backends"].split(":")
            ]
        config["unique_id"] = f"{self.engine_id}_dsv4_{store_suffix}"
        dp_rank = self._vllm_config.parallel_config.data_parallel_rank
        config["posix_gc_enable"] = (
            self._role != KVConnectorRole.WORKER and dp_rank == 0
        )
        return name, module_path, config

    def _create_deepseek_store(
        self,
        label: str,
        store_suffix: str,
        tensor_size_list: Optional[list[int]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        name, module_path, config = self._base_store_config(store_suffix)
        if self._role == KVConnectorRole.WORKER:
            if tensor_size_list is None:
                raise RuntimeError(
                    f"Worker DeepSeek V4 {label} store needs tensor sizes."
                )
            config["device_id"] = self.local_rank
            config["tensor_size_list"] = tensor_size_list
            config["shard_size"] = int(sum(tensor_size_list))
            config["block_size"] = int(sum(tensor_size_list))
            config["local_rank_size"] = 1
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
        logger.info(
            f"create DeepSeek V4 {label} {name} with config: "
            f"{self._summarize_store_config(config)}"
        )
        return UcmConnectorFactoryV1.create_connector(name, config, module_path)

    @staticmethod
    def _summarize_store_config(config: dict[str, object]) -> dict[str, object]:
        summary = dict(config)
        tensor_size_list = summary.pop("tensor_size_list", None)
        if tensor_size_list is not None:
            tensor_sizes = [int(size) for size in tensor_size_list]
            summary["tensor_count"] = len(tensor_sizes)
            summary["tensor_bytes"] = sum(tensor_sizes)
        return summary

    @staticmethod
    def _is_group0_name(name: str) -> bool:
        return name.endswith(".attn") or name.endswith(".attn.indexer.k_cache")

    @staticmethod
    def _is_swa_name(name: str) -> bool:
        return name.endswith(".attn.swa_cache")

    @staticmethod
    def _is_group3_name(name: str) -> bool:
        return name.endswith(".attn.indexer.compressor.state_cache")

    @staticmethod
    def _is_group4_name(name: str) -> bool:
        return (
            name.endswith(".attn.compressor.state_cache")
            and ".indexer." not in name
        )

    def _split_kv_caches_by_group(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> dict[int, dict[str, torch.Tensor]]:
        ordered = dict(
            sorted(
                kv_caches.items(),
                key=lambda item: (extract_layer_index(item[0]), item[0]),
            )
        )
        groups: dict[int, dict[str, torch.Tensor]] = {i: {} for i in range(5)}
        swa_items: list[tuple[str, torch.Tensor]] = []
        for name, value in ordered.items():
            if self._is_group3_name(name):
                groups[3][name] = value
            elif self._is_group4_name(name):
                groups[4][name] = value
            elif self._is_swa_name(name):
                swa_items.append((name, value))
            elif self._is_group0_name(name):
                groups[0][name] = value

        # vLLM splits DeepSeek V4 SWA groups in an interleaved fashion.
        groups[1].update(dict(swa_items[0::2]))
        groups[2].update(dict(swa_items[1::2]))
        return groups

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = kv_caches
        self.device = create_device()

        enable_affinity = os.getenv("VLLM_CPU_AFFINITY") == "1"
        worker_cores, store_cores = (
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        grouped = self._split_kv_caches_by_group(kv_caches)
        for group_id, group_caches in grouped.items():
            if not group_caches:
                logger.warning(f"DeepSeek V4 KV cache group {group_id} is empty.")
                continue
            layout = DeepSeekV4GroupKVCacheLayout(group_caches)
            self.group_layouts[group_id] = layout

        self.store = self._create_packed_store(self.group_layouts, store_cores)
        self.group0_store = self._create_group0_store(
            self.group_layouts,
            store_cores,
        )

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

    def _packed_key(self, canonical_hash: bytes) -> bytes:
        return self.request_hasher((b"deepseek_v4_packed", canonical_hash))

    def _local_stub_enabled(self) -> bool:
        return os.getenv("UCM_DEEPSEEK_V4_LOCAL_STUB", "0") == "1"

    def _local_stub_dir(self) -> str:
        stub_dir = os.getenv(
            "UCM_DEEPSEEK_V4_LOCAL_STUB_DIR",
            os.path.join(os.getcwd(), "deepseek_v4_ucm_stub"),
        )
        os.makedirs(stub_dir, exist_ok=True)
        return stub_dir

    def _local_stub_path(self, key: bytes) -> str:
        return os.path.join(self._local_stub_dir(), f"{key.hex()}.pt")

    def _local_stub_lookup_on_prefix(self, keys: list[bytes]) -> int:
        for idx, key in enumerate(keys):
            if not os.path.exists(self._local_stub_path(key)):
                return idx - 1
        return len(keys) - 1

    def _packed_tensor_size_list(
        self, group_layouts: dict[int, DeepSeekV4GroupKVCacheLayout]
    ) -> list[int]:
        tensor_size_list: list[int] = []
        for group_id in range(len(self.GROUP_BLOCK_SIZES)):
            layout = group_layouts.get(group_id)
            if layout is None:
                continue
            repeat = 1 if group_id == 0 else self.GROUP_TAIL_BLOCKS[group_id]
            assert repeat is not None
            tensor_size_list.extend(layout.tensor_size_list * repeat)
        return tensor_size_list

    @staticmethod
    def _group0_only_rows(
        packed_group_block_ids: DeepSeekV4PackedRows,
    ) -> DeepSeekV4PackedRows:
        return [
            (list(group_block_ids[0]),)
            for group_block_ids in packed_group_block_ids
        ]

    @staticmethod
    def _timing_enabled() -> bool:
        return os.getenv("UCM_DEEPSEEK_V4_TIMING", "0") == "1"

    @staticmethod
    def _debug_enabled() -> bool:
        return os.getenv("UCM_DEEPSEEK_V4_DEBUG", "0") == "1"

    def _packed_row_bytes(self, rows: DeepSeekV4PackedRows) -> int:
        total = 0
        for row in rows:
            for group_id, group_block_ids in enumerate(row):
                layout = self.group_layouts.get(group_id)
                if layout is None:
                    continue
                total += len(group_block_ids) * layout.shard_size
        return total

    def _required_group_block_indices(
        self,
        group_id: int,
        total_hit_tokens: int,
        min_external_tokens: int = 0,
    ) -> list[int]:
        if group_id == 0:
            start = min_external_tokens // self.HASH_BLOCK_SIZE
            end = total_hit_tokens // self.HASH_BLOCK_SIZE
            return list(range(start, end))

        group_block_size = self.GROUP_BLOCK_SIZES[group_id]
        total_group_blocks = total_hit_tokens // group_block_size
        tail_blocks = self.GROUP_TAIL_BLOCKS[group_id]
        assert tail_blocks is not None
        start = max(0, total_group_blocks - tail_blocks)
        start = max(start, min_external_tokens // group_block_size)
        return list(range(start, total_group_blocks))

    def _packed_group_indices(self, canonical_block_idx: int) -> list[list[int]]:
        end_tokens = (canonical_block_idx + 1) * self.HASH_BLOCK_SIZE
        return [[canonical_block_idx]] + [
            self._required_group_block_indices(group_id, end_tokens, 0)
            for group_id in range(1, len(self.GROUP_BLOCK_SIZES))
        ]

    def _scratch_block_tensor_views(
        self,
        group_id: int,
        block_pos: int,
    ) -> list[torch.Tensor]:
        key = (group_id, block_pos)
        scratch_views = self._packed_scratch_views.get(key)
        if scratch_views is None:
            layout = self.group_layouts[group_id]
            scratch_views = [
                torch.empty_like(tensor)
                for tensor in layout.extract_block_tensor_views([0])
            ]
            self._packed_scratch_views[key] = scratch_views
        return scratch_views

    def _scratch_block_addrs(self, group_id: int, block_pos: int) -> np.ndarray:
        return np.asarray(
            [
                tensor.data_ptr()
                for tensor in self._scratch_block_tensor_views(group_id, block_pos)
            ],
            dtype=np.uint64,
        )

    def _extract_packed_addrs(
        self,
        packed_group_block_ids: DeepSeekV4PackedRows,
        scratch_for_missing: bool = False,
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        for group_block_ids in packed_group_block_ids:
            row_parts: list[np.ndarray] = []
            for group_id, selected_ids in enumerate(group_block_ids):
                layout = self.group_layouts.get(group_id)
                if layout is None:
                    continue
                if not selected_ids:
                    continue
                for block_pos, block_id in enumerate(selected_ids):
                    if block_id < 0:
                        if not scratch_for_missing:
                            raise ValueError(
                                f"DeepSeek V4 packed group {group_id} block "
                                f"position {block_pos} needs a scratch target."
                            )
                        row_parts.append(
                            self._scratch_block_addrs(group_id, block_pos)
                        )
                    else:
                        row_parts.append(
                            layout.extract_block_addrs([block_id]).reshape(-1)
                        )
            if not row_parts:
                raise ValueError("DeepSeek V4 packed pointer row is empty.")
            rows.append(np.concatenate(row_parts).astype(np.uint64, copy=False))
        if not rows:
            return np.empty((0, 0), dtype=np.uint64)
        return np.vstack(rows)

    def _extract_packed_tensor_views(
        self,
        packed_group_block_ids: DeepSeekV4PackedRows,
        scratch_for_missing: bool = False,
    ) -> list[list[torch.Tensor]]:
        rows: list[list[torch.Tensor]] = []
        for group_block_ids in packed_group_block_ids:
            row: list[torch.Tensor] = []
            for group_id, selected_ids in enumerate(group_block_ids):
                layout = self.group_layouts.get(group_id)
                if layout is None or not selected_ids:
                    continue
                actual_ids: list[int] = []
                for block_pos, block_id in enumerate(selected_ids):
                    if block_id < 0:
                        if actual_ids:
                            row.extend(layout.extract_block_tensor_views(actual_ids))
                            actual_ids = []
                        if not scratch_for_missing:
                            raise ValueError(
                                f"DeepSeek V4 packed group {group_id} block "
                                f"position {block_pos} needs a scratch target."
                            )
                        row.extend(
                            self._scratch_block_tensor_views(group_id, block_pos)
                        )
                    else:
                        actual_ids.append(block_id)
                if actual_ids:
                    row.extend(layout.extract_block_tensor_views(actual_ids))
            if not row:
                raise ValueError("DeepSeek V4 packed tensor row is empty.")
            rows.append(row)
        return rows

    def _dump_local_stub(
        self,
        keys: list[bytes],
        packed_group_block_ids: DeepSeekV4PackedRows,
    ) -> None:
        rows = self._extract_packed_tensor_views(packed_group_block_ids)
        tensor_size_list = self._packed_tensor_size_list(self.group_layouts)
        for key, row in zip(keys, rows):
            if len(row) != len(tensor_size_list):
                raise ValueError(
                    f"DeepSeek V4 local stub dump row has {len(row)} tensors, "
                    f"expected {len(tensor_size_list)}."
                )
            payload = {
                "key_hex": key.hex(),
                "tensor_size_list": tensor_size_list,
                "tensors": [tensor.detach().cpu().clone() for tensor in row],
            }
            path = self._local_stub_path(key)
            tmp_path = f"{path}.{os.getpid()}.tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)

    def _load_local_stub(
        self,
        keys: list[bytes],
        packed_group_block_ids: DeepSeekV4PackedRows,
    ) -> None:
        rows = self._extract_packed_tensor_views(
            packed_group_block_ids,
            scratch_for_missing=True,
        )
        for key, row in zip(keys, rows):
            path = self._local_stub_path(key)
            payload = torch.load(path, map_location="cpu")
            tensors = payload.get("tensors", [])
            if len(tensors) != len(row):
                raise ValueError(
                    f"DeepSeek V4 local stub load row for {key.hex()} has "
                    f"{len(tensors)} tensors, expected {len(row)}."
                )
            for idx, (dst, src) in enumerate(zip(row, tensors)):
                if (
                    dst.numel() * dst.element_size()
                    != src.numel() * src.element_size()
                ):
                    raise ValueError(
                        f"DeepSeek V4 local stub tensor {idx} size mismatch: "
                        f"dst={dst.shape}/{dst.dtype}, src={src.shape}/{src.dtype}."
                    )
                dst.copy_(src.to(device=dst.device, dtype=dst.dtype).view_as(dst))

    def _capture_enabled(self) -> bool:
        return os.getenv("UCM_DEEPSEEK_V4_CAPTURE", "0") == "1"

    def _capture_tensor_enabled(self) -> bool:
        return os.getenv("UCM_DEEPSEEK_V4_CAPTURE_TENSORS", "1") != "0"

    def _capture_dir(self) -> str:
        capture_dir = os.getenv(
            "UCM_DEEPSEEK_V4_CAPTURE_DIR",
            os.path.join(os.getcwd(), "deepseek_v4_ucm_capture"),
        )
        os.makedirs(capture_dir, exist_ok=True)
        return capture_dir

    @staticmethod
    def _safe_capture_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

    def _capture_path(self, stage: str, request_id: str) -> str:
        rank = "scheduler" if self._role == KVConnectorRole.SCHEDULER else self.tp_rank
        name = (
            f"{int(time.time() * 1000000)}_"
            f"{self._safe_capture_name(stage)}_"
            f"rank{rank}_pid{os.getpid()}_"
            f"{self._safe_capture_name(request_id)}.pt"
        )
        return os.path.join(self._capture_dir(), name)

    def _capture_scheduler_state(
        self,
        stage: str,
        request_id: str,
        req_meta: DeepSeekV4RequestMeta,
        extra: Optional[dict] = None,
    ) -> None:
        if not self._capture_enabled():
            return
        payload = {
            "stage": stage,
            "role": "scheduler",
            "request_id": request_id,
            "hbm_hit_block_num": req_meta.hbm_hit_block_num,
            "total_hit_block_num": req_meta.total_hit_block_num,
            "num_token_ids": req_meta.num_token_ids,
            "token_processed": req_meta.token_processed,
            "ucm_block_ids_hex": [key.hex() for key in req_meta.ucm_block_ids],
            "packed_keys_hex": [
                self._packed_key(key).hex() for key in req_meta.ucm_block_ids
            ],
            "packed_block_ids": {
                idx: [list(group_ids) for group_ids in group_block_ids]
                for idx, group_block_ids in req_meta.packed_block_ids.items()
            },
            "group_block_sizes": list(self.GROUP_BLOCK_SIZES),
            "group_tail_blocks": list(self.GROUP_TAIL_BLOCKS),
            "extra": extra or {},
        }
        path = self._capture_path(stage, request_id)
        torch.save(payload, path)
        logger.info(f"Captured DeepSeek V4 scheduler metadata to {path}")

    def _capture_worker_payload(
        self,
        stage: str,
        request_id: str,
        keys: list[bytes],
        packed_group_block_ids: DeepSeekV4PackedRows,
        ptrs: Optional[np.ndarray] = None,
    ) -> None:
        if not self._capture_enabled():
            return

        payload: dict[str, object] = {
            "stage": stage,
            "role": "worker",
            "request_id": request_id,
            "pid": os.getpid(),
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "local_rank": self.local_rank,
            "keys_hex": [key.hex() for key in keys],
            "packed_group_block_ids": [
                [list(group_ids) for group_ids in row]
                for row in packed_group_block_ids
            ],
            "group_block_sizes": list(self.GROUP_BLOCK_SIZES),
            "group_tail_blocks": list(self.GROUP_TAIL_BLOCKS),
            "tensor_size_list": self._packed_tensor_size_list(self.group_layouts),
            "shard_size": int(sum(self._packed_tensor_size_list(self.group_layouts))),
            "ptr_shape": tuple(ptrs.shape) if ptrs is not None else None,
        }

        if self._capture_tensor_enabled():
            rows = []
            for row_idx, group_block_ids in enumerate(packed_group_block_ids):
                groups = []
                for group_id, selected_ids in enumerate(group_block_ids):
                    layout = self.group_layouts.get(group_id)
                    if layout is None or not selected_ids:
                        groups.append(
                            {
                                "group_id": group_id,
                                "block_ids": list(selected_ids),
                                "tensors": [],
                            }
                        )
                        continue
                    groups.append(
                        {
                            "group_id": group_id,
                            "block_ids": list(selected_ids),
                            "tensors": []
                            if any(block_id < 0 for block_id in selected_ids)
                            else layout.extract_block_tensors(selected_ids),
                        }
                    )
                rows.append({"row_idx": row_idx, "groups": groups})
            payload["rows"] = rows

        path = self._capture_path(stage, request_id)
        torch.save(payload, path)
        logger.info(f"Captured DeepSeek V4 worker payload to {path}")

    def _select_packed_group_block_ids(
        self,
        canonical_block_idx: int,
        blocks: "KVCacheBlocks",
        allow_null_tail: bool = False,
        ) -> DeepSeekV4PackedRow:
        selected: list[list[int]] = []
        group_indices_by_group = self._packed_group_indices(canonical_block_idx)
        for group_id, group_indices in enumerate(group_indices_by_group):
            group_selected: list[int] = []
            if group_id >= len(blocks.blocks):
                if group_indices:
                    raise ValueError(
                        f"DeepSeek V4 packed group {group_id} is missing from "
                        f"KVCacheBlocks for canonical block {canonical_block_idx}."
                    )
                selected.append(group_selected)
                continue

            group_blocks = blocks.blocks[group_id]
            for group_block_idx in group_indices:
                if group_block_idx >= len(group_blocks):
                    raise ValueError(
                        f"DeepSeek V4 packed group {group_id} block index "
                        f"{group_block_idx} is out of range "
                        f"(len={len(group_blocks)}) for canonical block "
                        f"{canonical_block_idx}."
                    )
                block = group_blocks[group_block_idx]
                if block.is_null:
                    if allow_null_tail and group_id != 0:
                        group_selected.append(-1)
                        continue
                    raise ValueError(
                        f"DeepSeek V4 packed group {group_id} block index "
                        f"{group_block_idx} maps to a null HBM block for "
                        f"canonical block {canonical_block_idx}."
                    )
                group_selected.append(block.block_id)
            selected.append(group_selected)
        return tuple(selected)

    def _record_packed_block_ids(
        self,
        req_meta: DeepSeekV4RequestMeta,
        blocks: "KVCacheBlocks",
        end_block: int,
    ) -> None:
        for canonical_block_idx in range(end_block):
            if canonical_block_idx in req_meta.packed_block_ids:
                continue
            allow_null_tail = canonical_block_idx < req_meta.total_hit_block_num - 1
            req_meta.packed_block_ids[canonical_block_idx] = (
                self._select_packed_group_block_ids(
                    canonical_block_idx,
                    blocks,
                    allow_null_tail=allow_null_tail,
                )
            )

    def _lookup_external_hit_blocks(self, external_keys: list[bytes]) -> int:
        if self._local_stub_enabled():
            return self._local_stub_lookup_on_prefix(external_keys) + 1

        packed_hit_blocks = self.store.lookup_on_prefix(external_keys) + 1
        if packed_hit_blocks <= 1:
            return packed_hit_blocks

        if self.group0_store is None:
            raise RuntimeError("DeepSeek V4 group0 store is not initialized.")
        group0_hit_blocks = self.group0_store.lookup_on_prefix(external_keys) + 1
        return min(packed_hit_blocks, group0_hit_blocks + 1)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert num_computed_tokens % self.HASH_BLOCK_SIZE == 0
        hbm_hit_block_num = num_computed_tokens // self.HASH_BLOCK_SIZE
        canonical_hashes = self.generate_hash(
            self.HASH_BLOCK_SIZE, request.all_token_ids, self._seed
        )

        if self.persist_token_threshold > request.num_tokens:
            return 0, False

        external_keys = [
            self._packed_key(block_hash)
            for block_hash in canonical_hashes[hbm_hit_block_num:]
        ]
        if not external_keys:
            return 0, False

        try:
            external_hit_blocks = self._lookup_external_hit_blocks(external_keys)
        except Exception as e:
            external_hit_blocks = 0
            logger.error(
                f"request {request.request_id} DeepSeek V4 packed lookup error. "
                f"{type(e).__name__}: {e}"
            )

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks
        external_hit_tokens = external_hit_blocks * self.HASH_BLOCK_SIZE
        num_total_hit_tokens = total_hit_block_num * self.HASH_BLOCK_SIZE
        if num_total_hit_tokens == request.num_tokens:
            external_hit_tokens -= 1

        self.requests_meta[request.request_id] = DeepSeekV4RequestMeta(
            ucm_block_ids=canonical_hashes,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
        )
        self._capture_scheduler_state(
            "lookup",
            request.request_id,
            self.requests_meta[request.request_id],
            {
                "num_computed_tokens": num_computed_tokens,
                "external_hit_blocks": external_hit_blocks,
                "external_hit_tokens": external_hit_tokens,
                "request_num_tokens": request.num_tokens,
                "all_token_ids": list(request.all_token_ids),
            },
        )

        logger.info_once(
            f"DeepSeek V4 request_id: {request.request_id}, "
            f"total_blocks_num: {len(canonical_hashes)}, "
            f"hit hbm: {hbm_hit_block_num}, "
            f"hit external: {external_hit_blocks}"
        )
        return external_hit_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_meta = self.requests_meta.get(request.request_id)
        if req_meta is None:
            return

        max_full_blocks = req_meta.num_token_ids // self.HASH_BLOCK_SIZE
        group0_blocks = len(blocks.blocks[0]) if blocks.blocks else 0
        end_block = min(max_full_blocks, group0_blocks)
        if end_block == 0:
            return

        try:
            self._record_packed_block_ids(req_meta, blocks, end_block)
        except Exception as e:
            logger.error(
                f"request {request.request_id} record DeepSeek V4 HBM-aligned "
                f"block ids failed. {type(e).__name__}: {e}"
            )
            raise

        if self._debug_enabled():
            block_lens = [len(group) for group in blocks.blocks]
            null_counts = [
                sum(1 for block in group if block.is_null)
                for group in blocks.blocks
            ]
            selected_lens = {
                idx: [len(group) for group in group_ids]
                for idx, group_ids in sorted(req_meta.packed_block_ids.items())
            }
            logger.info(
                f"DeepSeek V4 HBM block map request_id={request.request_id}, "
                f"num_external_tokens={num_external_tokens}, "
                f"block_lens={block_lens}, null_counts={null_counts}, "
                f"selected_lens={selected_lens}"
            )
        self._capture_scheduler_state(
            "after_alloc",
            request.request_id,
            req_meta,
            {
                "num_external_tokens": num_external_tokens,
                "block_lens": [len(group) for group in blocks.blocks],
                "null_counts": [
                    sum(1 for block in group if block.is_null)
                    for group in blocks.blocks
                ],
                "block_ids": [
                    [block.block_id for block in group]
                    for group in blocks.blocks
                ],
                "is_null": [
                    [block.is_null for block in group]
                    for group in blocks.blocks
                ],
            },
        )

    def _make_dispatch_meta(
        self,
        request_id: str,
        req_meta: DeepSeekV4RequestMeta,
        new_tokens: int,
        need_load: bool,
    ) -> DeepSeekV4RequestDispatchMeta:
        load_keys: list[bytes] = []
        load_group_block_ids: DeepSeekV4PackedRows = []
        if need_load and req_meta.total_hit_block_num > req_meta.hbm_hit_block_num:
            load_indices = list(
                range(req_meta.hbm_hit_block_num, req_meta.total_hit_block_num)
            )
            load_keys = [
                self._packed_key(req_meta.ucm_block_ids[idx])
                for idx in load_indices
            ]
            load_group_block_ids = [
                req_meta.packed_block_ids[idx]
                for idx in load_indices
            ]
            self._capture_scheduler_state(
                "dispatch_load",
                request_id,
                req_meta,
                {"load_indices": load_indices},
            )

        dump_keys: list[bytes] = []
        dump_group_block_ids: DeepSeekV4PackedRows = []
        if req_meta.token_processed < req_meta.num_token_ids:
            start_block = req_meta.token_processed // self.HASH_BLOCK_SIZE
            end_block = (req_meta.token_processed + new_tokens) // self.HASH_BLOCK_SIZE
            if end_block > start_block:
                dump_indices = list(range(start_block, end_block))
                dump_keys = [
                    self._packed_key(req_meta.ucm_block_ids[idx])
                    for idx in dump_indices
                ]
                dump_group_block_ids = [
                    req_meta.packed_block_ids[idx]
                    for idx in dump_indices
                ]
                self._capture_scheduler_state(
                    "dispatch_dump",
                    request_id,
                    req_meta,
                    {"dump_indices": dump_indices},
                )
            req_meta.token_processed += new_tokens

        return DeepSeekV4RequestDispatchMeta(
            (load_keys, load_group_block_ids),
            (dump_keys, dump_group_block_ids),
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta: dict[str, DeepSeekV4RequestDispatchMeta] = {}

        for request in scheduler_output.scheduled_new_reqs:
            req_meta = self.requests_meta.get(request.req_id)
            if req_meta:
                requests_dispatch_meta[request.req_id] = self._make_dispatch_meta(
                    request.req_id,
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request.req_id],
                    True,
        )

        cached = scheduler_output.scheduled_cached_reqs
        for request_id in cached.req_ids:
            req_meta = self.requests_meta.get(request_id)
            if not req_meta:
                continue
            resumed = request_id in cached.resumed_req_ids
            requests_dispatch_meta[request_id] = self._make_dispatch_meta(
                request_id,
                req_meta,
                scheduler_output.num_scheduled_tokens[request_id],
                resumed,
            )

        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMDeepSeekV4ConnectorMetadata(requests_dispatch_meta)

    def _submit_load_task(
        self,
        request_id: str,
        label: str,
        store: UcmKVStoreBaseV1,
        keys: list[bytes],
        packed_group_block_ids: DeepSeekV4PackedRows,
        ptrs: np.ndarray,
        capture_payload: bool,
        timing: bool,
    ) -> DeepSeekV4LoadTask:
        shard_indexs = [0] * len(keys)
        submit_start = time.perf_counter()
        task = store.load_data(keys, shard_indexs, ptrs)
        byte_count = self._packed_row_bytes(packed_group_block_ids)
        if timing:
            logger.info(
                f"DeepSeek V4 {label} load submit request_id={request_id} "
                f"keys={len(keys)} bytes={byte_count} "
                f"elapsed_s={time.perf_counter() - submit_start:.6f}"
            )
        return DeepSeekV4LoadTask(
            request_id=request_id,
            label=label,
            store=store,
            task=task,
            keys=keys,
            packed_group_block_ids=packed_group_block_ids,
            ptrs=ptrs,
            capture_payload=capture_payload,
            byte_count=byte_count,
        )

    def _wait_load_task(
        self,
        load_task: DeepSeekV4LoadTask,
        timing: bool,
    ) -> None:
        wait_start = time.perf_counter()
        try:
            load_task.store.wait(load_task.task)
            if timing:
                logger.info(
                    f"DeepSeek V4 {load_task.label} load wait "
                    f"request_id={load_task.request_id} "
                    f"keys={len(load_task.keys)} bytes={load_task.byte_count} "
                    f"elapsed_s={time.perf_counter() - wait_start:.6f}"
                )
            if load_task.capture_payload:
                self._capture_worker_payload(
                    "load_after",
                    load_task.request_id,
                    load_task.keys,
                    load_task.packed_group_block_ids,
                    load_task.ptrs,
                )
        except Exception as e:
            logger.error(
                f"request {load_task.request_id} wait DeepSeek V4 packed load "
                f"task label={load_task.label} "
                f"elapsed_s={time.perf_counter() - wait_start:.6f} "
                f"error. {type(e).__name__}: {e}"
            )

    def _submit_dump_task(
        self,
        label: str,
        store: UcmKVStoreBaseV1,
        keys: list[bytes],
        ptrs: np.ndarray,
        packed_group_block_ids: DeepSeekV4PackedRows,
        event_handle,
        timing: bool,
    ) -> DeepSeekV4DumpTask:
        shard_indexs = [0] * len(keys)
        submit_start = time.perf_counter()
        task = store.dump_data(keys, shard_indexs, ptrs, event_handle)
        byte_count = self._packed_row_bytes(packed_group_block_ids)
        if timing:
            logger.info(
                f"DeepSeek V4 {label} dump submit keys={len(keys)} "
                f"bytes={byte_count} "
                f"elapsed_s={time.perf_counter() - submit_start:.6f}"
            )
        return DeepSeekV4DumpTask(
            label=label,
            store=store,
            task=task,
            key_count=len(keys),
            byte_count=byte_count,
        )

    def _wait_dump_task(self, dump_task: DeepSeekV4DumpTask, timing: bool) -> None:
        wait_start = time.perf_counter()
        dump_task.store.wait(dump_task.task)
        if timing:
            logger.info(
                f"DeepSeek V4 {dump_task.label} dump wait "
                f"keys={dump_task.key_count} bytes={dump_task.byte_count} "
                f"elapsed_s={time.perf_counter() - wait_start:.6f}"
            )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMDeepSeekV4ConnectorMetadata)

        tasks: list[DeepSeekV4LoadTask] = []
        timing = self._timing_enabled()
        for request_id, request in metadata.request_meta.items():
            keys, packed_group_block_ids = request.load_block_ids
            if not keys:
                continue
            try:
                if self._local_stub_enabled():
                    ptrs = self._extract_packed_addrs(
                        packed_group_block_ids,
                        scratch_for_missing=True,
                    )
                    self._capture_worker_payload(
                        "load_before",
                        request_id,
                        keys,
                        packed_group_block_ids,
                        ptrs,
                    )
                    self._load_local_stub(keys, packed_group_block_ids)
                    self._capture_worker_payload(
                        "load_after",
                        request_id,
                        keys,
                        packed_group_block_ids,
                        ptrs,
                    )
                    continue

                if len(keys) > 1:
                    if self.group0_store is None:
                        raise RuntimeError(
                            "DeepSeek V4 group0 store is not initialized."
                        )
                    group0_keys = keys[:-1]
                    group0_block_ids = self._group0_only_rows(
                        packed_group_block_ids[:-1]
                    )
                    group0_ptrs = self._extract_packed_addrs(group0_block_ids)
                    tasks.append(
                        self._submit_load_task(
                            request_id,
                            "group0",
                            self.group0_store,
                            group0_keys,
                            group0_block_ids,
                            group0_ptrs,
                            False,
                            timing,
                        )
                    )

                full_keys = keys[-1:]
                full_group_block_ids = packed_group_block_ids[-1:]
                full_ptrs = self._extract_packed_addrs(
                    full_group_block_ids,
                    scratch_for_missing=True,
                )
                self._capture_worker_payload(
                    "load_before",
                    request_id,
                    full_keys,
                    full_group_block_ids,
                    full_ptrs,
                )
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "packed",
                        self.store,
                        full_keys,
                        full_group_block_ids,
                        full_ptrs,
                        True,
                        timing,
                    )
                )
            except Exception as e:
                logger.error(
                    f"request {request_id} submit DeepSeek V4 packed load task "
                    f"error. {type(e).__name__}: {e}"
                )

        for load_task in tasks:
            self._wait_load_task(load_task, timing)

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMDeepSeekV4ConnectorMetadata)

        keys: list[bytes] = []
        packed_group_block_rows: DeepSeekV4PackedRows = []
        packed_rows: list[np.ndarray] = []
        for request_id, request in metadata.request_meta.items():
            req_keys, packed_group_block_ids = request.dump_block_ids
            if not req_keys:
                continue
            try:
                rows = self._extract_packed_addrs(packed_group_block_ids)
                self._capture_worker_payload(
                    "dump_before",
                    request_id,
                    req_keys,
                    packed_group_block_ids,
                    rows,
                )
            except Exception as e:
                logger.error(
                    f"prepare DeepSeek V4 packed dump rows failed. "
                    f"{type(e).__name__}: {e}"
                )
                continue
            keys.extend(req_keys)
            packed_group_block_rows.extend(packed_group_block_ids)
            packed_rows.append(rows)

        if not keys:
            return

        if self._local_stub_enabled():
            if self.tp_rank == 0:
                try:
                    self._dump_local_stub(keys, packed_group_block_rows)
                except Exception as e:
                    logger.error(
                        f"dump DeepSeek V4 local stub failed. {type(e).__name__}: {e}"
                    )
            return

        if self.tp_rank != 0:
            return

        try:
            ptrs = np.vstack(packed_rows)
            event_handle = self._get_dump_event_handle()
            tasks: list[DeepSeekV4DumpTask] = []
            timing = self._timing_enabled()
            if self.group0_store is None:
                raise RuntimeError("DeepSeek V4 group0 store is not initialized.")
            group0_rows = self._group0_only_rows(packed_group_block_rows)
            group0_ptrs = self._extract_packed_addrs(group0_rows)
            tasks.append(
                self._submit_dump_task(
                    "group0",
                    self.group0_store,
                    keys,
                    group0_ptrs,
                    group0_rows,
                    event_handle,
                    timing,
                )
            )
            tasks.append(
                self._submit_dump_task(
                    "packed",
                    self.store,
                    keys,
                    ptrs,
                    packed_group_block_rows,
                    event_handle,
                    timing,
                )
            )
            for dump_task in tasks:
                self._wait_dump_task(dump_task, timing)
        except Exception as e:
            logger.error(
                f"dump DeepSeek V4 packed kv cache failed. {type(e).__name__}: {e}"
            )


class UCMConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.connector: KVConnectorBase_V1
        ucm_config = Config(vllm_config.kv_transfer_config)
        self.launch_config = ucm_config.get_config()
        logger.info(f"self.launch_config: {self.launch_config}")

        use_layerwise = (
            self.launch_config.get("use_layerwise", False)
            if self.launch_config is not None
            else False
        )

        pp_enabled = self._vllm_config.parallel_config.pipeline_parallel_size > 1
        if pp_enabled and not use_layerwise:
            raise RuntimeError(
                "Pipeline parallelism is not supported in UCMDirectConnector, please set use_layerwise=True."
            )

        use_lite = (
            self.launch_config.get("use_lite", False)
            if self.launch_config is not None
            else False
        )

        use_ratio_rate = (
            self.launch_config is not None and "hit_ratio" in self.launch_config
        )

        use_cp_parallel = (
            hasattr(self._vllm_config.parallel_config, "prefill_context_parallel_size")
            and hasattr(
                self._vllm_config.parallel_config, "decode_context_parallel_size"
            )
            and self._vllm_config.parallel_config.prefill_context_parallel_size
            * self._vllm_config.parallel_config.decode_context_parallel_size
            > 1
        )

        model_type = getattr(
            self._vllm_config.model_config.hf_text_config, "model_type", ""
        )
        use_deepseek_v4 = self.launch_config.get(
            "deepseek_v4", model_type == "deepseek_v4"
        )

        if use_deepseek_v4:
            self.connector = UCMDeepSeekV4Connector(vllm_config, role)
        elif use_lite:
            self.connector = UCMLiteConnector(vllm_config, role)
        elif use_ratio_rate:
            self.connector = UCMMockConnector(vllm_config, role)
        elif use_cp_parallel:
            self.connector = UCMCPConnector(vllm_config, role)
        elif use_layerwise:
            self.connector = UCMLayerWiseConnector(vllm_config, role)
        else:
            self.connector = UCMDirectConnector(vllm_config, role)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self.connector.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self.connector.update_state_after_alloc(request, blocks, num_external_tokens)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        self.connector.register_kv_caches(kv_caches)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self.connector.build_connector_meta(scheduler_output)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self.connector.bind_connector_metadata(connector_metadata)

    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bool: True if connector metadata exists, False otherwise.
        """
        return self.connector.has_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self.connector.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self.connector.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self.connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self.connector.wait_for_save()

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        return False, None

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self.connector.clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        return self.connector.get_block_ids_with_load_errors()
