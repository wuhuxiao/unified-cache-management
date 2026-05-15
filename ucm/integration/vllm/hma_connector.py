import copy
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.integration.vllm.device import create_device
from ucm.integration.vllm.ucm_connector import UCMDirectConnector
from ucm.logger import init_logger
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(frozen=True)
class KVCacheSegment:
    block_id: int
    offset: int = 0
    length: int = 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.block_id == other and self.offset == 0
        if not isinstance(other, KVCacheSegment):
            return False
        return (
            self.block_id == other.block_id
            and self.offset == other.offset
            and self.length == other.length
        )


@dataclass(frozen=True)
class KVCacheGroupMeta:
    group_id: int
    token_block_size: int
    tensor_block_size: int
    logical_blocks_per_hash_block: int
    hash_blocks_per_tensor_block: int
    tail_blocks: int | None
    window_spans: tuple[int, ...]


class KVCacheGroupLayout:
    """Flat pointer layout for one vLLM KV cache group.

    The cache views belonging to one KV group are not necessarily contiguous by
    layer id, so this layout flattens all registered tensors in a deterministic
    order.
    """

    def __init__(self, kvcaches: dict[str, torch.Tensor]) -> None:
        self.kvcaches = dict(sorted(kvcaches.items(), key=self._sort_key))
        self.base_ptrs: np.ndarray
        self.block_strides: np.ndarray
        self.token_strides: np.ndarray
        self.tensor_size_lists: np.ndarray
        self.tensor_size_per_token_lists: np.ndarray
        self.view_tensor_block_sizes: np.ndarray
        self._build_layout()
        self._tensor_tokens_cache: dict[tuple[int, int], np.ndarray] = {}
        self._segment_tensor_size_cache: dict[tuple[int, int], tuple[int, ...]] = {}

    @staticmethod
    def _sort_key(item: tuple[str, torch.Tensor]) -> tuple[int, str]:
        name, _ = item
        return (extract_layer_index(name), name)

    def _build_layout(self) -> None:
        ptrs: list[int] = []
        strides: list[int] = []
        token_strides: list[int] = []
        tensor_sizes: list[int] = []
        tensor_sizes_per_token: list[int] = []
        view_tensor_block_sizes: list[int] = []
        view_tensors: list[torch.Tensor] = []
        view_meta: list[tuple[str, tuple[int, ...], tuple[int, ...], str, int]] = []

        def handle_tensor(
            t: torch.Tensor,
            size_dims: Sequence[int],
            layer_name: str,
        ) -> None:
            ptrs.append(t[0].data_ptr())
            strides.append(t.stride(0) * t.element_size())
            tensor_size = math.prod([t.shape[i] for i in size_dims]) * t.element_size()
            tensor_sizes.append(tensor_size)
            token_dim = 1
            view_tensor_block_size = int(t.shape[token_dim])
            if view_tensor_block_size <= 0:
                raise ValueError(
                    f"KV cache tensor has empty block dimension: {t.shape}"
                )
            token_strides.append(t.stride(token_dim) * t.element_size())
            tensor_sizes_per_token.append(tensor_size // view_tensor_block_size)
            view_tensor_block_sizes.append(view_tensor_block_size)
            view_tensors.append(t)
            view_meta.append(
                (
                    layer_name,
                    tuple(t.shape),
                    tuple(t.stride()),
                    str(t.dtype),
                    view_tensor_block_size,
                )
            )

        def handle_kv_layer_tensor(tensor: torch.Tensor, layer_name: str) -> None:
            if tensor.dim() == 5:
                # [2, num_blocks, block_size, num_head, head_dim]
                handle_tensor(tensor[0], (-3, -2, -1), layer_name)
                handle_tensor(tensor[1], (-3, -2, -1), layer_name)
            elif tensor.dim() == 4:
                if tensor.shape[1] == 2:
                    # GPU kernels may register [num_blocks, 2, block_size, ...].
                    # Split the K/V axis before reading the token dimension.
                    handle_tensor(tensor[:, 0], (-2, -1), layer_name)
                    handle_tensor(tensor[:, 1], (-2, -1), layer_name)
                else:
                    # Ascend registers split KV/state tensors as
                    # [num_blocks, block_size, num_head, head_dim].
                    handle_tensor(tensor, (-3, -2, -1), layer_name)
            elif tensor.dim() == 3:
                # [num_blocks, block_size, head_dim]. Some DeepSeek V4 caches
                # use block_size=2 here and share a group with larger pages.
                handle_tensor(tensor, (-2, -1), layer_name)
            else:
                raise ValueError(
                    f"Unsupported KV cache tensor shape for "
                    f"{layer_name}: {tensor.shape}"
                )

        for layer_name, kv_layer in self.kvcaches.items():
            if isinstance(kv_layer, torch.Tensor):
                handle_kv_layer_tensor(kv_layer, layer_name)
            elif isinstance(kv_layer, Tuple):
                for tensor in kv_layer:
                    handle_kv_layer_tensor(tensor, layer_name)
            else:
                raise TypeError(
                    f"Unsupported KV cache type for " f"{layer_name}: {type(kv_layer)}"
                )

        if not ptrs:
            raise ValueError("KV cache group layout is empty.")

        self.base_ptrs = np.asarray(ptrs, dtype=np.uint64)
        self.block_strides = np.asarray(strides, dtype=np.uint64)
        self.token_strides = np.asarray(token_strides, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(tensor_sizes, dtype=np.uint64)
        self.tensor_size_per_token_lists = np.asarray(
            tensor_sizes_per_token, dtype=np.uint64
        )
        self.view_tensor_block_sizes = np.asarray(
            view_tensor_block_sizes, dtype=np.uint64
        )
        self.view_tensors = view_tensors
        self.view_meta = [
            {
                "name": name,
                "shape": shape,
                "stride": stride,
                "dtype": dtype,
                "view_tensor_block_size": view_tensor_block_size,
            }
            for name, shape, stride, dtype, view_tensor_block_size in view_meta
        ]
        logger.info(
            f"KV cache group layout: views={len(self.kvcaches)}, "
            f"ptrs={len(ptrs)}, tensor_block_bytes={self.tensor_block_bytes}, "
            f"view_tensor_block_sizes={sorted(set(view_tensor_block_sizes))}"
        )

    def extract_block_addrs(self, vllm_block_ids: list[int]) -> np.ndarray:
        vllm_block_ids_np = np.array(vllm_block_ids, np.uint64)
        return (
            vllm_block_ids_np[:, None] * self.block_strides[None, :]
            + self.base_ptrs[None, :]
        )

    def _logical_to_tensor_tokens(
        self,
        logical_tokens: int,
        group_tensor_block_size: int,
        view_tensor_block_size: int,
    ) -> int:
        scaled = logical_tokens * view_tensor_block_size
        if scaled % group_tensor_block_size != 0:
            raise ValueError(
                f"Logical segment of {logical_tokens} tokens does not align with "
                f"view tensor block size={view_tensor_block_size} and group "
                f"tensor block size={group_tensor_block_size}."
            )
        return scaled // group_tensor_block_size

    def _tensor_tokens_for_logical(
        self,
        logical_tokens: int,
        group_tensor_block_size: int,
    ) -> np.ndarray:
        key = (group_tensor_block_size, logical_tokens)
        cached = self._tensor_tokens_cache.get(key)
        if cached is not None:
            return cached

        scaled = self.view_tensor_block_sizes * np.uint64(logical_tokens)
        misaligned = scaled % np.uint64(group_tensor_block_size)
        if np.any(misaligned):
            raise ValueError(
                f"Logical segment of {logical_tokens} tokens does not align with "
                f"view tensor block sizes={self.view_tensor_block_sizes.tolist()} "
                f"and group tensor block size={group_tensor_block_size}."
            )
        tensor_tokens = scaled // np.uint64(group_tensor_block_size)
        self._tensor_tokens_cache[key] = tensor_tokens
        return tensor_tokens

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

    def extract_segment_addrs_flat(
        self,
        segment: KVCacheSegment,
        group_tensor_block_size: int,
    ) -> np.ndarray:
        if segment.block_id < 0:
            raise ValueError("Negative KV cache block id needs a scratch target.")
        # The offset conversion is reused heavily by load/dump metadata dispatch.
        tensor_offsets = self._tensor_tokens_for_logical(
            segment.offset,
            group_tensor_block_size,
        )
        return (
            np.uint64(segment.block_id) * self.block_strides
            + tensor_offsets * self.token_strides
            + self.base_ptrs
        ).astype(np.uint64, copy=False)

    def extract_segment_addrs(
        self,
        segments: list[KVCacheSegment],
        group_tensor_block_size: int,
    ) -> np.ndarray:
        if not segments:
            return np.empty((0, len(self.base_ptrs)), dtype=np.uint64)
        if len(segments) == 1:
            return self.extract_segment_addrs_flat(
                segments[0],
                group_tensor_block_size,
            ).reshape(1, -1)

        block_ids_np = np.fromiter(
            (segment.block_id for segment in segments),
            dtype=np.uint64,
            count=len(segments),
        )
        offsets_np = np.vstack(
            [
                self._tensor_tokens_for_logical(
                    segment.offset,
                    group_tensor_block_size,
                )
                for segment in segments
            ]
        )
        return (
            block_ids_np[:, None] * self.block_strides[None, :]
            + offsets_np * self.token_strides[None, :]
            + self.base_ptrs[None, :]
        ).astype(np.uint64, copy=False)

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

    def segment_tensor_size_list(
        self,
        logical_tokens: int,
        group_tensor_block_size: int,
    ) -> list[int]:
        key = (group_tensor_block_size, logical_tokens)
        cached = self._segment_tensor_size_cache.get(key)
        if cached is None:
            tensor_tokens = self._tensor_tokens_for_logical(
                logical_tokens,
                group_tensor_block_size,
            )
            cached = tuple(
                int(size)
                for size in (
                    self.tensor_size_per_token_lists * tensor_tokens
                ).tolist()
            )
            self._segment_tensor_size_cache[key] = cached
        return list(cached)

    def extract_block_tensor_views(
        self, vllm_block_ids: list[int]
    ) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []

        for block_id in vllm_block_ids:
            for tensor in self.view_tensors:
                tensors.append(tensor[block_id])
        return tensors

    def extract_segment_tensor_views(
        self,
        segments: list[KVCacheSegment],
        group_tensor_block_size: int,
    ) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []

        for segment in segments:
            tensor_offsets = self._tensor_tokens_for_logical(
                segment.offset,
                group_tensor_block_size,
            )
            tensor_lengths = self._tensor_tokens_for_logical(
                segment.length,
                group_tensor_block_size,
            )
            for tensor, tensor_offset, tensor_length in zip(
                self.view_tensors,
                tensor_offsets,
                tensor_lengths,
            ):
                tensors.append(
                    tensor[segment.block_id].narrow(
                        0,
                        int(tensor_offset),
                        int(tensor_length),
                    )
                )
        return tensors

    @property
    def tensor_size_list(self) -> list[int]:
        return self.tensor_size_lists.tolist()

    @property
    def shard_size(self) -> int:
        return int(self.tensor_size_lists.sum())

    @property
    def tensor_block_bytes(self) -> int:
        return self.shard_size

    @property
    def tensor_block_size(self) -> int:
        if len(set(self.view_tensor_block_sizes.tolist())) != 1:
            raise ValueError(
                "KV cache group layout has mixed view tensor block sizes: "
                f"{self.view_tensor_block_sizes.tolist()}"
            )
        return int(self.view_tensor_block_sizes[0])



# Multiple canonical hash blocks, each represented as one KVCacheGroupRow.
KVCacheGroupRows = list[list]


class FAWABlockSpanLayout:
    """Maps FAWA canonical hash blocks to per-group KV cache spans."""

    ASCEND_HASH_BLOCK_SIZE = 512
    ASCEND_REQUIRED_SPECS = frozenset(
        {"Compress4AttentionSpec", "C4IndexerSpec", "Compress128AttentionSpec"}
    )
    ASCEND_TENSOR_BLOCK_SPECS = {
        "Compress4AttentionSpec": 512,
        "C4IndexerSpec": 4096,
        "Compress128AttentionSpec": 16384,
    }
    ASCEND_MULTI_TENSOR_SPECS = {
        # vllm-ascend _reshape_kv_cache_tensors() extends C4 indexer as
        # [indexer_kv_cache, indexer_scale_cache]. Other DSV4 specs append one
        # tensor each in kv-cache-group order.
        "C4IndexerSpec": 2,
    }
    ASCEND_STATE_COMPRESS_RATIOS = {
        "C4AttnKVStateSpec": 4,
        "C4AttnScoreStateSpec": 4,
        "C4IndexerKVStateSpec": 4,
        "C4IndexerScoreStateSpec": 4,
        "C128AttnKVStateSpec": 128,
        "C128AttnScoreStateSpec": 128,
    }

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        fa_group_ids: tuple[int, ...],
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.fa_group_ids = fa_group_ids
        self.is_ascend = self._detect_ascend_layout()
        self.hash_block_size = self._get_hash_block_size()
        self.group_token_block_sizes = self._get_group_token_block_sizes()
        self.group_tensor_block_sizes = self._get_group_tensor_block_sizes()
        self.group_tensor_block_ratios = self._get_group_tensor_block_ratios()
        self.group_layer_tensor_indices = self._get_group_layer_tensor_indices()

    @staticmethod
    def group_specs(group_spec) -> tuple[object, ...]:
        nested_specs = getattr(group_spec.kv_cache_spec, "kv_cache_specs", None)
        return (
            tuple(nested_specs.values())
            if nested_specs
            else (group_spec.kv_cache_spec,)
        )

    @staticmethod
    def group_layer_specs(group_spec, layer_name: str) -> tuple[object, ...]:
        nested_specs = getattr(group_spec.kv_cache_spec, "kv_cache_specs", None)
        if nested_specs:
            return (nested_specs[layer_name],)
        return (group_spec.kv_cache_spec,)

    @staticmethod
    def group_spec_items(group_spec) -> tuple[tuple[str, object], ...]:
        nested_specs = getattr(group_spec.kv_cache_spec, "kv_cache_specs", None)
        if nested_specs:
            return tuple(nested_specs.items())
        return tuple(
            (layer_name, group_spec.kv_cache_spec)
            for layer_name in group_spec.layer_names
        )

    @staticmethod
    def spec_window_tokens(spec: object) -> Optional[int]:
        window_size = getattr(spec, "sliding_window", None) or getattr(
            spec, "attention_chunk_size", None
        )
        return int(window_size) if window_size is not None else None

    @classmethod
    def group_window_tokens(cls, group_spec) -> Optional[int]:
        window_sizes = {
            window_tokens
            for spec in cls.group_specs(group_spec)
            if (window_tokens := cls.spec_window_tokens(spec)) is not None
        }
        if not window_sizes:
            return None
        if len(window_sizes) != 1:
            raise RuntimeError(
                "FAWA KV cache group has mixed window sizes: " f"{sorted(window_sizes)}"
            )
        return window_sizes.pop()

    @classmethod
    def group_has_window(cls, group_spec) -> bool:
        return cls.group_window_tokens(group_spec) is not None

    @classmethod
    def is_ascend_kv_cache_config(
        cls, kv_cache_config: Optional["KVCacheConfig"]
    ) -> bool:
        if kv_cache_config is None:
            return False
        groups = kv_cache_config.kv_cache_groups
        if not any(type(group).__name__.startswith("Ascend") for group in groups):
            return False
        spec_names = {
            type(spec).__name__ for group in groups for spec in cls.group_specs(group)
        }
        return cls.ASCEND_REQUIRED_SPECS.issubset(spec_names)

    @staticmethod
    def spec_token_block_size(spec: object) -> int:
        block_size = getattr(spec, "block_size", None)
        if block_size is None:
            raise RuntimeError(
                f"FAWA KV cache spec {type(spec).__name__} has no block_size."
            )
        return int(block_size)

    @classmethod
    def spec_tensor_block_size(cls, spec: object) -> Optional[int]:
        spec_name = type(spec).__name__
        fixed_size = cls.ASCEND_TENSOR_BLOCK_SPECS.get(spec_name)
        if fixed_size is not None:
            return fixed_size
        compress_ratio = int(getattr(spec, "compress_ratio", 1))
        if compress_ratio <= 1:
            return None
        return cls.spec_token_block_size(spec) * compress_ratio

    def _detect_ascend_layout(self) -> bool:
        return self.is_ascend_kv_cache_config(self.kv_cache_config)

    def state_compress_ratio(self, group_id: int) -> Optional[int]:
        ratios = set()
        group = self.kv_cache_config.kv_cache_groups[group_id]
        for spec in self.group_specs(group):
            spec_name = type(spec).__name__
            ratio = getattr(spec, "compress_ratio", None)
            if ratio is None:
                ratio = self.ASCEND_STATE_COMPRESS_RATIOS.get(spec_name)
            if ratio is not None and int(ratio) > 1:
                ratios.add(int(ratio))
        if len(ratios) > 1:
            raise RuntimeError(
                f"FAWA Ascend group {group_id} has mixed compress ratios: "
                f"{sorted(ratios)}."
            )
        return ratios.pop() if ratios else None

    def is_swa_group(self, group_id: int) -> bool:
        group = self.kv_cache_config.kv_cache_groups[group_id]
        return any(
            type(spec).__name__ == "SWAAttentionSpec"
            for spec in self.group_specs(group)
        )

    def _get_hash_block_size(self) -> int:
        if self.is_ascend:
            return self.ASCEND_HASH_BLOCK_SIZE
        fa_block_sizes = {
            self.spec_token_block_size(
                self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            )
            for group_id in self.fa_group_ids
        }
        if len(fa_block_sizes) != 1:
            raise RuntimeError(
                "FAWA connector requires one FA token block size, got "
                f"{sorted(fa_block_sizes)}."
            )
        return fa_block_sizes.pop()

    def _ascend_group_token_block_size(self, group_spec) -> int:
        # Ascend DeepSeek V4 stores selected compressed groups at a 512-token
        # canonical segment inside larger tensor pages. Other groups keep the
        # token block size advertised by the KV cache spec.
        if any(
            self.spec_tensor_block_size(spec) is not None
            for spec in self.group_specs(group_spec)
        ):
            return self.hash_block_size
        return self.spec_token_block_size(group_spec.kv_cache_spec)

    def _get_group_token_block_sizes(self) -> tuple[int, ...]:
        groups = self.kv_cache_config.kv_cache_groups
        if self.is_ascend:
            group_token_block_sizes = tuple(
                self._ascend_group_token_block_size(group) for group in groups
            )
        else:
            raw_group_token_block_sizes = tuple(
                self.spec_token_block_size(group.kv_cache_spec) for group in groups
            )
            if not raw_group_token_block_sizes:
                raise RuntimeError("FAWA connector found no KV cache groups.")
            mutable_sizes = list(raw_group_token_block_sizes)
            for group_id in self.fa_group_ids:
                mutable_sizes[group_id] = self.hash_block_size
            group_token_block_sizes = tuple(mutable_sizes)

        for group_id, group_token_block_size in enumerate(group_token_block_sizes):
            if group_token_block_size <= 0:
                raise RuntimeError(
                    f"FAWA group {group_id} block size must be positive, "
                    f"got {group_token_block_size}."
                )
            if self.hash_block_size % group_token_block_size != 0:
                raise RuntimeError(
                    f"FAWA group {group_id} block size {group_token_block_size} "
                    f"must divide {self.hash_block_size}."
                )
        return group_token_block_sizes

    def _get_group_tensor_block_sizes(self) -> tuple[int, ...]:
        tensor_block_sizes: list[int] = []
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            detected = {
                tensor_block_size
                for spec in self.group_specs(group)
                if (tensor_block_size := self.spec_tensor_block_size(spec)) is not None
            }
            if len(detected) > 1:
                raise RuntimeError(
                    f"FAWA Ascend group {group_id} has mixed tensor "
                    f"block sizes: {sorted(detected)}."
                )
            tensor_block_sizes.append(
                detected.pop() if detected else self.group_token_block_sizes[group_id]
            )
        return tuple(tensor_block_sizes)

    def _get_group_tensor_block_ratios(self) -> tuple[int, ...]:
        ratios: list[int] = []
        for group_id, (
            group_token_block_size,
            group_tensor_block_size,
        ) in enumerate(zip(self.group_token_block_sizes, self.group_tensor_block_sizes)):
            if group_tensor_block_size % group_token_block_size != 0:
                raise RuntimeError(
                    f"FAWA group {group_id} logical block size "
                    f"{group_token_block_size} must divide tensor block size "
                    f"{group_tensor_block_size}."
                )
            ratios.append(group_tensor_block_size // group_token_block_size)
        return tuple(ratios)

    def _group_tensor_block_ratio(self, group_id: int) -> int:
        ratios = getattr(self, "group_tensor_block_ratios", None)
        if ratios is None:
            self.group_tensor_block_ratios = self._get_group_tensor_block_ratios()
            ratios = self.group_tensor_block_ratios
        return ratios[group_id]

    @classmethod
    def spec_tensor_count(cls, spec: object) -> int:
        return cls.ASCEND_MULTI_TENSOR_SPECS.get(type(spec).__name__, 1)

    def _get_group_layer_tensor_indices(
        self,
    ) -> dict[int, dict[str, tuple[int, ...]]]:
        if not self.is_ascend:
            return {}

        mapping: dict[int, dict[str, tuple[int, ...]]] = {}
        next_tensor_index_by_layer: dict[str, int] = {}
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            group_mapping: dict[str, tuple[int, ...]] = {}
            for layer_name in group.layer_names:
                tensor_count = sum(
                    self.spec_tensor_count(spec)
                    for spec in self.group_layer_specs(group, layer_name)
                )
                start = next_tensor_index_by_layer.get(layer_name, 0)
                end = start + tensor_count
                group_mapping[layer_name] = tuple(range(start, end))
                next_tensor_index_by_layer[layer_name] = end
            mapping[group_id] = group_mapping
        return mapping

    def group_tensor_indices(
        self, group_id: int, layer_name: str
    ) -> Optional[tuple[int, ...]]:
        if not self.is_ascend:
            return None
        try:
            return self.group_layer_tensor_indices[group_id][layer_name]
        except KeyError as exc:
            raise RuntimeError(
                f"FAWA Ascend layout has no tensor index mapping for "
                f"group {group_id}, layer {layer_name}."
            ) from exc

    def block_index_to_segment(
        self,
        group_id: int,
        group_block_idx: int,
        block_id: int,
    ) -> KVCacheSegment:
        group_token_block_size = self.group_token_block_sizes[group_id]
        token_blocks_per_tensor_block = self._group_tensor_block_ratio(group_id)
        tensor_block_idx = group_block_idx // token_blocks_per_tensor_block
        if tensor_block_idx != block_id:
            logger.debug(
                f"FAWA group {group_id} logical block idx {group_block_idx} "
                f"maps to tensor block idx {tensor_block_idx}, "
                f"allocated id {block_id}."
            )
        return KVCacheSegment(
            block_id=block_id,
            offset=(group_block_idx % token_blocks_per_tensor_block)
            * group_token_block_size,
            length=group_token_block_size,
        )

    def allocation_index(self, group_id: int, group_block_idx: int) -> int:
        return group_block_idx // self._group_tensor_block_ratio(group_id)

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


@dataclass
class UCMFAWAConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, FAWARequestDispatchMeta] = field(default_factory=dict)


@dataclass
class FAWALoadTask:
    request_id: str
    label: str
    store: UcmKVStoreBaseV1
    task: Task
    key_count: int
    anchor_vllm_block_ids: set[int] = field(default_factory=set)


@dataclass
class FAWADumpTask:
    label: str
    store: UcmKVStoreBaseV1
    task: Task
    key_count: int


class UCMFAWAConnector(UCMDirectConnector):
    """UCM connector for mixed full-attention and window KV cache groups.

    Full-attention groups are stored once per reusable prefix block and are
    loaded for every external prefix hit. WA groups store the tail blocks
    needed at each prefix boundary, and only the final matched boundary is
    loaded.
    """

    DEFAULT_HASH_BLOCK_SIZE = 256

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        self._defer_scheduler_store = True
        super().__init__(vllm_config, role, kv_cache_config)
        self.hash_block_size = self.DEFAULT_HASH_BLOCK_SIZE
        self.block_size = self.DEFAULT_HASH_BLOCK_SIZE
        self.group_layouts: dict[int, KVCacheGroupLayout] = {}
        self.group_metas: dict[int, KVCacheGroupMeta] = {}
        self.fa_group_ids, self.window_group_ids = self._partition_kv_cache_groups()
        if self._kv_cache_config is None:
            raise RuntimeError("FAWA connector requires kv_cache_config.")
        self.block_span_layout = self._create_block_span_layout()
        self._ascend_layout = False
        self.hash_block_size = self._get_hash_block_size()
        self.block_size = self.hash_block_size
        self.group_token_block_sizes = self._get_group_token_block_sizes()
        self.group_tensor_block_sizes = self._get_group_tensor_block_sizes()
        self.group_tensor_block_ratios = self._get_group_tensor_block_ratios()
        self.group_tail_blocks = self._get_group_tail_blocks()
        self.group_window_spans = self._get_group_window_spans()
        self.group_metas: dict[int, KVCacheGroupMeta] = {}
        self._init_group_metas()
        self.fa_store: Optional[UcmKVStoreBaseV1] = None
        self.wa_store: Optional[UcmKVStoreBaseV1] = None
        self.requests_meta: dict[str, FAWARequestMeta] = {}
        if role == KVConnectorRole.SCHEDULER:
            self.store = self._create_fa_store(None)
            self.fa_store = self.store
            self.wa_store = self._create_wa_store(None)
        logger.info(
            f"FAWA KV group config: fa_groups={self.fa_group_ids}, "
            f"window_groups={self.window_group_ids}, "
            f"ascend_layout={self._ascend_layout}, "
            f"token_block_sizes={self.group_token_block_sizes}, "
            f"group_tensor_block_sizes={self.group_tensor_block_sizes}, "
            f"tail_blocks={self.group_tail_blocks}, "
            f"window_spans={self.group_window_spans}"
        )
        logger.info("Init UCM FAWA connector.")

    @classmethod
    def can_handle_kv_cache_config(
        cls, kv_cache_config: Optional["KVCacheConfig"]
    ) -> bool:
        if kv_cache_config is None:
            return False
        if cls.can_handle_ascend_kv_cache_config(kv_cache_config):
            return False
        fa_groups, window_groups = cls._partition_group_specs(
            kv_cache_config.kv_cache_groups
        )
        return bool(fa_groups and window_groups)

    @classmethod
    def can_handle_ascend_kv_cache_config(
        cls, kv_cache_config: Optional["KVCacheConfig"]
    ) -> bool:
        return FAWABlockSpanLayout.is_ascend_kv_cache_config(kv_cache_config)

    def _create_block_span_layout(self) -> Optional[FAWABlockSpanLayout]:
        return None

    def _get_hash_block_size(self) -> int:
        fa_block_sizes = {
            self._spec_token_block_size(
                self._kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            )
            for group_id in self.fa_group_ids
        }
        if len(fa_block_sizes) != 1:
            raise RuntimeError(
                "FAWA connector requires one FA token block size, got "
                f"{sorted(fa_block_sizes)}."
            )
        return fa_block_sizes.pop()

    @staticmethod
    def _spec_token_block_size(spec: object) -> int:
        return FAWABlockSpanLayout.spec_token_block_size(spec)

    def _get_group_token_block_sizes(self) -> tuple[int, ...]:
        raw_group_token_block_sizes = tuple(
            self._spec_token_block_size(group.kv_cache_spec)
            for group in self._kv_cache_config.kv_cache_groups
        )
        if not raw_group_token_block_sizes:
            raise RuntimeError("FAWA connector found no KV cache groups.")
        mutable_sizes = list(raw_group_token_block_sizes)
        for group_id in self.fa_group_ids:
            mutable_sizes[group_id] = self.hash_block_size
        group_token_block_sizes = tuple(mutable_sizes)
        self._validate_group_token_block_sizes(group_token_block_sizes)
        return group_token_block_sizes

    def _get_group_tensor_block_sizes(self) -> tuple[int, ...]:
        return self.group_token_block_sizes

    def _validate_group_token_block_sizes(
        self, group_token_block_sizes: tuple[int, ...]
    ) -> None:
        for group_id, group_token_block_size in enumerate(group_token_block_sizes):
            if group_token_block_size <= 0:
                raise RuntimeError(
                    f"FAWA group {group_id} block size must be positive, "
                    f"got {group_token_block_size}."
                )
            if self.hash_block_size % group_token_block_size != 0:
                raise RuntimeError(
                    f"FAWA group {group_id} block size {group_token_block_size} "
                    f"must divide {self.hash_block_size}."
                )

    def _get_group_tensor_block_ratios(self) -> tuple[int, ...]:
        ratios: list[int] = []
        for group_id, (
            group_token_block_size,
            group_tensor_block_size,
        ) in enumerate(zip(self.group_token_block_sizes, self.group_tensor_block_sizes)):
            if group_tensor_block_size % group_token_block_size != 0:
                raise RuntimeError(
                    f"FAWA group {group_id} logical block size "
                    f"{group_token_block_size} must divide tensor block size "
                    f"{group_tensor_block_size}."
                )
            ratios.append(group_tensor_block_size // group_token_block_size)
        return tuple(ratios)

    def _group_tensor_block_ratio(self, group_id: int) -> int:
        ratios = getattr(self, "group_tensor_block_ratios", None)
        if ratios is None:
            self.group_tensor_block_ratios = self._get_group_tensor_block_ratios()
            ratios = self.group_tensor_block_ratios
        return ratios[group_id]

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

    def _create_fa_store(
        self,
        group_layouts: Optional[dict[int, KVCacheGroupLayout]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        tensor_size_list = None
        if self._role == KVConnectorRole.WORKER:
            if group_layouts is None:
                raise RuntimeError("Worker FA store needs layouts.")
            tensor_size_list = self._store_tensor_size_list(
                group_layouts,
                self.fa_group_ids,
            )
        return self._create_store(
            "FA",
            "fa",
            tensor_size_list,
            cpu_affinity_cores,
        )

    def _create_wa_store(
        self,
        group_layouts: Optional[dict[int, KVCacheGroupLayout]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        tensor_size_list = None
        if self._role == KVConnectorRole.WORKER:
            if group_layouts is None:
                raise RuntimeError("Worker WA store needs layouts.")
            tensor_size_list = self._store_tensor_size_list(
                group_layouts,
                self.window_group_ids,
            )
        return self._create_store(
            "WA",
            "wa",
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
        # MLA ranks share one logical store buffer; non-MLA stores are per rank.
        config.setdefault("share_buffer_enable", self.is_mla)
        if isinstance(config.get("storage_backends"), str):
            config["storage_backends"] = [
                path for path in config["storage_backends"].split(":")
            ]
        config["unique_id"] = f"{self.engine_id}_fawa_{store_suffix}"
        self._namespace_storage_backends(config, store_suffix)
        dp_rank = self._vllm_config.parallel_config.data_parallel_rank
        config["posix_gc_enable"] = (
            self._role != KVConnectorRole.WORKER and dp_rank == 0
        )
        return name, module_path, config

    @staticmethod
    def _namespace_storage_backends(
        config: dict[str, object],
        store_suffix: str,
    ) -> None:
        backends = config.get("storage_backends")
        if not isinstance(backends, list):
            return
        namespaced_backends: list[str] = []
        for backend in backends:
            backend_path = os.path.join(str(backend), f"fawa_{store_suffix}")
            os.makedirs(backend_path, exist_ok=True)
            namespaced_backends.append(backend_path)
        config["storage_backends"] = namespaced_backends

    def _create_store(
        self,
        label: str,
        store_suffix: str,
        tensor_size_list: Optional[list[int]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        name, module_path, config = self._base_store_config(store_suffix)
        if self._role == KVConnectorRole.WORKER:
            if tensor_size_list is None:
                raise RuntimeError(f"Worker FAWA {label} store needs tensor sizes.")
            config["device_id"] = self.local_rank
            config["tensor_size_list"] = tensor_size_list
            config["shard_size"] = int(sum(tensor_size_list))
            config["block_size"] = int(sum(tensor_size_list))
            # MLA stores aggregate TP shards under one logical rank group.
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
        logger.info(
            f"create FAWA {label} {name} with config: "
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
    def _partition_group_specs(
        group_specs,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        fa_group_ids: list[int] = []
        window_group_ids: list[int] = []
        for group_id, group_spec in enumerate(group_specs):
            if FAWABlockSpanLayout.group_has_window(group_spec):
                window_group_ids.append(group_id)
            else:
                fa_group_ids.append(group_id)
        return tuple(fa_group_ids), tuple(window_group_ids)

    def _partition_kv_cache_groups(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        fa_group_ids, window_group_ids = self._partition_group_specs(
            self._kv_cache_config.kv_cache_groups
        )
        if not fa_group_ids:
            raise RuntimeError("FAWA connector found no full-attention groups.")
        if not window_group_ids:
            raise RuntimeError("FAWA connector found no window groups.")
        return fa_group_ids, window_group_ids

    def _get_group_tail_blocks(self) -> tuple[Optional[int], ...]:
        tail_blocks: list[Optional[int]] = [None] * len(self.group_token_block_sizes)
        if self._kv_cache_config is None:
            raise RuntimeError("FAWA connector requires kv_cache_config.")
        for group_id in self.window_group_ids:
            group_spec = self._kv_cache_config.kv_cache_groups[group_id]
            group_token_block_size = self.group_token_block_sizes[group_id]
            window_tokens = FAWABlockSpanLayout.group_window_tokens(group_spec)
            if window_tokens is None:
                tail_blocks[group_id] = self.hash_block_size // group_token_block_size
                continue
            if self._is_compressor_state_group(group_id):
                tail_blocks[group_id] = self._compressor_state_tail_blocks(
                    group_id,
                    window_tokens,
                    group_token_block_size,
                )
                continue
            tail_blocks[group_id] = max(
                1, math.ceil(window_tokens / group_token_block_size)
            )
        return tuple(tail_blocks)

    def _get_group_window_spans(self) -> tuple[tuple[int, ...], ...]:
        spans: list[tuple[int, ...]] = []
        for group_id, tail_blocks in enumerate(self.group_tail_blocks):
            if tail_blocks is None:
                spans.append((self.group_token_block_sizes[group_id],))
            else:
                spans.append((self.group_token_block_sizes[group_id],) * tail_blocks)
        return tuple(spans)

    @staticmethod
    def _is_compressor_state_name(layer_name: str) -> bool:
        return ".compressor.state_cache" in layer_name

    @staticmethod
    def _compressor_state_prefix(layer_name: str) -> str:
        suffix = ".compressor.state_cache"
        if layer_name.endswith(suffix):
            return layer_name[: -len(suffix)]
        return layer_name.split(suffix, 1)[0]

    def _is_compressor_state_group(self, group_id: int) -> bool:
        if self._kv_cache_config is None:
            raise RuntimeError("FAWA connector requires kv_cache_config.")
        group_spec = self._kv_cache_config.kv_cache_groups[group_id]
        layer_names = tuple(group_spec.layer_names)
        return bool(layer_names) and all(
            self._is_compressor_state_name(name) for name in layer_names
        )

    def _group_compress_ratio(self, group_id: int) -> Optional[int]:
        if self._kv_cache_config is None:
            raise RuntimeError("FAWA connector requires kv_cache_config.")
        group_spec = self._kv_cache_config.kv_cache_groups[group_id]
        ratios = {
            int(ratio)
            for spec in FAWABlockSpanLayout.group_specs(group_spec)
            if (ratio := getattr(spec, "compress_ratio", 1)) and int(ratio) > 1
        }
        if len(ratios) > 1:
            raise RuntimeError(
                f"FAWA KV cache group {group_id} has mixed compress ratios: "
                f"{sorted(ratios)}"
            )
        if ratios:
            return ratios.pop()

        if not self._is_compressor_state_group(group_id):
            return None

        config_ratios = getattr(
            self._vllm_config.model_config.hf_config,
            "compress_ratios",
            None,
        )
        if config_ratios:
            for layer_name in group_spec.layer_names:
                layer_index = extract_layer_index(layer_name)
                if layer_index < len(config_ratios):
                    ratio = int(config_ratios[layer_index])
                    if ratio > 1:
                        ratios.add(ratio)
            if len(ratios) > 1:
                raise RuntimeError(
                    f"FAWA compressor state group {group_id} maps to mixed "
                    f"model config compress ratios: {sorted(ratios)}"
                )
            if ratios:
                return ratios.pop()

        prefixes = tuple(
            self._compressor_state_prefix(layer_name)
            for layer_name in group_spec.layer_names
        )
        for other_group in self._kv_cache_config.kv_cache_groups:
            for layer_name, spec in FAWABlockSpanLayout.group_spec_items(other_group):
                ratio = getattr(spec, "compress_ratio", 1)
                if not ratio or int(ratio) <= 1:
                    continue
                if any(
                    layer_name == prefix or layer_name.startswith(prefix + ".")
                    for prefix in prefixes
                ):
                    ratios.add(int(ratio))

        if len(ratios) > 1:
            raise RuntimeError(
                f"FAWA compressor state group {group_id} maps to mixed "
                f"compress ratios: {sorted(ratios)}"
            )
        return ratios.pop() if ratios else None

    def _compressor_state_tail_blocks(
        self,
        group_id: int,
        window_tokens: int,
        group_token_block_size: int,
    ) -> int:
        compress_ratio = self._group_compress_ratio(group_id)
        if compress_ratio is None:
            return max(1, math.ceil(window_tokens / group_token_block_size))
        if compress_ratio <= 0:
            raise RuntimeError(
                f"FAWA group {group_id} compress ratio must be positive, "
                f"got {compress_ratio}."
            )
        if window_tokens <= compress_ratio:
            return 0
        return math.ceil((window_tokens - compress_ratio) / group_token_block_size)

    def _split_kv_caches_by_vllm_groups(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> dict[int, dict[str, torch.Tensor]]:
        if self._kv_cache_config is None:
            raise RuntimeError("FAWA connector requires kv_cache_config.")
        groups: dict[int, dict[str, torch.Tensor]] = {}
        used_names: set[str] = set()
        for group_id, group_spec in enumerate(self._kv_cache_config.kv_cache_groups):
            group_caches: dict[str, torch.Tensor] = {}
            for name in group_spec.layer_names:
                if name not in kv_caches:
                    continue
                kv_cache = kv_caches[name]
                tensor_indices = (
                    self.block_span_layout.group_tensor_indices(group_id, name)
                    if self.block_span_layout is not None
                    else None
                )
                if tensor_indices is None:
                    group_caches[name] = kv_cache
                else:
                    if not isinstance(kv_cache, (list, tuple)):
                        raise TypeError(
                            f"FAWA Ascend KV cache {name} must be tuple/list-like, "
                            f"got {type(kv_cache)}."
                        )
                    missing = [
                        tensor_index
                        for tensor_index in tensor_indices
                        if tensor_index >= len(kv_cache)
                    ]
                    if missing:
                        raise RuntimeError(
                            f"FAWA Ascend KV cache {name} has {len(kv_cache)} "
                            f"tensors, missing indices {missing} for group "
                            f"{group_id}."
                        )
                    selected = tuple(
                        kv_cache[tensor_index] for tensor_index in tensor_indices
                    )
                    group_caches[name] = selected[0] if len(selected) == 1 else selected
            if group_caches:
                groups[group_id] = group_caches
                used_names.update(group_caches)

        missing_names = set(kv_caches) - used_names
        if missing_names:
            raise RuntimeError(
                "KV cache config did not include registered caches: "
                f"{sorted(missing_names)}"
            )

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

        grouped = self._split_kv_caches_by_vllm_groups(kv_caches)
        for group_id, group_caches in grouped.items():
            if not group_caches:
                logger.warning(f"KV cache group {group_id} is empty.")
                continue
            layout = KVCacheGroupLayout(group_caches)
            self.group_layouts[group_id] = layout

        self.store = self._create_fa_store(self.group_layouts, store_cores)
        self.fa_store = self.store
        self.wa_store = self._create_wa_store(
            self.group_layouts,
            store_cores,
        )

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

    def _store_tensor_size_list(
        self,
        group_layouts: dict[int, KVCacheGroupLayout],
        group_ids: tuple[int, ...],
    ) -> list[int]:
        tensor_size_list: list[int] = []
        for group_id in group_ids:
            layout = group_layouts.get(group_id)
            if layout is None:
                continue
            repeat = self.group_tail_blocks[group_id]
            if repeat is None:
                repeat = 1
            for segment_tokens in self.group_window_spans[group_id][:repeat]:
                segment_sizes = layout.segment_tensor_size_list(
                    segment_tokens,
                    self.group_tensor_block_sizes[group_id],
                )
                tensor_size_list.extend(segment_sizes)
        if not tensor_size_list:
            group_label = (
                "FA"
                if group_ids == self.fa_group_ids
                else "WA" if group_ids == self.window_group_ids else str(group_ids)
            )
            raise RuntimeError(f"Worker FAWA {group_label} layout is empty.")
        return tensor_size_list

    def _lookup_external_hit_blocks(self, external_keys: list[bytes]) -> int:
        if self.fa_store is None:
            raise RuntimeError("FA store is not initialized.")
        if self.wa_store is None:
            raise RuntimeError("WA store is not initialized.")
        fa_hit_blocks = self.fa_store.lookup_on_prefix(external_keys) + 1
        if fa_hit_blocks <= 0:
            return 0

        # WA rows represent window boundary state, so they are not required to
        # form a prefix. Search only inside the FA-contiguous hit range and use
        # the latest boundary that exists.
        window_hits = self.wa_store.lookup(external_keys[:fa_hit_blocks])
        for hit_idx in range(len(window_hits) - 1, -1, -1):
            if window_hits[hit_idx]:
                return hit_idx + 1
        return 0

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if num_computed_tokens % self.hash_block_size != 0:
            raise RuntimeError(
                f"FAWA requires aligned computed tokens, got "
                f"{num_computed_tokens} with block size {self.hash_block_size}."
            )
        hbm_hit_block_num = num_computed_tokens // self.hash_block_size
        canonical_hashes = self.generate_hash(
            self.hash_block_size, request.all_token_ids, self._seed
        )

        if self.persist_token_threshold > request.num_tokens:
            return 0, False

        external_keys = canonical_hashes[hbm_hit_block_num:]
        if not external_keys:
            return 0, False

        try:
            external_hit_blocks = self._lookup_external_hit_blocks(external_keys)
        except Exception as e:
            external_hit_blocks = 0
            logger.error(
                f"request {request.request_id} FAWA lookup error. "
                f"{type(e).__name__}: {e}"
            )

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks
        external_hit_tokens = external_hit_blocks * self.hash_block_size
        num_total_hit_tokens = total_hit_block_num * self.hash_block_size
        if num_total_hit_tokens == request.num_tokens:
            external_hit_tokens -= 1

        self.requests_meta[request.request_id] = FAWARequestMeta(
            ucm_block_ids=canonical_hashes,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
        )
        logger.info_once(
            f"FAWA request_id: {request.request_id}, "
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
        pass

    def _generate_dispatch_meta(
        self,
        req_meta: FAWARequestMeta,
        new_tokens: int,
        vllm_block_ids: tuple[list[int], ...],
        need_load: bool = True,
    ) -> FAWARequestDispatchMeta:
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
        for i, block_ids in enumerate(vllm_block_ids):
            req_meta.vllm_block_ids[i].extend(block_ids)

        load_block_keys, dump_block_keys = [], []
        load_start,load_end,dump_start,dump_end = 0,0,0,0
        load_vllm_block_ids, dump_vllm_block_ids= [], []
        if need_load:
            load_start = hbm_hit_block_num
            load_end = total_hit_block_num
            load_block_keys = ucm_block_ids[load_start:load_end]
            for g_id, meta in self.group_metas.items():
                g_start = int(load_start * meta.logical_blocks_per_hash_block)
                g_end = int(load_end * meta.logical_blocks_per_hash_block)
                load_vllm_block_ids.append(vllm_block_ids[g_id][g_start:g_end])
            

        if req_meta.token_processed < req_meta.num_token_ids:
            dump_start = req_meta.token_processed // self.hash_block_size
            dump_end = (req_meta.token_processed + new_tokens) // self.hash_block_size
            dump_block_keys = ucm_block_ids[dump_start:dump_end]
            for g_id, meta in self.group_metas.items():
                g_start = int(dump_start * meta.logical_blocks_per_hash_block)
                g_end = int(dump_end * meta.logical_blocks_per_hash_block)
                dump_vllm_block_ids.append(vllm_block_ids[g_id][g_start:g_end])
            req_meta.token_processed += new_tokens

        return FAWARequestDispatchMeta(
            load_keys=load_block_keys,
            load_hash_start=load_start,
            load_hash_end=load_end,
            load_vllm_block_ids=tuple(load_vllm_block_ids),
            dump_keys=dump_block_keys,
            dump_hash_start=dump_start,
            dump_hash_end=dump_end,
            dump_vllm_block_ids=tuple(dump_vllm_block_ids),
        )
    
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta: dict[str, FAWARequestDispatchMeta] = {}
        # for new request, we need to load and dump
        for request in scheduler_output.scheduled_new_reqs:
            request_id, vllm_block_ids = request.req_id, request.block_ids
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    vllm_block_ids,
                )


        
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                new_block_ids = []
                if scheduled_cached_reqs.new_block_ids[i] != None:
                    new_block_ids = scheduled_cached_reqs.new_block_ids[i]
                
                resumed_from_preemption = (
                    request_id in scheduled_cached_reqs.resumed_req_ids
                )
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    new_block_ids,
                    resumed_from_preemption,
                )

        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMFAWAConnectorMetadata(requests_dispatch_meta)

    def update_connector_output(self, connector_output) -> None:
        return None

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        return None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        return False, None

    def _submit_load_task(
        self,
        request_id: str,
        label: str,
        store: UcmKVStoreBaseV1,
        keys: list[bytes],
        ptrs: np.ndarray,
        anchor_vllm_block_ids: set[int],
    ) -> FAWALoadTask:
        shard_indices = [0] * len(keys)
        task = store.load_data(keys, shard_indices, ptrs)
        return FAWALoadTask(
            request_id=request_id,
            label=label,
            store=store,
            task=task,
            key_count=len(keys),
            anchor_vllm_block_ids=anchor_vllm_block_ids,
        )

    def _wait_load_task(
        self,
        load_task: FAWALoadTask,
    ) -> None:
        try:
            load_task.store.wait(load_task.task)
            logger.info(
                f"request {load_task.request_id} FAWA load "
                f"task label={load_task.label} succeeded, "
                f"blocks={load_task.key_count}"
            )
        except Exception as e:
            logger.error(
                f"request {load_task.request_id} wait FAWA load "
                f"task label={load_task.label} error. {type(e).__name__}: {e}"
            )
            self._invalid_block_ids.update(load_task.anchor_vllm_block_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get vLLM block IDs that failed to load through FAWA stores.

        Returns:
            Set of vLLM/HMA block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        res = self._invalid_block_ids
        self._invalid_block_ids = set()
        return res

    def _submit_dump_task(
        self,
        label: str,
        store: UcmKVStoreBaseV1,
        keys: list[bytes],
        ptrs: np.ndarray,
        event_handle,
    ) -> FAWADumpTask:
        shard_indices = [0] * len(keys)
        task = store.dump_data(keys, shard_indices, ptrs, event_handle)
        return FAWADumpTask(
            label=label,
            store=store,
            task=task,
            key_count=len(keys),
        )

    def _wait_dump_task(self, dump_task: FAWADumpTask) -> None:
        dump_task.store.wait(dump_task.task)
        logger.info(
            f"FAWA dump task label={dump_task.label} succeeded, "
            f"blocks={dump_task.key_count}"
        )

    def _extract_fa_ptr(self, store_keys, hash_start, hash_end, candidate_vllm_ids):
        """
        this function need to extract the data ptr, but for Ascend need to consider more
        hash_start, hash_end is the range of the contiguous part needs to be load or dump
        the relation of hash_start, hash_end and candidate_vllm_ids can refer _generate_dispatch_meta func
        for each hash_block_idx, we can get the vllm block id and offset for each group's tensor
        """
        pass

    def _extract_wa_ptr(self, store_keys, hash_start, hash_end, candidate_vllm_ids):
        """
        samilar as _extract_fa_ptr, but for Ascend need to consider more
        """
        pass

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMFAWAConnectorMetadata):
            raise RuntimeError(f"Unexpected FAWA metadata type: {type(metadata)}")

        tasks: list[FAWALoadTask] = []
        for request_id, request in metadata.request_meta.items():
            store_keys = request.load_keys
            hash_start = request.load_hash_start
            hash_end = request.load_hash_end
            candidate_vllm_ids = request.load_vllm_block_ids
            if not store_keys:
                continue
            try:
                # FA groups are loaded for every external-hit canonical block.
                fa_ptrs = self._extract_fa_ptr(store_keys, hash_start, hash_end, candidate_vllm_ids)
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "FA",
                        self.fa_store,
                        store_keys,
                        fa_ptrs,
                        candidate_vllm_ids,
                    )
                )

                if self.wa_store is None:
                    raise RuntimeError("WA store is not initialized.")
                # WA groups only need the final matched boundary.
                window_keys = store_keys[-1:]
                window_ptrs = self._extract_wa_ptr(store_keys, hash_end-1, hash_end, candidate_vllm_ids)
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "WA",
                        self.wa_store,
                        window_keys,
                        window_ptrs,
                        candidate_vllm_ids,
                    )
                )
            except Exception as e:
                logger.error(
                    f"request {request_id} submit FAWA load task "
                    f"error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(candidate_vllm_ids)

        for load_task in tasks:
            self._wait_load_task(load_task)

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMFAWAConnectorMetadata):
            raise RuntimeError(f"Unexpected FAWA metadata type: {type(metadata)}")

        if self.tp_rank != 0:
            return

        try:
            event_handle = self._get_dump_event_handle()
            if self.fa_store is None:
                raise RuntimeError("FA store is not initialized.")
            if self.wa_store is None:
                raise RuntimeError("WA store is not initialized.")


            total_keys: list[bytes] = []
            total_group_rows: KVCacheGroupRows = []
            fa_ptrs = []
            window_ptrs = []
            for request in metadata.request_meta.values():
                store_keys = request.dump_keys
                hash_start = request.dump_hash_start
                hash_end = request.dump_hash_end
                candidate_vllm_ids = request.dump_vllm_block_ids
                if not store_keys:
                    continue
                total_keys.extend(store_keys)
                req_fa_ptrs = self._extract_fa_ptr(store_keys, hash_start, hash_end, candidate_vllm_ids)
                req_window_ptrs = self._extract_wa_ptr(store_keys, hash_start, hash_end, candidate_vllm_ids)
                fa_ptrs.extend(req_fa_ptrs)
                window_ptrs.extend(req_window_ptrs)

            if not total_keys:
                return

            tasks: list[FAWADumpTask] = []

            tasks.append(
                self._submit_dump_task(
                    "FA",
                    self.fa_store,
                    total_keys,
                    fa_ptrs,
                    event_handle,
                )
            )
            tasks.append(
                self._submit_dump_task(
                    "WA",
                    self.wa_store,
                    total_keys,
                    window_ptrs,
                    event_handle,
                )
            )
            for dump_task in tasks:
                self._wait_dump_task(dump_task)
        except Exception as e:
            logger.error(f"dump FAWA kv cache failed. {type(e).__name__}: {e}")


class UCMAscendFAWAConnector(UCMFAWAConnector):
    """Ascend FAWA connector with segmented tensor KV block mapping."""

    def _create_block_span_layout(self) -> Optional[FAWABlockSpanLayout]:
        return FAWABlockSpanLayout(self._kv_cache_config, self.fa_group_ids)

    def _get_hash_block_size(self) -> int:
        if self.block_span_layout is None:
            raise RuntimeError("Ascend FAWA connector requires block span layout.")
        return self.block_span_layout.hash_block_size

    def _get_group_token_block_sizes(self) -> tuple[int, ...]:
        if self.block_span_layout is None:
            raise RuntimeError("Ascend FAWA connector requires block span layout.")
        self._ascend_layout = self.block_span_layout.is_ascend
        group_token_block_sizes = self.block_span_layout.group_token_block_sizes
        self._validate_group_token_block_sizes(group_token_block_sizes)
        return group_token_block_sizes

    def _get_group_tensor_block_sizes(self) -> tuple[int, ...]:
        if self.block_span_layout is None:
            raise RuntimeError("Ascend FAWA connector requires block span layout.")
        return self.block_span_layout.group_tensor_block_sizes

    def _ascend_window_tail_tokens(self, group_id: int) -> Optional[int]:
        group_spec = self._kv_cache_config.kv_cache_groups[group_id]
        window_tokens = FAWABlockSpanLayout.group_window_tokens(group_spec)
        if window_tokens is None or self.block_span_layout.is_swa_group(group_id):
            return window_tokens
        compress_ratio = self.block_span_layout.state_compress_ratio(group_id)
        if compress_ratio is None:
            return window_tokens
        return max(0, window_tokens - compress_ratio)

    def _get_group_tail_blocks(self) -> tuple[Optional[int], ...]:
        tail_blocks: list[Optional[int]] = [None] * len(self.group_token_block_sizes)
        for group_id in self.window_group_ids:
            group_token_block_size = self.group_token_block_sizes[group_id]
            window_tail_tokens = self._ascend_window_tail_tokens(group_id)
            if window_tail_tokens is None:
                tail_blocks[group_id] = self.hash_block_size // group_token_block_size
            elif window_tail_tokens == 0:
                tail_blocks[group_id] = 0
            elif self.block_span_layout.is_swa_group(group_id):
                tail_blocks[group_id] = max(
                    1,
                    math.ceil(window_tail_tokens / group_token_block_size),
                )
            else:
                tail_blocks[group_id] = math.ceil(
                    window_tail_tokens / group_token_block_size
                )
        return tuple(tail_blocks)

    def _get_group_window_spans(self) -> tuple[tuple[int, ...], ...]:
        spans: list[tuple[int, ...]] = []
        for group_id, tail_blocks in enumerate(self.group_tail_blocks):
            if tail_blocks is None:
                spans.append((self.group_token_block_sizes[group_id],))
                continue
            if tail_blocks == 0:
                spans.append(())
                continue
            window_tail_tokens = self._ascend_window_tail_tokens(group_id)
            if window_tail_tokens is None or self.block_span_layout.is_swa_group(
                group_id
            ):
                spans.append((self.group_token_block_sizes[group_id],) * tail_blocks)
                continue
            group_token_block_size = self.group_token_block_sizes[group_id]
            group_spans: list[int] = []
            while window_tail_tokens > 0:
                segment_tokens = min(group_token_block_size, window_tail_tokens)
                group_spans.append(segment_tokens)
                window_tail_tokens -= segment_tokens
            spans.append(tuple(reversed(group_spans)))
        return tuple(spans)

    def _group_block_range(self, group_id: int, computed_end_token: int) -> range:
        if (
            self.group_tail_blocks[group_id] is None
            or self.group_tail_blocks[group_id] == 0
            or self.block_span_layout.is_swa_group(group_id)
        ):
            return super()._group_block_range(group_id, computed_end_token)

        window_tail_tokens = self._ascend_window_tail_tokens(group_id)
        if window_tail_tokens is None:
            return super()._group_block_range(group_id, computed_end_token)
        if window_tail_tokens == 0:
            return range(0, 0)

        group_token_block_size = self.group_token_block_sizes[group_id]
        tail_end_token = computed_end_token
        tail_start_token = max(0, tail_end_token - window_tail_tokens)
        start_block = tail_start_token // group_token_block_size
        end_block = math.ceil(tail_end_token / group_token_block_size)
        return range(start_block, end_block)

    def _block_index_to_segment(
        self,
        group_id: int,
        group_block_idx: int,
        block_id: int,
        computed_end_token: Optional[int] = None,
    ) -> KVCacheSegment:
        segment = super()._block_index_to_segment(group_id, group_block_idx, block_id)
        return self._trim_window_segment(
            group_id,
            group_block_idx,
            computed_end_token,
            segment,
        )

    def _trim_window_segment(
        self,
        group_id: int,
        group_block_idx: int,
        computed_end_token: Optional[int],
        segment: KVCacheSegment,
    ) -> KVCacheSegment:
        if (
            group_id not in self.window_group_ids
            or self.group_tail_blocks[group_id] is None
            or self.group_tail_blocks[group_id] == 0
            or self.block_span_layout.is_swa_group(group_id)
        ):
            return segment

        window_tail_tokens = self._ascend_window_tail_tokens(group_id)
        if window_tail_tokens is None:
            return segment
        if window_tail_tokens == 0:
            return segment

        group_token_block_size = self.group_token_block_sizes[group_id]
        tail_end_token = (
            computed_end_token
            if computed_end_token is not None
            else (group_block_idx + 1) * group_token_block_size
        )
        tail_start_token = max(0, tail_end_token - window_tail_tokens)
        block_start_token = group_block_idx * group_token_block_size
        segment_start_in_block = max(0, tail_start_token - block_start_token)
        segment_end_in_block = min(
            group_token_block_size, tail_end_token - block_start_token
        )
        length = max(0, segment_end_in_block - segment_start_in_block)
        return KVCacheSegment(
            block_id=segment.block_id,
            offset=segment.offset + segment_start_in_block,
            length=length,
        )
