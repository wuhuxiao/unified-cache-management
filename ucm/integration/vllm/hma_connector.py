import copy
import logging
import math
import os
from collections.abc import Iterator, Mapping, Sequence as SequenceABC
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

    def extract_segment_addrs_flat(
        self,
        segment: KVCacheSegment,
        group_tensor_block_size: int,
    ) -> np.ndarray:
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

    def extract_segment_addrs_flat_batch(
        self,
        block_ids: np.ndarray,
        offsets: np.ndarray,
        group_tensor_block_size: int,
    ) -> np.ndarray:
        if len(block_ids) == 0:
            return np.empty((0, len(self.base_ptrs)), dtype=np.uint64)
        block_ids_np = np.asarray(block_ids, dtype=np.uint64)
        offsets_np = np.asarray(offsets, dtype=np.uint64)
        scaled = offsets_np[:, None] * self.view_tensor_block_sizes[None, :]
        divisor = np.uint64(group_tensor_block_size)
        if np.any(scaled % divisor):
            raise ValueError(
                f"Logical segment offsets do not align with view tensor block "
                f"sizes={self.view_tensor_block_sizes.tolist()} and group "
                f"tensor block size={group_tensor_block_size}."
            )
        tensor_offsets = scaled // divisor
        return (
            block_ids_np[:, None] * self.block_strides[None, :]
            + tensor_offsets * self.token_strides[None, :]
            + self.base_ptrs[None, :]
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


# One canonical hash block mapped to segment spans for each vLLM KV cache group.
KVCacheGroupRow = tuple[list[KVCacheSegment], ...]
# Multiple canonical hash blocks, each represented as one KVCacheGroupRow.
KVCacheGroupRows = list[KVCacheGroupRow]
# Full HMA allocation snapshot by kv-cache group.
KVCacheGroupAllocation = tuple[Sequence[int] | "BlockIdSequence", ...]


class CompactKVBlockPlan:
    """Dense row plan for canonical FAWA blocks.

    The hot scheduler path records primitive arrays here. Compatibility helpers
    materialize KVCacheSegment rows only when tests or legacy helpers inspect
    group_block_ids directly.
    """

    def __init__(
        self,
        segment_counts: tuple[int, ...],
        initial_capacity: int = 0,
    ) -> None:
        self.segment_counts = segment_counts
        self.recorded_end = 0
        self._capacity = max(0, initial_capacity)
        self.block_ids = [
            np.full((self._capacity, count), -1, dtype=np.int64)
            for count in segment_counts
        ]
        self.offsets = [
            np.zeros((self._capacity, count), dtype=np.int64)
            for count in segment_counts
        ]
        self.lengths = [
            np.zeros((self._capacity, count), dtype=np.int64)
            for count in segment_counts
        ]
        self._range_cache: dict[tuple[int, int], np.ndarray] = {}
        self._matrix_cache: dict[tuple[int, int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def clear(self) -> None:
        self.recorded_end = 0
        self._range_cache.clear()
        self._matrix_cache.clear()

    def _ensure_capacity(self, row_count: int) -> None:
        if row_count <= self._capacity:
            return
        new_capacity = max(row_count, max(1, self._capacity * 2))
        for arrays, fill_value in (
            (self.block_ids, -1),
            (self.offsets, 0),
            (self.lengths, 0),
        ):
            for group_id, current in enumerate(arrays):
                expanded = np.full(
                    (new_capacity, current.shape[1]),
                    fill_value,
                    dtype=current.dtype,
                )
                if self._capacity:
                    expanded[: self._capacity, :] = current
                arrays[group_id] = expanded
        self._capacity = new_capacity

    def row_range(self, start: int, end: int) -> np.ndarray:
        key = (start, end)
        cached = self._range_cache.get(key)
        if cached is None:
            cached = np.arange(start, end, dtype=np.int64)
            self._range_cache[key] = cached
        return cached

    def group_index_matrix(
        self,
        cache_key: tuple[int, int, int, int, int],
        canonical_block_indices: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        del canonical_block_indices
        return self._matrix_cache.get(cache_key)

    def set_group_index_matrix(
        self,
        cache_key: tuple[int, int, int, int, int],
        value: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self._matrix_cache[cache_key] = value

    def record_row(
        self,
        block_idx: int,
        group_segments: list[list[tuple[int, int, int]]],
    ) -> None:
        self._ensure_capacity(block_idx + 1)
        for group_id, segments in enumerate(group_segments):
            expected = self.segment_counts[group_id]
            if len(segments) != expected:
                raise RuntimeError(
                    f"FAWA compact row group {group_id} has {len(segments)} "
                    f"segments, expected {expected}."
                )
            for segment_idx, (block_id, offset, length) in enumerate(segments):
                self.block_ids[group_id][block_idx, segment_idx] = block_id
                self.offsets[group_id][block_idx, segment_idx] = offset
                self.lengths[group_id][block_idx, segment_idx] = length
        self.recorded_end = max(self.recorded_end, block_idx + 1)

    def materialize_row(self, block_idx: int) -> KVCacheGroupRow:
        if block_idx < 0 or block_idx >= self.recorded_end:
            raise KeyError(block_idx)
        row: list[list[KVCacheSegment]] = []
        for group_id, count in enumerate(self.segment_counts):
            group_segments: list[KVCacheSegment] = []
            for segment_idx in range(count):
                group_segments.append(
                    KVCacheSegment(
                        int(self.block_ids[group_id][block_idx, segment_idx]),
                        int(self.offsets[group_id][block_idx, segment_idx]),
                        int(self.lengths[group_id][block_idx, segment_idx]),
                    )
                )
            row.append(group_segments)
        return tuple(row)


class CompactKVRows(Mapping[int, KVCacheGroupRow]):
    def __init__(self, plan: CompactKVBlockPlan) -> None:
        self.plan = plan
        self._cache: dict[int, KVCacheGroupRow] = {}

    def __getitem__(self, block_idx: int) -> KVCacheGroupRow:
        cached = self._cache.get(block_idx)
        if cached is not None:
            return cached
        row = self.plan.materialize_row(block_idx)
        self._cache[block_idx] = row
        return row

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.plan.recorded_end))

    def __len__(self) -> int:
        return self.plan.recorded_end

    def __contains__(self, block_idx: object) -> bool:
        return (
            isinstance(block_idx, int)
            and block_idx >= 0
            and block_idx < self.plan.recorded_end
        )

    def clear(self) -> None:
        self.plan.clear()
        self._cache.clear()

    def __eq__(self, other: object) -> bool:
        if other == {}:
            return self.plan.recorded_end == 0
        return dict(self.items()) == other


class KVCacheRowsView(SequenceABC[KVCacheGroupRow]):
    def __init__(
        self,
        rows: Mapping[int, KVCacheGroupRow],
        indices: Sequence[int],
    ) -> None:
        self.rows = rows
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self.rows[idx] for idx in self.indices[item]]
        return self.rows[self.indices[item]]

    def __iter__(self) -> Iterator[KVCacheGroupRow]:
        for idx in self.indices:
            yield self.rows[idx]

    def __eq__(self, other: object) -> bool:
        return list(self) == other


class BlockIdSequence(SequenceABC[int]):
    """Block-id view over vLLM allocation blocks.

    vLLM exposes allocation groups as Python sequences of block objects. FAWA
    normalizes only the ids needed by the current compact row batch, avoiding
    assumptions about extra methods on vLLM's list-like containers.
    """

    def __init__(self, blocks: Sequence[object]) -> None:
        self.blocks = blocks
        self._dense: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._block_id(block) for block in self.blocks[index]]
        return self._block_id(self.blocks[index])

    def __iter__(self) -> Iterator[int]:
        for block in self.blocks:
            yield self._block_id(block)

    def __add__(self, other) -> list[int]:
        return list(self) + list(other)

    def __eq__(self, other: object) -> bool:
        return list(self) == other

    def as_array(self) -> np.ndarray:
        if self._dense is None:
            self._dense = np.fromiter(
                (self._block_id(block) for block in self.blocks),
                dtype=np.int64,
                count=len(self.blocks),
            )
        return self._dense

    def gather(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            return np.empty(indices.shape, dtype=np.int64)
        if self._dense is not None or indices.size >= len(self.blocks):
            return self.as_array()[indices]

        flat_indices = indices.reshape(-1)
        min_idx = int(flat_indices.min())
        max_idx = int(flat_indices.max())
        span = max_idx - min_idx + 1
        if span <= flat_indices.size * 2:
            span_block_ids = np.fromiter(
                (self._block_id(self.blocks[idx]) for idx in range(min_idx, max_idx + 1)),
                dtype=np.int64,
                count=span,
            )
            return span_block_ids[flat_indices - min_idx].reshape(indices.shape)

        unique_indices, inverse = np.unique(flat_indices, return_inverse=True)
        unique_block_ids = np.fromiter(
            (self._block_id(self.blocks[int(idx)]) for idx in unique_indices),
            dtype=np.int64,
            count=len(unique_indices),
        )
        return unique_block_ids[inverse].reshape(indices.shape)

    @staticmethod
    def _block_id(block: object) -> int:
        if isinstance(block, int):
            return block
        return int(getattr(block, "block_id"))


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
    # Canonical per-hash-block remote keys before _block_key() namespacing.
    ucm_block_ids: list[bytes] = field(default_factory=list)
    # Namespaced store keys derived from ucm_block_ids. Keep them cached because
    # scheduler metadata may be rebuilt many times for a long chunk-prefill req.
    block_keys: list[Optional[bytes]] = field(default_factory=list)
    # Number of 256-token hash blocks already hit in vLLM's HBM prefix cache.
    hbm_hit_block_num: int = 0
    # Total prefix hit in hash blocks, including HBM and external UCM hits.
    total_hit_block_num: int = 0
    # Logical request token count; persistence is capped by this, not allocation.
    num_token_ids: int = 0
    # Number of logical tokens already accounted for by scheduler progress.
    token_processed: int = 0
    # First canonical hash block that has not yet been emitted to store.
    store_block_cursor: int = 0
    # First canonical hash block whose per-group HMA row is not recorded yet.
    record_block_cursor: int = 0
    # canonical block idx -> per-kv-cache-group block ids needed to load/store it.
    group_block_ids: Mapping[int, KVCacheGroupRow] = field(init=False)
    # Full HMA allocation snapshot by kv-cache group, updated on alloc/chunk alloc.
    allocated_group_block_ids: KVCacheGroupAllocation = field(default_factory=tuple)
    compact_plan: Optional[CompactKVBlockPlan] = None

    def __post_init__(self) -> None:
        if self.compact_plan is None:
            object.__setattr__(self, "group_block_ids", {})
        else:
            object.__setattr__(self, "group_block_ids", CompactKVRows(self.compact_plan))


@dataclass
class FAWADispatchBlockPlan:
    keys: list[bytes]
    req_meta: FAWARequestMeta
    indices: tuple[int, ...]

    @property
    def rows(self) -> KVCacheRowsView:
        return KVCacheRowsView(self.req_meta.group_block_ids, self.indices)


@dataclass
class FAWARequestDispatchMeta:
    load_block_plan: Optional[FAWADispatchBlockPlan] = None
    dump_block_plan: Optional[FAWADispatchBlockPlan] = None

    @property
    def load_block_ids(self) -> tuple[list[bytes], KVCacheGroupRows]:
        if self.load_block_plan is None:
            return [], []
        return self.load_block_plan.keys, list(self.load_block_plan.rows)

    @property
    def dump_block_ids(self) -> tuple[list[bytes], KVCacheGroupRows]:
        if self.dump_block_plan is None:
            return [], []
        return self.dump_block_plan.keys, list(self.dump_block_plan.rows)


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
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        self._defer_scheduler_store = True
        super().__init__(vllm_config, role, kv_cache_config)
        self.hash_block_size = self.DEFAULT_HASH_BLOCK_SIZE
        self.block_size = self.DEFAULT_HASH_BLOCK_SIZE
        self.group_layouts: dict[int, KVCacheGroupLayout] = {}
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
        if self._kv_cache_config is None:
            raise RuntimeError("FAWA connector requires kv_cache_config.")
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

    def _block_key(self, canonical_hash: bytes) -> bytes:
        return self.request_hasher((b"fawa", canonical_hash))

    def _block_keys_for_indices(
        self,
        req_meta: FAWARequestMeta,
        indices: list[int],
    ) -> list[bytes]:
        if not indices:
            return []
        if len(req_meta.block_keys) < len(req_meta.ucm_block_ids):
            req_meta.block_keys.extend(
                [None] * (len(req_meta.ucm_block_ids) - len(req_meta.block_keys))
            )

        keys: list[bytes] = []
        for idx in indices:
            block_key = req_meta.block_keys[idx]
            if block_key is None:
                block_key = self._block_key(req_meta.ucm_block_ids[idx])
                req_meta.block_keys[idx] = block_key
            keys.append(block_key)
        return keys

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

    def _compact_segment_counts(self) -> tuple[int, ...]:
        # Each canonical hash block becomes one row. The row width can differ by
        # group: FA groups have one segment, WA groups have configured tail
        # segments, and zero-tail state groups have none.
        return tuple(
            self._expected_group_row_segments(group_id)
            for group_id in range(len(self.group_token_block_sizes))
        )

    def _ensure_compact_plan(self, req_meta: FAWARequestMeta) -> CompactKVBlockPlan:
        # The scheduler hot path writes primitive numpy arrays instead of
        # materializing KVCacheSegment objects for every request/block/group row.
        segment_counts = self._compact_segment_counts()
        if (
            req_meta.compact_plan is None
            or req_meta.compact_plan.segment_counts != segment_counts
        ):
            req_meta.compact_plan = CompactKVBlockPlan(
                segment_counts,
                req_meta.num_token_ids // self.hash_block_size,
            )
            req_meta.group_block_ids = CompactKVRows(req_meta.compact_plan)
        return req_meta.compact_plan

    def _group_block_range(self, group_id: int, computed_end_token: int) -> range:
        # The block row is determined by the logical token boundary, not by
        # how many tensor blocks vLLM has already allocated.
        if self.group_tail_blocks[group_id] is None:
            canonical_block_idx = computed_end_token // self.hash_block_size - 1
            return range(canonical_block_idx, canonical_block_idx + 1)

        total_group_blocks = (
            computed_end_token // self.group_token_block_sizes[group_id]
        )
        tail_blocks = self.group_tail_blocks[group_id]
        # Some compressor-state groups need no window tail for a prefix hit.
        if not tail_blocks:
            return range(total_group_blocks, total_group_blocks)
        start = max(0, total_group_blocks - tail_blocks)
        return range(start, total_group_blocks)

    def _expected_group_row_segments(self, group_id: int) -> int:
        tail_blocks = self.group_tail_blocks[group_id]
        return 1 if tail_blocks is None else int(tail_blocks)

    def _block_index_to_segment(
        self,
        group_id: int,
        group_block_idx: int,
        block_id: int,
        computed_end_token: Optional[int] = None,
    ) -> KVCacheSegment:
        if self.block_span_layout is None:
            token_blocks_per_tensor_block = self._group_tensor_block_ratio(group_id)
            return KVCacheSegment(
                block_id=block_id,
                offset=(group_block_idx % token_blocks_per_tensor_block)
                * self.group_token_block_sizes[group_id],
                length=self.group_token_block_sizes[group_id],
            )
        return self.block_span_layout.block_index_to_segment(
            group_id,
            group_block_idx,
            block_id,
        )

    def _extract_compact_group_addrs(
        self,
        plan: CompactKVBlockPlan,
        indices: Sequence[int],
        group_ids: tuple[int, ...],
    ) -> np.ndarray:
        if not indices:
            return np.empty((0, 0), dtype=np.uint64)
        row_indices = np.asarray(indices, dtype=np.int64)
        row_parts: list[np.ndarray] = []
        row_count = len(row_indices)
        for group_id in group_ids:
            layout = self.group_layouts.get(group_id)
            if layout is None:
                continue
            segment_count = plan.segment_counts[group_id]
            if segment_count == 0:
                continue
            group_block_ids = plan.block_ids[group_id][row_indices, :]
            group_offsets = plan.offsets[group_id][row_indices, :]
            for segment_idx in range(segment_count):
                block_ids = group_block_ids[:, segment_idx]
                offsets = group_offsets[:, segment_idx]
                row_parts.append(
                    layout.extract_segment_addrs_flat_batch(
                        block_ids,
                        offsets,
                        self.group_tensor_block_sizes[group_id],
                    )
                )
        if not row_parts:
            raise ValueError("KV cache pointer row is empty.")
        return np.concatenate(row_parts, axis=1).astype(np.uint64, copy=False)

    def _try_select_compact_group_segments(
        self,
        allocated_group_block_ids: KVCacheGroupAllocation,
        computed_end_token: int,
    ) -> Optional[list[list[tuple[int, int, int]]]]:
        """Build one compact row from the current vLLM allocation snapshot.

        Returning None means chunk prefill has not allocated enough HBM blocks yet.
        """
        selected: list[list[tuple[int, int, int]]] = []
        for group_id in range(len(self.group_token_block_sizes)):
            group_indices = self._group_block_range(group_id, computed_end_token)
            group_selected: list[tuple[int, int, int]] = []
            if group_id >= len(allocated_group_block_ids):
                if group_indices:
                    return None
                selected.append(group_selected)
                continue

            group_blocks = allocated_group_block_ids[group_id]
            for group_block_idx in group_indices:
                if self.block_span_layout is not None:
                    tensor_block_idx = self.block_span_layout.allocation_index(
                        group_id,
                        group_block_idx,
                    )
                else:
                    tensor_block_idx = (
                        group_block_idx // self._group_tensor_block_ratio(group_id)
                    )
                if tensor_block_idx >= len(group_blocks):
                    return None
                block_id = group_blocks[tensor_block_idx]
                segment = self._block_index_to_segment(
                    group_id,
                    group_block_idx,
                    block_id,
                    computed_end_token,
                )
                group_selected.append((segment.block_id, segment.offset, segment.length))
            expected_segments = self._expected_group_row_segments(group_id)
            if len(group_selected) != expected_segments:
                raise RuntimeError(
                    f"KV cache group {group_id} selected {len(group_selected)} "
                    f"segments for computed end token {computed_end_token}, "
                    f"expected {expected_segments}. This indicates the FAWA "
                    "hash block boundary does not cover the configured window "
                    "tail."
                )
            selected.append(group_selected)
        return selected

    def _group_block_index_matrix(
        self,
        group_id: int,
        canonical_block_indices: np.ndarray,
    ) -> np.ndarray:
        """Vectorized mapping from canonical rows to per-group block indices."""
        expected_segments = self._expected_group_row_segments(group_id)
        if expected_segments == 0:
            return np.empty((len(canonical_block_indices), 0), dtype=np.int64)
        if self.group_tail_blocks[group_id] is None:
            return canonical_block_indices[:, None].astype(np.int64, copy=False)

        computed_end_tokens = (canonical_block_indices + 1) * self.hash_block_size
        total_group_blocks = (
            computed_end_tokens // self.group_token_block_sizes[group_id]
        )
        positions = np.arange(expected_segments, dtype=np.int64)
        indices = total_group_blocks[:, None] - expected_segments + positions[None, :]
        if np.any(indices < 0):
            # Current FAWA layouts require one hash block boundary to cover the
            # whole WA tail window, so early-window padding is treated as a bad
            # configuration instead of silently creating scratch rows.
            first_row = int(np.argmax(np.any(indices < 0, axis=1)))
            computed_end_token = int(computed_end_tokens[first_row])
            raise RuntimeError(
                f"KV cache group {group_id} has fewer group blocks than the "
                f"configured window tail at computed end token "
                f"{computed_end_token}. This indicates the FAWA hash block "
                "boundary does not cover the configured window tail."
            )
        return indices

    def _cached_group_block_index_matrix(
        self,
        plan: CompactKVBlockPlan,
        group_id: int,
        canonical_block_indices: np.ndarray,
    ) -> np.ndarray:
        start = int(canonical_block_indices[0]) if len(canonical_block_indices) else 0
        end = int(canonical_block_indices[-1]) + 1 if len(canonical_block_indices) else 0
        key = (
            start,
            end,
            self.group_token_block_sizes[group_id],
            int(self.group_tail_blocks[group_id] or -1),
            plan.segment_counts[group_id],
        )
        cached = plan.group_index_matrix(key, canonical_block_indices)
        if cached is not None:
            return cached
        matrix = self._group_block_index_matrix(group_id, canonical_block_indices)
        plan.set_group_index_matrix(key, matrix)
        return matrix

    def _trim_compact_segment_arrays(
        self,
        group_id: int,
        canonical_block_indices: np.ndarray,
        group_block_indices: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del group_id, canonical_block_indices, group_block_indices
        return offsets, lengths

    def _allocation_index_matrix(
        self,
        group_id: int,
        group_block_indices: np.ndarray,
    ) -> np.ndarray:
        if self.block_span_layout is not None:
            ratio = self.block_span_layout._group_tensor_block_ratio(group_id)
        else:
            ratio = self._group_tensor_block_ratio(group_id)
        return group_block_indices // ratio

    def _record_ready_group_block_ids_fast(
        self,
        req_meta: FAWARequestMeta,
    ) -> None:
        """Record all newly complete canonical rows in a vectorized pass.

        Chunk prefill can append allocations over several scheduler steps. This
        method advances only through the longest contiguous range whose every
        required per-group tensor block has been allocated.
        """
        allocated_group_block_ids = req_meta.allocated_group_block_ids
        if not allocated_group_block_ids:
            return

        start_block = req_meta.record_block_cursor
        max_full_blocks = req_meta.num_token_ids // self.hash_block_size
        if start_block >= max_full_blocks:
            return

        plan = self._ensure_compact_plan(req_meta)
        all_indices = plan.row_range(start_block, max_full_blocks)
        end_block = max_full_blocks
        for group_id in range(len(self.group_token_block_sizes)):
            if plan.segment_counts[group_id] == 0:
                continue
            if group_id >= len(allocated_group_block_ids):
                return
            group_indices = self._cached_group_block_index_matrix(
                plan,
                group_id,
                all_indices,
            )
            allocation_indices = self._allocation_index_matrix(group_id, group_indices)
            row_valid_mask = np.all(
                allocation_indices < len(allocated_group_block_ids[group_id]),
                axis=1,
            )
            if not np.all(row_valid_mask):
                first_invalid = int(np.argmax(~row_valid_mask))
                end_block = min(end_block, start_block + first_invalid)
        if end_block <= start_block:
            return

        row_indices = plan.row_range(start_block, end_block)
        plan._ensure_capacity(end_block)
        for group_id in range(len(self.group_token_block_sizes)):
            expected_segments = plan.segment_counts[group_id]
            if expected_segments == 0:
                continue
            group_indices = self._cached_group_block_index_matrix(
                plan,
                group_id,
                row_indices,
            )
            allocation_indices = self._allocation_index_matrix(group_id, group_indices)
            group_blocks_source = allocated_group_block_ids[group_id]
            if isinstance(group_blocks_source, BlockIdSequence):
                group_blocks = group_blocks_source.gather(allocation_indices)
            else:
                group_blocks = np.asarray(group_blocks_source, dtype=np.int64)[
                    allocation_indices
                ]

            ratio = self._group_tensor_block_ratio(group_id)
            offsets = (
                group_indices % ratio
            ) * self.group_token_block_sizes[group_id]
            lengths = np.full(
                group_indices.shape,
                self.group_token_block_sizes[group_id],
                dtype=np.int64,
            )
            offsets, lengths = self._trim_compact_segment_arrays(
                group_id,
                row_indices,
                group_indices,
                offsets,
                lengths,
            )
            plan.block_ids[group_id][start_block:end_block, :] = group_blocks
            plan.offsets[group_id][start_block:end_block, :] = offsets
            plan.lengths[group_id][start_block:end_block, :] = lengths
        plan.recorded_end = max(plan.recorded_end, end_block)
        req_meta.record_block_cursor = end_block

    def _record_allocated_group_block_ids(
        self,
        req_meta: FAWARequestMeta,
        group_block_ids: tuple[list[int], ...] | None,
        replace: bool = False,
    ) -> None:
        """Merge vLLM allocation updates and derive ready canonical rows."""
        if group_block_ids is None:
            return
        # cached.new_block_ids contains only blocks allocated in this scheduler
        # step. Append it for chunk prefill; replace it for resumed requests.
        new_group_block_ids = tuple(
            np.asarray(group, dtype=np.int64) for group in group_block_ids
        )
        if replace or not req_meta.allocated_group_block_ids:
            req_meta.allocated_group_block_ids = new_group_block_ids
            if replace:
                # A resumed request may receive a fresh allocation snapshot; rows
                # derived from the old block ids must not survive into metadata.
                self._ensure_compact_plan(req_meta).clear()
                req_meta.record_block_cursor = 0
        else:
            if len(req_meta.allocated_group_block_ids) != len(new_group_block_ids):
                raise RuntimeError(
                    "FAWA cached allocation update has mismatched group count: "
                    f"current={len(req_meta.allocated_group_block_ids)}, "
                    f"new={len(new_group_block_ids)}."
                )
            req_meta.allocated_group_block_ids = tuple(
                np.concatenate((current, new))
                for current, new in zip(
                    req_meta.allocated_group_block_ids,
                    new_group_block_ids,
                )
            )
        self._record_ready_group_block_ids(req_meta)

    def _replace_allocated_blocks(
        self,
        req_meta: FAWARequestMeta,
        blocks: "KVCacheBlocks",
    ) -> None:
        # update_state_after_alloc receives a full allocation snapshot. Resumed
        # chunk-prefill requests can later replace that snapshot from
        # scheduled_cached_reqs.
        self._ensure_compact_plan(req_meta).clear()
        req_meta.record_block_cursor = 0
        req_meta.allocated_group_block_ids = tuple(
            BlockIdSequence(group_blocks) for group_blocks in blocks.blocks
        )
        self._record_ready_group_block_ids(req_meta)

    def _summarize_request_meta(self, req_meta: FAWARequestMeta) -> dict[str, object]:
        recorded_blocks = sorted(req_meta.group_block_ids)
        recorded_range = (
            (recorded_blocks[0], recorded_blocks[-1] + 1) if recorded_blocks else (0, 0)
        )
        sample_rows = {
            block_idx: [
                len(group_segments)
                for group_segments in req_meta.group_block_ids[block_idx]
            ]
            for block_idx in recorded_blocks[:3]
        }
        if len(recorded_blocks) > 3:
            last_block_idx = recorded_blocks[-1]
            sample_rows[last_block_idx] = [
                len(group_segments)
                for group_segments in req_meta.group_block_ids[last_block_idx]
            ]
        return {
            "ucm_blocks": len(req_meta.ucm_block_ids),
            "hbm_hit_blocks": req_meta.hbm_hit_block_num,
            "total_hit_blocks": req_meta.total_hit_block_num,
            "num_token_ids": req_meta.num_token_ids,
            "token_processed": req_meta.token_processed,
            "store_block_cursor": req_meta.store_block_cursor,
            "record_block_cursor": req_meta.record_block_cursor,
            "allocated_group_lens": [
                len(group_blocks) for group_blocks in req_meta.allocated_group_block_ids
            ],
            "recorded_block_count": len(recorded_blocks),
            "recorded_block_range": recorded_range,
            "sample_row_segment_lens": sample_rows,
        }

    def _record_ready_group_block_ids(
        self,
        req_meta: FAWARequestMeta,
    ) -> None:
        allocated_group_block_ids = req_meta.allocated_group_block_ids
        if not allocated_group_block_ids:
            return
        # Continue from the first unrecorded canonical block. Later chunk
        # prefill steps may append enough group blocks to complete more rows.
        # The cursor avoids rescanning already recorded rows on every scheduler tick.
        self._record_ready_group_block_ids_fast(req_meta)

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
        """Scheduler lookup hook for reusing a previous batch's prefix cache."""
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

        block_keys: list[Optional[bytes]] = [None] * len(canonical_hashes)
        external_keys: list[bytes] = []
        for idx in range(hbm_hit_block_num, len(canonical_hashes)):
            # Only namespace the suffix that can participate in external lookup.
            # Later dispatch paths fill other keys lazily if they are needed.
            block_key = self._block_key(canonical_hashes[idx])
            block_keys[idx] = block_key
            external_keys.append(block_key)
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
            block_keys=block_keys,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
            store_block_cursor=total_hit_block_num,
            record_block_cursor=0,
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
        """Scheduler allocation hook.

        vLLM calls this after HBM blocks are assigned. FAWA converts those block
        ids into canonical FA/WA rows that worker metadata can later load or
        dump without consulting scheduler state.
        """
        req_meta = self.requests_meta.get(request.request_id)
        if req_meta is None:
            return

        try:
            self._replace_allocated_blocks(req_meta, blocks)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"request {request.request_id} FAWA req_meta after "
                    f"replace_allocated_blocks: "
                    f"{self._summarize_request_meta(req_meta)}"
                )
        except Exception as e:
            logger.error(
                f"request {request.request_id} record FAWA HBM-aligned "
                f"block ids failed. {type(e).__name__}: {e}"
            )
            raise

    def _make_dispatch_meta(
        self,
        request_id: str,
        req_meta: FAWARequestMeta,
        new_tokens: int,
        need_load: bool,
    ) -> FAWARequestDispatchMeta:
        """Build one request's worker load/dump plan for this scheduler step."""
        load_keys: list[bytes] = []
        load_plan: Optional[FAWADispatchBlockPlan] = None
        if need_load and req_meta.total_hit_block_num > req_meta.hbm_hit_block_num:
            # External-hit loads must have a complete contiguous plan; vLLM has
            # already treated these tokens as computed.
            load_end_block = self._get_contiguous_recorded_end_block(
                req_meta,
                req_meta.hbm_hit_block_num,
                req_meta.total_hit_block_num,
            )
            if load_end_block < req_meta.total_hit_block_num:
                raise RuntimeError(
                    f"request {request_id} FAWA load plan is missing group block "
                    f"ids for canonical blocks "
                    f"[{load_end_block}, {req_meta.total_hit_block_num})."
                )
            load_keys, load_plan = self._block_keys_and_plan(
                req_meta,
                range(req_meta.hbm_hit_block_num, load_end_block),
            )

        dump_keys: list[bytes] = []
        dump_plan: Optional[FAWADispatchBlockPlan] = None
        computed_end_token = min(
            req_meta.num_token_ids,
            req_meta.token_processed + new_tokens,
        )
        if req_meta.store_block_cursor < req_meta.num_token_ids // self.hash_block_size:
            start_block = req_meta.store_block_cursor
            candidate_end_block = computed_end_token // self.hash_block_size
            # Store only full canonical blocks whose group rows are available.
            end_block = self._get_contiguous_recorded_end_block(
                req_meta,
                start_block,
                candidate_end_block,
            )
            if end_block > start_block:
                dump_keys, dump_plan = self._block_keys_and_plan(
                    req_meta,
                    range(start_block, end_block),
                )
                req_meta.store_block_cursor = end_block
            req_meta.token_processed = computed_end_token

        return FAWARequestDispatchMeta(
            load_plan,
            dump_plan,
        )

    def _get_contiguous_recorded_end_block(
        self,
        req_meta: FAWARequestMeta,
        start_block: int,
        stop_block: int,
    ) -> int:
        # group_block_ids is populated in canonical order; for ranges starting
        # before record_block_cursor we can answer without a dict membership loop.
        if start_block < req_meta.record_block_cursor:
            return min(stop_block, req_meta.record_block_cursor)
        end_block = start_block
        for block_idx in range(start_block, stop_block):
            if block_idx not in req_meta.group_block_ids:
                break
            end_block = block_idx + 1
        return end_block

    def _block_keys_and_plan(
        self,
        req_meta: FAWARequestMeta,
        block_indices: range,
    ) -> tuple[list[bytes], Optional[FAWADispatchBlockPlan]]:
        # Dispatch plans carry only row indices. Worker-side pointer matrices
        # are built lazily from the compact plan to avoid scheduler metadata
        # materializing every row as Python objects.
        indices = tuple(block_indices)
        keys = self._block_keys_for_indices(req_meta, list(indices))
        if not indices:
            return keys, None
        return keys, FAWADispatchBlockPlan(keys, req_meta, indices)

    @staticmethod
    def _scheduled_cached_new_block_ids(
        cached,
        index: int,
    ) -> tuple[list[int], ...] | None:
        if index >= len(cached.new_block_ids):
            return None
        return cached.new_block_ids[index]

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Scheduler-to-worker metadata hook.

        New requests may need an external-prefix load. Cached/chunk-prefill
        requests may append allocation rows and produce dump work for newly
        completed canonical blocks.
        """
        requests_dispatch_meta: dict[str, FAWARequestDispatchMeta] = {}

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
        for i, request_id in enumerate(cached.req_ids):
            req_meta = self.requests_meta.get(request_id)
            if not req_meta:
                continue
            resumed = request_id in cached.resumed_req_ids
            # Running chunk-prefill requests receive later allocations through
            # scheduled_cached_reqs.new_block_ids, not update_state_after_alloc().
            self._record_allocated_group_block_ids(
                req_meta,
                self._scheduled_cached_new_block_ids(cached, i),
                replace=resumed,
            )
            requests_dispatch_meta[request_id] = self._make_dispatch_meta(
                request_id,
                req_meta,
                scheduler_output.num_scheduled_tokens[request_id],
                resumed,
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
        self.requests_meta.pop(request.request_id, None)
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

    def _compact_anchor_vllm_block_ids(
        self,
        plan: FAWADispatchBlockPlan,
        indices: Optional[Sequence[int]] = None,
    ) -> set[int]:
        compact_plan = plan.req_meta.compact_plan
        if compact_plan is None:
            raise RuntimeError("FAWA dispatch plan is missing compact KV rows.")
        row_indices = np.asarray(indices if indices is not None else plan.indices)
        if len(row_indices) == 0 or compact_plan.segment_counts[0] == 0:
            return set()
        block_ids = compact_plan.block_ids[0][row_indices, :].reshape(-1)
        return {int(block_id) for block_id in block_ids}

    def _dispatch_group_addrs(
        self,
        dispatch_plan: FAWADispatchBlockPlan,
        group_ids: tuple[int, ...],
        indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Convert a scheduler dispatch plan into UCM pointer rows on worker."""
        compact_plan = dispatch_plan.req_meta.compact_plan
        selected_indices = tuple(indices) if indices is not None else dispatch_plan.indices
        if compact_plan is None:
            raise RuntimeError("FAWA dispatch plan is missing compact KV rows.")
        return self._extract_compact_group_addrs(
            compact_plan,
            selected_indices,
            group_ids,
        )

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

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Worker load hook.

        FA loads every externally matched canonical block. WA loads only the
        final matched boundary because WA rows are boundary state rather than a
        prefix sequence.
        """
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMFAWAConnectorMetadata):
            raise RuntimeError(f"Unexpected FAWA metadata type: {type(metadata)}")

        tasks: list[FAWALoadTask] = []
        for request_id, request in metadata.request_meta.items():
            load_plan = request.load_block_plan
            if load_plan is None or not load_plan.keys:
                continue
            try:
                if self.fa_store is None:
                    raise RuntimeError("FA store is not initialized.")
                # FA groups are loaded for every external-hit canonical block.
                fa_ptrs = self._dispatch_group_addrs(
                    load_plan,
                    self.fa_group_ids,
                )
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "FA",
                        self.fa_store,
                        load_plan.keys,
                        fa_ptrs,
                        self._compact_anchor_vllm_block_ids(load_plan),
                    )
                )

                if self.wa_store is None:
                    raise RuntimeError("WA store is not initialized.")
                # WA groups only need the final matched boundary.
                window_keys = load_plan.keys[-1:]
                window_indices = load_plan.indices[-1:]
                window_ptrs = self._dispatch_group_addrs(
                    load_plan,
                    self.window_group_ids,
                    indices=window_indices,
                )
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "WA",
                        self.wa_store,
                        window_keys,
                        window_ptrs,
                        self._compact_anchor_vllm_block_ids(load_plan, window_indices),
                    )
                )
            except Exception as e:
                logger.error(
                    f"request {request_id} submit FAWA load task "
                    f"error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    self._compact_anchor_vllm_block_ids(load_plan)
                )

        for load_task in tasks:
            self._wait_load_task(load_task)

    def wait_for_save(self) -> None:
        """Worker save hook.

        Rank 0 owns FA/WA persistence for the current MLA TP path. Dump rows are
        batched across requests to keep store-task overhead independent of
        request count where possible.
        """
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
            dump_plans: list[FAWADispatchBlockPlan] = []
            for request in metadata.request_meta.values():
                dump_plan = request.dump_block_plan
                if dump_plan is None or not dump_plan.keys:
                    continue
                total_keys.extend(dump_plan.keys)
                dump_plans.append(dump_plan)

            if not total_keys:
                return

            tasks: list[FAWADumpTask] = []
            # Batch all request rows in one worker call; this keeps metadata
            # dispatch order while avoiding one FA/WA task pair per request.
            fa_parts = [
                self._dispatch_group_addrs(dump_plan, self.fa_group_ids)
                for dump_plan in dump_plans
            ]
            fa_ptrs = np.concatenate(fa_parts, axis=0)
            tasks.append(
                self._submit_dump_task(
                    "FA",
                    self.fa_store,
                    total_keys,
                    fa_ptrs,
                    event_handle,
                )
            )
            window_parts = [
                self._dispatch_group_addrs(
                    dump_plan,
                    self.window_group_ids,
                )
                for dump_plan in dump_plans
            ]
            window_ptrs = np.concatenate(window_parts, axis=0)
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

    def _trim_compact_segment_arrays(
        self,
        group_id: int,
        canonical_block_indices: np.ndarray,
        group_block_indices: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            group_id not in self.window_group_ids
            or self.group_tail_blocks[group_id] is None
            or self.group_tail_blocks[group_id] == 0
            or self.block_span_layout.is_swa_group(group_id)
        ):
            return offsets, lengths

        window_tail_tokens = self._ascend_window_tail_tokens(group_id)
        if window_tail_tokens is None or window_tail_tokens == 0:
            return offsets, lengths

        group_token_block_size = self.group_token_block_sizes[group_id]
        tail_end_tokens = (canonical_block_indices + 1) * self.hash_block_size
        tail_start_tokens = np.maximum(0, tail_end_tokens - window_tail_tokens)
        block_start_tokens = group_block_indices * group_token_block_size
        segment_start_in_block = np.maximum(
            0,
            tail_start_tokens[:, None] - block_start_tokens,
        )
        segment_end_in_block = np.minimum(
            group_token_block_size,
            tail_end_tokens[:, None] - block_start_tokens,
        )
        segment_lengths = np.maximum(0, segment_end_in_block - segment_start_in_block)
        trimmed_offsets = offsets + segment_start_in_block
        return trimmed_offsets.astype(np.int64), segment_lengths.astype(np.int64)
