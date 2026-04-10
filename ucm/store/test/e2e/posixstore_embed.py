# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import argparse
import mmap
import os
import secrets
import time

import numpy as np

from ucm.store.factory_v1 import UcmConnectorFactoryV1, UcmKVStoreBaseV1


def parse_size(value: str) -> int:
    """Parse a human-readable size string like '8M', '512K', '1G' into bytes."""
    suffixes = {"K": 1024, "M": 1024**2, "G": 1024**3}
    value = value.strip().upper()
    if value[-1] in suffixes:
        return int(float(value[:-1]) * suffixes[value[-1]])
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POSIX store embedded test")
    parser.add_argument(
        "--storage_backends",
        nargs="+",
        default=["./build/data"],
        help="List of storage backend paths (default: ./build/data)",
    )
    parser.add_argument(
        "--block_size",
        type=parse_size,
        default=parse_size("1M"),
        help="Block size, supports human-readable suffixes K/M/G (default: 1M)",
    )
    parser.add_argument(
        "--data_trans_concurrency",
        type=int,
        default=8,
        help="Data transfer concurrency (default: 8)",
    )
    parser.add_argument(
        "--lookup_concurrency",
        type=int,
        default=8,
        help="Lookup concurrency (default: 8)",
    )
    parser.add_argument(
        "--io_direct",
        action="store_true",
        default=False,
        help="Enable O_DIRECT I/O (default: False)",
    )
    parser.add_argument(
        "--round_number",
        type=int,
        default=64,
        help="Number of test rounds (default: 64)",
    )
    parser.add_argument(
        "--block_num",
        type=int,
        default=1024,
        help="Number of blocks per round (default: 1024)",
    )
    parser.add_argument(
        "--posix_io_engine",
        type=str,
        default="psync",
        help="POSIX I/O engine: psync, aio, io_uring (default: psync)",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Device ID for worker (-1 for scheduler, default: 0)",
    )
    return parser.parse_args()


def setup_connector(
    backends: list[str],
    block_size: int,
    data_trans_concur: int,
    lookup_concur: int,
    io_direct: bool,
    is_worker: bool,
    io_engine: str,
    device_id: int,
) -> UcmKVStoreBaseV1:
    config = {
        "store_pipeline": "Posix",
        "storage_backends": backends,
        "tensor_size": block_size,
        "shard_size": block_size,
        "block_size": block_size,
        "posix_io_engine": io_engine,
        "posix_data_trans_concurrency": data_trans_concur,
        "posix_lookup_concurrency": lookup_concur,
        "io_direct": io_direct,
        "device_id": device_id if is_worker else -1,
    }
    return UcmConnectorFactoryV1.create_connector(
        "UcmPipelineStore", config, "ucm.store.pipeline.connector"
    )


def make_aligned_array(
    size: int, alignment: int = 4096, dtype: type = np.uint8
) -> np.ndarray:
    """Create a memory-aligned numpy array suitable for O_DIRECT I/O."""
    itemsize = np.dtype(dtype).itemsize
    total_bytes = size * itemsize
    mm = mmap.mmap(-1, total_bytes + alignment)
    raw_array = np.frombuffer(mm, dtype=np.uint8, count=total_bytes + alignment)
    raw_ptr = raw_array.__array_interface__["data"][0]
    aligned_addr = (raw_ptr + alignment - 1) & ~(alignment - 1)
    offset = aligned_addr - raw_ptr
    return raw_array[offset : offset + total_bytes].view(dtype=dtype)


def generate_block_ids(count: int) -> list[bytes]:
    return [secrets.token_bytes(16) for _ in range(count)]


def run_round(
    scheduler: UcmKVStoreBaseV1,
    worker: UcmKVStoreBaseV1,
    block_num: int,
    data_ptrs_write: list,
    data_ptrs_read: list,
) -> dict[str, float]:
    """Execute a single test round and return timing metrics."""
    block_ids = generate_block_ids(block_num)
    shard_idxes = [0] * block_num

    # Pre-dump: blocks should not exist yet
    t0 = time.perf_counter()
    founds = scheduler.lookup(block_ids)
    cost_lookup_before = time.perf_counter() - t0
    assert not any(founds), "Blocks should not exist before dump"

    # Prefix lookup before dump
    t0 = time.perf_counter()
    found_idx = scheduler.lookup_on_prefix(block_ids)
    cost_prefix_before = time.perf_counter() - t0
    assert found_idx == -1, "Prefix lookup should fail before dump"

    # Dump data
    t0 = time.perf_counter()
    handle = worker.dump_data(block_ids, shard_idxes, data_ptrs_write)
    worker.wait(handle)
    cost_dump = time.perf_counter() - t0

    # Lookup after dump: all blocks should exist
    t0 = time.perf_counter()
    founds = scheduler.lookup(block_ids)
    cost_lookup_after = time.perf_counter() - t0
    assert all(founds), "All blocks should exist after dump"

    # Prefix lookup after dump: should find the last block
    t0 = time.perf_counter()
    found_idx = scheduler.lookup_on_prefix(block_ids)
    cost_prefix_after = time.perf_counter() - t0
    assert found_idx == block_num - 1, "Prefix lookup should find last block"

    # Load data
    t0 = time.perf_counter()
    handle = worker.load_data(block_ids, shard_idxes, data_ptrs_read)
    worker.wait(handle)
    cost_load = time.perf_counter() - t0

    data_size = len(block_ids) * data_ptrs_write[0][0].itemsize
    return {
        "lookup_before": cost_lookup_before * 1e3,
        "prefix_before": cost_prefix_before * 1e3,
        "dump": cost_dump * 1e3,
        "lookup_after": cost_lookup_after * 1e3,
        "prefix_after": cost_prefix_after * 1e3,
        "load": cost_load * 1e3,
        "bw_dump": data_size / cost_dump / 1e9,
        "bw_load": data_size / cost_load / 1e9,
    }


def main():
    args = parse_args()
    block_size = args.block_size
    block_num = args.block_num

    worker = setup_connector(
        args.storage_backends,
        block_size,
        args.data_trans_concurrency,
        args.lookup_concurrency,
        args.io_direct,
        is_worker=True,
        io_engine=args.posix_io_engine,
        device_id=args.device_id,
    )
    scheduler = setup_connector(
        args.storage_backends,
        block_size,
        args.data_trans_concurrency,
        args.lookup_concurrency,
        args.io_direct,
        is_worker=False,
        io_engine=args.posix_io_engine,
        device_id=args.device_id,
    )

    # Prepare aligned buffers
    data_write = [make_aligned_array(block_size) for _ in range(block_num)]
    data_read = [make_aligned_array(block_size) for _ in range(block_num)]
    data_ptrs_write = [[d.ctypes.data] for d in data_write]
    data_ptrs_read = [[d.ctypes.data] for d in data_read]

    for idx in range(args.round_number):
        metrics = run_round(scheduler, worker, block_num, data_ptrs_write, data_ptrs_read)
        print(
            f"[{idx:03}/{args.round_number:03}] [{block_size}] [{block_num}] "
            f"lookup_before={metrics['lookup_before']:.3f}ms, "
            f"prefix_before={metrics['prefix_before']:.3f}ms, "
            f"dump={metrics['dump']:.3f}ms, "
            f"lookup_after={metrics['lookup_after']:.3f}ms, "
            f"prefix_after={metrics['prefix_after']:.3f}ms, "
            f"load={metrics['load']:.3f}ms, "
            f"bw_dump={metrics['bw_dump']:.3f}GB/s, "
            f"bw_load={metrics['bw_load']:.3f}GB/s."
        )


if __name__ == "__main__":
    os.environ["UC_LOGGER_LEVEL"] = "info"
    main()

# numactl examples:
#   Bind to specific NUMA node and its local CPUs:
#     numactl --cpunodebind=<node> --membind=<node> python -m ucm.store.test.e2e.posixstore_embed
#   Bind to specific CPU cores and NUMA node:
#     numactl --physcpubind=0-7 --membind=0 python -m ucm.store.test.e2e.posixstore_embed
#   Full example with all arguments:
#     numactl --cpunodebind=0 --membind=0 python -m ucm.store.test.e2e.posixstore_embed \
#       --block_size 8M --block_num 512 --round_number 128 \
#       --storage_backends /data/ucm/cache --data_trans_concurrency 16 \
#       --io_direct --posix_io_engine psync --device_id 0
