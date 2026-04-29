#!/usr/bin/env python3
"""Validate captured DeepSeek V4 UCM packed KV metadata and tensors."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WORKER_STAGES = {"dump_before", "load_before", "load_after"}
SCHEDULER_STAGES = {"lookup", "after_alloc", "dispatch_dump", "dispatch_load"}


@dataclass(frozen=True)
class FlatTensor:
    group_id: int
    group_block_id: int
    block_pos: int
    tensor_index: int
    layer_name: str
    view_name: str
    data: torch.Tensor

    @property
    def logical_id(self) -> tuple[int, int, str, str, tuple[int, ...], torch.dtype]:
        return (
            self.group_id,
            self.tensor_index,
            self.layer_name,
            self.view_name,
            tuple(self.data.shape),
            self.data.dtype,
        )

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()


def load_capture(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def capture_files(capture_dir: Path) -> list[Path]:
    return sorted(capture_dir.glob("*.pt"))


def group_by_stage(files: Iterable[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in files:
        payload = load_capture(path)
        grouped.setdefault(payload["stage"], []).append(path)
    return grouped


def find_one(
    grouped: dict[str, list[Path]],
    stage: str,
    request_id: str | None = None,
    rank: int | None = None,
) -> Path:
    matches = []
    for path in grouped.get(stage, []):
        payload = load_capture(path)
        if request_id is not None and payload.get("request_id") != request_id:
            continue
        if rank is not None and payload.get("tp_rank") != rank:
            continue
        matches.append(path)
    if len(matches) != 1:
        details = ", ".join(path.name for path in matches) or "none"
        raise AssertionError(
            f"Expected one {stage} capture request={request_id} rank={rank}, got {details}"
        )
    return matches[0]


def scheduler_packed_rows(payload: dict) -> list[list[list[int]]]:
    packed = payload.get("packed_block_ids", {})
    return [
        [list(group_ids) for group_ids in groups]
        for _, groups in sorted(packed.items(), key=lambda item: int(item[0]))
    ]


def worker_packed_rows(payload: dict) -> list[list[list[int]]]:
    return [
        [list(group_ids) for group_ids in row]
        for row in payload.get("packed_group_block_ids", [])
    ]


def row_counts(row: list[list[int]]) -> list[int]:
    return [len(group_ids) for group_ids in row]


def expected_group_counts(payload: dict, row_idx: int = 0) -> list[int]:
    block_sizes = list(payload["group_block_sizes"])
    tail_blocks = list(payload["group_tail_blocks"])
    end_tokens = (row_idx + 1) * int(payload.get("hash_block_size", 256))
    counts: list[int] = []
    for group_id, block_size in enumerate(block_sizes):
        if group_id == 0:
            counts.append(1)
            continue
        tail = tail_blocks[group_id]
        if tail is None:
            counts.append(end_tokens // block_size)
        else:
            counts.append(min(int(tail), end_tokens // block_size))
    return counts


def expand_worker_row(payload: dict, row_idx: int = 0) -> list[FlatTensor]:
    row = payload["rows"][row_idx]
    flat: list[FlatTensor] = []
    for group in row["groups"]:
        group_id = int(group["group_id"])
        block_ids = list(group["block_ids"])
        tensors = group["tensors"]
        if not block_ids:
            if tensors:
                raise AssertionError(f"group {group_id} has tensors but no block ids")
            continue
        for block_pos, block_id in enumerate(block_ids):
            for tensor_index, entry in enumerate(tensors):
                data = entry["data"]
                if data.shape[0] != len(block_ids):
                    raise AssertionError(
                        f"group {group_id} tensor {tensor_index} leading dim "
                        f"{data.shape[0]} != block count {len(block_ids)}"
                    )
                flat.append(
                    FlatTensor(
                        group_id=group_id,
                        group_block_id=int(block_id),
                        block_pos=block_pos,
                        tensor_index=tensor_index,
                        layer_name=entry["layer_name"],
                        view_name=entry["view_name"],
                        data=data[block_pos].contiguous(),
                    )
                )
    return flat


def tensor_max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    if torch.equal(a, b):
        return 0.0
    if a.is_floating_point():
        return float((a.float() - b.float()).abs().max().item())
    return float((a.to(torch.int64) - b.to(torch.int64)).abs().max().item())


def assert_tensor_size_order(payload: dict, path: Path) -> None:
    flat = expand_worker_row(payload)
    tensor_size_list = list(payload["tensor_size_list"])
    actual = [entry.nbytes for entry in flat]
    if actual != tensor_size_list:
        for idx, (got, expected) in enumerate(zip(actual, tensor_size_list)):
            if got != expected:
                raise AssertionError(
                    f"{path.name}: tensor_size_list mismatch at {idx}: "
                    f"payload tensor bytes={got}, config bytes={expected}"
                )
        raise AssertionError(
            f"{path.name}: tensor count mismatch: payload={len(actual)}, "
            f"tensor_size_list={len(tensor_size_list)}"
        )
    shard_size = int(payload["shard_size"])
    if sum(tensor_size_list) != shard_size:
        raise AssertionError(
            f"{path.name}: shard_size={shard_size}, sum(tensor_size_list)="
            f"{sum(tensor_size_list)}"
        )


def compare_flat_payloads(
    reference: list[FlatTensor],
    candidate: list[FlatTensor],
    *,
    compare_block_ids: bool,
) -> list[str]:
    mismatches: list[str] = []
    if len(reference) != len(candidate):
        return [f"tensor count {len(candidate)} != {len(reference)}"]
    for idx, (ref, cur) in enumerate(zip(reference, candidate)):
        if ref.logical_id != cur.logical_id:
            mismatches.append(
                f"tensor {idx}: logical metadata {cur.logical_id} != {ref.logical_id}"
            )
            continue
        if compare_block_ids and ref.group_block_id != cur.group_block_id:
            mismatches.append(
                f"tensor {idx}: block id {cur.group_block_id} != {ref.group_block_id}"
            )
            continue
        diff = tensor_max_diff(ref.data, cur.data)
        if diff != 0:
            mismatches.append(
                f"tensor {idx}: group={cur.group_id} name={cur.layer_name} "
                f"view={cur.view_name} max_diff={diff}"
            )
    return mismatches


def validate_row_counts(grouped: dict[str, list[Path]]) -> None:
    for stage in SCHEDULER_STAGES:
        for path in grouped.get(stage, []):
            payload = load_capture(path)
            for row_idx, row in enumerate(scheduler_packed_rows(payload)):
                counts = row_counts(row)
                expected = expected_group_counts(payload, row_idx)
                if counts != expected:
                    raise AssertionError(
                        f"{path.name}: group counts {counts} != {expected}"
                    )
    for stage in WORKER_STAGES:
        for path in grouped.get(stage, []):
            payload = load_capture(path)
            for row_idx, row in enumerate(worker_packed_rows(payload)):
                counts = row_counts(row)
                expected = expected_group_counts(payload, row_idx)
                if counts != expected:
                    raise AssertionError(
                        f"{path.name}: group counts {counts} != {expected}"
                    )


def validate_keys(grouped: dict[str, list[Path]]) -> tuple[str, str]:
    dump_dispatch = load_capture(find_one(grouped, "dispatch_dump"))
    load_dispatch = load_capture(find_one(grouped, "dispatch_load"))
    dump_indices = dump_dispatch["extra"]["dump_indices"]
    load_indices = load_dispatch["extra"]["load_indices"]
    dump_keys = [dump_dispatch["packed_keys_hex"][idx] for idx in dump_indices]
    load_keys = [load_dispatch["packed_keys_hex"][idx] for idx in load_indices]
    if dump_keys != load_keys:
        raise AssertionError(f"dispatch dump/load keys differ: {dump_keys} != {load_keys}")

    dump_worker = load_capture(find_one(grouped, "dump_before", rank=0))
    load_worker = load_capture(find_one(grouped, "load_before", rank=0))
    if dump_worker["keys_hex"] != dump_keys:
        raise AssertionError(
            f"rank0 dump keys {dump_worker['keys_hex']} != scheduler {dump_keys}"
        )
    if load_worker["keys_hex"] != load_keys:
        raise AssertionError(
            f"rank0 load keys {load_worker['keys_hex']} != scheduler {load_keys}"
        )
    return dump_dispatch["request_id"], load_dispatch["request_id"]


def validate_worker_tensor_sizes(grouped: dict[str, list[Path]]) -> None:
    for stage in WORKER_STAGES:
        for path in grouped.get(stage, []):
            assert_tensor_size_order(load_capture(path), path)


def validate_tp_rank_consistency(grouped: dict[str, list[Path]]) -> bool:
    dump_paths = sorted(grouped.get("dump_before", []))
    by_rank = {load_capture(path)["tp_rank"]: path for path in dump_paths}
    if 0 not in by_rank:
        raise AssertionError("No rank0 dump_before payload found.")
    reference = expand_worker_row(load_capture(by_rank[0]))
    all_identical = True
    for rank, path in sorted(by_rank.items()):
        if rank == 0:
            continue
        mismatches = compare_flat_payloads(
            reference,
            expand_worker_row(load_capture(path)),
            compare_block_ids=True,
        )
        if mismatches:
            all_identical = False
            print(f"TP rank {rank} differs from rank0:")
            for item in mismatches[:8]:
                print(f"  {item}")
    return all_identical


def validate_stub_semantics(grouped: dict[str, list[Path]]) -> None:
    dump_reference = expand_worker_row(load_capture(find_one(grouped, "dump_before", rank=0)))
    for path in sorted(grouped.get("load_before", [])):
        payload = load_capture(path)
        mismatches = compare_flat_payloads(
            dump_reference,
            expand_worker_row(payload),
            compare_block_ids=False,
        )
        metadata_mismatches = [
            item for item in mismatches if "max_diff=" not in item
        ]
        if metadata_mismatches:
            raise AssertionError(
                f"{path.name}: simulated load metadata mismatch: "
                + "; ".join(metadata_mismatches[:4])
            )


def diagnose_actual_load_after(grouped: dict[str, list[Path]], strict: bool) -> None:
    dump_reference = expand_worker_row(load_capture(find_one(grouped, "dump_before", rank=0)))
    failures: list[str] = []
    for path in sorted(grouped.get("load_after", [])):
        payload = load_capture(path)
        mismatches = compare_flat_payloads(
            dump_reference,
            expand_worker_row(payload),
            compare_block_ids=False,
        )
        if mismatches:
            failures.append(f"{path.name}: {mismatches[0]}")
    if failures:
        print("Actual captured UCM load_after does not match rank0 dump_before:")
        for failure in failures:
            print(f"  {failure}")
        if strict:
            raise AssertionError("actual load_after mismatch")
    else:
        print("Actual captured UCM load_after matches rank0 dump_before.")


def run_ucm_store_check(grouped: dict[str, list[Path]], device_id: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --run-ucm-store.")

    from ucm.store.pipeline.connector import UcmPipelineStore

    payload = load_capture(find_one(grouped, "dump_before", rank=0))
    key = bytes.fromhex(payload["keys_hex"][0])
    tensor_size_list = list(payload["tensor_size_list"])
    source = [entry.data.contiguous().cuda(device_id) for entry in expand_worker_row(payload)]
    destination = [torch.empty_like(tensor) for tensor in source]

    config = {
        "store_pipeline": "Cache|Empty",
        "unique_id": f"dsv4_offline_{secrets.token_hex(8)}",
        "tensor_size_list": tensor_size_list,
        "shard_size": int(sum(tensor_size_list)),
        "block_size": int(sum(tensor_size_list)),
        "share_buffer_enable": True,
        "cache_buffer_capacity_gb": 32,
        "cache_load_exclusive_buffer_number": 16,
        "waiting_queue_depth": 16,
        "running_queue_depth": 1024,
        "timeout_ms": 10000,
    }
    worker = UcmPipelineStore(config | {"device_id": device_id})
    scheduler = UcmPipelineStore(config)
    assert not any(scheduler.lookup([key]))
    src_ptrs = np.array([[tensor.data_ptr() for tensor in source]], dtype=np.uint64)
    dst_ptrs = np.array([[tensor.data_ptr() for tensor in destination]], dtype=np.uint64)
    task = worker.dump_data([key], [0], src_ptrs)
    worker.wait(task)
    if not all(scheduler.lookup([key])):
        raise AssertionError("UCM CacheStore lookup missed after dump.")
    task = worker.load_data([key], [0], dst_ptrs)
    worker.wait(task)
    torch.cuda.synchronize(device_id)
    for idx, (src, dst) in enumerate(zip(source, destination)):
        if not torch.equal(src, dst):
            diff = tensor_max_diff(src.cpu(), dst.cpu())
            raise AssertionError(f"UCM CacheStore tensor {idx} mismatch, max_diff={diff}")


def _ucm_store_mp_worker(
    rank: int,
    world_size: int,
    unique_id: str,
    key_hex: str,
    tensor_size_list: list[int],
    cpu_tensors: list[torch.Tensor],
    barrier: multiprocessing.Barrier,
    result_queue: multiprocessing.Queue,
) -> None:
    try:
        from ucm.store.pipeline.connector import UcmPipelineStore

        torch.cuda.set_device(rank)
        config = {
            "store_pipeline": "Cache|Empty",
            "unique_id": unique_id,
            "tensor_size_list": tensor_size_list,
            "shard_size": int(sum(tensor_size_list)),
            "block_size": int(sum(tensor_size_list)),
            "share_buffer_enable": True,
            "cache_buffer_capacity_gb": 32,
            "cache_load_exclusive_buffer_number": 16,
            "waiting_queue_depth": 16,
            "running_queue_depth": 1024,
            "timeout_ms": 10000,
        }
        store = UcmPipelineStore(config | {"device_id": rank})
        key = bytes.fromhex(key_hex)
        source = [tensor.contiguous().cuda(rank) for tensor in cpu_tensors]
        if rank == 0:
            ptrs = np.array([[tensor.data_ptr() for tensor in source]], dtype=np.uint64)
            task = store.dump_data([key], [0], ptrs)
            store.wait(task)
            torch.cuda.synchronize(rank)
        barrier.wait()

        destination = [torch.empty_like(tensor) for tensor in source]
        ptrs = np.array([[tensor.data_ptr() for tensor in destination]], dtype=np.uint64)
        task = store.load_data([key], [0], ptrs)
        store.wait(task)
        torch.cuda.synchronize(rank)
        for idx, (expected, actual) in enumerate(zip(cpu_tensors, destination)):
            actual_cpu = actual.cpu()
            if not torch.equal(expected, actual_cpu):
                diff = tensor_max_diff(expected, actual_cpu)
                result_queue.put((rank, False, f"tensor {idx} max_diff={diff}"))
                return
        result_queue.put((rank, True, ""))
    except Exception as exc:
        result_queue.put((rank, False, f"{type(exc).__name__}: {exc}"))


def run_ucm_store_mp_check(grouped: dict[str, list[Path]], world_size: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"CUDA with at least {world_size} devices is required for --run-ucm-store-mp."
        )

    payload = load_capture(find_one(grouped, "dump_before", rank=0))
    key_hex = payload["keys_hex"][0]
    tensor_size_list = list(payload["tensor_size_list"])
    cpu_tensors = [entry.data.contiguous() for entry in expand_worker_row(payload)]
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(world_size)
    result_queue = ctx.Queue()
    unique_id = f"dsv4_offline_mp_{secrets.token_hex(8)}"
    processes = [
        ctx.Process(
            target=_ucm_store_mp_worker,
            args=(
                rank,
                world_size,
                unique_id,
                key_hex,
                tensor_size_list,
                cpu_tensors,
                barrier,
                result_queue,
            ),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    results = [result_queue.get() for _ in processes]
    for process in processes:
        process.join()
    failures = [
        f"rank {rank}: {message}"
        for rank, ok, message in sorted(results)
        if not ok
    ]
    exit_failures = [
        f"pid {process.pid} exitcode={process.exitcode}"
        for process in processes
        if process.exitcode != 0
    ]
    if failures or exit_failures:
        raise AssertionError(
            "Multiprocess UCM CacheStore check failed: "
            + "; ".join(failures + exit_failures)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture_dir",
        nargs="?",
        default="/vllm-workspace/deepseek_v4_ucm_capture",
        help="Directory containing DeepSeek V4 UCM capture .pt files.",
    )
    parser.add_argument(
        "--strict-actual-load",
        action="store_true",
        help="Fail if captured load_after tensors differ from the dump payload.",
    )
    parser.add_argument(
        "--run-ucm-store",
        action="store_true",
        help="Run a minimal CUDA CacheStore dump/load check using the rank0 payload.",
    )
    parser.add_argument(
        "--run-ucm-store-mp",
        action="store_true",
        help="Run a 4-process CUDA CacheStore check using rank0 dump and all-rank load.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_dir = Path(args.capture_dir)
    files = capture_files(capture_dir)
    if not files:
        raise SystemExit(f"No .pt capture files found in {capture_dir}")
    grouped = group_by_stage(files)

    validate_row_counts(grouped)
    dump_request, load_request = validate_keys(grouped)
    validate_worker_tensor_sizes(grouped)
    tp_identical = validate_tp_rank_consistency(grouped)
    validate_stub_semantics(grouped)
    diagnose_actual_load_after(grouped, args.strict_actual_load)
    if args.run_ucm_store:
        run_ucm_store_check(grouped, args.device_id)
        print("Minimal UCM CacheStore dump/load check passed.")
    if args.run_ucm_store_mp:
        run_ucm_store_mp_check(grouped, args.world_size)
        print("Multiprocess UCM CacheStore dump/load check passed.")

    print("Offline DeepSeek V4 packed layout checks passed.")
    print(f"Dump request: {dump_request}; load request: {load_request}")
    if tp_identical:
        print("TP rank payloads are identical; single-shard rank0 dump is valid.")
    else:
        print("TP rank payloads differ; rank-sharded storage is required.")


if __name__ == "__main__":
    main()
