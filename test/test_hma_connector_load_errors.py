import numpy as np

from ucm.integration.vllm.hma_connector import (
    FAWARequestDispatchMeta,
    UCMFAWAConnectorMetadata,
    FAWALoadTask,
    UCMFAWAConnector,
)


class FailingStore:
    def load_data(self, keys, shard_indices, ptrs):
        return ("load", tuple(keys), tuple(shard_indices), ptrs)

    def wait(self, task):
        raise RuntimeError("load failed")


class RecordingStore:
    def __init__(self):
        self.loads = []
        self.dumps = []
        self.waited = []

    def load_data(self, keys, shard_indices, ptrs):
        task = ("load", tuple(keys), tuple(shard_indices), ptrs)
        self.loads.append(task)
        return task

    def dump_data(self, keys, shard_indices, ptrs, event_handle):
        task = ("dump", tuple(keys), tuple(shard_indices), ptrs, event_handle)
        self.dumps.append(task)
        return task

    def wait(self, task):
        self.waited.append(task)


def test_wait_load_task_reports_vllm_block_ids_once():
    connector = object.__new__(UCMFAWAConnector)
    connector._invalid_block_ids = set()

    task = FAWALoadTask(
        request_id="req-0",
        label="FA",
        store=FailingStore(),
        task=object(),
        key_count=2,
        anchor_vllm_block_ids={3, 5},
    )

    connector._wait_load_task(task)

    assert connector.get_block_ids_with_load_errors() == {3, 5}
    assert connector.get_block_ids_with_load_errors() == set()


def test_first_group_anchor_ids_ignore_negative_values():
    connector = object.__new__(UCMFAWAConnector)

    assert connector._first_group_anchor_ids(([3, -1, 7], [5], [])) == {3, 7}
    assert connector._first_group_anchor_ids(([], [5], [])) == set()


def test_first_group_anchor_ids_for_hash_range_uses_candidate_base():
    connector = object.__new__(UCMFAWAConnector)

    assert connector._first_group_anchor_ids_for_hash_range(
        ([11, -1, 13], [21], [31]),
        4,
        5,
        3,
    ) == set()
    assert connector._first_group_anchor_ids_for_hash_range(
        ([11, -1, 13], [21], [31]),
        5,
        6,
        3,
    ) == {13}


def test_start_load_kv_uses_flat_request_metadata_and_boundary_anchors():
    connector = object.__new__(UCMFAWAConnector)
    connector._invalid_block_ids = set()
    connector.fa_store = RecordingStore()
    connector.wa_store = RecordingStore()

    load_meta = FAWARequestDispatchMeta(
        load_keys=[b"a", b"b"],
        load_hash_start=3,
        load_hash_end=5,
        load_vllm_block_ids=([11, 13], [21], [31]),
    )
    connector._get_connector_metadata = lambda: UCMFAWAConnectorMetadata(
        {"req-0": load_meta}
    )
    fa_calls = []
    wa_calls = []

    def extract_fa(keys, hash_start, hash_end, candidate_ids):
        fa_calls.append((keys, hash_start, hash_end, candidate_ids))
        return "fa-ptrs"

    def extract_wa(keys, hash_start, hash_end, candidate_ids):
        wa_calls.append((keys, hash_start, hash_end, candidate_ids))
        return "wa-ptrs"

    connector._extract_fa_ptr = extract_fa
    connector._extract_wa_ptr = extract_wa

    connector.start_load_kv(None)

    assert fa_calls == [([b"a", b"b"], 3, 5, ([11, 13], [21], [31]))]
    assert wa_calls == [([b"b"], 4, 5, ([11, 13], [21], [31]))]
    assert connector.fa_store.loads[0][:3] == (
        "load",
        (b"a", b"b"),
        (0, 0),
    )
    assert connector.wa_store.loads[0][:3] == ("load", (b"b",), (0,))
    assert connector.fa_store.loads[0][3] == "fa-ptrs"
    assert connector.wa_store.loads[0][3] == "wa-ptrs"
    assert connector.fa_store.waited[0][3] == "fa-ptrs"
    assert connector.wa_store.waited[0][3] == "wa-ptrs"
    assert connector._invalid_block_ids == set()

    fa_task = connector.fa_store.waited[0]
    wa_task = connector.wa_store.waited[0]
    assert fa_task[3] == "fa-ptrs"
    assert wa_task[3] == "wa-ptrs"


def test_start_load_kv_reports_wa_failure_with_final_boundary_anchor_only():
    connector = object.__new__(UCMFAWAConnector)
    connector._invalid_block_ids = set()
    connector.fa_store = RecordingStore()
    connector.wa_store = FailingStore()

    load_meta = FAWARequestDispatchMeta(
        load_keys=[b"a", b"b"],
        load_hash_start=3,
        load_hash_end=5,
        load_vllm_block_ids=([11, 13], [21], [31]),
    )
    connector._get_connector_metadata = lambda: UCMFAWAConnectorMetadata(
        {"req-0": load_meta}
    )
    connector._extract_fa_ptr = lambda *args: "fa-ptrs"
    connector._extract_wa_ptr = lambda *args: "wa-ptrs"

    connector.start_load_kv(None)

    assert connector.get_block_ids_with_load_errors() == {13}


def test_wait_for_save_batches_flat_dump_metadata():
    connector = object.__new__(UCMFAWAConnector)
    connector.tp_rank = 0
    connector.fa_store = RecordingStore()
    connector.wa_store = RecordingStore()
    connector._get_dump_event_handle = lambda: "event"
    connector._get_connector_metadata = lambda: UCMFAWAConnectorMetadata(
        {
            "req-0": FAWARequestDispatchMeta(
                dump_keys=[b"a"],
                dump_hash_start=0,
                dump_hash_end=1,
                dump_vllm_block_ids=([1], [10]),
            ),
            "req-1": FAWARequestDispatchMeta(),
            "req-2": FAWARequestDispatchMeta(
                dump_keys=[b"b", b"c"],
                dump_hash_start=4,
                dump_hash_end=6,
                dump_vllm_block_ids=([2, 3], [20, 21]),
            ),
        }
    )

    fa_calls = []
    wa_calls = []

    def extract_fa(keys, hash_start, hash_end, candidate_ids):
        fa_calls.append((keys, hash_start, hash_end, candidate_ids))
        return np.array([[hash_start]], dtype=np.uint64)

    def extract_wa(keys, hash_start, hash_end, candidate_ids):
        wa_calls.append((keys, hash_start, hash_end, candidate_ids))
        return np.array([[hash_end]], dtype=np.uint64)

    connector._extract_fa_ptr = extract_fa
    connector._extract_wa_ptr = extract_wa

    connector.wait_for_save()

    assert fa_calls == [
        ([b"a"], 0, 1, ([1], [10])),
        ([b"b", b"c"], 4, 6, ([2, 3], [20, 21])),
    ]
    assert wa_calls == [
        ([b"a"], 0, 1, ([1], [10])),
        ([b"b", b"c"], 4, 6, ([2, 3], [20, 21])),
    ]
    assert connector.fa_store.dumps[0][1:3] == ((b"a", b"b", b"c"), (0, 0, 0))
    assert connector.wa_store.dumps[0][1:3] == ((b"a", b"b", b"c"), (0, 0, 0))
    np.testing.assert_array_equal(
        connector.fa_store.dumps[0][3],
        np.array([[0], [4]], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        connector.wa_store.dumps[0][3],
        np.array([[1], [6]], dtype=np.uint64),
    )
