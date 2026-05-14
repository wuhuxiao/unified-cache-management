from ucm.integration.vllm.hma_connector import (
    FAWALoadTask,
    UCMFAWAConnector,
)


class FailingStore:
    def wait(self, task):
        raise RuntimeError("load failed")


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
