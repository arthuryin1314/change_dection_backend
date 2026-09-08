import threading

from utils.classification_result_lock import classification_result_lock


def test_result_lock_serializes_readers_and_rebuilders(tmp_path):
    result_directory = tmp_path / "result-1"
    result_directory.mkdir()
    entered = threading.Event()

    def wait_for_lock():
        with classification_result_lock(result_directory):
            entered.set()

    with classification_result_lock(result_directory):
        worker = threading.Thread(target=wait_for_lock)
        worker.start()
        assert entered.wait(0.1) is False

    worker.join(timeout=2)
    assert entered.is_set()
    assert worker.is_alive() is False
