from __future__ import annotations

import threading
import time

import pytest

from deploy.control_thread import StoppablePeriodicThread


def test_stop_and_join_prevent_future_target_calls() -> None:
    first_call = threading.Event()
    count_lock = threading.Lock()
    call_count = 0

    def target() -> None:
        nonlocal call_count
        with count_lock:
            call_count += 1
        first_call.set()

    worker = StoppablePeriodicThread(interval=0.005, target=target, name="test-loop")
    worker.start()

    assert first_call.wait(timeout=1.0)
    worker.stop()
    assert worker.join(timeout=1.0)
    assert not worker.is_alive()

    with count_lock:
        count_after_join = call_count
    time.sleep(0.02)
    with count_lock:
        assert call_count == count_after_join


def test_target_exception_is_recorded_propagated_and_stops_loop() -> None:
    failure = ValueError("control target failed")
    target_called = threading.Event()
    call_count = 0

    def target() -> None:
        nonlocal call_count
        call_count += 1
        target_called.set()
        raise failure

    worker = StoppablePeriodicThread(interval=0.005, target=target, name="failing-loop")
    worker.start()

    assert target_called.wait(timeout=1.0)
    with pytest.raises(ValueError, match="control target failed") as exc_info:
        worker.join(timeout=1.0)

    assert exc_info.value is failure
    assert worker.exception is failure
    assert not worker.is_alive()
    assert call_count == 1


def test_join_can_report_recorded_exception_without_raising() -> None:
    failure = RuntimeError("record me")

    def target() -> None:
        raise failure

    worker = StoppablePeriodicThread(interval=0.005, target=target)
    worker.start()

    assert worker.join(timeout=1.0, raise_on_error=False)
    assert worker.exception is failure
    assert not worker.is_alive()
