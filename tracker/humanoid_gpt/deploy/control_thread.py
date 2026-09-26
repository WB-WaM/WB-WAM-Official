"""Deterministic lifecycle helper for periodic control loops.

Unlike ``unitree_sdk2py.utils.thread.RecurrentThread``, this wrapper exposes an
explicit stop event, uses a non-daemon thread by default, and retains target
exceptions so the owning thread can fail closed before publishing a final
damping command.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable


class StoppablePeriodicThread:
    """Run ``target`` periodically until stopped or the target raises.

    The first invocation runs immediately. Later invocations are aligned to a
    monotonic fixed-period schedule. If an invocation overruns one or more
    periods, missed slots are skipped instead of being replayed back-to-back.

    ``join()`` returns whether the thread stopped before the optional timeout.
    By default it re-raises a target exception in the joining thread after the
    worker has stopped. The original exception remains available via
    :attr:`exception` regardless of ``raise_on_error``.
    """

    def __init__(
        self,
        interval: float,
        target: Callable[[], None],
        name: str | None = None,
        daemon: bool = False,
    ) -> None:
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("interval must be a positive finite number")
        if not callable(target):
            raise TypeError("target must be callable")

        self._interval = float(interval)
        self._target = target
        self._stop_event = threading.Event()
        self._exception: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=name,
            daemon=bool(daemon),
        )

    @property
    def exception(self) -> BaseException | None:
        """Exception raised by ``target``, or ``None`` if it has not failed."""

        return self._exception

    def start(self) -> None:
        """Start the worker thread."""

        self._thread.start()

    def stop(self) -> None:
        """Request shutdown and wake an in-progress periodic wait."""

        self._stop_event.set()

    def join(
        self,
        timeout: float | None = None,
        *,
        raise_on_error: bool = True,
    ) -> bool:
        """Wait for shutdown and optionally propagate a target exception.

        Returns ``False`` if ``timeout`` expires while the worker is still
        alive. A recorded target exception is raised only after the worker has
        fully stopped.
        """

        self._thread.join(timeout)
        stopped = not self._thread.is_alive()
        if stopped and raise_on_error and self._exception is not None:
            raise self._exception
        return stopped

    def is_alive(self) -> bool:
        """Return whether the worker thread is currently running."""

        return self._thread.is_alive()

    def _run(self) -> None:
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            try:
                self._target()
            except BaseException as exc:
                self._exception = exc
                self._stop_event.set()
                return

            if self._stop_event.is_set():
                return

            next_deadline += self._interval
            now = time.monotonic()
            if next_deadline <= now:
                missed_periods = math.floor((now - next_deadline) / self._interval) + 1
                next_deadline += missed_periods * self._interval

            if self._stop_event.wait(max(0.0, next_deadline - time.monotonic())):
                return
