from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from .depth_io import save_raw_depth

_THREAD_SENTINEL = object()
_DEPTH_WRITE = "write"
_DEPTH_BARRIER = "barrier"
_DEPTH_STOP = "stop"
_DEPTH_ERROR = "error"
_DEPTH_POLL_SECONDS = 0.05


class _ThreadBarrier:
    def __init__(self) -> None:
        self.done = threading.Event()


class AsyncJSONLWriter:
    def __init__(self) -> None:
        self._queue: queue.Queue[object] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="collector-jsonl-writer", daemon=True)
        self._thread.start()

    def write(self, path: Path, payload: dict) -> None:
        text = json.dumps(payload, ensure_ascii=True) + "\n"
        self._queue.put((path, text))

    def flush(self) -> None:
        self._queue.join()

    def close(self, cancel: bool = False) -> None:
        if cancel:
            self._drain_queue()
        self._queue.put(_THREAD_SENTINEL)
        self._thread.join(timeout=5.0)

    def _drain_queue(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not _THREAD_SENTINEL:
                self._queue.task_done()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _THREAD_SENTINEL:
                    return
                path, text = item
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(text)
            finally:
                self._queue.task_done()


class AsyncImageWriter:
    def __init__(self, *, wait_timeout_s: float = 30.0) -> None:
        self._queue: queue.Queue[object] = queue.Queue(maxsize=256)
        self._first_error: tuple[Path, BaseException] | None = None
        self._error_count = 0
        self._wait_timeout_s = float(wait_timeout_s)
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="collector-image-writer", daemon=True)
        self._thread.start()

    def write(
        self,
        path: Path,
        rgb: np.ndarray | None,
        *,
        encoded_jpeg: bytes | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("image writer is closed")
        if encoded_jpeg is not None:
            self._put((path, None, bytes(encoded_jpeg)))
            return
        if rgb is None:
            raise ValueError("rgb must be provided when encoded_jpeg is not available")
        self._put((path, rgb.copy(), None))

    def flush(self) -> None:
        if self._closed:
            return
        barrier = _ThreadBarrier()
        self._put(barrier)
        deadline = time.monotonic() + self._wait_timeout_s
        while not barrier.done.wait(timeout=_DEPTH_POLL_SECONDS):
            if not self._thread.is_alive():
                raise RuntimeError("image writer thread stopped unexpectedly")
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for image writer to flush")
        self._raise_if_failed()

    def close(self, cancel: bool = False) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        try:
            if cancel:
                self._drain_queue()
            else:
                self.flush()
        except BaseException as exc:
            failure = exc
        finally:
            self._closed = True
            if self._thread.is_alive():
                try:
                    self._put(_THREAD_SENTINEL, allow_closed=True)
                except BaseException as exc:
                    failure = failure or exc
                self._thread.join(timeout=self._wait_timeout_s)
            if self._thread.is_alive():
                failure = failure or RuntimeError("image writer thread did not stop")
        if failure is not None:
            raise failure

    def _put(self, item: object, *, allow_closed: bool = False) -> None:
        if self._closed and not allow_closed:
            raise RuntimeError("image writer is closed")
        deadline = time.monotonic() + self._wait_timeout_s
        while True:
            if not self._thread.is_alive():
                raise RuntimeError("image writer thread is not running")
            try:
                self._queue.put(item, timeout=_DEPTH_POLL_SECONDS)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for image writer queue")

    def _drain_queue(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not _THREAD_SENTINEL:
                self._queue.task_done()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _THREAD_SENTINEL:
                    return
                if isinstance(item, _ThreadBarrier):
                    item.done.set()
                    continue
                path, rgb, encoded_jpeg = item
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if encoded_jpeg is not None:
                        path.write_bytes(encoded_jpeg)
                    else:
                        assert rgb is not None
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        if not cv2.imwrite(str(path), bgr):
                            raise OSError(f"cv2.imwrite returned false for {path}")
                except BaseException as exc:
                    self._error_count += 1
                    if self._first_error is None:
                        self._first_error = (path, exc)
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._first_error is None:
            return
        path, exc = self._first_error
        message = f"image writer failed for {path}: {type(exc).__name__}: {exc}"
        if self._error_count > 1:
            message += f" ({self._error_count} failures total)"
        raise RuntimeError(message) from exc


def _depth_writer_main(
    task_queue,
    response_sender,
) -> None:
    error_count = 0
    error_reported = False
    try:
        while True:
            kind, token, depth = task_queue.get()
            if kind == _DEPTH_STOP:
                return
            if kind == _DEPTH_BARRIER:
                response_sender.send((_DEPTH_BARRIER, token, error_count))
                continue
            if kind != _DEPTH_WRITE or depth is None:
                raise RuntimeError(f"invalid depth writer task: {kind}")
            try:
                save_raw_depth(Path(token), depth)
            except BaseException as exc:
                error_count += 1
                if not error_reported:
                    response_sender.send((_DEPTH_ERROR, (token, type(exc).__name__, str(exc))))
                    error_reported = True
    finally:
        response_sender.close()


class DepthWriterProcess:
    def __init__(self, *, wait_timeout_s: float = 30.0) -> None:
        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=128)
        self._response_receiver, response_sender = context.Pipe(duplex=False)
        self._first_error: tuple[str, str, str] | None = None
        self._error_count = 0
        self._next_barrier = 0
        self._wait_timeout_s = float(wait_timeout_s)
        self._closed = False
        self._process = context.Process(
            target=_depth_writer_main,
            args=(self._queue, response_sender),
            daemon=True,
        )
        self._process.start()
        response_sender.close()

    def write(self, path: Path, depth: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("depth writer is closed")
        self._put((_DEPTH_WRITE, str(path), depth.copy()))

    def flush(self) -> None:
        if self._closed:
            return
        token = str(self._next_barrier)
        self._next_barrier += 1
        self._put((_DEPTH_BARRIER, token, None))
        self._wait_for_barrier(token)
        self._raise_if_failed()

    def close(self, cancel: bool = False) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        graceful = not cancel
        if graceful:
            try:
                self.flush()
            except BaseException as exc:
                failure = exc
        try:
            if self._process.is_alive() and graceful:
                try:
                    self._put((_DEPTH_STOP, "", None))
                    self._process.join(timeout=self._wait_timeout_s)
                except BaseException as exc:
                    failure = failure or exc
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
            if self._process.is_alive() and hasattr(self._process, "kill"):
                self._process.kill()
                self._process.join(timeout=5.0)
            if self._process.is_alive():
                failure = failure or RuntimeError("depth writer process did not stop")
            elif graceful and self._process.exitcode not in {0, None}:
                failure = failure or RuntimeError(
                    f"depth writer process exited with code {self._process.exitcode}"
                )
        finally:
            self._closed = True
            try:
                if cancel or self._process.is_alive() or self._process.exitcode not in {0, None}:
                    self._queue.cancel_join_thread()
                self._queue.close()
                if not cancel and self._process.exitcode == 0:
                    self._queue.join_thread()
            except BaseException as exc:
                failure = failure or exc
            try:
                self._response_receiver.close()
            except BaseException as exc:
                failure = failure or exc
        if failure is not None:
            raise failure

    def _put(self, task) -> None:
        deadline = time.monotonic() + self._wait_timeout_s
        while True:
            if not self._process.is_alive():
                self._drain_responses()
                self._raise_if_failed()
                raise RuntimeError("depth writer process is not running")
            try:
                self._queue.put(task, timeout=_DEPTH_POLL_SECONDS)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for depth writer queue")

    def _wait_for_barrier(self, token: str) -> None:
        deadline = time.monotonic() + self._wait_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("timed out waiting for depth writer to flush")
            try:
                if self._response_receiver.poll(min(_DEPTH_POLL_SECONDS, remaining)):
                    kind, payload, *extra = self._response_receiver.recv()
                    if kind == _DEPTH_ERROR:
                        if self._first_error is None:
                            self._first_error = payload
                    elif kind == _DEPTH_BARRIER:
                        self._error_count = max(self._error_count, int(extra[0]))
                        if payload == token:
                            if not self._process.is_alive():
                                self._raise_if_failed()
                                raise RuntimeError("depth writer process stopped unexpectedly")
                            return
            except (EOFError, OSError) as exc:
                self._raise_if_failed()
                raise RuntimeError("depth writer response channel closed") from exc
            if not self._process.is_alive():
                self._drain_responses()
                self._raise_if_failed()
                raise RuntimeError("depth writer process stopped unexpectedly")

    def _drain_responses(self) -> None:
        try:
            while self._response_receiver.poll():
                kind, payload, *extra = self._response_receiver.recv()
                if kind == _DEPTH_ERROR and self._first_error is None:
                    self._first_error = payload
                elif kind == _DEPTH_BARRIER:
                    self._error_count = max(self._error_count, int(extra[0]))
        except (EOFError, OSError):
            pass

    def _raise_if_failed(self) -> None:
        if self._first_error is None:
            return
        path, error_type, message = self._first_error
        detail = f"depth writer failed for {path}: {error_type}: {message}"
        if self._error_count > 1:
            detail += f" ({self._error_count} failures total)"
        raise RuntimeError(detail)
