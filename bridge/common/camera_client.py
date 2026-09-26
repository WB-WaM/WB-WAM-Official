from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from collector.sonic.core.remote_camera_protocol import (
    COMMAND_GET_FRAME,
    COMMAND_LIST_CAMERAS,
    CameraServiceProfile,
    decode_frame_reply,
    decode_list_cameras_response,
    encode_request,
)


@dataclass(frozen=True)
class RemoteCameraFrame:
    rgb: np.ndarray
    depth: np.ndarray | None
    receive_monotonic_ns: int


class RemoteCameraClient:
    def __init__(self, *, endpoint: str, camera_key: str, timeout_ms: int = 1000):
        import zmq

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._endpoint = endpoint
        self._camera_key = camera_key
        self._timeout_ms = timeout_ms
        self._socket = None

    @staticmethod
    def list_cameras(*, endpoint: str, timeout_ms: int = 1000) -> list[CameraServiceProfile]:
        import zmq

        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            socket.connect(endpoint)
            socket.send(encode_request(COMMAND_LIST_CAMERAS))
            return decode_list_cameras_response(socket.recv())
        finally:
            socket.close(0)

    def _ensure_socket(self):
        if self._socket is not None:
            return self._socket
        socket = self._context.socket(self._zmq.REQ)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        socket.setsockopt(self._zmq.SNDTIMEO, self._timeout_ms)
        socket.connect(self._endpoint)
        self._socket = socket
        return socket

    @staticmethod
    def _decode_rgb(rgb_jpeg: bytes) -> np.ndarray:
        import cv2

        rgb_bgr = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError("remote camera returned invalid JPEG")
        return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    def _get_frame(self, *, include_depth: bool) -> RemoteCameraFrame:
        socket = self._ensure_socket()
        try:
            socket.send(
                encode_request(
                    COMMAND_GET_FRAME,
                    camera_key=self._camera_key,
                    include_depth=include_depth,
                )
            )
            meta, rgb_jpeg, depth_payload = decode_frame_reply(socket.recv_multipart())
            receive_monotonic_ns = time.monotonic_ns()
        except Exception:
            self.close()
            raise
        depth = None
        if include_depth:
            if depth_payload is None:
                raise RuntimeError(f"remote camera {self._camera_key!r} did not return depth")
            depth = np.frombuffer(depth_payload, dtype=np.uint16).reshape((meta.height, meta.width))
        return RemoteCameraFrame(
            rgb=self._decode_rgb(rgb_jpeg),
            depth=depth,
            receive_monotonic_ns=receive_monotonic_ns,
        )

    def get_rgb_sample(self) -> RemoteCameraFrame:
        return self._get_frame(include_depth=False)

    def get_rgb(self) -> np.ndarray:
        return self.get_rgb_sample().rgb

    def get_rgbd_sample(self) -> RemoteCameraFrame:
        return self._get_frame(include_depth=True)

    def get_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        sample = self.get_rgbd_sample()
        assert sample.depth is not None
        return sample.rgb, sample.depth

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
