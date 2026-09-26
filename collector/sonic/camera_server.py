from __future__ import annotations

import argparse
import logging
import threading
import time

import cv2
import numpy as np

from core.constants import (
    CAMERA_FPS,
    SUPPORTED_CAMERA_FPS,
    normalize_camera_fps,
)
from core.camera_discovery import discover_cameras
from core.camera_manager import StreamProfile
from core.remote_camera_protocol import (
    COMMAND_GET_FRAME,
    COMMAND_LIST_CAMERAS,
    CameraServiceProfile,
    FrameResponseMeta,
    RemoteCameraProtocolError,
    decode_request,
    encode_error_response,
    encode_frame_meta,
    encode_list_cameras_response,
)
from core.schema import CameraInfo


LOGGER = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import zmq
except ImportError:
    zmq = None


class ServerManagedCamera:
    def __init__(
        self,
        camera_info: CameraInfo,
        *,
        width: int,
        height: int,
        fps: int,
        record_depth: bool,
        jpeg_quality: int,
    ):
        self.camera_info = camera_info
        self.record_depth = record_depth
        self.jpeg_quality = jpeg_quality
        self.profile = StreamProfile(width=width, height=height, fps=fps)
        self.pipeline = None
        self.align = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_meta: FrameResponseMeta | None = None
        self.latest_rgb_jpeg: bytes | None = None
        self.latest_depth: bytes | None = None
        self._sequence = 0

    def start(self) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 is required to run camera_server.py")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.camera_info.serial)
        config.enable_stream(rs.stream.color, self.profile.width, self.profile.height, rs.format.bgr8, self.profile.fps)
        if self.record_depth:
            config.enable_stream(rs.stream.depth, self.profile.width, self.profile.height, rs.format.z16, self.profile.fps)
        align = rs.align(rs.stream.color) if self.record_depth else None
        pipeline.start(config)

        self.pipeline = pipeline
        self.align = align
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, name=f"camera-server-{self.camera_info.key}", daemon=True)
        self.thread.start()
        LOGGER.info(
            "camera_server started %s at %dx%d@%d",
            self.camera_info.key,
            self.profile.width,
            self.profile.height,
            self.profile.fps,
        )

    def _capture_loop(self) -> None:
        assert self.pipeline is not None

        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=250)
                if self.align is not None:
                    frames = self.align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if self.record_depth else None
                if not color_frame or (self.record_depth and not depth_frame):
                    continue

                color_bgr = np.asanyarray(color_frame.get_data())
                success, encoded_rgb = cv2.imencode(
                    ".jpg",
                    color_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not success:
                    raise RuntimeError(f"failed to encode RGB frame for camera {self.camera_info.key}")

                depth_bytes = None
                if depth_frame is not None:
                    depth_bytes = np.asanyarray(depth_frame.get_data()).tobytes()

                self._sequence += 1
                meta = FrameResponseMeta(
                    camera_key=self.camera_info.key,
                    model=self.camera_info.model,
                    serial=self.camera_info.serial,
                    sequence=self._sequence,
                    capture_monotonic_ns=time.monotonic_ns(),
                    capture_realtime_ns=time.time_ns(),
                    width=self.profile.width,
                    height=self.profile.height,
                    fps=self.profile.fps,
                    has_depth=depth_bytes is not None,
                )
                with self.lock:
                    self.latest_meta = meta
                    self.latest_rgb_jpeg = encoded_rgb.tobytes()
                    self.latest_depth = depth_bytes
            except RuntimeError:
                continue
            except Exception as exc:
                LOGGER.warning("camera_server capture loop error for %s: %s", self.camera_info.key, exc)
                time.sleep(0.1)

    def get_latest_reply(self, *, include_depth: bool) -> list[bytes]:
        with self.lock:
            if self.latest_meta is None or self.latest_rgb_jpeg is None:
                raise RuntimeError(f"camera {self.camera_info.key} has no frame yet")
            meta = FrameResponseMeta(
                camera_key=self.latest_meta.camera_key,
                model=self.latest_meta.model,
                serial=self.latest_meta.serial,
                sequence=self.latest_meta.sequence,
                capture_monotonic_ns=self.latest_meta.capture_monotonic_ns,
                capture_realtime_ns=self.latest_meta.capture_realtime_ns,
                width=self.latest_meta.width,
                height=self.latest_meta.height,
                fps=self.latest_meta.fps,
                has_depth=include_depth and self.latest_depth is not None,
            )
            reply = [encode_frame_meta(meta), self.latest_rgb_jpeg]
            if include_depth and self.latest_depth is not None:
                reply.append(self.latest_depth)
            return reply

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.align = None


class CameraServer:
    def __init__(self, *, bind: str, width: int, height: int, fps: int, record_depth: bool, jpeg_quality: int):
        if zmq is None:
            raise RuntimeError("pyzmq is required to run camera_server.py")
        fps = normalize_camera_fps(fps)
        self.bind = bind
        self.width = width
        self.height = height
        self.fps = fps
        self.record_depth = record_depth
        self.jpeg_quality = jpeg_quality
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(bind)
        cameras = discover_cameras()
        self.managed = {
            camera.key: ServerManagedCamera(
                camera,
                width=width,
                height=height,
                fps=fps,
                record_depth=record_depth,
                jpeg_quality=jpeg_quality,
            )
            for camera in cameras
        }

    def start(self) -> None:
        if not self.managed:
            raise RuntimeError("no RealSense cameras detected for camera_server.py")
        for managed in self.managed.values():
            managed.start()
        LOGGER.info("camera_server listening on %s", self.bind)

    def serve_forever(self) -> None:
        try:
            while True:
                request = self.socket.recv()
                try:
                    command, payload = decode_request(request)
                    if command == COMMAND_LIST_CAMERAS:
                        self.socket.send(self._list_cameras())
                    elif command == COMMAND_GET_FRAME:
                        camera_key = str(payload.get("camera_key", "")).strip()
                        if not camera_key:
                            raise RemoteCameraProtocolError("GET_FRAME requires camera_key")
                        include_depth = bool(payload.get("include_depth", self.record_depth))
                        self.socket.send_multipart(self._get_frame(camera_key, include_depth=include_depth))
                    else:
                        raise RemoteCameraProtocolError(f"unknown command: {command}")
                except Exception as exc:
                    self.socket.send(encode_error_response(str(exc)))
        finally:
            self.stop()

    def _list_cameras(self) -> bytes:
        cameras = [
            CameraServiceProfile(
                key=managed.camera_info.key,
                model=managed.camera_info.model,
                serial=managed.camera_info.serial,
                width=managed.profile.width,
                height=managed.profile.height,
                fps=managed.profile.fps,
                has_depth=self.record_depth,
                product_line=managed.camera_info.product_line,
                firmware_version=managed.camera_info.firmware_version,
                usb_type=managed.camera_info.usb_type,
            )
            for managed in self.managed.values()
        ]
        return encode_list_cameras_response(cameras)

    def _get_frame(self, camera_key: str, *, include_depth: bool) -> list[bytes]:
        managed = self.managed.get(camera_key)
        if managed is None:
            raise RemoteCameraProtocolError(f"unknown camera key: {camera_key}")
        return managed.get_latest_reply(include_depth=include_depth)

    def stop(self) -> None:
        for managed in self.managed.values():
            managed.stop()
        try:
            self.socket.close(0)
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve remote RealSense frames to collector.")
    parser.add_argument("--bind", type=str, default="tcp://0.0.0.0:5560")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=CAMERA_FPS, choices=SUPPORTED_CAMERA_FPS)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--record-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )
    server = CameraServer(
        bind=args.bind,
        width=args.width,
        height=args.height,
        fps=args.fps,
        record_depth=args.record_depth,
        jpeg_quality=args.jpeg_quality,
    )
    server.start()
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
