from __future__ import annotations

import logging
import socket
import struct
import threading
import time

from .camera_manager import MultiCameraManager
from .gst_streamer import GstVideoStreamer, StreamConfig
from .xr_protocol import parse_camera_request, parse_network_protocol, recv_exact


LOGGER = logging.getLogger(__name__)


class XRListenServer:
    def __init__(
        self,
        listen_address: str,
        camera_manager: MultiCameraManager,
        primary_camera: str,
        video_host: str = "",
    ):
        self.listen_ip, self.listen_port = self._parse_listen_address(listen_address)
        self.camera_manager = camera_manager
        self.primary_camera = primary_camera
        self.video_host = video_host.strip()
        self.stop_event = threading.Event()
        self.streaming_event = threading.Event()
        self.control_thread: threading.Thread | None = None
        self.streaming_thread: threading.Thread | None = None
        self.server_socket: socket.socket | None = None
        self.streamer: GstVideoStreamer | None = None
        self.stream_config: StreamConfig | None = None
        self._stream_lock = threading.Lock()

    @staticmethod
    def _parse_listen_address(listen_address: str) -> tuple[str, int]:
        if ":" not in listen_address:
            raise ValueError("--listen must look like <ip>:<port>")
        ip, port_text = listen_address.rsplit(":", 1)
        return ip, int(port_text)

    def set_primary_camera(self, camera_key: str) -> None:
        self.primary_camera = camera_key

    @staticmethod
    def _request_ip_is_unusable(ip: str) -> bool:
        ip = str(ip or "").strip().lower()
        return not ip or ip in {"0.0.0.0", "::", "localhost"} or ip.startswith("127.")

    def _stream_ip_for_request(self, request, peer_ip: str = "") -> str:
        if self.video_host:
            return self.video_host
        request_ip = str(request.ip or "").strip()
        if not self._request_ip_is_unusable(request_ip):
            return request_ip
        if peer_ip:
            LOGGER.info("XR request IP %r is not usable; falling back to peer IP %s", request.ip, peer_ip)
            return peer_ip
        return request_ip

    def start(self) -> None:
        if self.control_thread is not None and self.control_thread.is_alive():
            return
        self.stop_event.clear()
        self.control_thread = threading.Thread(target=self._control_loop, name="xr-listen", daemon=True)
        self.control_thread.start()
        LOGGER.info("XR listen service on %s:%d", self.listen_ip, self.listen_port)

    def _control_loop(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.listen_ip, self.listen_port))
        server.listen(1)
        server.settimeout(1.0)
        self.server_socket = server

        try:
            while not self.stop_event.is_set():
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise

                LOGGER.info("XR headset connected from %s", address)
                with connection:
                    while not self.stop_event.is_set():
                        try:
                            header = recv_exact(connection, 4)
                            body_len = struct.unpack(">I", header)[0]
                            body = recv_exact(connection, body_len)
                            command, payload = parse_network_protocol(body)
                            if command == "OPEN_CAMERA":
                                request = parse_camera_request(payload)
                                self._open_stream(request, peer_ip=address[0])
                            elif command == "CLOSE_CAMERA":
                                self._close_stream()
                            else:
                                LOGGER.warning("unknown XR command: %s", command)
                        except ConnectionError:
                            break
                        except Exception as exc:
                            LOGGER.warning("XR control error: %s", exc)
                            self._close_stream()
                            break
        finally:
            self._close_stream()
            try:
                server.close()
            except Exception:
                pass
            self.server_socket = None

    def _open_stream(self, request, peer_ip: str = "") -> None:
        with self._stream_lock:
            self._close_stream_locked()
            stream_ip = self._stream_ip_for_request(request, peer_ip=peer_ip)
            if request.port <= 0:
                raise ValueError(f"invalid XR video port: {request.port}")
            if self.video_host and self.video_host != request.ip:
                LOGGER.info("XR video host override: request_ip=%s stream_ip=%s", request.ip, stream_ip)
            LOGGER.info(
                "XR OPEN_CAMERA: request_ip=%s peer_ip=%s stream_ip=%s port=%d size=%dx%d fps=%d bitrate=%d camera=%s",
                request.ip,
                peer_ip or "<unknown>",
                stream_ip,
                request.port,
                request.width,
                request.height,
                request.fps,
                request.bitrate,
                request.camera or "<default>",
            )
            if request.width == 2560 and request.height == 720:
                config = StreamConfig(
                    ip=stream_ip,
                    port=request.port,
                    bitrate=request.bitrate if request.bitrate > 0 else 4_000_000,
                    output_width=2560,
                    output_height=720,
                    output_fps=60,
                    stereo_mode=True,
                )
            else:
                config = StreamConfig(
                    ip=stream_ip,
                    port=request.port,
                    bitrate=request.bitrate if request.bitrate > 0 else 4_000_000,
                    output_width=1280,
                    output_height=720,
                    output_fps=30,
                    stereo_mode=False,
                )

            self.streamer = GstVideoStreamer()
            self.streamer.start(config)
            self.stream_config = config
            self.streaming_event.set()
            self.streaming_thread = threading.Thread(target=self._stream_loop, name="xr-stream", daemon=True)
            self.streaming_thread.start()
            LOGGER.info("XR stream opened using primary camera %s", self.primary_camera)

    def _stream_loop(self) -> None:
        after_sequence: int | None = None
        last_camera = self.primary_camera

        while self.streaming_event.is_set() and not self.stop_event.is_set():
            if self.streamer is None or self.stream_config is None:
                return

            camera_key = self.primary_camera
            if camera_key != last_camera:
                after_sequence = None
                last_camera = camera_key

            try:
                frame_result = self.camera_manager.wait_for_camera_frame(
                    camera_key,
                    after_sequence=after_sequence,
                    timeout_ms=250,
                )
                if frame_result is None:
                    continue

                after_sequence, frame = frame_result
                profile = self.camera_manager.get_active_profile(camera_key)
                capture_fps = profile.fps if profile is not None else 30
                repeat = self._repeat_for_capture_fps(capture_fps)
                self.streamer.push_rgb_frame(frame.rgb, repeat=repeat)
            except Exception as exc:
                LOGGER.warning("XR stream loop stopped: %s", exc)
                break

        with self._stream_lock:
            self._close_stream_locked()

    def _repeat_for_capture_fps(self, capture_fps: int) -> int:
        if self.stream_config is None or capture_fps <= 0:
            return 1

        output_fps = int(self.stream_config.output_fps)
        if output_fps <= 0 or output_fps % int(capture_fps) != 0:
            return 1
        return max(1, output_fps // int(capture_fps))

    def _close_stream(self) -> None:
        with self._stream_lock:
            self._close_stream_locked()

    def _close_stream_locked(self) -> None:
        self.streaming_event.clear()
        if self.streaming_thread is not None and self.streaming_thread.is_alive():
            current = threading.current_thread()
            if self.streaming_thread is not current:
                self.streaming_thread.join(timeout=3.0)
        self.streaming_thread = None
        if self.streamer is not None:
            self.streamer.stop()
        self.streamer = None
        self.stream_config = None

    def stop(self) -> None:
        self.stop_event.set()
        self.streaming_event.clear()
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except Exception:
                pass
        if self.control_thread is not None and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        self.control_thread = None
        self._close_stream()
