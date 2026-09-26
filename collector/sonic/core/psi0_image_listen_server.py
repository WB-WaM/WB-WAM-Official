from __future__ import annotations

import html
import json
import logging
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Psi0ImageListenServer:
    def __init__(
        self,
        listen_address: str,
        camera_manager,
        primary_camera: str,
        *,
        jpeg_quality: int = 80,
    ) -> None:
        self.listen_ip, self.listen_port = self._parse_listen_address(listen_address)
        self.camera_manager = camera_manager
        self.primary_camera = primary_camera
        self.jpeg_quality = jpeg_quality
        self.stop_event = threading.Event()
        self.server: _ThreadingHTTPServer | None = None
        self.server_thread: threading.Thread | None = None

    @staticmethod
    def _parse_listen_address(listen_address: str) -> tuple[str, int]:
        if ":" not in listen_address:
            raise ValueError("--listen must look like <ip>:<port>")
        ip, port_text = listen_address.rsplit(":", 1)
        return ip, int(port_text)

    def set_primary_camera(self, camera_key: str) -> None:
        self.primary_camera = camera_key

    def start(self) -> None:
        if self.server_thread is not None and self.server_thread.is_alive():
            return

        self.stop_event.clear()
        handler = self._make_handler()
        self.server = _ThreadingHTTPServer((self.listen_ip, self.listen_port), handler)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            name="psi0-image-listen",
            daemon=True,
        )
        self.server_thread.start()
        LOGGER.info(
            "Psi0-style image listen service on http://%s:%d/",
            self._display_host(),
            self.listen_port,
        )

    def stop(self) -> None:
        self.stop_event.set()
        if self.server is not None:
            try:
                self.server.shutdown()
            except Exception:
                pass
            try:
                self.server.server_close()
            except Exception:
                pass
        if self.server_thread is not None and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)
        self.server = None
        self.server_thread = None

    def _display_host(self) -> str:
        if self.listen_ip not in {"0.0.0.0", "", "::"}:
            return self.listen_ip
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "CollectorPsi0Image/1.0"

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path in {"", "/"}:
                    outer._handle_index(self, parse_qs(parsed.query))
                    return
                if parsed.path == "/stream.mjpg":
                    outer._handle_mjpeg_stream(self, parse_qs(parsed.query))
                    return
                if parsed.path == "/snapshot.jpg":
                    outer._handle_snapshot(self, parse_qs(parsed.query))
                    return
                if parsed.path == "/cameras.json":
                    outer._handle_cameras_json(self)
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                LOGGER.info("psi0_image_http: " + format, *args)

        return Handler

    def _camera_keys(self) -> list[str]:
        return list(self.camera_manager.camera_keys)

    def _resolve_camera_key(self, requested_key: str | None) -> str:
        if requested_key and requested_key in self._camera_keys():
            return requested_key
        return self.primary_camera

    def _latest_snapshot(self, camera_key: str):
        return self.camera_manager.get_latest_frame_snapshots().get(camera_key)

    def _wait_for_frame(self, camera_key: str, after_sequence: int | None, timeout_ms: int):
        return self.camera_manager.wait_for_camera_frame(
            camera_key,
            after_sequence=after_sequence,
            timeout_ms=timeout_ms,
        )

    def _encode_frame_jpeg(self, frame) -> bytes:
        if frame.encoded_color_jpeg is not None:
            return bytes(frame.encoded_color_jpeg)

        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not success:
            raise RuntimeError(f"failed to encode JPEG for camera {frame.key}")
        return encoded.tobytes()

    def _handle_index(self, handler: BaseHTTPRequestHandler, query: dict[str, list[str]]) -> None:
        requested_camera = (query.get("camera") or [""])[0].strip()
        active_camera = self._resolve_camera_key(requested_camera)
        camera_links = []
        for key in self._camera_keys():
            active_attr = " class='active'" if key == active_camera else ""
            camera_links.append(
                f"<a{active_attr} href='/?camera={html.escape(key)}'>{html.escape(key)}</a>"
            )

        body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Collector Psi0 Image Preview</title>
  <style>
    body {{
      margin: 0;
      font-family: sans-serif;
      background: #101418;
      color: #e8eef5;
    }}
    header {{
      padding: 16px 20px;
      border-bottom: 1px solid #23313f;
      background: #131c24;
    }}
    h1 {{
      margin: 0 0 10px 0;
      font-size: 18px;
    }}
    nav {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    nav a {{
      color: #9cc8ff;
      text-decoration: none;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid #33506d;
    }}
    nav a.active {{
      color: #101418;
      background: #9cc8ff;
      border-color: #9cc8ff;
    }}
    main {{
      padding: 20px;
    }}
    img {{
      display: block;
      width: min(100%, 1280px);
      height: auto;
      border-radius: 12px;
      border: 1px solid #23313f;
      background: #000;
    }}
    p {{
      color: #a8b6c4;
      max-width: 1280px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Collector Psi0-style image preview</h1>
    <nav>{''.join(camera_links)}</nav>
  </header>
  <main>
    <img src="/stream.mjpg?camera={html.escape(active_camera)}" alt="camera stream">
    <p>Open this page in the headset browser. Current camera: <strong>{html.escape(active_camera)}</strong>.</p>
  </main>
</body>
</html>
"""
        encoded = body.encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(encoded)))
        handler.end_headers()
        handler.wfile.write(encoded)

    def _handle_cameras_json(self, handler: BaseHTTPRequestHandler) -> None:
        payload = {
            "primary_camera": self.primary_camera,
            "cameras": self._camera_keys(),
        }
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(encoded)))
        handler.end_headers()
        handler.wfile.write(encoded)

    def _handle_snapshot(self, handler: BaseHTTPRequestHandler, query: dict[str, list[str]]) -> None:
        requested_camera = (query.get("camera") or [""])[0].strip()
        camera_key = self._resolve_camera_key(requested_camera)
        snapshot = self._latest_snapshot(camera_key)
        if snapshot is None:
            handler.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "camera frame unavailable")
            return

        _, frame = snapshot
        jpeg = self._encode_frame_jpeg(frame)
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Content-Length", str(len(jpeg)))
        handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        handler.end_headers()
        handler.wfile.write(jpeg)

    def _handle_mjpeg_stream(
        self,
        handler: BaseHTTPRequestHandler,
        query: dict[str, list[str]],
    ) -> None:
        requested_camera = (query.get("camera") or [""])[0].strip()
        boundary = b"--frame"
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Age", "0")
        handler.send_header("Cache-Control", "no-cache, private")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.end_headers()

        after_sequence: int | None = None
        active_camera = ""

        while not self.stop_event.is_set():
            camera_key = self._resolve_camera_key(requested_camera)
            if camera_key != active_camera:
                active_camera = camera_key
                after_sequence = None

            try:
                frame_result = self._wait_for_frame(camera_key, after_sequence=after_sequence, timeout_ms=1000)
                if frame_result is None:
                    snapshot = self._latest_snapshot(camera_key)
                    if snapshot is None:
                        continue
                    after_sequence, frame = snapshot
                else:
                    after_sequence, frame = frame_result

                jpeg = self._encode_frame_jpeg(frame)
                handler.wfile.write(boundary + b"\r\n")
                handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                handler.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionError, ConnectionResetError):
                return
            except Exception as exc:
                LOGGER.warning("psi0 image stream stopped for %s: %s", camera_key, exc)
                return
