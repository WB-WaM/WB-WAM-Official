from __future__ import annotations

import logging
import socket
import struct
from dataclasses import dataclass

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)

GST_IMPORT_ERROR: Exception | None = None
REQUIRED_GST_ELEMENTS = ("appsrc", "queue", "videoconvert", "x264enc", "h264parse", "appsink")

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
except Exception as exc:  # gi may exist while Gst or its typelibs/plugins are missing.
    gi = None
    Gst = None
    GST_IMPORT_ERROR = exc



def gst_unavailable_reason() -> str | None:
    if Gst is None:
        detail = f": {GST_IMPORT_ERROR}" if GST_IMPORT_ERROR is not None else ""
        return f"Python gi/Gst import failed{detail}"

    missing = [name for name in REQUIRED_GST_ELEMENTS if Gst.ElementFactory.find(name) is None]
    if missing:
        return "missing GStreamer element(s): " + ", ".join(missing)
    return None


def require_gst_environment() -> None:
    reason = gst_unavailable_reason()
    if reason is None:
        return
    raise RuntimeError(
        "XR gst video backend is unavailable: "
        f"{reason}. Install GStreamer Python bindings and plugins, including x264enc, "
        "or set XR_LISTEN_BACKEND=psi0_image for browser MJPEG preview."
    )


@dataclass(frozen=True)
class StreamConfig:
    ip: str
    port: int
    bitrate: int
    output_width: int
    output_height: int
    output_fps: int
    stereo_mode: bool


def letterbox_bgr_frame(rgb: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    src_height, src_width = rgb.shape[:2]
    if src_height <= 0 or src_width <= 0:
        raise ValueError("input frame must have a positive size")

    scale = min(target_width / src_width, target_height / src_height)
    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))
    resized_bgr = cv2.cvtColor(
        cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_RGB2BGR,
    )

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized_bgr
    return canvas


def compose_output_bgr(rgb: np.ndarray, config: StreamConfig) -> np.ndarray:
    eye_width = config.output_width // 2 if config.stereo_mode else config.output_width
    eye_bgr = letterbox_bgr_frame(rgb, eye_width, config.output_height)
    if config.stereo_mode:
        return np.concatenate([eye_bgr, eye_bgr], axis=1)
    return eye_bgr


class TcpPacketSender:
    def __init__(self) -> None:
        self.sock: socket.socket | None = None

    def connect(self, ip: str, port: int) -> None:
        self.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect((ip, port))
        self.sock = sock
        LOGGER.info("XR video connected to %s:%d", ip, port)

    def send_packet(self, payload: bytes) -> None:
        if self.sock is None:
            raise ConnectionError("video socket not connected")
        packet = struct.pack(">I", len(payload)) + payload
        self.sock.sendall(packet)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None


class GstVideoStreamer:
    def __init__(self) -> None:
        self.video_sender = TcpPacketSender()
        self.gst_pipeline = None
        self.appsrc = None
        self.appsink = None
        self.config: StreamConfig | None = None
        self.frame_index = 0
        self.send_failed = False

    def _pipeline_desc(self, config: StreamConfig) -> str:
        key_int_max = max(1, int(config.output_fps // 2))
        return (
            f"appsrc name=mysource is-live=true block=true format=time "
            f"caps=video/x-raw,format=BGR,width={config.output_width},height={config.output_height},"
            f"framerate={config.output_fps}/1 ! "
            f"queue leaky=downstream max-size-buffers=4 ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc bitrate={max(1, config.bitrate // 1000)} speed-preset=ultrafast "
            f"tune=zerolatency key-int-max={key_int_max} bframes=0 byte-stream=true aud=true ! "
            f"h264parse config-interval=-1 ! "
            f"video/x-h264,stream-format=byte-stream,alignment=au,profile=constrained-baseline ! "
            f"appsink name=mysink emit-signals=true sync=false max-buffers=4 drop=true"
        )

    def start(self, config: StreamConfig) -> None:
        require_gst_environment()
        self.stop()
        self.config = config
        self.frame_index = 0
        self.send_failed = False
        self.video_sender.connect(config.ip, config.port)
        self.gst_pipeline = Gst.parse_launch(self._pipeline_desc(config))
        self.appsrc = self.gst_pipeline.get_by_name("mysource")
        self.appsink = self.gst_pipeline.get_by_name("mysink")
        self.appsink.connect("new-sample", self._on_new_sample)
        self.gst_pipeline.set_state(Gst.State.PLAYING)

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        ok, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR

        try:
            payload = bytes(mapinfo.data)
            if payload:
                self.video_sender.send_packet(payload)
        except Exception as exc:
            LOGGER.warning("XR packet send failed: %s", exc)
            self.send_failed = True
        finally:
            buffer.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def push_rgb_frame(self, rgb: np.ndarray, repeat: int = 1) -> None:
        if self.config is None or self.appsrc is None:
            raise RuntimeError("streamer not started")
        if self.send_failed:
            raise RuntimeError("XR stream send failed")

        bgr = compose_output_bgr(rgb, self.config)
        raw = bgr.tobytes()
        duration_ns = int(1e9 / self.config.output_fps)
        for _ in range(max(1, repeat)):
            buffer = Gst.Buffer.new_allocate(None, len(raw), None)
            buffer.fill(0, raw)
            pts_ns = int(self.frame_index * duration_ns)
            buffer.pts = pts_ns
            buffer.dts = pts_ns
            buffer.duration = duration_ns
            result = self.appsrc.emit("push-buffer", buffer)
            if result != Gst.FlowReturn.OK:
                raise RuntimeError(f"push-buffer failed: {result}")
            self.frame_index += 1

    def stop(self) -> None:
        if self.appsrc is not None:
            try:
                self.appsrc.emit("end-of-stream")
            except Exception:
                pass
        if self.gst_pipeline is not None:
            try:
                self.gst_pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.video_sender.close()
        self.gst_pipeline = None
        self.appsrc = None
        self.appsink = None
        self.config = None
        self.frame_index = 0
        self.send_failed = False
