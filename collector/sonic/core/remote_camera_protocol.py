from __future__ import annotations

import json
from dataclasses import dataclass

from .schema import CameraInfo


COMMAND_LIST_CAMERAS = "LIST_CAMERAS"
COMMAND_GET_FRAME = "GET_FRAME"


class RemoteCameraProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraServiceProfile:
    key: str
    model: str
    serial: str
    width: int
    height: int
    fps: int
    has_depth: bool
    product_line: str = ""
    firmware_version: str = ""
    usb_type: str = ""

    def to_camera_info(self) -> CameraInfo:
        return CameraInfo(
            key=self.key,
            model=self.model,
            serial=self.serial,
            product_line=self.product_line,
            firmware_version=self.firmware_version,
            usb_type=self.usb_type,
        )

    def to_record(self) -> dict:
        return {
            "key": self.key,
            "model": self.model,
            "serial": self.serial,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "has_depth": self.has_depth,
            "product_line": self.product_line,
            "firmware_version": self.firmware_version,
            "usb_type": self.usb_type,
        }

    @classmethod
    def from_record(cls, record: dict) -> "CameraServiceProfile":
        return cls(
            key=str(record["key"]),
            model=str(record["model"]),
            serial=str(record.get("serial", "")),
            width=int(record["width"]),
            height=int(record["height"]),
            fps=int(record["fps"]),
            has_depth=bool(record.get("has_depth", False)),
            product_line=str(record.get("product_line", "")),
            firmware_version=str(record.get("firmware_version", "")),
            usb_type=str(record.get("usb_type", "")),
        )


@dataclass(frozen=True)
class FrameResponseMeta:
    camera_key: str
    model: str
    serial: str
    sequence: int
    capture_monotonic_ns: int
    capture_realtime_ns: int
    width: int
    height: int
    fps: int
    has_depth: bool

    def to_record(self) -> dict:
        return {
            "ok": True,
            "camera_key": self.camera_key,
            "model": self.model,
            "serial": self.serial,
            "sequence": self.sequence,
            "capture_monotonic_ns": self.capture_monotonic_ns,
            "capture_realtime_ns": self.capture_realtime_ns,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "has_depth": self.has_depth,
        }

    @classmethod
    def from_record(cls, record: dict) -> "FrameResponseMeta":
        return cls(
            camera_key=str(record["camera_key"]),
            model=str(record["model"]),
            serial=str(record.get("serial", "")),
            sequence=int(record["sequence"]),
            capture_monotonic_ns=int(record["capture_monotonic_ns"]),
            capture_realtime_ns=int(record["capture_realtime_ns"]),
            width=int(record["width"]),
            height=int(record["height"]),
            fps=int(record["fps"]),
            has_depth=bool(record.get("has_depth", False)),
        )


def _encode_json(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _decode_json(message: bytes) -> dict:
    try:
        payload = json.loads(message.decode("utf-8"))
    except Exception as exc:
        raise RemoteCameraProtocolError(f"invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise RemoteCameraProtocolError("JSON payload must be an object")
    return payload


def encode_request(command: str, **payload) -> bytes:
    request = {"command": command}
    request.update(payload)
    return _encode_json(request)


def decode_request(message: bytes) -> tuple[str, dict]:
    payload = _decode_json(message)
    try:
        command = str(payload.pop("command"))
    except KeyError as exc:
        raise RemoteCameraProtocolError("request missing command") from exc
    return command, payload


def encode_list_cameras_response(cameras: list[CameraServiceProfile]) -> bytes:
    return _encode_json(
        {
            "ok": True,
            "cameras": [camera.to_record() for camera in cameras],
        }
    )


def decode_list_cameras_response(message: bytes) -> list[CameraServiceProfile]:
    payload = _decode_json(message)
    if not payload.get("ok", False):
        raise RemoteCameraProtocolError(str(payload.get("error", "remote camera request failed")))
    cameras = payload.get("cameras")
    if not isinstance(cameras, list):
        raise RemoteCameraProtocolError("camera list response missing cameras array")
    return [CameraServiceProfile.from_record(record) for record in cameras]


def encode_error_response(error: str, *, command: str | None = None) -> bytes:
    payload = {"ok": False, "error": error}
    if command is not None:
        payload["command"] = command
    return _encode_json(payload)


def encode_frame_meta(meta: FrameResponseMeta) -> bytes:
    return _encode_json(meta.to_record())


def decode_frame_reply(parts: list[bytes]) -> tuple[FrameResponseMeta, bytes, bytes | None]:
    if not parts:
        raise RemoteCameraProtocolError("empty frame response")

    meta_payload = _decode_json(parts[0])
    if not meta_payload.get("ok", False):
        raise RemoteCameraProtocolError(str(meta_payload.get("error", "frame request failed")))
    if len(parts) < 2:
        raise RemoteCameraProtocolError("frame response missing RGB payload")

    meta = FrameResponseMeta.from_record(meta_payload)
    rgb_jpeg = parts[1]
    depth_payload = parts[2] if len(parts) >= 3 else None

    if meta.has_depth and depth_payload is None:
        raise RemoteCameraProtocolError("frame response declared depth but did not include depth payload")
    if not meta.has_depth:
        depth_payload = None

    return meta, rgb_jpeg, depth_payload
