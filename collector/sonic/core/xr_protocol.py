from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraRequest:
    width: int
    height: int
    fps: int
    bitrate: int
    enable_mv_hevc: int
    render_mode: int
    port: int
    camera: str
    ip: str


def recv_exact(sock, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data


def read_int32_le(buffer: bytes, offset: int) -> int:
    if offset + 4 > len(buffer):
        raise ValueError("buffer too short for int32")
    return struct.unpack_from("<i", buffer, offset)[0]


def read_compact_string(buffer: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(buffer):
        raise ValueError("buffer too short for compact string length")
    length = buffer[offset]
    offset += 1
    if offset + length > len(buffer):
        raise ValueError("buffer too short for compact string")
    value = buffer[offset : offset + length].decode("utf-8", errors="ignore")
    return value, offset + length


def parse_network_protocol(body: bytes) -> tuple[str, bytes]:
    offset = 0
    cmd_len = read_int32_le(body, offset)
    offset += 4
    if cmd_len < 0 or offset + cmd_len > len(body):
        raise ValueError("invalid command length")
    command = body[offset : offset + cmd_len].decode("utf-8", errors="ignore").rstrip("\x00")
    offset += cmd_len

    data_len = read_int32_le(body, offset)
    offset += 4
    if data_len < 0 or offset + data_len > len(body):
        raise ValueError("invalid data length")
    return command, body[offset : offset + data_len]


def parse_camera_request(data: bytes) -> CameraRequest:
    if len(data) < 31:
        raise ValueError("camera request too short")
    if data[0] != 0xCA or data[1] != 0xFE:
        raise ValueError("bad magic bytes")
    version = data[2]
    if version != 1:
        raise ValueError(f"unsupported protocol version: {version}")

    offset = 3
    width = read_int32_le(data, offset)
    offset += 4
    height = read_int32_le(data, offset)
    offset += 4
    fps = read_int32_le(data, offset)
    offset += 4
    bitrate = read_int32_le(data, offset)
    offset += 4
    enable_mv_hevc = read_int32_le(data, offset)
    offset += 4
    render_mode = read_int32_le(data, offset)
    offset += 4
    port = read_int32_le(data, offset)
    offset += 4
    camera, offset = read_compact_string(data, offset)
    ip, offset = read_compact_string(data, offset)
    _ = offset

    return CameraRequest(
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        enable_mv_hevc=enable_mv_hevc,
        render_mode=render_mode,
        port=port,
        camera=camera,
        ip=ip,
    )
