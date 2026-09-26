"""SONIC-compatible topic/header/binary decoder."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

HEADER_SIZE = 1280
DTYPES = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}


@dataclass(frozen=True, slots=True)
class DecodedMessage:
    topic: str
    version: int
    payload: dict[str, np.ndarray]


def decode_topic_message(message: bytes, topic: str) -> DecodedMessage:
    prefix = topic.encode()
    if not message.startswith(prefix):
        raise ValueError(f"message does not start with topic {topic!r}")
    if len(message) < len(prefix) + HEADER_SIZE:
        raise ValueError("truncated topic header")
    header_start = len(prefix)
    header_end = header_start + HEADER_SIZE
    header = json.loads(message[header_start:header_end].rstrip(b"\0").decode())
    payload = memoryview(message)[header_end:]
    cursor = 0
    result: dict[str, np.ndarray] = {}
    for field in header.get("fields", []):
        name = str(field["name"])
        dtype_name = str(field["dtype"])
        if dtype_name not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype_name!r}")
        dtype = DTYPES[dtype_name]
        shape = tuple(int(value) for value in field["shape"])
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        size = count * dtype.itemsize
        if cursor + size > len(payload):
            raise ValueError(f"truncated field {name!r}")
        result[name] = (
            np.frombuffer(payload[cursor : cursor + size], dtype=dtype, count=count).reshape(shape).copy()
        )
        cursor += size
    if cursor != len(payload):
        raise ValueError(f"message has {len(payload) - cursor} trailing bytes")
    return DecodedMessage(topic, int(header.get("v", 0)), result)
