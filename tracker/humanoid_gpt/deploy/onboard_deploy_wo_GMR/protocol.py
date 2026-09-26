"""Wire format for the host -> robot mocap stream.

A single UDP datagram carries one retargeted G1 frame (qpos_full + optional
hand + optional BrainCo dexterous-hand qpos).  We keep the format small and
fixed-layout so encoding/decoding is cheap and the packet stays well below
the WiFi MTU (1500 B) to avoid IP fragmentation.

Layout (little-endian, fixed-size header + variable trailing sections)::

    offset             size  field
    -----------------  ----  --------------------------------------------
    0                  4     magic       b"HMCP"  (Humanoid MoCap Packet)
    4                  1     version     uint8    schema version (=1)
    5                  1     flags       uint8    bit0: has_hand
                                                  bit1: has_brainco
    6                  4     seq         uint32   per-stream monotonic seq
    10                 8     send_ts     float64  sender wall-clock time.time()
    18                 2     n_qpos      uint16   number of float32 qpos entries
    20                 4*n   qpos        float32  G1 full qpos
                                                  (root_pos3 + root_rot4 + dof29 = 36)
    20+4*n             16    hand        float32[4]  [l_open, l_dist, r_open, r_dist]
                                                     (only if has_hand)
    20+4*n+[16]        96    brainco     float32[24] BrainCo dex-hand qpos
                                                     (left12 then right12 -- this
                                                     mirrors the layout produced
                                                     by GMR's brainco/brainco2/
                                                     brainco3 targets and is
                                                     what brainco_qpos24_to_cmd12
                                                     expects.  Only if has_brainco.)

Default packet sizes for the G1 (n_qpos=36):

- Dex3 mode      (has_hand=1, has_brainco=0): 20 +144 +16     = 180 B
- BrainCo mode   (has_hand=1, has_brainco=1): 20 +144 +16 +96 = 276 B
- Body-only mode (has_hand=0, has_brainco=0): 20 +144         = 164 B

Both sides agree on layout via :func:`encode_frame` / :func:`decode_frame`.
The protocol carries no other state -- recovery from drops is the receiver's
job (jitter buffer + sequence-gap counter).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


MAGIC = b"HMCP"
VERSION = 1
FLAG_HAS_HAND = 0x01
FLAG_HAS_BRAINCO = 0x02

# Fixed prefix: magic(4) + version(1) + flags(1) + seq(4) + send_ts(8) + n_qpos(2)
_HEADER_STRUCT = struct.Struct("<4sBBIdH")
HEADER_SIZE = _HEADER_STRUCT.size  # 20

HAND_FLOATS = 4
HAND_SIZE = HAND_FLOATS * 4  # 16

BRAINCO_QPOS_FLOATS = 24
BRAINCO_QPOS_SIZE = BRAINCO_QPOS_FLOATS * 4  # 96

# Default G1 layout: 3 root_pos + 4 root_rot + 29 dof = 36 floats
G1_DOF_FULL = 36

DEFAULT_PORT = 51234

# Upper bound for a single datagram; gives plenty of headroom over current
# packets (276 B with BrainCo) while staying well below typical WiFi MTU.
MAX_PACKET_SIZE = 1024


@dataclass
class MocapPacket:
    seq: int
    send_ts: float
    qpos: np.ndarray                   # shape (n_qpos,), dtype float32
    hand: np.ndarray | None            # shape (4,), float32, or None
    brainco_qpos: np.ndarray | None    # shape (24,), float32, or None


def encode_frame(
    seq: int,
    send_ts: float,
    qpos: np.ndarray,
    hand: np.ndarray | None = None,
    brainco_qpos: np.ndarray | None = None,
) -> bytes:
    """Serialize a single mocap frame.

    ``qpos`` must be a 1-D float32 array.  ``hand`` (4 floats) and
    ``brainco_qpos`` (24 floats) are optional and independently flagged.
    """
    if qpos.dtype != np.float32:
        qpos = qpos.astype(np.float32, copy=False)
    if qpos.ndim != 1:
        raise ValueError(f"qpos must be 1-D, got shape {qpos.shape}")

    flags = 0
    hand_bytes = b""
    if hand is not None:
        if hand.dtype != np.float32:
            hand = hand.astype(np.float32, copy=False)
        if hand.shape != (HAND_FLOATS,):
            raise ValueError(f"hand must be shape ({HAND_FLOATS},), got {hand.shape}")
        flags |= FLAG_HAS_HAND
        hand_bytes = hand.tobytes()

    brainco_bytes = b""
    if brainco_qpos is not None:
        if brainco_qpos.dtype != np.float32:
            brainco_qpos = brainco_qpos.astype(np.float32, copy=False)
        if brainco_qpos.shape != (BRAINCO_QPOS_FLOATS,):
            raise ValueError(
                f"brainco_qpos must be shape ({BRAINCO_QPOS_FLOATS},), "
                f"got {brainco_qpos.shape}"
            )
        flags |= FLAG_HAS_BRAINCO
        brainco_bytes = brainco_qpos.tobytes()

    header = _HEADER_STRUCT.pack(
        MAGIC, VERSION, flags, seq & 0xFFFFFFFF, float(send_ts), qpos.size
    )
    return header + qpos.tobytes() + hand_bytes + brainco_bytes


def decode_frame(buf: bytes) -> MocapPacket:
    """Inverse of :func:`encode_frame`.

    Raises ``ValueError`` if the magic or version do not match, or if the
    payload length disagrees with the declared ``n_qpos``/flags.
    """
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"packet too short: {len(buf)} < {HEADER_SIZE}")
    magic, version, flags, seq, send_ts, n_qpos = _HEADER_STRUCT.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version: {version}")

    has_hand = bool(flags & FLAG_HAS_HAND)
    has_brainco = bool(flags & FLAG_HAS_BRAINCO)

    qpos_bytes = n_qpos * 4
    expected = (
        HEADER_SIZE
        + qpos_bytes
        + (HAND_SIZE if has_hand else 0)
        + (BRAINCO_QPOS_SIZE if has_brainco else 0)
    )
    if len(buf) != expected:
        raise ValueError(
            f"packet length {len(buf)} != expected {expected} "
            f"(n_qpos={n_qpos}, has_hand={has_hand}, has_brainco={has_brainco})"
        )

    off = HEADER_SIZE
    qpos = np.frombuffer(buf, dtype=np.float32, count=n_qpos, offset=off).copy()
    off += qpos_bytes

    hand: np.ndarray | None = None
    if has_hand:
        hand = np.frombuffer(buf, dtype=np.float32, count=HAND_FLOATS, offset=off).copy()
        off += HAND_SIZE

    brainco: np.ndarray | None = None
    if has_brainco:
        brainco = np.frombuffer(
            buf, dtype=np.float32, count=BRAINCO_QPOS_FLOATS, offset=off
        ).copy()
        off += BRAINCO_QPOS_SIZE

    return MocapPacket(
        seq=seq, send_ts=send_ts, qpos=qpos, hand=hand, brainco_qpos=brainco
    )


def packet_size(
    n_qpos: int = G1_DOF_FULL,
    has_hand: bool = True,
    has_brainco: bool = False,
) -> int:
    """Return the byte size of a packet with the given layout."""
    return (
        HEADER_SIZE
        + n_qpos * 4
        + (HAND_SIZE if has_hand else 0)
        + (BRAINCO_QPOS_SIZE if has_brainco else 0)
    )
