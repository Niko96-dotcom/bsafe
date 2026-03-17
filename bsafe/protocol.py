"""IPC message format: constants, framing, parsing."""

import struct
from dataclasses import dataclass

# Message types: Swift → Python
MSG_FRAME = 0x01

# Message types: Python → Swift
CMD_START = 0x10
CMD_STOP = 0x11
CMD_SHUTDOWN = 0xFF

# Header: 4 bytes length + 1 byte type
HEADER_SIZE = 5
HEADER_FMT = "!IB"  # network byte-order: uint32 length, uint8 type

# Max message payload size (50 MB — generous for 4K JPEG frames)
MAX_MESSAGE_SIZE = 50 * 1024 * 1024

# Frame payload header: display_id(4) + width(4) + height(4) + timestamp_ns(8) = 20 bytes
FRAME_META_SIZE = 20
FRAME_META_FMT = "!IIIq"


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    display_id: int
    width: int
    height: int
    timestamp_ns: int


def pack_message(msg_type: int, payload: bytes = b"") -> bytes:
    """Pack a length-prefixed message: [4B length][1B type][payload]."""
    length = 1 + len(payload)  # type byte + payload
    return struct.pack(HEADER_FMT, length, msg_type) + payload


def read_message(recv_fn) -> tuple[int, bytes]:
    """Read one message from a stream.

    recv_fn(n) should return up to n bytes (like socket.recv).
    Returns (msg_type, payload).
    Raises ConnectionError if the stream closes mid-message.
    Raises ValueError if the message exceeds MAX_MESSAGE_SIZE.
    """
    header = _recv_exact(recv_fn, HEADER_SIZE)
    length, msg_type = struct.unpack(HEADER_FMT, header)
    payload_size = length - 1
    if payload_size > MAX_MESSAGE_SIZE:
        raise ValueError(
            f"Message too large: {payload_size} bytes exceeds {MAX_MESSAGE_SIZE} byte limit"
        )
    payload = _recv_exact(recv_fn, payload_size) if payload_size > 0 else b""
    return msg_type, payload


def parse_frame_payload(payload: bytes) -> tuple[FrameMetadata, bytes]:
    """Parse a FRAME message payload into metadata + JPEG data."""
    if len(payload) < FRAME_META_SIZE:
        raise ValueError(f"Frame payload too short: {len(payload)} < {FRAME_META_SIZE}")
    display_id, width, height, timestamp_ns = struct.unpack(
        FRAME_META_FMT, payload[:FRAME_META_SIZE]
    )
    jpeg_data = payload[FRAME_META_SIZE:]
    return FrameMetadata(display_id, width, height, timestamp_ns), jpeg_data


def pack_cmd_start(fps: int) -> bytes:
    """Pack a CMD_START message with the desired FPS."""
    return pack_message(CMD_START, struct.pack("!B", fps))


def _recv_exact(recv_fn, n: int) -> bytes:
    """Read exactly n bytes from recv_fn."""
    buf = bytearray()
    while len(buf) < n:
        chunk = recv_fn(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while reading message")
        buf.extend(chunk)
    return bytes(buf)
