"""Unit tests for message framing round-trips."""

import io
import struct

from bsafe.protocol import (
    CMD_SHUTDOWN,
    CMD_START,
    CMD_STOP,
    FRAME_META_FMT,
    FRAME_META_SIZE,
    HEADER_FMT,
    MAX_MESSAGE_SIZE,
    MSG_FRAME,
    FrameMetadata,
    pack_cmd_start,
    pack_message,
    parse_frame_payload,
    read_message,
)


def _make_recv(data: bytes):
    """Create a recv_fn from bytes."""
    stream = io.BytesIO(data)
    return stream.read


def test_pack_and_read_empty_payload():
    raw = pack_message(CMD_STOP)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_STOP
    assert payload == b""


def test_pack_and_read_with_payload():
    data = b"hello"
    raw = pack_message(CMD_START, data)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_START
    assert payload == data


def test_pack_cmd_start():
    raw = pack_cmd_start(fps=3)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == CMD_START
    assert struct.unpack("!B", payload)[0] == 3


def test_frame_round_trip():
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg-data"
    meta = FrameMetadata(display_id=1, width=1920, height=1080, timestamp_ns=123456789)
    frame_payload = (
        struct.pack(FRAME_META_FMT, meta.display_id, meta.width, meta.height, meta.timestamp_ns)
        + jpeg
    )

    raw = pack_message(MSG_FRAME, frame_payload)
    msg_type, payload = read_message(_make_recv(raw))
    assert msg_type == MSG_FRAME

    parsed_meta, parsed_jpeg = parse_frame_payload(payload)
    assert parsed_meta == meta
    assert parsed_jpeg == jpeg


def test_parse_frame_payload_too_short():
    import pytest

    with pytest.raises(ValueError, match="too short"):
        parse_frame_payload(b"short")


def test_read_message_connection_closed():
    import pytest

    with pytest.raises(ConnectionError):
        read_message(_make_recv(b""))


def test_read_message_partial_header():
    import pytest

    with pytest.raises(ConnectionError):
        read_message(_make_recv(b"\x00\x00"))


def test_multiple_messages():
    raw = pack_message(CMD_START, b"\x03") + pack_message(CMD_SHUTDOWN)
    stream = io.BytesIO(raw)
    recv = stream.read

    t1, p1 = read_message(recv)
    assert t1 == CMD_START
    assert p1 == b"\x03"

    t2, p2 = read_message(recv)
    assert t2 == CMD_SHUTDOWN
    assert p2 == b""


def test_frame_metadata_size():
    assert struct.calcsize(FRAME_META_FMT) == FRAME_META_SIZE


def test_read_message_rejects_oversized():
    import pytest

    # Craft a header claiming a payload larger than MAX_MESSAGE_SIZE
    huge_length = MAX_MESSAGE_SIZE + 2  # +1 for type byte, +1 to exceed
    header = struct.pack(HEADER_FMT, huge_length, MSG_FRAME)
    with pytest.raises(ValueError, match="too large"):
        read_message(_make_recv(header))
