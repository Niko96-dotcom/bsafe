"""Integration test: mock client → server → queue."""

import os
import socket
import struct
import tempfile
import time

import pytest

from bsafe.ipc import FrameServer
from bsafe.protocol import CMD_START, FRAME_META_FMT, MSG_FRAME, pack_message, read_message


def _make_frame_payload(display_id=1, width=640, height=480, timestamp_ns=100000, jpeg=b"\xff\xd8"):
    meta = struct.pack(FRAME_META_FMT, display_id, width, height, timestamp_ns)
    return meta + jpeg


@pytest.fixture
def sock_path():
    """Provide a short socket path that fits within AF_UNIX limits."""
    fd, path = tempfile.mkstemp(prefix="bsafe-", suffix=".sock", dir="/tmp")
    os.close(fd)
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_frame_server_receives_frames(sock_path):
    server = FrameServer(sock_path, fps=3, maxsize=5)
    server.start()

    # Connect as mock Swift client
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)

    # Read CMD_START from server
    msg_type, payload = read_message(client.recv)
    assert msg_type == CMD_START
    fps = struct.unpack("!B", payload)[0]
    assert fps == 3

    # Send a frame
    frame_payload = _make_frame_payload()
    client.sendall(pack_message(MSG_FRAME, frame_payload))

    # Wait for queue
    meta, jpeg = server.frame_queue.get(timeout=2)
    assert meta.display_id == 1
    assert meta.width == 640
    assert meta.height == 480
    assert jpeg == b"\xff\xd8"

    client.close()
    server.shutdown()


def test_frame_server_drops_when_full(sock_path):
    server = FrameServer(sock_path, fps=1, maxsize=2)
    server.start()

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)

    # Read CMD_START
    read_message(client.recv)

    # Send more frames than the queue can hold
    frame_payload = _make_frame_payload()
    sent = 8
    for _ in range(sent):
        client.sendall(pack_message(MSG_FRAME, frame_payload))

    # Wait for server thread to process all sent frames
    time.sleep(0.5)

    # Drain the queue and count — should be at most maxsize (2)
    received = 0
    while not server.frame_queue.empty():
        server.frame_queue.get_nowait()
        received += 1

    assert received <= 2
    assert received < sent  # some were dropped

    client.close()
    server.shutdown()


def test_frame_server_cleans_stale_socket(sock_path):
    # Create a stale socket file
    with open(sock_path, "w") as f:
        f.write("")

    server = FrameServer(sock_path, fps=1)
    server.start()

    # Should be able to connect
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)
    read_message(client.recv)

    client.close()
    server.shutdown()
