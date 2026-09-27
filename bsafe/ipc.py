"""Unix domain socket server: receives frames from Swift helper into a mailbox."""

import logging
import os
import queue
import socket
import threading
import time
from collections import deque

from bsafe.protocol import (
    CMD_SHUTDOWN,
    MSG_FRAME,
    pack_cmd_censor,
    pack_cmd_censor_style,
    pack_cmd_start,
    pack_message,
    parse_frame_payload,
    read_message,
)

logger = logging.getLogger(__name__)


class FrameMailbox:
    """Bounded latest-frame per-display mailbox.

    Keeps at most one pending frame per display. A busy display replaces its
    own pending frame instead of queuing behind (or ahead of) quiet displays,
    so a busy monitor cannot starve a quiet one. Consumers take pending
    displays in first-pending order (fair round-robin across displays).

    Compatible subset of :class:`queue.Queue` used by live code:
    ``get(block=True, timeout=None)``, ``get_nowait()``, ``put_nowait()``,
    ``empty()``, ``qsize()``. ``get`` returns ``(meta, jpeg, receipt_mono)``
    where ``receipt_mono`` is a local ``time.monotonic()`` timestamp taken on
    receipt (never the cross-process Swift clock).
    """

    def __init__(self, maxsize: int = 10):
        self.maxsize = maxsize
        self._latest: dict[int, tuple] = {}
        self._order: deque[int] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        # Counters for --stats (no per-frame warnings).
        self.replaced = 0
        self.dropped_full = 0

    def _store(self, meta, jpeg_data: bytes, receipt_mono: float) -> None:
        display_id = meta.display_id
        with self._not_empty:
            if display_id in self._latest:
                self._latest[display_id] = (meta, jpeg_data, receipt_mono)
                self.replaced += 1
                logger.debug("Frame replaced: display=%d (latest-frame mailbox)", display_id)
                return
            if self.maxsize > 0 and len(self._latest) >= self.maxsize:
                self.dropped_full += 1
                logger.debug(
                    "Frame dropped: display=%d mailbox at capacity %d",
                    display_id,
                    self.maxsize,
                )
                return
            self._latest[display_id] = (meta, jpeg_data, receipt_mono)
            self._order.append(display_id)
            self._not_empty.notify()

    def put_nowait(self, item) -> None:
        """Store ``(meta, jpeg)`` (or ``(meta, jpeg, receipt)``), replacing."""
        meta, jpeg_data = item[0], item[1]
        if len(item) > 2:
            receipt_mono = float(item[2])
        else:
            receipt_mono = time.monotonic()
        self._store(meta, jpeg_data, receipt_mono)

    def put(self, item, block: bool = True, timeout=None) -> None:
        """Non-blocking store; never blocks (drops only on display cap)."""
        self.put_nowait(item)

    def get(self, block: bool = True, timeout=None):
        """Return next ``(meta, jpeg, receipt_mono)`` in fair display order."""
        with self._not_empty:
            if not block:
                if not self._latest:
                    raise queue.Empty
            elif timeout is None:
                while not self._latest:
                    self._not_empty.wait()
            else:
                end = time.monotonic() + timeout
                while not self._latest:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self._not_empty.wait(remaining)
            while self._order:
                display_id = self._order.popleft()
                item = self._latest.pop(display_id, None)
                if item is not None:
                    return item
            raise queue.Empty

    def get_nowait(self):
        return self.get(block=False)

    def peek(self, display_id: int):
        """Return latest pending ``(meta, jpeg, receipt_mono)`` without consuming.

        Returns None when no frame is pending for the display. Never reorders,
        removes, or counts (no replaced/dropped accounting).
        """
        with self._lock:
            return self._latest.get(display_id)

    def empty(self) -> bool:
        with self._lock:
            return not self._latest

    def qsize(self) -> int:
        with self._lock:
            return len(self._latest)

    def __len__(self) -> int:
        return self.qsize()


class FrameServer:
    """Binds a Unix socket, accepts one client, sends CMD_START, reads frames into a queue."""

    def __init__(
        self,
        socket_path: str,
        fps: int = 45,
        maxsize: int = 10,
        blur: float = 0.0,
        pixels: float = 0.0,
        censor_text: str | None = None,
    ):
        self.socket_path = socket_path
        self.fps = fps
        self.blur = blur
        self.pixels = pixels
        self.censor_text = censor_text
        self.frame_queue: FrameMailbox = FrameMailbox(maxsize=maxsize)
        self._sock: socket.socket | None = None
        self._client: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()

    def start(self):
        """Bind socket and start the reader thread."""
        # Remove stale socket file
        if os.path.exists(self.socket_path):
            logger.warning("Removing stale socket: %s", self.socket_path)
            os.unlink(self.socket_path)

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.socket_path)
        self._sock.listen(1)
        self._sock.settimeout(30.0)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Accept a client and read frames until stopped."""
        try:
            logger.info("Waiting for Swift helper to connect...")
            self._client, _ = self._sock.accept()
            logger.info("Swift helper connected")

            # Send CMD_START with desired FPS
            self._send_to_client(pack_cmd_start(self.fps))

            # Send censor style config if blur, pixels, or text is set
            if self.blur > 0.0 or self.pixels > 0.0 or self.censor_text is not None:
                self._send_to_client(
                    pack_cmd_censor_style(self.blur, self.pixels, self.censor_text)
                )

            # Read frames
            recv_fn = self._client.recv
            while not self._stop_event.is_set():
                try:
                    msg_type, payload = read_message(recv_fn)
                except ConnectionError, OSError:
                    logger.info("Swift helper disconnected")
                    break

                if msg_type == MSG_FRAME:
                    meta, jpeg_data = parse_frame_payload(payload)
                    logger.debug(
                        "Frame received: display=%d %dx%d (%d bytes JPEG)",
                        meta.display_id,
                        meta.width,
                        meta.height,
                        len(jpeg_data),
                    )
                    self.frame_queue.put_nowait((meta, jpeg_data))
                else:
                    logger.debug("Ignoring message type 0x%02x", msg_type)

        except TimeoutError:
            logger.error("Timed out waiting for Swift helper to connect")
        except Exception:
            logger.exception("Error in frame server")
        finally:
            self._cleanup_client()

    def _send_to_client(self, data: bytes):
        """Thread-safe send to the connected client."""
        with self._write_lock:
            if self._client:
                self._client.sendall(data)

    def send_censor(
        self,
        display_id: int,
        frame_width: int,
        frame_height: int,
        boxes: list[tuple[int, int, int, int]],
    ):
        """Send CMD_CENSOR with overlay box coordinates to Swift."""
        self._send_to_client(pack_cmd_censor(display_id, frame_width, frame_height, boxes))

    def shutdown(self):
        """Send CMD_SHUTDOWN and stop the server."""
        self._stop_event.set()
        try:
            self._send_to_client(pack_message(CMD_SHUTDOWN))
        except OSError:
            pass
        self._cleanup_client()
        self._cleanup_socket()
        if self._thread:
            self._thread.join(timeout=3)

    def _cleanup_client(self):
        if self._client:
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None

    def _cleanup_socket(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
