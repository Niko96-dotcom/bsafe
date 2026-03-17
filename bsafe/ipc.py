"""Unix domain socket server: receives frames from Swift helper into a queue."""

import logging
import os
import queue
import socket
import threading

from bsafe.protocol import (
    CMD_SHUTDOWN,
    MSG_FRAME,
    pack_cmd_start,
    pack_message,
    parse_frame_payload,
    read_message,
)

logger = logging.getLogger(__name__)


class FrameServer:
    """Binds a Unix socket, accepts one client, sends CMD_START, reads frames into a queue."""

    def __init__(self, socket_path: str, fps: int = 3, maxsize: int = 10):
        self.socket_path = socket_path
        self.fps = fps
        self.frame_queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._sock: socket.socket | None = None
        self._client: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

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
            self._client.sendall(pack_cmd_start(self.fps))

            # Read frames
            recv_fn = self._client.recv
            while not self._stop_event.is_set():
                try:
                    msg_type, payload = read_message(recv_fn)
                except ConnectionError:
                    logger.info("Swift helper disconnected")
                    break

                if msg_type == MSG_FRAME:
                    meta, jpeg_data = parse_frame_payload(payload)
                    try:
                        self.frame_queue.put_nowait((meta, jpeg_data))
                    except queue.Full:
                        logger.warning("Frame queue full, dropping frame")
                else:
                    logger.debug("Ignoring message type 0x%02x", msg_type)

        except TimeoutError:
            logger.error("Timed out waiting for Swift helper to connect")
        except Exception:
            logger.exception("Error in frame server")
        finally:
            self._cleanup_client()

    def shutdown(self):
        """Send CMD_SHUTDOWN and stop the server."""
        self._stop_event.set()
        if self._client:
            try:
                self._client.sendall(pack_message(CMD_SHUTDOWN))
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
