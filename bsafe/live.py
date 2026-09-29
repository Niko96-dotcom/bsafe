"""Live v2 screen censoring loop: raw frames, GPU detection, seq-tagged boxes.

Python grants frame credits (at most one outstanding per display). Swift answers
with the newest raw BGRA frame. Python runs full-frame NudeNet at the frame's
native resolution, replies with boxes tagged with the frame seq, then grants the
next credit. Swift catches boxes up and merges them into its native tracks.
Frames stay in memory only; never written to disk, never logged.
"""

import logging
import math
import queue
import threading
import time

import numpy as np

from bsafe.style import dim, info

logger = logging.getLogger(__name__)

ALLOWED_DETECT_SCALES = (1.0, 1.25, 1.5, 2.0)
STATS_INTERVAL_S = 2.0
_ERROR_LOG_INTERVAL_S = 5.0

AUTO_EXTRA_SCALES: tuple[float, ...] = (0.5,)
MAX_EXTRA_SCALES = 3

STILL_SHIFT_PX = 0.5

# Detections from this confidence up to fastdetect.STRONG_CONFIDENCE only keep
# existing tracks alive (docs/tuning-log.md, keep-only detections).
DEFAULT_KEEP_CONF = 0.1


def parse_extra_scales(raw) -> tuple[float, ...] | None:
    """Parse an --extra-scales value. Returns None for auto (raw is None).

    Accepts a TOML/CLI string (comma-separated floats, "none"/"" disables),
    a single float/int, or a list/tuple of floats. Raises ValueError on
    unparseable input.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return (float(raw),)
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return ()
        try:
            return tuple(float(v) for v in raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"invalid --extra-scales {raw!r}: expected floats") from e
    s = str(raw).strip()
    if s == "" or s.lower() == "none":
        return ()
    parts = [p.strip() for p in s.split(",")]
    if any(p == "" for p in parts):
        raise ValueError(f"invalid --extra-scales {raw!r}: expected comma-separated floats")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"invalid --extra-scales {raw!r}: expected comma-separated floats") from e


STILL_SCALE_MIN = 0.25
STILL_SCALE_MAX = 2.0


def parse_still_scales(raw) -> tuple[float, ...]:
    """Parse a --still-scales value: 1-3 distinct frame-relative factors in [0.25, 2.0].

    ``None`` means the default ``(1.0,)``. Accepts a comma-separated string, a
    single number, or a list/tuple. Raises ValueError on invalid input.
    """
    if raw is None:
        return (1.0,)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        parts = [raw]
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        parts = [p.strip() for p in str(raw).split(",")]
    try:
        scales = tuple(float(p) for p in parts)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid --still-scales {raw!r}: expected comma-separated floats") from e
    if not 1 <= len(scales) <= 3:
        raise ValueError(f"invalid --still-scales {raw!r}: expected 1 to 3 values")
    if len(set(scales)) != len(scales):
        raise ValueError(f"invalid --still-scales {raw!r}: duplicate values")
    for f in scales:
        if not (math.isfinite(f) and STILL_SCALE_MIN <= f <= STILL_SCALE_MAX):
            raise ValueError(
                f"invalid --still-scales {raw!r}: each value must be in "
                f"[{STILL_SCALE_MIN}, {STILL_SCALE_MAX}]"
            )
    return scales


def validate_extra_scales(
    scales: tuple[float, ...] | list[float] | None, detect_scale: float
) -> None:
    """Validate point-size extra scales against --detect-scale. Raises ValueError."""
    if scales is None:
        return
    scales = tuple(scales)
    if len(scales) > MAX_EXTRA_SCALES:
        raise ValueError(f"invalid --extra-scales {list(scales)}: at most 3 values")
    if len(set(scales)) != len(scales):
        raise ValueError(f"invalid --extra-scales {list(scales)}: duplicate values")
    for v in scales:
        if not (0 < v < detect_scale):
            raise ValueError(
                f"invalid --extra-scales {v}: each scale must satisfy 0 < s < --detect-scale {detect_scale}"
            )


def resolve_extra_scales(model, detect_scale: float, raw) -> tuple[float, ...]:
    """Resolve raw --extra-scales (None = auto) to point-size scales. Raises ValueError."""
    from bsafe.detector import get_model_backend

    parsed = parse_extra_scales(raw)
    if parsed is None:
        backend = get_model_backend(model)
        if backend == "erax":
            return ()
        return tuple(AUTO_EXTRA_SCALES)
    backend = get_model_backend(model)
    if backend == "erax":
        # EraX ignores extra scales (CLI warns); skip validation.
        return ()
    validate_extra_scales(parsed, detect_scale)
    return tuple(parsed)


def validate_live_options(model, detect_scale: float, extra_scales=None) -> None:
    """Reject unsupported detect scales and EraX combinations. Raises ValueError."""
    if detect_scale not in ALLOWED_DETECT_SCALES:
        allowed = ", ".join(str(s) for s in ALLOWED_DETECT_SCALES)
        raise ValueError(f"unsupported --detect-scale {detect_scale}. Allowed values: {allowed}")
    from bsafe.detector import get_model_backend

    backend = get_model_backend(model)
    if backend == "erax" and detect_scale != 1.0:
        raise ValueError("--detect-scale is NudeNet-only; use 1.0 with EraX models")
    if extra_scales is None:
        return
    # Parse for every backend so unparseable input fails before spawn;
    # only a successfully parsed value is ignored for EraX (CLI warns).
    parsed = parse_extra_scales(extra_scales)
    if parsed is None:
        return
    if backend == "erax":
        return
    validate_extra_scales(parsed, detect_scale)


class LiveStats:
    """Window + lifetime detection timing stats (no frames/pixels saved)."""

    def __init__(self, time_fn=time.monotonic, interval_s: float = STATS_INTERVAL_S):
        self._time_fn = time_fn
        self._interval = interval_s
        self._start = time_fn()
        self._win_start = self._start
        self._last_log = self._start
        self._win_n = 0
        self._win_detect = 0.0
        self._win_detect_max = 0.0
        self._win_r2s = 0.0
        self._life_n = 0
        self._life_detect = 0.0
        self._life_detect_max = 0.0
        self._life_r2s = 0.0

    def record(self, detect_s: float, receive_to_send_s: float) -> None:
        d = max(0.0, float(detect_s))
        r = max(0.0, float(receive_to_send_s))
        self._win_n += 1
        self._win_detect += d
        self._win_detect_max = max(self._win_detect_max, d)
        self._win_r2s += r
        self._life_n += 1
        self._life_detect += d
        self._life_detect_max = max(self._life_detect_max, d)
        self._life_r2s += r

    def maybe_log(self) -> str | None:
        now = self._time_fn()
        if now - self._last_log < self._interval:
            return None
        self._last_log = now
        line = self._format(now, window=True)
        self._win_n = 0
        self._win_detect = 0.0
        self._win_detect_max = 0.0
        self._win_r2s = 0.0
        self._win_start = now
        return line

    def summary(self) -> str:
        return self._format(self._time_fn(), window=False)

    def _format(self, now: float, window: bool) -> str:
        if window:
            n = self._win_n
            elapsed = now - self._win_start
            sum_d = self._win_detect
            max_d = self._win_detect_max
            sum_r = self._win_r2s
            scope = "window"
        else:
            n = self._life_n
            elapsed = now - self._start
            sum_d = self._life_detect
            max_d = self._life_detect_max
            sum_r = self._life_r2s
            scope = "total"
        rate = (n / elapsed) if elapsed > 0 and n else 0.0
        avg_detect_ms = (sum_d / n * 1000.0) if n else 0.0
        max_detect_ms = (max_d * 1000.0) if n else 0.0
        avg_r2s_ms = (sum_r / n * 1000.0) if n else 0.0
        return (
            f"live stats ({scope}): frames={n} rate={rate:.1f}/s "
            f"avg_detect_ms={avg_detect_ms:.1f} max_detect_ms={max_detect_ms:.1f} "
            f"avg_receive_to_send_ms={avg_r2s_ms:.1f}"
        )


def bgra_view(meta, pixels) -> np.ndarray:
    """Zero-copy BGRA view over raw frame pixels as HxWx4 uint8."""
    return np.frombuffer(pixels, dtype=np.uint8).reshape(meta.height, meta.width, 4)


class StillnessTracker:
    """Per-display stillness from phase-correlation shift (metadata only).

    Downscales each BGRA frame to 1/4 size grayscale and measures the shift
    against the previous frame with cv2.phaseCorrelate. A frame is still when
    the full-resolution shift magnitude is below STILL_SHIFT_PX.
    """

    def __init__(self):
        self._state: dict = {}

    def reset(self, display_id) -> None:
        """Drop stillness state for one display (e.g. on display info)."""
        self._state.pop(display_id, None)

    def update(self, display_id, frame_bgra, now) -> float | None:
        """Record a frame; return seconds still, or None when not still."""
        import math

        import cv2

        gray = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2GRAY)
        curr = cv2.resize(gray, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        h, w = curr.shape
        entry = self._state.get(display_id)
        if entry is None or entry["prev"].shape != curr.shape:
            self._state[display_id] = {
                "prev": curr,
                "window": cv2.createHanningWindow((w, h), cv2.CV_32F),
                "still_since": None,
            }
            return None
        # phaseCorrelate can modify its inputs in place (seen with small frames),
        # so correlate against a copy and keep the pristine curr as prev.
        (dx, dy), _ = cv2.phaseCorrelate(entry["prev"], curr.copy(), entry["window"])
        magnitude = math.hypot(dx, dy) * 4
        entry["prev"] = curr
        if magnitude < STILL_SHIFT_PX:
            if entry["still_since"] is None:
                entry["still_since"] = now
            return now - entry["still_since"]
        entry["still_since"] = None
        return None


class StillWorker:
    """Runs still-pass detections off the light-detection thread.

    One job at a time: ``submit`` returns False while a job is running. The
    finished result is picked up with ``take``. The detector is only touched
    under ``lock`` so ``prepare`` from the main thread cannot race a run.
    """

    def __init__(self, detector):
        self._detector = detector
        self.lock = threading.Lock()
        self._cv = threading.Condition()
        self._job = None
        self._result = None
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="bsafe-still", daemon=True)
        self._thread.start()

    def submit(self, job) -> bool:
        """Queue ``(display_id, seq, frame, scale)``; False while busy."""
        with self._cv:
            if self._busy or self._stop:
                return False
            self._busy = True
            self._job = job
            self._cv.notify()
            return True

    def take(self):
        """Return ``(display_id, seq, detections)`` once, or None."""
        with self._cv:
            result, self._result = self._result, None
            return result

    def close(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout)

    def _run(self) -> None:
        from bsafe.fastdetect import detect_at_scale

        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                display_id, seq, frame, scale = self._job
                self._job = None
            try:
                with self.lock:
                    dets = list(detect_at_scale(self._detector, frame, scale))
            except Exception:
                logger.exception("Live still detection failed")
                dets = []
            with self._cv:
                self._result = (display_id, seq, dets)
                self._busy = False


class LiveSession:
    """Testable heart of cmd_start: credits, detection, seq-tagged boxes."""

    def __init__(
        self,
        server,
        detector,
        *,
        censor_classes,
        padding: float,
        full_censor: bool,
        is_nudenet: bool,
        stats: LiveStats | None = None,
        print_fn=print,
        time_fn=time.monotonic,
        min_padding: int = 0,
        extra_factors: tuple[float, ...] = (),
        still_detector=None,
        still_period_s: float = 2.0,
        still_after_s: float = 0.5,
        still_scales: tuple[float, ...] = (1.0,),
        still_async: bool = False,
        strong_confidence: float | None = None,
    ):
        self._server = server
        self._detector = detector
        self._censor_classes = censor_classes
        self._padding = padding
        self._min_padding = min_padding
        self._full_censor = full_censor
        self._is_nudenet = is_nudenet
        self._stats = stats
        self._print_fn = print_fn
        self._time_fn = time_fn
        self._last_exc_log = float("-inf")
        self._extra_factors = tuple(extra_factors)
        self._still_detector = still_detector
        self._still_period_s = still_period_s
        self._still_after_s = still_after_s
        self._still_scales = tuple(still_scales) or (1.0,)
        self._still_runs: dict = {}
        self._still_tracker = StillnessTracker() if still_detector is not None else None
        self._last_still_run: dict = {}
        # Async still pass: the heavy model runs on its own thread so it never
        # stalls the light loop. Its boxes are re-sent for the last replied
        # frame of that display while the screen is still still.
        self._still_worker = (
            StillWorker(still_detector) if still_detector is not None and still_async else None
        )
        # display_id -> (seq, width, height, light detections, seconds_still)
        self._last_reply: dict = {}
        # With a keep floor, detections below this only keep tracks alive.
        self._strong_confidence = strong_confidence

    def close(self) -> None:
        """Stop the async still worker (if any)."""
        if self._still_worker is not None:
            self._still_worker.close()

    def _build_boxes(self, detections, width, height):
        """Return (censor boxes, keep-only boxes) for one frame."""
        from bsafe.censor import FULL_CENSOR_MULTIPLIER, build_censor_box_sets, expand_boxes

        boxes, keep = build_censor_box_sets(
            detections,
            self._censor_classes,
            self._padding,
            width,
            height,
            self._min_padding,
            self._strong_confidence,
        )
        if self._full_censor:
            boxes = expand_boxes(boxes, FULL_CENSOR_MULTIPLIER, width, height)
            keep = expand_boxes(keep, FULL_CENSOR_MULTIPLIER, width, height)
        return boxes, keep

    def flush_still_results(self) -> bool:
        """Send finished async still-pass boxes; True when something was sent.

        The result is merged with the light detections of the display's last
        replied frame and re-sent for that seq, but only while that frame is
        the one the still pass ran on or the screen is still still (otherwise
        the boxes would land on moved content).
        """
        if self._still_worker is None:
            return False
        result = self._still_worker.take()
        if result is None:
            return False
        display_id, seq, still_dets = result
        last = self._last_reply.get(display_id)
        if not still_dets or last is None:
            return False
        last_seq, width, height, light_dets, seconds_still = last
        if last_seq != seq and seconds_still is None:
            return False
        boxes, keep = self._build_boxes(list(light_dets) + still_dets, width, height)
        self._server.send_censor_seq(display_id, width, height, last_seq, boxes, keep)
        return True

    def _emit(self, text: str) -> None:
        try:
            self._print_fn(text, flush=True)
        except TypeError:
            self._print_fn(text)

    def handle_display_info(self, display_info) -> None:
        cw = display_info.capture_width
        ch = display_info.capture_height
        pw = display_info.points_width
        ph = display_info.points_height
        did = display_info.display_id
        self._emit(
            f"{info(f'Display {did}')}: capturing {info(f'{cw}x{ch}')} px {dim(f'({pw}x{ph} pt)')}"
        )
        if hasattr(self._detector, "prepare"):
            try:
                t0 = self._time_fn()
                from bsafe.fastdetect import prepare_multiscale

                prepare_multiscale(self._detector, cw, ch, self._extra_factors)
                if self._still_detector is not None and hasattr(self._still_detector, "prepare"):
                    from bsafe.fastdetect import extra_size

                    lock = (
                        self._still_worker.lock
                        if self._still_worker is not None
                        else threading.Lock()
                    )
                    with lock:
                        for f in self._still_scales:
                            self._still_detector.prepare(*extra_size(cw, ch, f))
                elapsed_ms = (self._time_fn() - t0) * 1000.0
                provider = getattr(self._detector, "provider", "cpu")
                self._emit(
                    f"Detector ready for {info(f'{cw}x{ch}')} "
                    f"({provider}, {info(f'{elapsed_ms:.1f} ms')})"
                )
            except Exception:
                now = self._time_fn()
                if now - self._last_exc_log >= _ERROR_LOG_INTERVAL_S:
                    logger.exception("Live detector prepare failed")
                    self._last_exc_log = now
        if self._still_tracker is not None:
            self._still_tracker.reset(did)
        self._server.request_frame(did)

    def drain_display_events(self) -> int:
        count = 0
        while True:
            try:
                display_info = self._server.display_events.get_nowait()
            except queue.Empty:
                break
            self.handle_display_info(display_info)
            count += 1
        return count

    def step(self, timeout: float = 0.1) -> bool:
        self.drain_display_events()
        self.flush_still_results()
        try:
            meta, pixels, receipt_mono = self._server.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return False
        detect_start = self._time_fn()
        detect_s = 0.0
        seconds_still = None
        try:
            frame = bgra_view(meta, pixels)
            if self._is_nudenet:
                if self._extra_factors:
                    from bsafe.fastdetect import detect_multiscale

                    detections = detect_multiscale(self._detector, frame, self._extra_factors)
                else:
                    detections = self._detector.detect_bgra(frame)
            else:
                import cv2

                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                detections = self._detector.detect_frame(bgr)
            if (
                self._still_detector is not None
                and self._is_nudenet
                and self._still_tracker is not None
            ):
                now = self._time_fn()
                attempted_still = False
                try:
                    seconds_still = self._still_tracker.update(meta.display_id, frame, now)
                    if seconds_still is not None and seconds_still >= self._still_after_s:
                        last_run = self._last_still_run.get(meta.display_id)
                        if last_run is None or now - last_run >= self._still_period_s:
                            run = self._still_runs.get(meta.display_id, 0)
                            scale = self._still_scales[run % len(self._still_scales)]
                            if self._still_worker is not None:
                                # Copy: the frame buffer belongs to the reader.
                                job = (meta.display_id, meta.seq, np.array(frame), scale)
                                if self._still_worker.submit(job):
                                    self._still_runs[meta.display_id] = run + 1
                                    self._last_still_run[meta.display_id] = now
                            else:
                                attempted_still = True
                                from bsafe.fastdetect import detect_at_scale

                                self._still_runs[meta.display_id] = run + 1
                                still_dets = detect_at_scale(self._still_detector, frame, scale)
                                if still_dets:
                                    detections = list(detections) + list(still_dets)
                                self._last_still_run[meta.display_id] = now
                except Exception:
                    now_exc = self._time_fn()
                    if now_exc - self._last_exc_log >= _ERROR_LOG_INTERVAL_S:
                        logger.exception("Live still detection failed")
                        self._last_exc_log = now_exc
                    if not attempted_still:
                        self._still_tracker.reset(meta.display_id)
                    self._last_still_run[meta.display_id] = now
            detect_s = self._time_fn() - detect_start
            logger.debug("Processed frame: %d detection(s)", len(detections))
            if detections:
                for d in detections:
                    logger.debug(
                        "[detection] %s (%.2f) at %s on display %d",
                        d.class_name,
                        d.confidence,
                        d.box,
                        meta.display_id,
                    )
            boxes, keep = self._build_boxes(detections, meta.width, meta.height)
            if self._still_worker is not None:
                self._last_reply[meta.display_id] = (
                    meta.seq,
                    meta.width,
                    meta.height,
                    list(detections),
                    seconds_still,
                )
        except Exception:
            now = self._time_fn()
            if now - self._last_exc_log >= _ERROR_LOG_INTERVAL_S:
                logger.exception("Live detection failed")
                self._last_exc_log = now
            detect_s = now - detect_start
            # Fail closed: an empty reply would count as a missed pass and let
            # tracks expire. Send nothing; Swift keeps the current boxes and
            # pixelates the whole display if detection stays down.
            boxes = None
        if boxes is not None:
            self._server.send_censor_seq(
                meta.display_id, meta.width, meta.height, meta.seq, boxes, keep
            )
        self._server.request_frame(meta.display_id)
        if self._stats is not None:
            send_mono = self._time_fn()
            self._stats.record(detect_s, send_mono - receipt_mono)
            line = self._stats.maybe_log()
            if line is not None:
                self._emit(dim(line))
        return True
