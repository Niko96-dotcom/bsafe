"""Live responsiveness helpers: staleness, stats, and detail-scan tiling.

Screen overlays react after content appears; nothing here guarantees
prevention. Timing values are local receipt-through-processing budgets
(``time.monotonic`` on receipt to send), not capture-to-render latency.
"""

import logging
import time

from bsafe.detector import DEFAULT_INFERENCE_RESOLUTION, Detection

logger = logging.getLogger(__name__)

DEFAULT_MAX_FRAME_AGE_MS = 250
STATS_INTERVAL_S = 2.0
DETAIL_TILE_OVERLAP_FRAC = 0.25
DETAIL_DEDUPE_IOU = 0.5


def is_frame_stale(receipt_mono: float, now_mono: float, max_age_ms: int) -> bool:
    """Return True when receipt-through-processing budget is exceeded."""
    if max_age_ms is None or max_age_ms <= 0:
        return False
    return (now_mono - receipt_mono) * 1000.0 > max_age_ms


def validate_live_options(model, inference_resolution: int, detail_scan: bool) -> None:
    """Reject unsupported EraX flag combinations. Raises ValueError."""
    from bsafe.detector import get_model_backend

    backend = get_model_backend(model)
    if backend == "erax":
        if detail_scan:
            raise ValueError("--detail-scan is NudeNet-only and has no effect with EraX")
        if inference_resolution != DEFAULT_INFERENCE_RESOLUTION:
            raise ValueError("--inference-resolution is NudeNet-only and has no effect with EraX")


def compute_tile_rects(
    frame_w: int, frame_h: int, overlap_frac: float = DETAIL_TILE_OVERLAP_FRAC
) -> list[tuple[int, int, int, int]]:
    """Return overlapping 2x2 tile rects as ``(x0, y0, w, h)``.

    Tiles cover the full frame; the middle seam overlaps so objects on the
    split are fully visible in at least one tile. Small frames collapse
    duplicate origins, yielding fewer than 4 tiles.
    """
    if frame_w <= 0 or frame_h <= 0:
        return []
    overlap_x = int(frame_w * overlap_frac)
    overlap_y = int(frame_h * overlap_frac)
    tile_w = max(1, min(frame_w, (frame_w + overlap_x + 1) // 2))
    tile_h = max(1, min(frame_h, (frame_h + overlap_y + 1) // 2))
    xs = sorted({0, frame_w - tile_w})
    ys = sorted({0, frame_h - tile_h})
    return [(x0, y0, tile_w, tile_h) for y0 in ys for x0 in xs]


def remap_box(
    tile_box: tuple[int, int, int, int],
    tile_origin: tuple[int, int],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int] | None:
    """Translate a tile-local ``(x, y, w, h)`` to full-frame coords, clamped."""
    x, y, w, h = tile_box
    x0, y0 = tile_origin
    x1 = max(0, x + x0)
    y1 = max(0, y + y0)
    x2 = min(frame_w, x + x0 + w)
    y2 = min(frame_h, y + y0 + h)
    nw, nh = x2 - x1, y2 - y1
    if nw <= 0 or nh <= 0:
        return None
    return (x1, y1, nw, nh)


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def dedupe_detections(
    detections: list[Detection], iou_threshold: float = DETAIL_DEDUPE_IOU
) -> list[Detection]:
    """Class-aware NMS: keep highest confidence per overlapping same-class group."""
    kept: list[Detection] = []
    by_class: dict[str, list[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.class_name, []).append(d)
    for items in by_class.values():
        ordered = sorted(items, key=lambda d: d.confidence, reverse=True)
        suppressed = [False] * len(ordered)
        for i, cand in enumerate(ordered):
            if suppressed[i]:
                continue
            kept.append(cand)
            for j in range(i + 1, len(ordered)):
                if suppressed[j]:
                    continue
                if _box_iou(cand.box, ordered[j].box) > iou_threshold:
                    suppressed[j] = True
    return kept


class LiveStats:
    """Bounded aggregate timing stats (no frames/images saved).

    Lifetime counters (``total``/``stale``) and lifetime sums are cumulative
    and never reset, so :meth:`summary` stays accurate after interval flushes.
    Window counters/sums cover only drawn frames since the last interval log.
    Averages are drawn-only (stale frames excluded) receive-to-send-start ages
    (local monotonic receipt to send start), not capture-to-render latency.
    """

    def __init__(self, time_fn=time.monotonic, interval_s: float = STATS_INTERVAL_S):
        self._time_fn = time_fn
        self._interval = interval_s
        self._last_log = time_fn()
        self.total = 0
        self.stale = 0
        self._win_n = 0
        self._win_queue = 0.0
        self._win_inf = 0.0
        self._win_total = 0.0
        self._life_n = 0
        self._life_queue = 0.0
        self._life_inf = 0.0
        self._life_total = 0.0
        self._last_replaced = 0

    def record(self, queue_age_s: float, inference_age_s: float, total_age_s: float) -> None:
        self.total += 1
        self._win_n += 1
        self._win_queue += max(0.0, queue_age_s)
        self._win_inf += max(0.0, inference_age_s)
        self._win_total += max(0.0, total_age_s)
        self._life_n += 1
        self._life_queue += max(0.0, queue_age_s)
        self._life_inf += max(0.0, inference_age_s)
        self._life_total += max(0.0, total_age_s)

    def record_stale(self) -> None:
        self.total += 1
        self.stale += 1

    def maybe_log(self, replaced: int) -> str | None:
        now = self._time_fn()
        if now - self._last_log < self._interval:
            return None
        self._last_log = now
        return self._format(replaced, window=True)

    def summary(self, replaced: int) -> str:
        return self._format(replaced, window=False)

    def _format(self, replaced: int, window: bool) -> str:
        if window:
            n = self._win_n
            sum_q, sum_i, sum_t = self._win_queue, self._win_inf, self._win_total
            replaced_val = max(0, replaced - self._last_replaced)
            scope = "window"
        else:
            n = self._life_n
            sum_q, sum_i, sum_t = self._life_queue, self._life_inf, self._life_total
            replaced_val = replaced
            scope = "total"
        if n:
            avg_q = sum_q / n * 1000.0
            avg_i = sum_i / n * 1000.0
            avg_t = sum_t / n * 1000.0
        else:
            avg_q = avg_i = avg_t = 0.0
        line = (
            f"live stats ({scope}; receive-to-send-start, drawn-only, "
            f"not capture-to-render): "
            f"total_n={self.total} total_stale={self.stale} drawn_n={n} replaced={replaced_val} "
            f"avg_receive_to_send_ms={avg_t:.1f} "
            f"avg_queue_ms={avg_q:.1f} avg_inference_ms={avg_i:.1f}"
        )
        if window:
            self._win_n = 0
            self._win_queue = 0.0
            self._win_inf = 0.0
            self._win_total = 0.0
            self._last_replaced = replaced
        return line


class DetailScanDetector:
    """Live-only wrapper: full frame plus overlapping 2x2 tiles (NudeNet only).

    Default (``enabled=False``) delegates to a single pass so file-processing
    paths stay unchanged. No result cache is kept across frames.
    """

    def __init__(
        self,
        detector,
        enabled: bool = False,
        overlap_frac: float = DETAIL_TILE_OVERLAP_FRAC,
        iou_threshold: float = DETAIL_DEDUPE_IOU,
    ):
        self._detector = detector
        self.enabled = enabled
        self.overlap_frac = overlap_frac
        self.iou_threshold = iou_threshold

    def close(self) -> None:
        self._detector.close()

    def detect(self, jpeg_bytes: bytes) -> list[Detection]:
        if not self.enabled:
            return self._detector.detect(jpeg_bytes)
        import cv2
        import numpy as np

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return self._detector.detect(jpeg_bytes)
        return self.detect_frame(frame)

    def detect_frame(self, frame) -> list[Detection]:
        if not self.enabled:
            return self._detector.detect_frame(frame)
        frame_h, frame_w = frame.shape[:2]
        merged: list[Detection] = list(self._detector.detect_frame(frame))
        for x0, y0, tw, th in compute_tile_rects(frame_w, frame_h, self.overlap_frac):
            tile = frame[y0 : y0 + th, x0 : x0 + tw]
            if tile.size == 0:
                continue
            for d in self._detector.detect_frame(tile):
                remapped = remap_box(d.box, (x0, y0), frame_w, frame_h)
                if remapped is None:
                    continue
                merged.append(
                    Detection(class_name=d.class_name, confidence=d.confidence, box=remapped)
                )
        return dedupe_detections(merged, self.iou_threshold)
