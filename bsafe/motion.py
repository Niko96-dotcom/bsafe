"""Isolated motion compensation: map detector boxes to the newest frame.

Translation-only per-box Lucas-Kanade shift. Stateless; falls back to
clamped input boxes when motion is unmeasurable.
"""

import cv2
import numpy as np

_MAX_DIM = 960
_MAX_CORNERS_PER_BOX = 40
_MIN_INLIERS = 3
_MIN_INLIER_RATIO = 0.4
_FB_MAX_PX = 2.0
_SPREAD_MAX_PX = 3.0
_MAX_SHIFT_FRAC = 0.5
_PAD_PX = 2


def _decode_gray(jpeg: bytes) -> np.ndarray | None:
    try:
        if not isinstance(jpeg, (bytes, bytearray)) or len(jpeg) == 0:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None or img.ndim != 2:
            return None
        return img
    except Exception:
        return None


def _clamp_box(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int] | None:
    try:
        x, y, bw, bh = (int(v) for v in box)
    except Exception:
        return None
    if bw <= 0 or bh <= 0:
        return None
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    nw, nh = x2 - x1, y2 - y1
    if nw <= 0 or nh <= 0:
        return None
    return (x1, y1, nw, nh)


def _compensate_one(
    ref_s: np.ndarray,
    cur_s: np.ndarray,
    box: tuple[int, int, int, int],
    scale: float,
    w: int,
    h: int,
) -> tuple[int, int, int, int] | None:
    """Translate one box or return its clamped fallback (None if invalid)."""
    fallback = _clamp_box(box, w, h)
    if fallback is None:
        return None
    try:
        x, y, bw, bh = (int(v) for v in box)
        sh_w, sh_h = ref_s.shape[1], ref_s.shape[0]
        sx0, sy0 = int(round(x * scale)), int(round(y * scale))
        sw, sh = int(round(bw * scale)), int(round(bh * scale))
        if sw < 8 or sh < 8:
            return fallback
        ex0 = max(0, sx0 - _PAD_PX)
        ey0 = max(0, sy0 - _PAD_PX)
        ex1 = min(sh_w, sx0 + sw + _PAD_PX)
        ey1 = min(sh_h, sy0 + sh + _PAD_PX)
        if ex1 - ex0 < 8 or ey1 - ey0 < 8:
            return fallback
        roi = ref_s[ey0:ey1, ex0:ex1]
        pts = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=_MAX_CORNERS_PER_BOX,
            qualityLevel=0.01,
            minDistance=4,
            blockSize=7,
            useHarrisDetector=False,
        )
        if pts is None or len(pts) < _MIN_INLIERS:
            return fallback
        pts_ref = pts.astype(np.float32).copy()
        pts_ref[:, 0, 0] += ex0
        pts_ref[:, 0, 1] += ey0
        px, py = pts_ref[:, 0, 0], pts_ref[:, 0, 1]
        inside = ((px >= sx0) & (px < sx0 + sw) & (py >= sy0) & (py < sy0 + sh)).ravel()
        # Keep edge points: allow border-touching features found via padding.
        if int(np.count_nonzero(inside)) >= _MIN_INLIERS:
            pts_ref = pts_ref[inside]
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        fwd, st_f, _ = cv2.calcOpticalFlowPyrLK(
            ref_s, cur_s, pts_ref, None, winSize=(21, 21), maxLevel=3, criteria=criteria
        )
        if fwd is None or st_f is None:
            return fallback
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(
            cur_s, ref_s, fwd, None, winSize=(21, 21), maxLevel=3, criteria=criteria
        )
        if back is None or st_b is None:
            return fallback
        fb_err = np.linalg.norm((pts_ref - back).reshape(-1, 2), axis=1)
        keep = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb_err <= _FB_MAX_PX)
        n_in = int(np.count_nonzero(keep))
        if n_in < _MIN_INLIERS or n_in / len(pts_ref) < _MIN_INLIER_RATIO:
            return fallback
        disp = (fwd[keep] - pts_ref[keep]).reshape(-1, 2).astype(np.float64)
        med = np.median(disp, axis=0)
        spread = np.std(disp, axis=0)
        if not np.all(np.isfinite(med)) or not np.all(np.isfinite(spread)):
            return fallback
        if float(np.max(spread)) > _SPREAD_MAX_PX:
            return fallback
        if float(np.hypot(med[0], med[1])) > _MAX_SHIFT_FRAC * min(sh_w, sh_h):
            return fallback
        dx, dy = float(med[0]) / scale, float(med[1]) / scale
        if not np.isfinite(dx) or not np.isfinite(dy):
            return fallback
        moved = (int(round(x + dx)), int(round(y + dy)), int(bw), int(bh))
        clipped = _clamp_box(moved, w, h)
        return clipped if clipped is not None else fallback
    except Exception:
        return fallback


def compensate_boxes(
    reference_jpeg: bytes,
    current_jpeg: bytes,
    boxes: list[tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int, int, int]]:
    """Map reference ``(x, y, w, h)`` boxes onto the current frame.

    Translation only (extent conserved, clipped at edges). Invalid boxes
    are dropped; decode/size/tracking failure returns clamped inputs.
    """
    items = list(boxes) if boxes else []
    if not items:
        return []
    try:
        w, h = int(frame_width), int(frame_height)
    except Exception:
        return []
    if w <= 0 or h <= 0:
        return []
    safe = [(b, _clamp_box(b, w, h)) for b in items]

    def _fallback() -> list[tuple[int, int, int, int]]:
        return [c for _, c in safe if c is not None]

    ref = _decode_gray(reference_jpeg)
    cur = _decode_gray(current_jpeg)
    if ref is None or cur is None or ref.shape != cur.shape:
        return _fallback()
    ih, iw = ref.shape[:2]
    if iw != w or ih != h:
        return _fallback()
    try:
        scale = min(1.0, _MAX_DIM / float(max(iw, ih)))
        if scale < 1.0:
            nw = max(1, int(round(iw * scale)))
            nh = max(1, int(round(ih * scale)))
            ref_s = cv2.resize(ref, (nw, nh), interpolation=cv2.INTER_AREA)
            cur_s = cv2.resize(cur, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            ref_s, cur_s = ref, cur
        out: list[tuple[int, int, int, int]] = []
        for b, _ in safe:
            moved = _compensate_one(ref_s, cur_s, b, scale, w, h)
            if moved is not None:
                out.append(moved)
        return out
    except Exception:
        return _fallback()
