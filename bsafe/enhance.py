"""Low-light video enhancement: pre-scan, adaptive luma correction, FFmpeg denoise."""

from __future__ import annotations

import bisect
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import cv2
import numpy as np

from bsafe.style import bold, dim, info, warn

# ---------------------------------------------------------------------------
# Denoise defaults (hqdn3d)
# ---------------------------------------------------------------------------
_HQDN3D_LUMA_SPATIAL = 1.5
_HQDN3D_CHROMA_SPATIAL = 1.5
_HQDN3D_LUMA_TEMPORAL = 6
_HQDN3D_CHROMA_TEMPORAL = 6

_DENOISE_TIMEOUT = 3600  # seconds — same as re-encode timeout in video.py

# ---------------------------------------------------------------------------
# Enhancement parameter defaults / thresholds
# ---------------------------------------------------------------------------
_BRIGHT_CUTOFF = 140  # median luma above this → skip enhancement
_FADE_LOW = 100  # median luma below this → full enhancement
_PITCH_BLACK = 20  # median below this → cap gamma (can't rescue)
_SHADOW_THRESHOLD = 50  # pixel value below this counts as shadow
_CLIP_THRESHOLD = 240  # pixel value above this counts as clipped highlight
_NOISY_SHADOW_RATIO = 0.6  # shadow_ratio above this + low median → cap gamma
_NOISY_MEDIAN = 40  # median below this + high shadow_ratio → noisy
_CLIP_RATIO_LIMIT = 0.05  # clip_ratio above this → reduce gamma

_GAMMA_MAX = 1.3
_GAMMA_PITCH_BLACK_CAP = 1.15
_GAMMA_NOISY_CAP = 1.1
_CONTRAST_MAX = 1.08
_SATURATION_MAX = 1.03

_WINDOW_SECONDS = 2  # stat aggregation window
_EMA_ALPHA = 0.3  # temporal smoothing between windows

# Pre-scan: target ~1 sample per second
_SAMPLE_INTERVAL_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class FrameStats(NamedTuple):
    """Luma statistics for a single frame."""

    median: float
    p5: float
    p95: float
    shadow_ratio: float
    clip_ratio: float


class EnhanceParams(NamedTuple):
    """Enhancement parameters computed from frame statistics."""

    gamma: float
    contrast: float
    saturation: float
    skip: bool


@dataclass(frozen=True, slots=True)
class WindowParams:
    """Enhancement parameters for a range of frames."""

    start_frame: int
    end_frame: int  # exclusive
    gamma: float
    contrast: float
    saturation: float
    skip: bool


class WindowIndex:
    """Pre-computed index for O(log n) frame → window lookup.

    Avoids rebuilding the starts list on every ``lookup_params`` call.
    """

    __slots__ = ("_windows", "_starts")

    def __init__(self, windows: list[WindowParams]) -> None:
        self._windows = windows
        self._starts = [wp.start_frame for wp in windows]

    @property
    def all_skip(self) -> bool:
        """True when every window has ``skip=True``."""
        return all(wp.skip for wp in self._windows)

    def lookup(self, frame_index: int) -> tuple[float, float, float]:
        """Return ``(gamma, contrast, saturation)`` for *frame_index*."""
        idx = bisect.bisect_right(self._starts, frame_index) - 1
        if idx < 0:
            return 1.0, 1.0, 1.0
        wp = self._windows[idx]
        if frame_index < wp.end_frame:
            if wp.skip:
                return 1.0, 1.0, 1.0
            return wp.gamma, wp.contrast, wp.saturation
        return 1.0, 1.0, 1.0


# ---------------------------------------------------------------------------
# Pre-scan
# ---------------------------------------------------------------------------
def prescan_video(
    path: str,
    fps: float,
    total_frames: int,
) -> WindowIndex:
    """Sample frames and compute per-window enhancement parameters.

    Returns a ``WindowIndex`` covering all frames. If the video is bright
    enough, every window will have ``skip=True``.
    """
    if total_frames <= 0:
        return WindowIndex([WindowParams(0, max(total_frames, 0), 1.0, 1.0, 1.0, skip=True)])

    sample_every = max(1, round(fps * _SAMPLE_INTERVAL_SECONDS))
    window_size = max(1, round(fps * _WINDOW_SECONDS))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for pre-scan: {path}")

    # Collect per-sample stats.
    # NOTE: CAP_PROP_POS_FRAMES seeking may land on nearby keyframes depending
    # on codec — acceptable here since we only need approximate luma stats.
    samples: list[tuple[int, dict]] = []  # (frame_index, stats)
    try:
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            stats = _frame_stats(frame)
            samples.append((frame_idx, stats))
            frame_idx += sample_every
    finally:
        cap.release()

    if not samples:
        return WindowIndex([WindowParams(0, total_frames, 1.0, 1.0, 1.0, skip=True)])

    # Group samples into windows and compute per-window params
    raw_windows: list[WindowParams] = []
    window_start = 0
    while window_start < total_frames:
        window_end = min(window_start + window_size, total_frames)
        # Samples in this window
        win_samples = [s for fi, s in samples if window_start <= fi < window_end]
        if not win_samples:
            # No samples in window — reuse last computed params or skip
            if raw_windows:
                prev = raw_windows[-1]
                raw_windows.append(
                    WindowParams(
                        window_start,
                        window_end,
                        prev.gamma,
                        prev.contrast,
                        prev.saturation,
                        prev.skip,
                    )
                )
            else:
                raw_windows.append(WindowParams(window_start, window_end, 1.0, 1.0, 1.0, skip=True))
            window_start = window_end
            continue

        avg = _average_stats(win_samples)
        ep = _compute_params(avg.median, avg.p5, avg.p95, avg.shadow_ratio, avg.clip_ratio)
        raw_windows.append(
            WindowParams(window_start, window_end, ep.gamma, ep.contrast, ep.saturation, ep.skip)
        )
        window_start = window_end

    return WindowIndex(_smooth_windows(raw_windows))


def _frame_stats(frame: np.ndarray) -> FrameStats:
    """Compute luma statistics for a single BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return FrameStats(
        median=float(np.median(gray)),
        p5=float(np.percentile(gray, 5)),
        p95=float(np.percentile(gray, 95)),
        shadow_ratio=float(np.mean(gray < _SHADOW_THRESHOLD)),
        clip_ratio=float(np.mean(gray > _CLIP_THRESHOLD)),
    )


def _average_stats(stats_list: list[FrameStats]) -> FrameStats:
    """Average stats across multiple samples."""
    n = len(stats_list)
    return FrameStats(
        median=sum(s.median for s in stats_list) / n,
        p5=sum(s.p5 for s in stats_list) / n,
        p95=sum(s.p95 for s in stats_list) / n,
        shadow_ratio=sum(s.shadow_ratio for s in stats_list) / n,
        clip_ratio=sum(s.clip_ratio for s in stats_list) / n,
    )


def _compute_params(
    median: float,
    p5: float,
    p95: float,
    shadow_ratio: float,
    clip_ratio: float,
) -> EnhanceParams:
    """Determine enhancement parameters from luma statistics."""
    if median > _BRIGHT_CUTOFF:
        return EnhanceParams(1.0, 1.0, 1.0, skip=True)

    # Scale: 0.0 at bright cutoff, 1.0 at fade-low and below
    if median > _FADE_LOW:
        scale = (_BRIGHT_CUTOFF - median) / (_BRIGHT_CUTOFF - _FADE_LOW)
    else:
        scale = 1.0

    gamma = 1.0 + (_GAMMA_MAX - 1.0) * scale
    contrast = 1.0 + (_CONTRAST_MAX - 1.0) * scale
    saturation = 1.0 + (_SATURATION_MAX - 1.0) * scale

    # Highlight protection: reduce gamma when highlights are already clipping
    if clip_ratio > _CLIP_RATIO_LIMIT:
        gamma = max(1.0, gamma * 0.7)

    # Pitch-black protection: don't aggressively rescue nearly-black footage
    if median < _PITCH_BLACK:
        gamma = min(gamma, _GAMMA_PITCH_BLACK_CAP)
        contrast = 1.0  # no contrast boost for pitch-black

    # Noisy-dark protection: avoid lifting noise in very dark, noisy footage
    if shadow_ratio > _NOISY_SHADOW_RATIO and median < _NOISY_MEDIAN:
        gamma = min(gamma, _GAMMA_NOISY_CAP)

    return EnhanceParams(gamma, contrast, saturation, skip=False)


def _smooth_windows(windows: list[WindowParams]) -> list[WindowParams]:
    """Apply EMA smoothing to enhancement parameters across consecutive windows."""
    if len(windows) <= 1:
        return windows

    smoothed: list[WindowParams] = [windows[0]]
    for i in range(1, len(windows)):
        prev = smoothed[-1]
        curr = windows[i]

        if curr.skip and prev.skip:
            smoothed.append(curr)
            continue

        # Smooth numeric params — treat skip windows as (1.0, 1.0, 1.0) for blending
        pg, pc, ps = (
            (prev.gamma, prev.contrast, prev.saturation) if not prev.skip else (1.0, 1.0, 1.0)
        )
        cg, cc, cs = (
            (curr.gamma, curr.contrast, curr.saturation) if not curr.skip else (1.0, 1.0, 1.0)
        )

        sg = _EMA_ALPHA * cg + (1 - _EMA_ALPHA) * pg
        sc = _EMA_ALPHA * cc + (1 - _EMA_ALPHA) * pc
        ss = _EMA_ALPHA * cs + (1 - _EMA_ALPHA) * ps

        # Preserve skip if original window wanted to skip and blended result is near unity
        skip = curr.skip and sg < 1.005 and sc < 1.005 and ss < 1.005

        smoothed.append(WindowParams(curr.start_frame, curr.end_frame, sg, sc, ss, skip))

    return smoothed


# ---------------------------------------------------------------------------
# Per-frame enhancement
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _gamma_lut(milligamma: int) -> np.ndarray:
    """Build a gamma LUT, cached so identical gamma values reuse the table.

    Accepts gamma scaled to integer thousandths (e.g. 1300 for gamma=1.3)
    so the ``@lru_cache`` key is a stable ``int``, not a fragile ``float``.
    """
    gamma = milligamma / 1000.0
    inv_gamma = 1.0 / gamma
    return np.array(
        [min(255, int(((i / 255.0) ** inv_gamma) * 255.0 + 0.5)) for i in range(256)],
        dtype=np.uint8,
    )


def enhance_frame(
    frame: np.ndarray,
    gamma: float,
    contrast: float,
    saturation: float,
) -> None:
    """Apply luma-focused enhancement to a BGR frame in-place."""
    if gamma == 1.0 and contrast == 1.0 and saturation == 1.0:
        return

    # --- Gamma + contrast in YCrCb (luma-focused) ---
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]

    if gamma != 1.0:
        lut = _gamma_lut(round(gamma * 1000))
        y = cv2.LUT(y, lut)

    if contrast != 1.0:
        mean = float(np.mean(y))
        y = np.clip((y.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)

    ycrcb[:, :, 0] = y
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # --- Saturation in HSV ---
    if saturation != 1.0:
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation, 0, 255).astype(
            np.uint8
        )
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    np.copyto(frame, enhanced)


# ---------------------------------------------------------------------------
# FFmpeg temporal denoise
# ---------------------------------------------------------------------------
def denoise_video(input_path: str, output_path: str) -> bool:
    """Run FFmpeg hqdn3d temporal+spatial denoise on *input_path*.

    Returns ``True`` on success, ``False`` if FFmpeg is unavailable or fails.
    The caller should treat failure as non-fatal (enhancement still works
    without denoising, just with more noise in the input).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(
            f"  {warn('Warning:')} ffmpeg not found — skipping temporal denoise",
            file=sys.stderr,
        )
        return False

    vf = (
        f"hqdn3d={_HQDN3D_LUMA_SPATIAL}:{_HQDN3D_CHROMA_SPATIAL}"
        f":{_HQDN3D_LUMA_TEMPORAL}:{_HQDN3D_CHROMA_TEMPORAL}"
    )
    cmd = [
        ffmpeg,
        "-i",
        input_path,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-an",
        "-y",
        output_path,
    ]

    print(f"  Denoising with {bold('hqdn3d')}...", flush=True)
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=_DENOISE_TIMEOUT,
            check=True,
        )
    except subprocess.TimeoutExpired:
        print(
            f"  {warn('Warning:')} denoise timed out after {_DENOISE_TIMEOUT}s — skipping",
            file=sys.stderr,
        )
        return False
    except subprocess.CalledProcessError as exc:
        stderr_tail = (exc.stderr or b"").decode(errors="replace")[-200:]
        print(
            f"  {warn('Warning:')} denoise failed — skipping\n    {dim(stderr_tail)}",
            file=sys.stderr,
        )
        return False

    print(f"  Denoise {info('complete')}", flush=True)
    return True
