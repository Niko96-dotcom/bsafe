"""Python-side censor rendering on numpy BGR frames."""

import cv2
import numpy as np


def apply_black_boxes(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    """Fill regions with black."""
    for x, y, w, h in boxes:
        frame[y : y + h, x : x + w] = 0


def apply_blur(frame: np.ndarray, boxes: list[tuple[int, int, int, int]], intensity: float) -> None:
    """Gaussian blur on each ROI. Kernel size proportional to box size * intensity."""
    for x, y, w, h in boxes:
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        # Kernel size based on the larger dimension, scaled by intensity
        k = int(max(w, h) * 0.3 * intensity)
        k = max(3, k | 1)  # ensure odd and >= 3
        frame[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (k, k), 0)


def apply_pixelation(
    frame: np.ndarray, boxes: list[tuple[int, int, int, int]], intensity: float
) -> None:
    """Downscale then upscale each ROI for a pixelation effect."""
    for x, y, w, h in boxes:
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        # Pixel block size proportional to the larger dimension * intensity
        block = int(max(w, h) * 0.05 * intensity)
        block = max(2, block)
        small_w = max(1, w // block)
        small_h = max(1, h // block)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        frame[y : y + h, x : x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_text(frame: np.ndarray, boxes: list[tuple[int, int, int, int]], text: str) -> None:
    """Draw white text centered in each box."""
    for x, y, w, h in boxes:
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Scale font to fit roughly within the box
        scale = min(w, h) / 100.0
        scale = max(0.3, scale)
        thickness = max(1, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), thickness)


def render_censors(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    blur: float = 0.0,
    pixels: float = 0.0,
    censor_text: str | None = None,
) -> None:
    """Apply the appropriate censor effect: blur > pixels > black, then optional text."""
    if not boxes:
        return
    if blur > 0.0:
        apply_blur(frame, boxes, blur)
    elif pixels > 0.0:
        apply_pixelation(frame, boxes, pixels)
    else:
        apply_black_boxes(frame, boxes)
    if censor_text:
        apply_text(frame, boxes, censor_text)
