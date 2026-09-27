"""Tests for isolated motion compensation (synthetic textured rectangles only)."""

import copy
import math

import cv2
import numpy as np

from bsafe.motion import compensate_boxes

W, H = 320, 240
_TOL = 8


def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return bytes(buf.tobytes())


def _bg(val: int = 100) -> np.ndarray:
    return np.full((H, W, 3), val, dtype=np.uint8)


def _textured_block(bw: int, bh: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(bh, bw, 3)).astype(np.uint8)


def _place(bg: np.ndarray, block: np.ndarray, x: int, y: int) -> np.ndarray:
    frame = bg.copy()
    bh, bw = block.shape[:2]
    frame[y : y + bh, x : x + bw] = block
    return frame


def _assert_bounded(boxes: list[tuple[int, int, int, int]]) -> None:
    for x, y, bw, bh in boxes:
        assert all(isinstance(v, int) for v in (x, y, bw, bh))
        assert all(math.isfinite(v) for v in (x, y, bw, bh))
        assert bw > 0 and bh > 0
        assert 0 <= x and 0 <= y and x + bw <= W and y + bh <= H


def test_translation_maps_box():
    block = _textured_block(60, 60, seed=7)
    ref = _place(_bg(), block, 50, 40)
    cur = _place(_bg(), block, 90, 100)  # +40 x, +60 y
    out = compensate_boxes(_jpeg(ref), _jpeg(cur), [(50, 40, 60, 60)], W, H)
    assert len(out) == 1
    x, y, bw, bh = out[0]
    assert (bw, bh) == (60, 60)
    assert abs(x - 90) <= _TOL, out
    assert abs(y - 100) <= _TOL, out
    _assert_bounded(out)


def test_scene_cut_returns_unchanged():
    rng = np.random.default_rng(11)
    ref = rng.integers(0, 256, size=(H, W, 3)).astype(np.uint8)
    rng2 = np.random.default_rng(99)
    cur = rng2.integers(0, 256, size=(H, W, 3)).astype(np.uint8)
    boxes = [(30, 30, 50, 50)]
    out = compensate_boxes(_jpeg(ref), _jpeg(cur), boxes, W, H)
    assert out == boxes
    _assert_bounded(out)


def test_low_texture_returns_unchanged():
    ref = _bg(128)
    cur = _bg(128)
    boxes = [(40, 40, 60, 60)]
    out = compensate_boxes(_jpeg(ref), _jpeg(cur), boxes, W, H)
    assert out == boxes
    _assert_bounded(out)


def test_frame_dimension_mismatch_safe():
    ref = _bg()
    cur = _bg()
    boxes = [(10, 10, 40, 40)]
    out = compensate_boxes(_jpeg(ref), _jpeg(cur), boxes, W + 10, H)
    assert out == boxes
    for x, y, bw, bh in out:
        assert all(isinstance(v, int) and math.isfinite(v) for v in (x, y, bw, bh))
    # Mismatched payload shapes must also be safe.
    small = np.full((120, 160, 3), 100, dtype=np.uint8)
    out2 = compensate_boxes(_jpeg(ref), _jpeg(small), boxes, W, H)
    assert out2 == boxes


def test_clipped_edges_bounded_and_invalid_dropped():
    ref = _bg(128)
    cur = _bg(128)
    boxes = [(W - 20, H - 20, 40, 40), (10, 10, 0, 5), (10, 10, -4, 10)]
    out = compensate_boxes(_jpeg(ref), _jpeg(cur), boxes, W, H)
    assert out == [(W - 20, H - 20, 20, 20)]
    _assert_bounded(out)


def test_zero_boxes_empty():
    ref = _bg()
    out = compensate_boxes(_jpeg(ref), _jpeg(ref), [], W, H)
    assert out == []


def test_input_immutable():
    block = _textured_block(60, 60, seed=7)
    ref = _place(_bg(), block, 50, 40)
    cur = _place(_bg(), block, 90, 100)
    boxes = [(50, 40, 60, 60)]
    snapshot = copy.deepcopy(boxes)
    compensate_boxes(_jpeg(ref), _jpeg(cur), boxes, W, H)
    assert boxes == snapshot
