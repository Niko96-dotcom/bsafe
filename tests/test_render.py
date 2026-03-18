import numpy as np

from bsafe.render import (
    apply_black_boxes,
    apply_blur,
    apply_pixelation,
    apply_text,
    render_censors,
)


def _white_frame(w=100, h=100):
    """Create a white BGR frame."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_apply_black_boxes_fills_region():
    frame = _white_frame()
    apply_black_boxes(frame, [(10, 10, 20, 20)])
    roi = frame[10:30, 10:30]
    assert np.all(roi == 0)
    # Outside region unchanged
    assert frame[0, 0, 0] == 255


def test_apply_black_boxes_empty():
    frame = _white_frame()
    original = frame.copy()
    apply_black_boxes(frame, [])
    assert np.array_equal(frame, original)


def test_apply_blur_modifies_region():
    frame = _white_frame()
    # Draw a black rectangle so blur has something to work with
    frame[20:40, 20:40] = 0
    original = frame.copy()
    apply_blur(frame, [(10, 10, 50, 50)], 1.0)
    # The blurred region should differ from original
    roi_orig = original[10:60, 10:60]
    roi_new = frame[10:60, 10:60]
    assert not np.array_equal(roi_orig, roi_new)


def test_apply_pixelation_modifies_region():
    # Create a frame with a gradient so pixelation is visible
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        frame[i, :] = i * 2  # gradient
    original = frame.copy()
    apply_pixelation(frame, [(10, 10, 50, 50)], 1.0)
    roi_orig = original[10:60, 10:60]
    roi_new = frame[10:60, 10:60]
    assert not np.array_equal(roi_orig, roi_new)


def test_apply_text_modifies_region():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    original = frame.copy()
    apply_text(frame, [(50, 50, 100, 100)], "NSFW")
    assert not np.array_equal(frame, original)


def test_render_censors_uses_blur():
    frame = _white_frame()
    frame[20:40, 20:40] = 0
    original = frame.copy()
    render_censors(frame, [(10, 10, 50, 50)], blur=1.0)
    assert not np.array_equal(frame, original)


def test_render_censors_uses_pixelation():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        frame[i, :] = i * 2
    original = frame.copy()
    render_censors(frame, [(10, 10, 50, 50)], pixels=1.0)
    assert not np.array_equal(frame, original)


def test_render_censors_uses_black_by_default():
    frame = _white_frame()
    render_censors(frame, [(10, 10, 20, 20)])
    roi = frame[10:30, 10:30]
    assert np.all(roi == 0)


def test_render_censors_empty_boxes():
    frame = _white_frame()
    original = frame.copy()
    render_censors(frame, [])
    assert np.array_equal(frame, original)


def test_apply_blur_zero_size_roi():
    """Boxes at boundaries with zero-size ROI should not crash."""
    frame = _white_frame(50, 50)
    apply_blur(frame, [(50, 50, 0, 0)], 1.0)  # ROI is empty


def test_apply_pixelation_zero_size_roi():
    frame = _white_frame(50, 50)
    apply_pixelation(frame, [(50, 50, 0, 0)], 1.0)
