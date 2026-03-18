"""Tests for censor filtering and box padding."""

from bsafe.censor import (
    DEFAULT_CENSOR_CLASSES,
    build_censor_boxes,
    filter_detections,
    merge_overlapping_boxes,
    pad_box,
)
from bsafe.detector import Detection


def _det(class_name, box=(100, 100, 80, 60), confidence=0.9):
    return Detection(class_name=class_name, confidence=confidence, box=box)


def test_filter_detections_keeps_matching():
    dets = [
        _det("FEMALE_BREAST_EXPOSED"),
        _det("FACE_FEMALE"),
        _det("MALE_GENITALIA_EXPOSED"),
    ]
    result = filter_detections(dets, DEFAULT_CENSOR_CLASSES)
    assert len(result) == 2
    assert result[0].class_name == "FEMALE_BREAST_EXPOSED"
    assert result[1].class_name == "MALE_GENITALIA_EXPOSED"


def test_filter_detections_empty_input():
    assert filter_detections([], DEFAULT_CENSOR_CLASSES) == []


def test_filter_detections_none_match():
    dets = [_det("FACE_FEMALE"), _det("BELLY_EXPOSED")]
    assert filter_detections(dets, DEFAULT_CENSOR_CLASSES) == []


def test_pad_box_no_padding():
    # (x=100, y=100, w=80, h=60) with 0 padding → same box
    assert pad_box((100, 100, 80, 60), 0.0, 1920, 1080) == (100, 100, 80, 60)


def test_pad_box_with_padding():
    # box (x=100, y=100, w=100, h=100), padding 0.2 → expand 20px each side
    x, y, w, h = pad_box((100, 100, 100, 100), 0.2, 1920, 1080)
    assert x == 80
    assert y == 80
    assert w == 140
    assert h == 140


def test_pad_box_clamps_to_bounds():
    # Box near top-left corner (x=5, y=5, w=40, h=40), large padding
    x, y, w, h = pad_box((5, 5, 40, 40), 0.5, 100, 100)
    assert x == 0
    assert y == 0
    assert w >= 40  # at least original width
    assert h >= 40


def test_pad_box_clamps_to_bottom_right():
    # Box near bottom-right, padding would exceed display
    # (x=950, y=550, w=40, h=40) in 1000x600 display, padding 0.5 → 20px each side
    x, y, w, h = pad_box((950, 550, 40, 40), 0.5, 1000, 600)
    assert x + w <= 1000
    assert y + h <= 600


def test_build_censor_boxes():
    # BUTTOCKS_EXPOSED is intentionally not censored (see censor.py)
    dets = [
        _det("FEMALE_BREAST_EXPOSED", box=(100, 100, 80, 60)),
        _det("FACE_FEMALE", box=(300, 300, 80, 60)),
        _det("BUTTOCKS_EXPOSED", box=(500, 500, 80, 60)),
    ]
    boxes = build_censor_boxes(dets, DEFAULT_CENSOR_CLASSES, 0.0, 1920, 1080)
    assert len(boxes) == 1
    assert boxes[0] == (100, 100, 80, 60)


def test_build_censor_boxes_empty():
    assert build_censor_boxes([], DEFAULT_CENSOR_CLASSES, 0.2, 1920, 1080) == []


# --- merge_overlapping_boxes tests ---


def test_merge_non_overlapping_stay_separate():
    boxes = [(0, 0, 50, 50), (200, 200, 50, 50)]
    result = merge_overlapping_boxes(boxes)
    assert len(result) == 2
    assert (0, 0, 50, 50) in result
    assert (200, 200, 50, 50) in result


def test_merge_overlapping_pair():
    boxes = [(0, 0, 60, 60), (30, 30, 60, 60)]
    result = merge_overlapping_boxes(boxes)
    assert len(result) == 1
    assert result[0] == (0, 0, 90, 90)


def test_merge_transitive_overlap():
    # A overlaps B, B overlaps C → all three merge into one
    boxes = [(0, 0, 40, 40), (30, 0, 40, 40), (60, 0, 40, 40)]
    result = merge_overlapping_boxes(boxes)
    assert len(result) == 1
    assert result[0] == (0, 0, 100, 40)


def test_merge_single_box_unchanged():
    boxes = [(10, 20, 30, 40)]
    assert merge_overlapping_boxes(boxes) == [(10, 20, 30, 40)]


def test_merge_empty():
    assert merge_overlapping_boxes([]) == []
