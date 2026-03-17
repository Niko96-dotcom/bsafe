"""Tests for NudeNet wrapper."""

import os

import pytest

from bsafe.detector import Detection, Detector

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_FRAME = os.path.join(FIXTURES_DIR, "sample_frame.jpg")

needs_nudenet = pytest.mark.skipif(
    os.environ.get("BSAFE_SKIP_SLOW") == "1",
    reason="Skipped: BSAFE_SKIP_SLOW=1 (NudeNet downloads ~200MB model)",
)


@pytest.fixture(scope="module")
def detector():
    return Detector(min_confidence=0.3)


def test_detection_dataclass():
    d = Detection(class_name="EXPOSED_BREAST_F", confidence=0.85, box=(10, 20, 100, 200))
    assert d.class_name == "EXPOSED_BREAST_F"
    assert d.confidence == 0.85
    assert d.box == (10, 20, 100, 200)


@needs_nudenet
def test_detect_returns_list(detector):
    with open(SAMPLE_FRAME, "rb") as f:
        jpeg_bytes = f.read()
    results = detector.detect(jpeg_bytes)
    assert isinstance(results, list)
    # sample_frame.jpg is a safe image, so detections should be empty or low-confidence
    for r in results:
        assert isinstance(r, Detection)


@needs_nudenet
def test_detect_filters_low_confidence():
    strict = Detector(min_confidence=0.99)
    with open(SAMPLE_FRAME, "rb") as f:
        jpeg_bytes = f.read()
    results = strict.detect(jpeg_bytes)
    for r in results:
        assert r.confidence >= 0.99
