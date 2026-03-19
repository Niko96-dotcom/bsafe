"""Tests for detection backends."""

import os
from unittest.mock import MagicMock, patch

import pytest

from bsafe.detector import (
    Detection,
    Detector,
    ModelInfo,
    _ERAX_CLASS_MAP,
    _EraXBackend,
    get_model_backend,
    resolve_model,
)

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


# --- resolve_model tests ---


def test_resolve_model_default():
    info = resolve_model(None)
    assert info == ModelInfo(backend="nudenet", path=None)


def test_resolve_model_320n():
    info = resolve_model("320n")
    assert info == ModelInfo(backend="nudenet", path=None)


def test_resolve_model_unknown_raises():
    with pytest.raises(ValueError, match="unknown model 'nonexistent'"):
        resolve_model("nonexistent")


def test_resolve_model_erax_names():
    for name in ("erax-nano", "erax-small", "erax-medium"):
        assert get_model_backend(name) == "erax"


def test_resolve_model_erax_missing_file():
    with patch("os.path.isfile", return_value=False):
        with pytest.raises(ValueError, match="model file not found"):
            resolve_model("erax-nano")


# --- get_model_backend tests ---


def test_get_model_backend_nudenet():
    assert get_model_backend(None) == "nudenet"
    assert get_model_backend("320n") == "nudenet"
    assert get_model_backend("640m") == "nudenet"


def test_get_model_backend_erax():
    assert get_model_backend("erax-nano") == "erax"
    assert get_model_backend("erax-small") == "erax"
    assert get_model_backend("erax-medium") == "erax"


def test_get_model_backend_unknown_raises():
    with pytest.raises(ValueError, match="unknown model"):
        get_model_backend("nonexistent")


# --- EraX class mapping tests ---


def test_erax_class_map_basic():
    assert _ERAX_CLASS_MAP["anus"] == ["ANUS_EXPOSED"]
    assert _ERAX_CLASS_MAP["penis"] == ["MALE_GENITALIA_EXPOSED"]
    assert _ERAX_CLASS_MAP["vagina"] == ["FEMALE_GENITALIA_EXPOSED"]
    assert _ERAX_CLASS_MAP["nipple"] == ["FEMALE_BREAST_EXPOSED"]


# --- EraX backend parsing tests ---


def test_erax_backend_import_error():
    with patch.dict("sys.modules", {"ultralytics": None}):
        with pytest.raises(ImportError, match="uv sync --extra erax"):
            _EraXBackend(0.5, "/fake/model.pt")


def test_erax_backend_parse_results():
    """Test that _EraXBackend correctly parses ultralytics results."""
    import numpy as np

    # Create a mock EraX backend (bypass __init__)
    backend = object.__new__(_EraXBackend)
    backend.min_confidence = 0.3

    # Build mock ultralytics result
    mock_result = MagicMock()
    mock_result.names = {0: "nipple", 1: "penis"}

    mock_boxes = MagicMock()
    mock_boxes.conf = [0.9, 0.8]
    mock_boxes.cls = [0, 1]
    mock_boxes.xyxy = [
        np.array([10.0, 20.0, 110.0, 120.0]),
        np.array([200.0, 300.0, 280.0, 380.0]),
    ]
    mock_boxes.__len__ = lambda self: 2
    mock_result.boxes = mock_boxes

    detections = backend._parse_results([mock_result])

    # nipple → 1 detection, penis → 1
    assert len(detections) == 2

    class_names = [d.class_name for d in detections]
    assert class_names.count("FEMALE_BREAST_EXPOSED") == 1
    assert class_names.count("MALE_GENITALIA_EXPOSED") == 1

    # Check box conversion (xyxy → xywh)
    assert detections[0].box == (10, 20, 100, 100)


def test_erax_backend_filters_low_confidence():
    """Test that low-confidence detections are filtered out."""
    import numpy as np

    backend = object.__new__(_EraXBackend)
    backend.min_confidence = 0.8

    mock_result = MagicMock()
    mock_result.names = {0: "nipple"}
    mock_boxes = MagicMock()
    mock_boxes.conf = [0.5]  # below threshold
    mock_boxes.cls = [0]
    mock_boxes.xyxy = [np.array([10.0, 20.0, 110.0, 120.0])]
    mock_boxes.__len__ = lambda self: 1
    mock_result.boxes = mock_boxes

    detections = backend._parse_results([mock_result])
    assert len(detections) == 0
