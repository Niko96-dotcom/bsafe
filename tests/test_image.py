import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from bsafe.detector import Detection
from bsafe.image import SUPPORTED_EXTENSIONS, process_image


def _create_test_image(path, width=64, height=64):
    """Create a minimal synthetic image for testing."""
    frame = np.full((height, width, 3), (100, 150, 200), dtype=np.uint8)
    cv2.imwrite(path, frame)


@pytest.fixture
def test_image(tmp_path):
    """Create a test image and return its path."""
    path = str(tmp_path / "test.jpg")
    _create_test_image(path)
    return path


@pytest.fixture
def mock_detector():
    """Mock the Detector to avoid loading NudeNet."""
    with patch("bsafe.image.Detector") as MockDetector:
        instance = MagicMock()
        instance.detect_frame.return_value = []
        MockDetector.return_value = instance
        yield instance


def test_process_image_creates_output(test_image, mock_detector):
    output = process_image(test_image)
    assert os.path.exists(output)
    assert ".bsafe." in output


def test_output_path_generation(test_image, mock_detector):
    output = process_image(test_image)
    expected = test_image.replace(".jpg", ".bsafe.jpg")
    assert output == expected


def test_missing_input_raises():
    with pytest.raises(FileNotFoundError):
        process_image("/nonexistent/path/image.jpg")


def test_bad_extension_raises(tmp_path):
    bad = tmp_path / "image.gif"
    bad.write_text("fake")
    with pytest.raises(ValueError, match="unsupported format"):
        process_image(str(bad))


def test_output_already_exists_raises(test_image, mock_detector):
    output_path = test_image.replace(".jpg", ".bsafe.jpg")
    with open(output_path, "w") as f:
        f.write("existing")
    with pytest.raises(ValueError, match="already exists"):
        process_image(test_image)


def test_supported_extensions():
    assert ".jpg" in SUPPORTED_EXTENSIONS
    assert ".jpeg" in SUPPORTED_EXTENSIONS
    assert ".png" in SUPPORTED_EXTENSIONS
    assert ".bmp" in SUPPORTED_EXTENSIONS
    assert ".webp" in SUPPORTED_EXTENSIONS
    assert ".tif" in SUPPORTED_EXTENSIONS
    assert ".tiff" in SUPPORTED_EXTENSIONS


def test_process_image_with_blur(test_image, mock_detector):
    output = process_image(test_image, blur=1.0)
    assert os.path.exists(output)


def test_process_image_with_pixels(test_image, mock_detector):
    output = process_image(test_image, pixels=1.0)
    assert os.path.exists(output)


def test_process_image_with_censor_text(test_image, mock_detector):
    output = process_image(test_image, censor_text="NSFW")
    assert os.path.exists(output)


def test_process_image_png(tmp_path, mock_detector):
    path = str(tmp_path / "test.png")
    _create_test_image(path)
    output = process_image(path)
    assert output.endswith(".bsafe.png")
    assert os.path.exists(output)


def test_process_image_with_detections(tmp_path):
    """Exercise the detection → censor → render pipeline with a mock detection."""
    path = str(tmp_path / "test.png")
    frame = np.full((64, 64, 3), 200, dtype=np.uint8)
    cv2.imwrite(path, frame)

    det = Detection(class_name="FEMALE_BREAST_EXPOSED", confidence=0.9, box=(5, 5, 30, 30))
    with patch("bsafe.image.Detector") as MockDetector:
        instance = MagicMock()
        instance.detect_frame.return_value = [det]
        MockDetector.return_value = instance
        output = process_image(path, confidence=0.5)
        instance.detect_frame.assert_called_once()
    assert os.path.exists(output)
    original = cv2.imread(path)
    censored = cv2.imread(output)
    assert not np.array_equal(original, censored)
