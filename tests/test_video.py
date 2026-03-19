import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from bsafe.video import (
    SUPPORTED_EXTENSIONS,
    _CHUNK_FRAMES,
    _process_chunk,
    process_video,
)


def _create_test_video(path, frames=5, width=64, height=64, fps=10.0):
    """Create a minimal synthetic video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(frames):
        # Solid color frames (different each frame for variety)
        color = (i * 50 % 256, 100, 200)
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def test_video(tmp_path):
    """Create a test video and return its path."""
    path = str(tmp_path / "test.mp4")
    _create_test_video(path)
    return path


def _inprocess_chunk_subprocess(**kwargs):
    """Run _process_chunk in the current process so mocks apply."""
    _process_chunk(**kwargs)
    return 0


@pytest.fixture
def mock_detector():
    """Mock the Detector to avoid loading NudeNet."""
    with (
        patch("bsafe.video.Detector") as MockDetector,
        patch("bsafe.video._run_chunk_subprocess", side_effect=_inprocess_chunk_subprocess),
    ):
        instance = MagicMock()
        instance.detect.return_value = []
        instance.detect_frame.return_value = []
        MockDetector.return_value = instance
        yield instance


def test_process_video_creates_output(test_video, mock_detector):
    output = process_video(test_video)
    assert os.path.exists(output)
    assert ".bsafe." in output

    # Verify output has correct frame count
    cap = cv2.VideoCapture(output)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count == 5


def test_output_path_generation(test_video, mock_detector):
    output = process_video(test_video)
    expected = test_video.replace(".mp4", ".bsafe.mp4")
    assert output == expected


def test_missing_input_raises():
    with pytest.raises(FileNotFoundError):
        process_video("/nonexistent/path/video.mp4")


def test_bad_extension_raises(tmp_path):
    bad = tmp_path / "video.avi"
    bad.write_text("fake")
    with pytest.raises(ValueError, match="unsupported format"):
        process_video(str(bad))


def test_custom_output_path(test_video, mock_detector, tmp_path):
    custom = str(tmp_path / "custom_output.mp4")
    output = process_video(test_video, output_path=custom)
    assert output == custom
    assert os.path.exists(custom)


def test_output_dir_missing_raises(test_video):
    with pytest.raises(ValueError, match="output directory does not exist"):
        process_video(test_video, output_path="/nonexistent/dir/out.mp4")


def test_output_already_exists_raises(test_video, mock_detector):
    # Create the output file first
    output_path = test_video.replace(".mp4", ".bsafe.mp4")
    with open(output_path, "w") as f:
        f.write("existing")
    with pytest.raises(ValueError, match="already exists"):
        process_video(test_video)


def test_supported_extensions():
    assert ".mp4" in SUPPORTED_EXTENSIONS
    assert ".m4v" in SUPPORTED_EXTENSIONS
    assert ".mov" in SUPPORTED_EXTENSIONS


def test_fps_override(test_video, mock_detector):
    """With fps_override < native FPS, detection should run less often."""
    process_video(test_video, fps_override=2)
    # With 10 FPS native and 2 FPS detection, detect_every=5
    # 5 frames total, so detect_frame should be called once (frame 0)
    assert mock_detector.detect_frame.call_count == 1


def test_process_video_with_blur(test_video, mock_detector):
    output = process_video(test_video, blur=1.0)
    assert os.path.exists(output)


def test_process_video_with_pixels(test_video, mock_detector):
    output = process_video(test_video, pixels=1.0)
    assert os.path.exists(output)


def test_process_video_with_censor_text(test_video, mock_detector):
    output = process_video(test_video, censor_text="NSFW")
    assert os.path.exists(output)


def test_chunks_cleaned_up_after_success(test_video, mock_detector):
    """Chunk directory should be removed after successful processing."""
    chunks_dir = test_video.replace(".mp4", ".bsafe.chunks")
    process_video(test_video)
    assert not os.path.exists(chunks_dir)


def test_resume_from_partial(tmp_path, mock_detector):
    """If chunks exist from a previous run, they should be reused."""
    # Create a video with more frames than one chunk
    path = str(tmp_path / "test.mp4")
    total = _CHUNK_FRAMES + 10  # just over 1 chunk
    _create_test_video(path, frames=total)

    # Simulate a previous partial run: create a complete first chunk
    chunks_dir = str(tmp_path / "test.bsafe.chunks")
    os.makedirs(chunks_dir)
    chunk_0 = os.path.join(chunks_dir, "chunk_0000.mp4")
    _create_test_video(chunk_0, frames=_CHUNK_FRAMES)

    output = process_video(path)
    assert os.path.exists(output)
    # Chunks should be cleaned up
    assert not os.path.exists(chunks_dir)


def test_chunk_retry_on_subprocess_failure(test_video, mock_detector):
    """Chunk should complete after a transient subprocess failure."""
    call_count = 0

    def _fail_then_succeed(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return -9  # simulate OOM kill
        # Actually process the chunk on retry
        return _inprocess_chunk_subprocess(**kwargs)

    # Override the mock_detector's _run_chunk_subprocess patch
    with patch("bsafe.video._run_chunk_subprocess", side_effect=_fail_then_succeed):
        output = process_video(test_video)
        assert os.path.exists(output)
        assert call_count == 2


def test_chunk_retry_exhausted(test_video, mock_detector):
    """RuntimeError should be raised after all retries are exhausted."""
    with patch(
        "bsafe.video._run_chunk_subprocess",
        return_value=-9,
    ):
        with pytest.raises(RuntimeError, match="--chunk-frames"):
            process_video(test_video)
