"""Tests for bsafe.enhance — low-light video enhancement."""

import os
import subprocess
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

from bsafe.enhance import (
    WindowIndex,
    WindowParams,
    _compute_params,
    _smooth_windows,
    denoise_video,
    enhance_frame,
    prescan_video,
)


# ---------------------------------------------------------------------------
# _compute_params
# ---------------------------------------------------------------------------
class TestComputeParams:
    def test_bright_footage_skips(self):
        result = _compute_params(median=150, p5=80, p95=220, shadow_ratio=0.05, clip_ratio=0.01)
        assert result.skip is True
        assert result.gamma == 1.0
        assert result.contrast == 1.0
        assert result.saturation == 1.0

    def test_dark_footage_enhances(self):
        result = _compute_params(median=40, p5=5, p95=100, shadow_ratio=0.3, clip_ratio=0.0)
        assert result.skip is False
        assert result.gamma > 1.0
        assert result.contrast > 1.0
        assert result.saturation > 1.0

    def test_pitch_black_caps_gamma(self):
        result = _compute_params(median=15, p5=2, p95=30, shadow_ratio=0.8, clip_ratio=0.0)
        assert result.skip is False
        # Pitch-black cap at 1.15, but noisy cap at 1.1 may apply too
        assert result.gamma <= 1.15
        # Pitch-black sets contrast to 1.0
        assert result.contrast == 1.0

    def test_noisy_dark_caps_gamma(self):
        result = _compute_params(median=30, p5=3, p95=60, shadow_ratio=0.7, clip_ratio=0.0)
        assert result.skip is False
        assert result.gamma <= 1.1

    def test_highlight_protection_reduces_gamma(self):
        no_clip = _compute_params(median=60, p5=10, p95=180, shadow_ratio=0.2, clip_ratio=0.0)
        clip = _compute_params(median=60, p5=10, p95=250, shadow_ratio=0.2, clip_ratio=0.1)
        assert clip.gamma < no_clip.gamma

    def test_fade_zone(self):
        """Median 100-140 should produce partial enhancement."""
        result = _compute_params(median=120, p5=50, p95=200, shadow_ratio=0.1, clip_ratio=0.01)
        assert result.skip is False
        assert 1.0 < result.gamma < 1.3
        assert 1.0 < result.contrast < 1.08


# ---------------------------------------------------------------------------
# enhance_frame
# ---------------------------------------------------------------------------
class TestEnhanceFrame:
    def test_noop_when_all_unity(self):
        frame = np.full((100, 100, 3), 40, dtype=np.uint8)
        original = frame.copy()
        enhance_frame(frame, 1.0, 1.0, 1.0)
        np.testing.assert_array_equal(frame, original)

    def test_brightens_dark_frame(self):
        frame = np.full((100, 100, 3), 30, dtype=np.uint8)
        mean_before = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        enhance_frame(frame, 1.3, 1.0, 1.0)
        mean_after = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        assert mean_after > mean_before

    def test_preserves_shape(self):
        frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        shape_before = frame.shape
        enhance_frame(frame, 1.2, 1.05, 1.02)
        assert frame.shape == shape_before

    def test_modifies_in_place(self):
        frame = np.full((50, 50, 3), 40, dtype=np.uint8)
        ref = frame  # same object
        enhance_frame(frame, 1.3, 1.08, 1.03)
        # ref should point to the modified data
        assert ref is frame
        assert not np.all(ref == 40)

    def test_contrast_only(self):
        frame = np.full((50, 50, 3), 80, dtype=np.uint8)
        enhance_frame(frame, 1.0, 1.08, 1.0)
        # With uniform input, contrast around mean should keep values similar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        assert np.std(gray) < 5  # still fairly uniform

    def test_no_clipping_on_bright_input(self):
        """Enhancement should not produce values > 255 or < 0."""
        frame = np.full((50, 50, 3), 200, dtype=np.uint8)
        enhance_frame(frame, 1.3, 1.08, 1.03)
        assert frame.max() <= 255
        assert frame.min() >= 0


# ---------------------------------------------------------------------------
# WindowIndex.lookup
# ---------------------------------------------------------------------------
class TestWindowIndex:
    def test_finds_correct_window(self):
        index = WindowIndex(
            [
                WindowParams(0, 100, 1.2, 1.05, 1.02, skip=False),
                WindowParams(100, 200, 1.1, 1.03, 1.01, skip=False),
            ]
        )
        g, c, s = index.lookup(50)
        assert g == 1.2
        assert c == 1.05

        g, c, s = index.lookup(150)
        assert g == 1.1

    def test_skip_window_returns_unity(self):
        index = WindowIndex([WindowParams(0, 100, 1.2, 1.05, 1.02, skip=True)])
        g, c, s = index.lookup(50)
        assert g == 1.0
        assert c == 1.0
        assert s == 1.0

    def test_out_of_range_returns_unity(self):
        index = WindowIndex([WindowParams(0, 100, 1.2, 1.05, 1.02, skip=False)])
        g, c, s = index.lookup(200)
        assert g == 1.0

    def test_boundary_frame(self):
        index = WindowIndex(
            [
                WindowParams(0, 100, 1.2, 1.05, 1.02, skip=False),
                WindowParams(100, 200, 1.1, 1.03, 1.01, skip=False),
            ]
        )
        # Frame 100 should be in the second window (start_frame <= idx < end_frame)
        g, _, _ = index.lookup(100)
        assert g == 1.1

    def test_all_skip_property(self):
        index = WindowIndex(
            [
                WindowParams(0, 100, 1.0, 1.0, 1.0, skip=True),
                WindowParams(100, 200, 1.0, 1.0, 1.0, skip=True),
            ]
        )
        assert index.all_skip is True

        index2 = WindowIndex(
            [
                WindowParams(0, 100, 1.2, 1.05, 1.02, skip=False),
                WindowParams(100, 200, 1.0, 1.0, 1.0, skip=True),
            ]
        )
        assert index2.all_skip is False


# ---------------------------------------------------------------------------
# _smooth_windows (EMA)
# ---------------------------------------------------------------------------
class TestSmoothWindows:
    def test_single_window_unchanged(self):
        windows = [WindowParams(0, 100, 1.3, 1.08, 1.03, skip=False)]
        result = _smooth_windows(windows)
        assert len(result) == 1
        assert result[0].gamma == 1.3

    def test_smoothing_reduces_jumps(self):
        windows = [
            WindowParams(0, 100, 1.3, 1.08, 1.03, skip=False),
            WindowParams(100, 200, 1.0, 1.0, 1.0, skip=False),
        ]
        result = _smooth_windows(windows)
        # Second window should be smoothed toward first, not exactly 1.0
        assert result[1].gamma > 1.0

    def test_all_skip_preserved(self):
        windows = [
            WindowParams(0, 100, 1.0, 1.0, 1.0, skip=True),
            WindowParams(100, 200, 1.0, 1.0, 1.0, skip=True),
        ]
        result = _smooth_windows(windows)
        assert all(w.skip for w in result)


# ---------------------------------------------------------------------------
# prescan_video
# ---------------------------------------------------------------------------
def _create_synthetic_video(path, num_frames, pixel_value, fps=30.0):
    """Create a synthetic video with uniform color for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (160, 120))
    for _ in range(num_frames):
        frame = np.full((120, 160, 3), pixel_value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestPrescanVideo:
    def test_dark_video_returns_enhancement_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "dark.mp4")
            _create_synthetic_video(video_path, 90, pixel_value=30, fps=30.0)

            index = prescan_video(video_path, 30.0, 90)
            assert not index.all_skip
            # Spot-check: frame 0 should get enhancement
            g, c, s = index.lookup(0)
            assert g > 1.0

    def test_bright_video_skips_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "bright.mp4")
            _create_synthetic_video(video_path, 90, pixel_value=180, fps=30.0)

            index = prescan_video(video_path, 30.0, 90)
            assert index.all_skip

    def test_empty_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "empty.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (160, 120))
            writer.release()

            index = prescan_video(video_path, 30.0, 0)
            assert index.all_skip


# ---------------------------------------------------------------------------
# denoise_video
# ---------------------------------------------------------------------------
class TestDenoiseVideo:
    def test_no_ffmpeg_returns_false(self):
        with patch("bsafe.enhance.shutil.which", return_value=None):
            result = denoise_video("/fake/input.mp4", "/fake/output.mp4")
            assert result is False

    def test_ffmpeg_failure_returns_false(self):
        with (
            patch("bsafe.enhance.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch(
                "bsafe.enhance.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"error"),
            ),
        ):
            result = denoise_video("/fake/input.mp4", "/fake/output.mp4")
            assert result is False

    def test_ffmpeg_success_returns_true(self):
        with (
            patch("bsafe.enhance.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("bsafe.enhance.subprocess.run"),
        ):
            result = denoise_video("/fake/input.mp4", "/fake/output.mp4")
            assert result is True
