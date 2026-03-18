"""Tests for temporal smoothing and box tracking."""

from bsafe.tracking import BoxTracker


def test_persistence_after_disappearance():
    tracker = BoxTracker(persist_frames=3, smooth_alpha=1.0)
    # Frame 1: box appears
    result = tracker.update(1, [(100, 100, 50, 50)])
    assert len(result) == 1
    # Frame 2: box disappears — should persist
    result = tracker.update(1, [])
    assert len(result) == 1
    # Frames 3-4: still persisting
    result = tracker.update(1, [])
    assert len(result) == 1
    result = tracker.update(1, [])
    assert len(result) == 1
    # Frame 5: expired (missing for 4 frames > persist_frames=3)
    result = tracker.update(1, [])
    assert len(result) == 0


def test_position_smoothing():
    tracker = BoxTracker(persist_frames=8, smooth_alpha=0.5)
    # Frame 1: box at (100, 100, 50, 50)
    tracker.update(1, [(100, 100, 50, 50)])
    # Frame 2: box jumps to (120, 100, 50, 50) — smoothed position should be ~110
    result = tracker.update(1, [(120, 100, 50, 50)])
    assert len(result) == 1
    x, y, w, h = result[0]
    assert x == 110  # EMA: 0.5 * 120 + 0.5 * 100
    assert y == 100
    assert w == 50
    assert h == 50


def test_per_display_independence():
    tracker = BoxTracker(persist_frames=8, smooth_alpha=1.0)
    # Display 1 has a box, display 2 doesn't
    result1 = tracker.update(1, [(10, 10, 20, 20)])
    result2 = tracker.update(2, [])
    assert len(result1) == 1
    assert len(result2) == 0
    # Display 2 gets a box, display 1 loses it
    result1 = tracker.update(1, [])
    result2 = tracker.update(2, [(50, 50, 30, 30)])
    assert len(result1) == 1  # persisting
    assert len(result2) == 1


def test_new_unmatched_box_creates_track():
    tracker = BoxTracker(persist_frames=8, smooth_alpha=1.0)
    result = tracker.update(1, [(0, 0, 10, 10)])
    assert len(result) == 1
    # Add a second non-overlapping box
    result = tracker.update(1, [(0, 0, 10, 10), (200, 200, 10, 10)])
    assert len(result) == 2


def test_empty_input_first_call():
    tracker = BoxTracker(persist_frames=8, smooth_alpha=1.0)
    result = tracker.update(1, [])
    assert result == []
