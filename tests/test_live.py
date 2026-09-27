"""Live mailbox, staleness, stats, and detail-scan tiling tests (deterministic)."""

import argparse
import queue
import time
from unittest.mock import MagicMock

import pytest

from bsafe.detector import Detection
from bsafe.ipc import FrameMailbox
from bsafe.live import (
    DetailScanDetector,
    LiveStats,
    compute_tile_rects,
    dedupe_detections,
    is_frame_stale,
    remap_box,
    validate_live_options,
)
from bsafe.protocol import FrameMetadata
from bsafe.tracking import BoxTracker


def _meta(display_id=1, w=640, h=480):
    return FrameMetadata(display_id, w, h, 0)


# --- FrameMailbox ---


def test_mailbox_replaces_same_display_latest():
    box = FrameMailbox(maxsize=10)
    box.put_nowait((_meta(1), b"old"))
    box.put_nowait((_meta(1), b"new"))
    assert box.qsize() == 1
    assert box.replaced == 1
    meta, jpeg, _receipt = box.get(timeout=1)
    assert jpeg == b"new"
    assert meta.display_id == 1


def test_mailbox_fair_across_displays():
    box = FrameMailbox(maxsize=10)
    box.put_nowait((_meta(1), b"d1-v1"))
    box.put_nowait((_meta(2), b"d2-v1"))
    first = box.get(timeout=1)
    assert first[0].display_id == 1
    # Busy display posts again while quiet display still pending.
    box.put_nowait((_meta(1), b"d1-v2"))
    second = box.get(timeout=1)
    assert second[0].display_id == 2
    assert second[1] == b"d2-v1"
    third = box.get(timeout=1)
    assert third[0].display_id == 1
    assert third[1] == b"d1-v2"


def test_mailbox_timeout_and_nowait():
    box = FrameMailbox(maxsize=5)
    assert box.empty()
    with pytest.raises(queue.Empty):
        box.get(timeout=0.05)
    with pytest.raises(queue.Empty):
        box.get_nowait()


def test_mailbox_overload_bounded_single_display():
    box = FrameMailbox(maxsize=10)
    for i in range(100):
        box.put_nowait((_meta(7), f"f{i}".encode()))
    assert box.qsize() == 1
    meta, jpeg, _ = box.get(timeout=1)
    assert jpeg == b"f99"
    assert box.empty()


def test_mailbox_honors_display_capacity():
    box = FrameMailbox(maxsize=2)
    box.put_nowait((_meta(1), b"a"))
    box.put_nowait((_meta(2), b"b"))
    box.put_nowait((_meta(3), b"c"))  # beyond capacity
    assert box.qsize() == 2
    assert box.dropped_full == 1
    seen = {box.get(timeout=1)[0].display_id, box.get(timeout=1)[0].display_id}
    assert seen == {1, 2}


def test_mailbox_receipt_monotonic():
    box = FrameMailbox(maxsize=5)
    t0 = time.monotonic()
    box.put_nowait((_meta(1), b"x"))
    t1 = time.monotonic()
    _, _, receipt = box.get(timeout=1)
    assert t0 <= receipt <= t1


# --- Staleness ---


def test_is_frame_stale_budget():
    now = 100.0
    assert not is_frame_stale(now - 0.1, now, 250)
    assert is_frame_stale(now - 0.3, now, 250)
    assert not is_frame_stale(now - 10.0, now, 0)  # 0 disables


def test_stale_check_ignores_newer_existence():
    # Fresh result must not be discarded merely because a newer frame exists.
    now = 50.0
    assert not is_frame_stale(now - 0.05, now, 250)


def test_tracker_clear_resets_display():
    tracker = BoxTracker(persist_frames=8, smooth_alpha=1.0)
    tracker.update(1, [(10, 10, 20, 20)])
    tracker.clear(1)
    assert tracker.update(1, []) == []
    # Other displays unaffected.
    tracker.update(2, [(0, 0, 5, 5)])
    tracker.clear(1)
    assert tracker.update(2, []) != []


# --- LiveStats ---


def test_live_stats_aggregates_and_labels_scope():
    times = [0.0]

    stats = LiveStats(time_fn=lambda: times[0], interval_s=2.0)
    stats.record(0.05, 0.1, 0.2)
    stats.record(0.05, 0.1, 0.2)
    assert stats.maybe_log(replaced=3) is None  # interval not elapsed
    times[0] = 2.5
    line = stats.maybe_log(replaced=3)
    assert line is not None
    assert "not capture-to-render" in line
    assert "replaced=3" in line
    assert "avg_receive_to_send_ms" in line
    # Bounded: single line, no pixel data.
    assert "\n" not in line
    assert "jpeg" not in line.lower()


def test_live_stats_stale_and_summary():
    stats = LiveStats(time_fn=time.monotonic)
    stats.record(0.01, 0.02, 0.05)
    stats.record_stale()
    assert stats.total == 2
    assert stats.stale == 1
    summary = stats.summary(replaced=1)
    assert "stale=1" in summary
    assert "not capture-to-render" in summary


# --- Tiling ---


def test_tile_rects_cover_frame():
    for w, h in [(640, 480), (1920, 1080), (100, 100), (10, 10)]:
        rects = compute_tile_rects(w, h)
        assert 1 <= len(rects) <= 4
        for x0, y0, tw, th in rects:
            assert 0 <= x0 and x0 + tw <= w
            assert 0 <= y0 and y0 + th <= h
        # Union covers all corners.
        covered = set()
        for x0, y0, tw, th in rects:
            for px, py in [(0, 0), (w - 1, h - 1), (w - 1, 0), (0, h - 1), (w // 2, h // 2)]:
                if x0 <= px < x0 + tw and y0 <= py < y0 + th:
                    covered.add((px, py))
        assert len(covered) == 5


def test_tile_rects_small_frames():
    assert compute_tile_rects(1, 1) == [(0, 0, 1, 1)]
    rects = compute_tile_rects(2, 1)
    assert len(rects) == 2
    for x0, y0, tw, th in rects:
        assert x0 + tw <= 2 and y0 + th <= 1


def test_tile_rects_overlap_middle_seam():
    rects = compute_tile_rects(1000, 800)
    assert len(rects) == 4
    # Left and right tiles must overlap in x; top and bottom in y.
    xs = sorted({x0 for x0, _, _, _ in rects})
    assert len(xs) == 2
    left = [r for r in rects if r[0] == xs[0]][0]
    right = [r for r in rects if r[0] == xs[1]][0]
    assert left[0] + left[2] > right[0]


def test_remap_box_translate_and_clamp():
    assert remap_box((10, 10, 20, 20), (100, 50), 1000, 800) == (110, 60, 20, 20)
    # Clamped at frame edges.
    assert remap_box((90, 90, 30, 30), (900, 700), 1000, 800) == (990, 790, 10, 10)
    assert remap_box((0, 0, 5, 5), (0, 0), 640, 480) == (0, 0, 5, 5)
    assert remap_box((0, 0, 0, 5), (0, 0), 640, 480) is None


def test_dedupe_suppresses_same_class_duplicates():
    a = Detection("FEMALE_BREAST_EXPOSED", 0.9, (100, 100, 50, 50))
    dup = Detection("FEMALE_BREAST_EXPOSED", 0.7, (102, 102, 50, 50))
    other_class = Detection("MALE_GENITALIA_EXPOSED", 0.8, (102, 102, 50, 50))
    out = dedupe_detections([dup, a, other_class])
    assert len(out) == 2
    kept_names = sorted(d.class_name for d in out)
    assert kept_names == ["FEMALE_BREAST_EXPOSED", "MALE_GENITALIA_EXPOSED"]
    kept_breast = [d for d in out if d.class_name == "FEMALE_BREAST_EXPOSED"][0]
    assert kept_breast.confidence == 0.9


def test_dedupe_keeps_separate_boxes():
    a = Detection("ANUS_EXPOSED", 0.9, (0, 0, 10, 10))
    b = Detection("ANUS_EXPOSED", 0.8, (200, 200, 10, 10))
    assert len(dedupe_detections([a, b])) == 2


# --- Detail wrapper ---


def test_detail_disabled_single_pass():
    inner = MagicMock()
    inner.detect.return_value = [Detection("ANUS_EXPOSED", 0.9, (1, 1, 5, 5))]
    wrapper = DetailScanDetector(inner, enabled=False)
    out = wrapper.detect(b"jpeg")
    assert out == inner.detect.return_value
    assert inner.detect.call_count == 1
    assert inner.detect_frame.call_count == 0


def test_detail_enabled_full_plus_tiles_remapped():
    import numpy as np

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def _detect_frame(arr):
        # Report a box at tile-local (1,1,10,10) for every call.
        _detect_frame.calls += 1
        return [Detection("ANUS_EXPOSED", 0.9, (1, 1, 10, 10))]

    _detect_frame.calls = 0
    inner = MagicMock()
    inner.detect_frame.side_effect = _detect_frame
    wrapper = DetailScanDetector(inner, enabled=True)
    out = wrapper.detect_frame(frame)
    # 1 full + 4 tiles merged, then deduped to distinct tile positions.
    assert _detect_frame.calls == 5
    assert len(out) >= 1
    for d in out:
        x, y, w, h = d.box
        assert 0 <= x and x + w <= 100
        assert 0 <= y and y + h <= 100
    # Full-frame box (1,1) must be present; tile boxes shifted by origins.
    boxes = sorted(d.box for d in out)
    assert (1, 1, 10, 10) in boxes


def test_detail_no_cache_across_frames():
    import numpy as np

    inner = MagicMock()
    inner.detect_frame.return_value = [Detection("ANUS_EXPOSED", 0.9, (0, 0, 5, 5))]
    wrapper = DetailScanDetector(inner, enabled=True)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    wrapper.detect_frame(frame)
    first_calls = inner.detect_frame.call_count
    inner.detect_frame.return_value = []
    out = wrapper.detect_frame(frame)
    assert inner.detect_frame.call_count > first_calls
    assert out == []


# --- Options validation ---


def test_validate_live_options_nudenet_ok():
    validate_live_options(None, 320, False)
    validate_live_options("320n", 640, True)


def test_validate_live_options_erax_rejected():
    with pytest.raises(ValueError, match="NudeNet-only"):
        validate_live_options("erax-nano", 320, True)
    with pytest.raises(ValueError, match="NudeNet-only"):
        validate_live_options("erax-small", 640, False)


def test_cli_start_defaults_and_validation():
    from bsafe.cli import _build_parser, _validate_live_args

    parser, *_ = _build_parser()
    args = parser.parse_args(["start"])
    assert args.max_frame_age_ms == 250
    assert args.stats is False
    assert args.inference_resolution == 320
    assert args.detail_scan is False
    assert args.motion_compensation is False
    _validate_live_args(args)  # default NudeNet passes

    args = parser.parse_args(
        ["start", "--max-frame-age-ms", "100", "--stats", "--inference-resolution", "640"]
    )
    assert args.max_frame_age_ms == 100
    assert args.stats is True
    assert args.inference_resolution == 640

    args = parser.parse_args(["start", "--motion-compensation"])
    assert args.motion_compensation is True
    args = parser.parse_args(["start", "--no-motion-compensation"])
    assert args.motion_compensation is False

    bad = argparse.Namespace(
        model="erax-nano", max_frame_age_ms=250, inference_resolution=320, detail_scan=True
    )
    with pytest.raises(SystemExit):
        _validate_live_args(bad)

    bad_age = argparse.Namespace(
        model="320n", max_frame_age_ms=-5, inference_resolution=320, detail_scan=False
    )
    with pytest.raises(SystemExit):
        _validate_live_args(bad_age)


def test_live_stats_summary_survives_window_flush():
    times = [0.0]
    stats = LiveStats(time_fn=lambda: times[0], interval_s=2.0)
    stats.record(0.05, 0.10, 0.20)
    stats.record(0.05, 0.10, 0.20)
    times[0] = 2.5
    window_line = stats.maybe_log(replaced=2)
    assert window_line is not None
    assert "window" in window_line
    assert "drawn-only" in window_line
    assert "receive-to-send-start" in window_line
    # Lifetime summary stays accurate after the window flush.
    stats.record(0.05, 0.10, 0.40)
    summary = stats.summary(replaced=2)
    assert "total" in summary
    assert "n=3" in summary
    assert "stale=0" in summary
    # Lifetime avg over 3 drawn frames: (0.2+0.2+0.4)/3 = 266.7ms.
    assert "avg_receive_to_send_ms=266.7" in summary
    assert "drawn-only" in summary


def test_live_stats_stale_excluded_and_replaced_delta():
    times = [0.0]
    stats = LiveStats(time_fn=lambda: times[0], interval_s=2.0)
    stats.record(0.05, 0.10, 0.20)
    stats.record_stale()
    times[0] = 2.5
    window_line = stats.maybe_log(replaced=5)
    assert "n=2" in window_line  # totals cumulative
    assert "stale=1" in window_line
    assert "replaced=5" in window_line  # first delta equals cumulative
    assert "avg_receive_to_send_ms=200.0" in window_line  # stale excluded
    times[0] = 5.0
    stats.record(0.01, 0.01, 0.10)
    window_line2 = stats.maybe_log(replaced=8)
    assert "replaced=3" in window_line2  # window delta, not cumulative
    assert "n=3" in window_line2
    summary = stats.summary(replaced=8)
    assert "replaced=8" in summary  # totals cumulative
    assert "n=3" in summary
    assert "stale=1" in summary


def test_mailbox_peek_nonconsuming():
    box = FrameMailbox(maxsize=10)
    assert box.peek(1) is None
    box.put_nowait((_meta(1), b"v1"))
    box.put_nowait((_meta(1), b"v2"))
    replaced_before = box.replaced
    peeked = box.peek(1)
    assert peeked is not None
    assert peeked[1] == b"v2"
    # No consume/reorder/count.
    assert box.qsize() == 1
    assert box.replaced == replaced_before
    assert box.peek(1)[1] == b"v2"
    meta, jpeg, _receipt = box.get(timeout=1)
    assert jpeg == b"v2"
    assert box.peek(1) is None
