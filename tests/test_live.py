"""Live v2 session, stats, and validation tests (fakes only, no model/helper)."""

import queue
import time

import pytest

from bsafe.detector import Detection
from bsafe.live import (
    ALLOWED_DETECT_SCALES,
    LiveSession,
    LiveStats,
    bgra_view,
    validate_live_options,
)
from bsafe.protocol import DisplayInfo, RawFrameMeta


def test_validate_live_options_ok():
    validate_live_options(None, 1.0)
    validate_live_options("320n", 1.0)
    for scale in ALLOWED_DETECT_SCALES:
        validate_live_options("320n", scale)


def test_validate_live_options_bad_scale():
    with pytest.raises(ValueError, match="Allowed values"):
        validate_live_options(None, 1.3)
    with pytest.raises(ValueError, match="Allowed values"):
        validate_live_options("320n", 2.5)


def test_validate_live_options_erax():
    validate_live_options("erax-nano", 1.0)
    with pytest.raises(ValueError, match="NudeNet-only"):
        validate_live_options("erax-nano", 1.5)
    with pytest.raises(ValueError, match="NudeNet-only"):
        validate_live_options("erax-small", 2.0)


def test_live_stats_window_and_lifetime():
    times = [10.0]
    stats = LiveStats(time_fn=lambda: times[0], interval_s=2.0)
    stats.record(0.01, 0.05)
    stats.record(0.03, 0.07)
    assert stats.maybe_log() is None
    times[0] = 12.0
    line = stats.maybe_log()
    assert line is not None
    assert "live stats (window)" in line
    assert "frames=2" in line
    assert "rate=1.0/s" in line
    assert "avg_detect_ms=20.0" in line
    assert "max_detect_ms=30.0" in line
    assert "avg_receive_to_send_ms=60.0" in line
    # Window reset: immediate call suppressed.
    assert stats.maybe_log() is None
    times[0] = 13.0
    stats.record(0.02, 0.04)
    times[0] = 15.0
    summary = stats.summary()
    assert "live stats (total)" in summary
    assert "frames=3" in summary
    assert "rate=0.6/s" in summary


def test_live_stats_empty_window():
    times = [0.0]
    stats = LiveStats(time_fn=lambda: times[0], interval_s=2.0)
    assert stats.maybe_log() is None
    times[0] = 3.0
    line = stats.maybe_log()
    assert line is not None
    assert "frames=0" in line
    summary = stats.summary()
    assert "frames=0" in summary


def test_bgra_view_shape_and_zero_copy():
    import numpy as np

    w, h = 4, 3
    buf = bytearray(w * h * 4)
    for i in range(len(buf)):
        buf[i] = i % 256
    meta = RawFrameMeta(1, w, h, 0, 7)
    view = bgra_view(meta, memoryview(buf))
    assert isinstance(view, np.ndarray)
    assert view.shape == (h, w, 4)
    assert view.dtype == np.uint8
    buf[0] = 123
    assert view[0, 0, 0] == 123


class _FakeQueue:
    def __init__(self, items):
        self._items = list(items)

    def get(self, timeout=None):
        if self._items:
            return self._items.pop(0)
        raise queue.Empty


class _FakeServer:
    def __init__(self, frames):
        self.frame_queue = _FakeQueue(frames)
        self.display_events: queue.Queue = queue.Queue()
        self.sends: list = []
        self.requests: list = []
        self.calls: list = []

    def send_censor_seq(self, display_id, w, h, seq, boxes):
        self.sends.append((display_id, w, h, seq, list(boxes)))
        self.calls.append(("send", display_id, w, h, seq, list(boxes)))

    def request_frame(self, display_id):
        self.requests.append(display_id)
        self.calls.append(("request", display_id))


class _FakeNude:
    def __init__(self, detections=None, raise_=False):
        self._detections = list(detections or [])
        self._raise = raise_
        self.prepare_calls: list = []
        self.bgra_shapes: list = []
        self.provider = "coreml"

    def prepare(self, w, h):
        self.prepare_calls.append((w, h))

    def detect_bgra(self, frame):
        self.bgra_shapes.append(tuple(frame.shape))
        if self._raise:
            raise RuntimeError("boom")
        return list(self._detections)

    def close(self):
        pass


class _FakeEraX:
    def __init__(self):
        self.frames: list = []

    def detect_frame(self, frame):
        self.frames.append(frame)
        return []


def _display(display_id=1, cw=640, ch=480, pw=640, ph=480):
    return DisplayInfo(display_id, cw, ch, pw, ph)


def _session(server, detector, **kwargs):
    defaults = {
        "censor_classes": frozenset({"FEMALE_BREAST_EXPOSED"}),
        "padding": 0.0,
        "full_censor": False,
        "is_nudenet": True,
    }
    defaults.update(kwargs)
    return LiveSession(server, detector, **defaults)


def test_handle_display_info_prepare_and_credit():
    server = _FakeServer([])
    det = _FakeNude()
    printed: list = []
    sess = _session(server, det, print_fn=printed.append, time_fn=time.monotonic)
    sess.handle_display_info(_display(1, 1728, 1117, 1728, 1117))
    assert det.prepare_calls == [(1728, 1117)]
    assert server.requests == [1]
    assert any("Display 1" in p and "1728x1117" in p for p in printed)
    assert any("Detector ready" in p and "1728x1117" in p for p in printed)


def test_handle_display_info_prepare_failure_still_requests():
    server = _FakeServer([])
    printed: list = []

    class _FailPrepare:
        provider = "coreml"

        def prepare(self, w, h):
            raise RuntimeError("prepare boom")

    sess = _session(server, _FailPrepare(), print_fn=printed.append, time_fn=time.monotonic)
    sess.handle_display_info(_display(1, 640, 480, 640, 480))
    assert server.requests == [1]
    assert any("Display 1" in p for p in printed)
    assert not any("Detector ready" in p for p in printed)


def test_handle_display_info_no_prepare():
    server = _FakeServer([])
    det = _FakeEraX()
    printed: list = []
    sess = _session(server, det, print_fn=printed.append, time_fn=time.monotonic)
    sess.handle_display_info(_display(2, 800, 600, 800, 600))
    assert server.requests == [2]
    assert any("Display 2" in p for p in printed)


def test_drain_display_events():
    server = _FakeServer([])
    det = _FakeNude()
    sess = _session(server, det, print_fn=lambda *a, **k: None)
    server.display_events.put(_display(1, 640, 480, 640, 480))
    server.display_events.put(_display(2, 800, 600, 800, 600))
    assert sess.drain_display_events() == 2
    assert server.requests == [1, 2]
    assert sess.drain_display_events() == 0


def _raw_frame(display_id=1, w=640, h=480, seq=42):
    pixels = bytearray(w * h * 4)
    meta = RawFrameMeta(display_id, w, h, 0, seq)
    return meta, memoryview(pixels), 100.0


def test_step_sends_boxes_then_requests_in_order():
    meta, pixels, receipt = _raw_frame(seq=42)
    server = _FakeServer([(meta, pixels, receipt)])
    det = _FakeNude([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    sess = _session(server, det, print_fn=lambda *a, **k: None)
    assert sess.step(0.01) is True
    assert len(server.sends) == 1
    did, w, h, seq, boxes = server.sends[0]
    assert (did, w, h, seq) == (1, 640, 480, 42)
    assert boxes == [(10, 10, 20, 20)]
    assert server.requests == [1]
    assert server.calls[0][0] == "send"
    assert server.calls[1][0] == "request"


def test_step_padding_honored():
    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    det = _FakeNude([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    sess = _session(server, det, padding=0.5, print_fn=lambda *a, **k: None)
    assert sess.step(0.01) is True
    assert server.sends[0][4] == [(0, 0, 40, 40)]


def test_step_min_padding_honored():
    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    det = _FakeNude([Detection("FEMALE_BREAST_EXPOSED", 0.9, (100, 100, 20, 20))])
    sess = _session(server, det, min_padding=48, print_fn=lambda *a, **k: None)
    assert sess.step(0.01) is True
    assert server.sends[0][4] == [(52, 52, 116, 116)]


def test_step_min_padding_default_is_zero():
    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    det = _FakeNude([Detection("FEMALE_BREAST_EXPOSED", 0.9, (100, 100, 20, 20))])
    sess = _session(server, det, print_fn=lambda *a, **k: None)
    assert sess.step(0.01) is True
    assert server.sends[0][4] == [(100, 100, 20, 20)]


def test_step_full_censor_honored():
    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    det = _FakeNude([Detection("FEMALE_BREAST_EXPOSED", 0.9, (100, 100, 20, 20))])
    sess = _session(server, det, full_censor=True, print_fn=lambda *a, **k: None)
    assert sess.step(0.01) is True
    assert server.sends[0][4] == [(80, 80, 60, 60)]


def test_step_zero_detections_sends_empty():
    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    det = _FakeNude([])
    sess = _session(server, det, print_fn=lambda *a, **k: None)
    assert sess.step(0.01) is True
    assert server.sends[0][4] == []
    assert server.requests == [1]


def test_step_detector_error_sends_empty_and_requests_next():
    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    det = _FakeNude(raise_=True)
    sess = _session(server, det, print_fn=lambda *a, **k: None)
    assert sess.step(0.01) is True
    assert server.sends[0][4] == []
    assert server.requests == [1]


def test_step_erax_gets_bgr():
    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    det = _FakeEraX()
    sess = _session(server, det, is_nudenet=False, print_fn=lambda *a, **k: None)
    assert sess.step(0.01) is True
    assert len(det.frames) == 1
    assert det.frames[0].shape == (480, 640, 3)
    assert server.sends[0][4] == []
    assert server.requests == [1]


def test_step_empty_returns_false_and_sends_nothing():
    server = _FakeServer([])
    det = _FakeNude([])
    sess = _session(server, det, print_fn=lambda *a, **k: None)
    assert sess.step(0.01) is False
    assert server.sends == []
    assert server.requests == []


def test_parse_extra_scales():
    from bsafe.live import parse_extra_scales

    assert parse_extra_scales(None) is None
    assert parse_extra_scales("none") == ()
    assert parse_extra_scales("") == ()
    assert parse_extra_scales("0.5") == (0.5,)
    assert parse_extra_scales("0.5,0.75") == (0.5, 0.75)
    assert parse_extra_scales(0.5) == (0.5,)
    assert parse_extra_scales([0.5, 0.75]) == (0.5, 0.75)


def test_validate_extra_scales_rejects():
    from bsafe.live import validate_extra_scales

    with pytest.raises(ValueError):
        validate_extra_scales((1.0,), 1.0)
    with pytest.raises(ValueError):
        validate_extra_scales((0.0,), 1.0)
    with pytest.raises(ValueError):
        validate_extra_scales((-0.5,), 1.0)
    with pytest.raises(ValueError):
        validate_extra_scales((0.5, 0.5), 1.0)
    with pytest.raises(ValueError):
        validate_extra_scales((0.1, 0.2, 0.3, 0.4), 1.0)
    validate_extra_scales((0.5,), 1.0)
    validate_extra_scales((), 1.0)


def test_resolve_extra_scales_auto():
    from bsafe.live import resolve_extra_scales

    assert resolve_extra_scales(None, 1.0, None) == (0.5,)
    assert resolve_extra_scales("320n", 1.0, None) == (0.5,)
    assert resolve_extra_scales("erax-nano", 1.0, None) == ()


def test_step_uses_multiscale_when_factors_set(monkeypatch):
    import bsafe.fastdetect as fast_mod

    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    det = _FakeNude([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    seen = {}

    def _fake_multiscale(detector, frame, factors):
        seen["factors"] = tuple(factors)
        seen["shape"] = tuple(frame.shape)
        assert detector is det
        return [Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))]

    monkeypatch.setattr(fast_mod, "detect_multiscale", _fake_multiscale)
    sess = _session(server, det, print_fn=lambda *a, **k: None, extra_factors=(0.5,))
    assert sess.step(0.01) is True
    assert seen["factors"] == (0.5,)
    assert seen["shape"] == (480, 640, 4)
    assert server.sends[0][4] == [(10, 10, 20, 20)]


def test_step_without_factors_calls_detect_bgra():
    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    det = _FakeNude([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    sess = _session(server, det, print_fn=lambda *a, **k: None)
    assert sess.step(0.01) is True
    assert det.bgra_shapes == [(480, 640, 4)]


def test_handle_display_info_warms_extra_shapes():
    from bsafe.fastdetect import extra_size

    server = _FakeServer([])
    det = _FakeNude()
    sess = _session(
        server, det, print_fn=lambda *a, **k: None, extra_factors=(0.5,), time_fn=time.monotonic
    )
    sess.handle_display_info(_display(1, 200, 100, 200, 100))
    assert det.prepare_calls[0] == (200, 100)
    assert det.prepare_calls[1:] == [extra_size(200, 100, 0.5)]
    assert server.requests == [1]


def test_validate_live_options_erax_parses_extra_scales():
    # Unparseable input fails even for EraX (before spawn).
    with pytest.raises(ValueError, match="extra-scales"):
        validate_live_options("erax-nano", 1.0, "bogus")
    with pytest.raises(ValueError, match="extra-scales"):
        validate_live_options("erax-nano", 1.0, "0.5,,0.25")
    # Successfully parsed values are ignored for EraX.
    validate_live_options("erax-nano", 1.0, "0.5")
    validate_live_options("erax-nano", 1.0, "none")
    validate_live_options("erax-nano", 1.0, None)


def _textured_bgra(w=64, h=48, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)


def _textured_frame(display_id=1, w=64, h=48, seq=1, seed=7, shift_px=0):
    import numpy as np

    arr = _textured_bgra(w, h, seed)
    if shift_px:
        arr = np.roll(arr, shift_px, axis=1)
    meta = RawFrameMeta(display_id, w, h, 0, seq)
    return meta, memoryview(arr.tobytes()), 100.0


class _FakeStill:
    def __init__(self, detections=None, raise_=False):
        self._detections = list(detections or [])
        self._raise = raise_
        self.calls = 0
        self.prepare_calls: list = []
        self.shapes: list = []

    def prepare(self, w, h):
        self.prepare_calls.append((w, h))

    def detect_bgra(self, frame):
        self.calls += 1
        self.shapes.append(frame.shape[:2])
        if self._raise:
            raise RuntimeError("still boom")
        return list(self._detections)

    def close(self):
        pass


def test_stillness_tracker_first_none_then_still_grows():
    from bsafe.live import StillnessTracker

    tracker = StillnessTracker()
    frame = _textured_bgra()
    assert tracker.update(1, frame, 10.0) is None
    assert tracker.update(2, frame, 10.0) is None
    first = tracker.update(1, frame, 10.2)
    assert first is not None and abs(first) < 0.05
    assert tracker.update(1, frame, 11.0) == pytest.approx(0.8, abs=0.05)


def test_stillness_tracker_shift_resets():
    import numpy as np

    from bsafe.live import StillnessTracker

    tracker = StillnessTracker()
    frame = _textured_bgra(seed=3)
    assert tracker.update(1, frame, 0.0) is None
    first = tracker.update(1, frame, 0.5)
    assert first is not None
    moved = np.roll(frame, 8, axis=1)
    assert tracker.update(1, moved, 1.0) is None
    assert tracker.update(1, moved, 1.5) == pytest.approx(0.0, abs=0.05)
    assert tracker.update(1, moved, 2.0) == pytest.approx(0.5, abs=0.05)


def test_stillness_tracker_shape_change_resets():
    from bsafe.live import StillnessTracker

    tracker = StillnessTracker()
    small = _textured_bgra(w=64, h=48, seed=3)
    big = _textured_bgra(w=80, h=48, seed=3)
    assert tracker.update(1, small, 0.0) is None
    assert tracker.update(1, small, 0.5) is not None
    assert tracker.update(1, big, 1.0) is None
    assert tracker.update(1, big, 1.5) == pytest.approx(0.0, abs=0.05)


def test_step_still_runs_when_still_and_rate_limited():
    now = [100.0]
    frames = [_textured_frame(seq=i) for i in range(5)]
    server = _FakeServer(frames)
    fast = _FakeNude([])
    still = _FakeStill([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    sess = _session(
        server,
        fast,
        still_detector=still,
        still_period_s=2.0,
        still_after_s=0.5,
        print_fn=lambda *a, **k: None,
        time_fn=lambda: now[0],
    )
    assert sess.step(0.01) is True
    assert still.calls == 0
    now[0] = 100.2
    assert sess.step(0.01) is True
    assert still.calls == 0
    now[0] = 100.8
    assert sess.step(0.01) is True
    assert still.calls == 1
    assert server.sends[-1][4] == [(10, 10, 20, 20)]
    now[0] = 101.0
    assert sess.step(0.01) is True
    assert still.calls == 1
    assert server.sends[-1][4] == []
    now[0] = 103.0
    assert sess.step(0.01) is True
    assert still.calls == 2
    assert server.sends[-1][4] == [(10, 10, 20, 20)]


def test_step_still_skipped_when_moving():
    now = [50.0]
    frames = [_textured_frame(seq=i, shift_px=i * 8) for i in range(4)]
    server = _FakeServer(frames)
    still = _FakeStill([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    sess = _session(
        server,
        _FakeNude([]),
        still_detector=still,
        print_fn=lambda *a, **k: None,
        time_fn=lambda: now[0],
    )
    for _ in range(4):
        now[0] += 1.0
        assert sess.step(0.01) is True
    assert still.calls == 0
    assert all(send[4] == [] for send in server.sends)


def test_step_still_skipped_on_erax():
    now = [50.0]
    frames = [_textured_frame(seq=i) for i in range(3)]
    server = _FakeServer(frames)
    still = _FakeStill([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    sess = _session(
        server,
        _FakeEraX(),
        is_nudenet=False,
        still_detector=still,
        print_fn=lambda *a, **k: None,
        time_fn=lambda: now[0],
    )
    for _ in range(3):
        now[0] += 1.0
        assert sess.step(0.01) is True
    assert still.calls == 0


def test_step_still_exception_keeps_fast_detections():
    now = [0.0]
    frames = [_textured_frame(seq=i) for i in range(3)]
    server = _FakeServer(frames)
    fast = _FakeNude([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    still = _FakeStill(raise_=True)
    sess = _session(
        server,
        fast,
        still_detector=still,
        print_fn=lambda *a, **k: None,
        time_fn=lambda: now[0],
    )
    assert sess.step(0.01) is True
    now[0] = 0.1
    assert sess.step(0.01) is True
    now[0] = 1.0
    assert sess.step(0.01) is True
    assert still.calls == 1
    assert server.sends[-1][4] == [(10, 10, 20, 20)]


def test_step_still_tracker_exception_keeps_fast_detections():
    now = [0.0]
    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    fast = _FakeNude([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    still = _FakeStill([Detection("FEMALE_BREAST_EXPOSED", 0.9, (30, 30, 10, 10))])
    sess = _session(
        server,
        fast,
        still_detector=still,
        print_fn=lambda *a, **k: None,
        time_fn=lambda: now[0],
    )

    class _BoomTracker:
        def __init__(self):
            self.reset_calls: list = []

        def update(self, display_id, frame, now):
            raise RuntimeError("tracker boom")

        def reset(self, display_id):
            self.reset_calls.append(display_id)

    boom = _BoomTracker()
    sess._still_tracker = boom
    assert sess.step(0.01) is True
    assert server.sends[0][4] == [(10, 10, 20, 20)]
    assert boom.reset_calls == [1]
    assert still.calls == 0


def test_step_without_still_detector_uses_no_motion_estimation(monkeypatch):
    import cv2

    import bsafe.live as live_mod

    def _boom(*args, **kwargs):
        raise AssertionError("phaseCorrelate must not be called")

    def _boom_tracker(*args, **kwargs):
        raise AssertionError("StillnessTracker must not be constructed")

    monkeypatch.setattr(cv2, "phaseCorrelate", _boom)
    monkeypatch.setattr(live_mod, "StillnessTracker", _boom_tracker)
    meta, pixels, receipt = _raw_frame()
    server = _FakeServer([(meta, pixels, receipt)])
    sess = _session(
        server,
        _FakeNude([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))]),
        print_fn=lambda *a, **k: None,
    )
    assert sess.step(0.01) is True
    assert server.sends[0][4] == [(10, 10, 20, 20)]


def test_handle_display_info_prepares_still_detector():
    server = _FakeServer([])
    det = _FakeNude()
    still = _FakeStill()
    sess = _session(
        server,
        det,
        still_detector=still,
        print_fn=lambda *a, **k: None,
        extra_factors=(0.5,),
        time_fn=time.monotonic,
    )
    sess.handle_display_info(_display(1, 200, 100, 200, 100))
    assert still.prepare_calls == [(200, 100)]
    assert det.prepare_calls[0] == (200, 100)
    assert server.requests == [1]


def test_handle_display_info_resets_stillness():
    now = [10.0]
    server = _FakeServer([_textured_frame(seq=i) for i in range(3)])
    still = _FakeStill([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    sess = _session(
        server,
        _FakeNude([]),
        still_detector=still,
        still_after_s=0.5,
        print_fn=lambda *a, **k: None,
        time_fn=lambda: now[0],
    )
    assert sess.step(0.01) is True
    now[0] = 10.1
    assert sess.step(0.01) is True
    assert still.calls == 0
    sess.handle_display_info(_display(1, 64, 48, 64, 48))
    now[0] = 11.0
    assert sess.step(0.01) is True
    assert still.calls == 0


def test_step_still_cycles_scales_and_maps_boxes_back():
    now = [100.0]
    server = _FakeServer([_textured_frame(seq=i) for i in range(8)])
    still = _FakeStill([Detection("FEMALE_BREAST_EXPOSED", 0.9, (10, 10, 20, 20))])
    sess = _session(
        server,
        _FakeNude([]),
        still_detector=still,
        still_period_s=1.0,
        still_after_s=0.5,
        still_scales=(1.0, 1.5),
        print_fn=lambda *a, **k: None,
        time_fn=lambda: now[0],
    )
    sess.step(0.01)
    now[0] = 100.2
    sess.step(0.01)
    now[0] = 100.8
    sess.step(0.01)
    assert still.shapes == [(48, 64)]
    assert server.sends[-1][4] == [(10, 10, 20, 20)]
    now[0] = 101.9
    sess.step(0.01)
    assert still.shapes == [(48, 64), (72, 96)]
    assert server.sends[-1][4] == [(7, 7, 13, 13)]
    now[0] = 103.0
    sess.step(0.01)
    assert still.shapes[-1] == (48, 64)


def test_handle_display_info_prepares_every_still_scale():
    still = _FakeStill()
    sess = _session(
        _FakeServer([]),
        _FakeNude(),
        still_detector=still,
        still_scales=(1.0, 1.5),
        print_fn=lambda *a, **k: None,
        time_fn=time.monotonic,
    )
    sess.handle_display_info(_display(1, 200, 100, 200, 100))
    assert still.prepare_calls == [(200, 100), (300, 150)]


def test_parse_still_scales():
    from bsafe.live import parse_still_scales

    assert parse_still_scales(None) == (1.0,)
    assert parse_still_scales("1.0,1.5") == (1.0, 1.5)
    assert parse_still_scales(1.5) == (1.5,)
    assert parse_still_scales([1, 0.5]) == (1.0, 0.5)
    for bad in ("", "a", "1.0,1.0", "0.1", "2.5", "nan", "1,1.25,1.5,2", "1,,2"):
        with pytest.raises(ValueError):
            parse_still_scales(bad)
