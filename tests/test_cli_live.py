"""cmd_start loop tests with mocked detector/server/helper (no network/model)."""

from unittest.mock import MagicMock, patch

from bsafe.cli import _build_parser, cmd_start
from bsafe.detector import Detection
from bsafe.protocol import FrameMetadata


def _args(extra=None):
    parser, *_ = _build_parser()
    argv = ["start", "--persist-frames", "8", "--smooth-alpha", "1.0"]
    argv += extra or []
    args = parser.parse_args(argv)
    return args


class _Clock:
    def __init__(self, start=100.0):
        self.t = start

    def mono(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _meta(display_id=1, w=640, h=480):
    return FrameMetadata(display_id, w, h, 0)


class _FakeQueue:
    """Fake mailbox: get() dequeues preset frames, peek() returns pending."""

    def __init__(self, frames, pending_by_display=None):
        self._frames = list(frames)
        self._pending = dict(pending_by_display or {})
        self.replaced = 0
        self.peek_calls = []

    def get(self, timeout=None):
        if self._frames:
            return self._frames.pop(0)
        raise KeyboardInterrupt

    def peek(self, display_id):
        self.peek_calls.append(display_id)
        return self._pending.get(display_id)


class _FakeServer:
    def __init__(self, fake_queue):
        self.frame_queue = fake_queue
        self.sends = []

    def start(self):
        pass

    def send_censor(self, display_id, w, h, boxes):
        self.sends.append((display_id, w, h, list(boxes)))

    def shutdown(self):
        pass


class _FakeHelper:
    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


def _run(
    args,
    frames,
    pending=None,
    detections=None,
    motion_fn=None,
    clock=None,
    stats_interval=2.0,
    det_advance=0.02,
    motion_advance=0.01,
):
    """Run cmd_start with fakes; returns (sends, server, mocks)."""
    from bsafe import live as live_mod

    clock = clock or _Clock()
    det_list = detections if detections is not None else []
    fake_queue = _FakeQueue(frames, pending)
    server = _FakeServer(fake_queue)
    helper = _FakeHelper()
    real_stats = live_mod.LiveStats

    def _make_detector(*a, **k):
        m = MagicMock()

        def _detect(jpeg):
            clock.advance(det_advance)
            if callable(det_list):
                return list(det_list())
            return list(det_list)

        m.detect.side_effect = _detect
        m.close.return_value = None
        return m

    def _make_stats(*a, **k):
        return real_stats(time_fn=clock.mono, interval_s=stats_interval)

    patches = [
        patch("bsafe.detector.Detector", side_effect=_make_detector),
        patch("bsafe.ipc.FrameServer", return_value=server),
        patch("bsafe.swift_helper.spawn_helper", return_value=helper),
        patch("bsafe.live.LiveStats", side_effect=_make_stats),
        patch("time.monotonic", side_effect=lambda: clock.mono()),
    ]
    if motion_fn is not None:

        def _motion(ref_jpeg, cur_jpeg, boxes, w, h, *a, **k):
            clock.advance(motion_advance)
            return motion_fn(ref_jpeg, cur_jpeg, boxes, w, h, *a, **k)

        patches.append(patch("bsafe.motion.compensate_boxes", side_effect=_motion))
    for p in patches:
        p.start()
    try:
        try:
            cmd_start(args)
        except KeyboardInterrupt:
            pass  # cmd_start handles it internally; safety net
    finally:
        for p in patches:
            p.stop()
    return server.sends, server, fake_queue


def _det(box=(10, 10, 20, 20)):
    return [Detection("ANUS_EXPOSED", 0.9, box)]


def test_fresh_with_newer_pending_sends_compensated_and_tracker_stays_inference():
    from bsafe import tracking as tracking_mod

    clock = _Clock(100.0)
    receipt = clock.t
    frames = [(_meta(1), b"ref", receipt)]
    pending = {1: (_meta(1), b"cur", receipt + 0.01)}
    args = _args(["--motion-compensation", "--max-frame-age-ms", "250"])
    real_tracker = tracking_mod.BoxTracker
    updates, clears = [], []

    class SpyTracker(real_tracker):
        def update(self, display_id, boxes):
            updates.append((display_id, list(boxes)))
            return super().update(display_id, boxes)

        def clear(self, display_id):
            clears.append(display_id)
            return super().clear(display_id)

    with (
        patch("bsafe.tracking.BoxTracker", SpyTracker),
        patch(
            "bsafe.motion.compensate_boxes",
            side_effect=lambda r, c, b, w, h, *a, **k: (
                clock.advance(0.01),
                [(30, 30, 20, 20)],
            )[1],
        ) as motion_mock,
    ):
        sends, _, fq = _run(
            args,
            frames,
            pending=pending,
            detections=_det(),
            clock=clock,
            det_advance=0.02,
        )
    assert len(sends) == 1
    assert sends[0][3] == [(30, 30, 20, 20)]
    # Tracker got inference coords, not projected; no feedback, no clear.
    assert updates and updates[0][1] == [(10, 10, 20, 20)]
    assert clears == []
    assert motion_mock.called
    # Peek must not consume the pending frame.
    assert fq.peek(1) is not None
    assert fq.peek(1)[1] == b"cur"


def test_stale_sends_empty_and_clears_tracker():
    from bsafe import tracking as tracking_mod

    clock = _Clock(50.0)
    receipt = clock.t
    frames = [(_meta(1), b"jpeg", receipt)]
    args = _args(["--max-frame-age-ms", "250"])
    real_tracker = tracking_mod.BoxTracker
    clears = []

    class SpyTracker(real_tracker):
        def clear(self, display_id):
            clears.append(display_id)
            return super().clear(display_id)

    with patch("bsafe.tracking.BoxTracker", SpyTracker):
        sends, _, _ = _run(args, frames, detections=_det(), clock=clock, det_advance=0.5)
    assert len(sends) == 1
    assert sends[0][3] == []  # stale sends [], not detections
    assert clears == [1]


def test_stale_clears_prior_tracks():
    clock = _Clock(200.0)
    r0 = clock.t
    # Fresh frame populates tracker, stale frame clears, fresh empty stays empty.
    frames = [(_meta(1), b"f0", r0), (_meta(1), b"f1", r0), (_meta(1), b"f2", r0 + 0.5)]
    args = _args(["--max-frame-age-ms", "250"])
    calls = {"n": 0}
    # Clock: small advance for fresh frames, large for the stale one.
    advances = [0.02, 0.5, 0.02]

    from bsafe import live as live_mod

    real_stats = live_mod.LiveStats
    fake_queue = _FakeQueue(frames)
    server = _FakeServer(fake_queue)
    helper = _FakeHelper()

    def _make_detector(*a, **k):
        m = MagicMock()

        def _detect(jpeg):
            idx = min(calls["n"], 2)
            clock.advance(advances[idx])
            calls["n"] += 1
            if calls["n"] <= 2:
                return _det()
            return []

        m.detect.side_effect = _detect
        return m

    with (
        patch("bsafe.detector.Detector", side_effect=_make_detector),
        patch("bsafe.ipc.FrameServer", return_value=server),
        patch("bsafe.swift_helper.spawn_helper", return_value=helper),
        patch(
            "bsafe.live.LiveStats",
            side_effect=lambda *a, **k: real_stats(time_fn=clock.mono),
        ),
        patch("time.monotonic", side_effect=lambda: clock.mono()),
    ):
        try:
            cmd_start(args)
        except KeyboardInterrupt:
            pass
    assert len(server.sends) == 3
    assert server.sends[0][3] == [(10, 10, 20, 20)]
    assert server.sends[1][3] == []  # stale
    # Tracker was cleared, so no persistence into the final fresh frame.
    assert server.sends[2][3] == []


def test_all_stale_interval_stats_emitted(capsys):
    clock = _Clock(0.0)
    frames = [(_meta(1), b"a", 0.0), (_meta(1), b"b", 0.0), (_meta(1), b"c", 0.0)]
    args = _args(["--stats", "--max-frame-age-ms", "250"])
    sends, _, _ = _run(args, frames, detections=_det(), clock=clock, det_advance=1.2)
    assert sends == [(1, 640, 480, []), (1, 640, 480, []), (1, 640, 480, [])]
    out = capsys.readouterr().out
    assert "live stats" in out
    assert "window" in out


def test_summary_after_flush_lifetime_correct(capsys):
    clock = _Clock(100.0)
    r = clock.t
    frames = [(_meta(1), b"f0", r), (_meta(1), b"f1", r), (_meta(1), b"f2", r)]
    args = _args(["--stats", "--max-frame-age-ms", "0"])
    sends, _, _ = _run(
        args,
        frames,
        detections=_det(),
        clock=clock,
        det_advance=0.02,
        stats_interval=0.05,
    )
    assert len(sends) == 3
    out = capsys.readouterr().out
    assert "live stats" in out
    # Shutdown summary is cumulative even after interval flushes.
    assert "n=3" in out
    assert "total" in out


def test_motion_dim_mismatch_sends_original():
    clock = _Clock(100.0)
    receipt = clock.t
    frames = [(_meta(1, 640, 480), b"ref", receipt)]
    pending = {1: (_meta(1, 800, 600), b"cur", receipt + 0.01)}
    args = _args(["--motion-compensation", "--max-frame-age-ms", "250"])
    with patch(
        "bsafe.motion.compensate_boxes",
        side_effect=AssertionError("must not compensate"),
    ):
        sends, _, _ = _run(args, frames, pending=pending, detections=_det(), clock=clock)
    assert sends[0][3] == [(10, 10, 20, 20)]


def test_motion_default_disabled_no_compensation():
    parser, *_ = _build_parser()
    assert parser.parse_args(["start"]).motion_compensation is False
    clock = _Clock(100.0)
    receipt = clock.t
    frames = [(_meta(1), b"ref", receipt)]
    pending = {1: (_meta(1), b"cur", receipt + 0.01)}
    args = _args([])  # default disabled
    with patch(
        "bsafe.motion.compensate_boxes",
        side_effect=AssertionError("must not compensate"),
    ):
        sends, _, fq = _run(args, frames, pending=pending, detections=_det(), clock=clock)
    assert sends[0][3] == [(10, 10, 20, 20)]
    assert fq.peek_calls == []  # never even peeks when disabled


def test_final_deadline_after_motion_stale_sends_empty():
    clock = _Clock(100.0)
    receipt = clock.t
    frames = [(_meta(1), b"ref", receipt)]
    pending = {1: (_meta(1), b"cur", receipt + 0.01)}
    args = _args(["--motion-compensation", "--max-frame-age-ms", "250"])

    def _motion(r, c, boxes, w, h, *a, **k):
        clock.advance(0.5)  # compensation pushes send-start past deadline
        return [(99, 99, 20, 20)]

    with patch("bsafe.motion.compensate_boxes", side_effect=_motion):
        sends, _, _ = _run(
            args,
            frames,
            pending=pending,
            detections=_det(),
            clock=clock,
            det_advance=0.02,
            motion_advance=0.0,
        )
    assert sends[0][3] == []


def test_older_pending_not_used():
    clock = _Clock(100.0)
    receipt = clock.t
    frames = [(_meta(1), b"ref", receipt)]
    pending = {1: (_meta(1), b"cur", receipt - 0.05)}  # older, not newer
    args = _args(["--motion-compensation", "--max-frame-age-ms", "250"])
    with patch(
        "bsafe.motion.compensate_boxes",
        side_effect=AssertionError("must not compensate"),
    ):
        sends, _, _ = _run(args, frames, pending=pending, detections=_det(), clock=clock)
    assert sends[0][3] == [(10, 10, 20, 20)]


def test_fresh_not_dropped_when_newer_frame_exists():
    # Fresh inference is sent (compensated) even though a newer frame exists.
    clock = _Clock(300.0)
    receipt = clock.t
    frames = [(_meta(1), b"ref", receipt)]
    pending = {1: (_meta(1), b"cur", receipt + 0.01)}
    args = _args(["--motion-compensation", "--max-frame-age-ms", "250"])
    with patch("bsafe.motion.compensate_boxes", return_value=[(11, 11, 20, 20)]) as m:
        sends, _, _ = _run(args, frames, pending=pending, detections=_det(), clock=clock)
    assert m.called
    assert sends[0][3] == [(11, 11, 20, 20)]


def test_lookahead_default_off():
    parser, *_ = _build_parser()
    args = parser.parse_args(["start"])
    assert args.motion_lookahead_ms == 0
    assert args.motion_compensation is False


def test_lookahead_ratio_derived_from_interval():
    clock = _Clock(100.0)
    receipt = clock.t
    frames = [(_meta(1), b"ref", receipt)]
    pending = {1: (_meta(1), b"cur", receipt + 0.01)}
    args = _args(
        ["--motion-compensation", "--motion-lookahead-ms", "10", "--max-frame-age-ms", "250"]
    )
    captured = {}

    def _cap(r, c, boxes, w, h, *a, **k):
        captured["ratio"] = k.get("lead_ratio")
        return [(30, 30, 20, 20)]

    with patch("bsafe.motion.compensate_boxes", side_effect=_cap):
        sends, _, _ = _run(
            args, frames, pending=pending, detections=_det(), clock=clock, det_advance=0.02
        )
    assert sends[0][3] == [(30, 30, 20, 20)]
    # 10ms / 10ms interval = 1.0
    assert captured.get("ratio") is not None
    assert abs(captured["ratio"] - 1.0) < 1e-6


def test_lookahead_ratio_clamped_and_config_value_used():
    clock = _Clock(100.0)
    receipt = clock.t
    frames = [(_meta(1), b"ref", receipt)]
    pending = {1: (_meta(1), b"cur", receipt + 0.01)}
    args = _args(["--motion-compensation", "--max-frame-age-ms", "250"])
    args.motion_lookahead_ms = 100.0  # as if from config file
    captured = {}

    def _cap(r, c, boxes, w, h, *a, **k):
        captured["ratio"] = k.get("lead_ratio")
        return [(30, 30, 20, 20)]

    with patch("bsafe.motion.compensate_boxes", side_effect=_cap):
        sends, _, _ = _run(
            args, frames, pending=pending, detections=_det(), clock=clock, det_advance=0.02
        )
    assert sends[0][3] == [(30, 30, 20, 20)]
    # 100ms / 10ms = 10 -> clamped to 2.0
    assert abs(captured["ratio"] - 2.0) < 1e-6


def test_lookahead_zero_passes_zero_ratio():
    clock = _Clock(100.0)
    receipt = clock.t
    frames = [(_meta(1), b"ref", receipt)]
    pending = {1: (_meta(1), b"cur", receipt + 0.02)}
    args = _args(["--motion-compensation", "--max-frame-age-ms", "250"])
    captured = {}

    def _cap(r, c, boxes, w, h, *a, **k):
        captured["ratio"] = k.get("lead_ratio")
        return [(30, 30, 20, 20)]

    with patch("bsafe.motion.compensate_boxes", side_effect=_cap):
        _run(args, frames, pending=pending, detections=_det(), clock=clock, det_advance=0.02)
    assert abs(captured.get("ratio", 99.0) - 0.0) < 1e-9


def test_lookahead_requires_compensation():
    import pytest

    from bsafe.cli import _validate_live_args

    args = _args(["--motion-lookahead-ms", "10"])
    with pytest.raises(SystemExit):
        _validate_live_args(args)


def test_lookahead_rejects_out_of_range():
    import pytest

    from bsafe.cli import _validate_live_args

    for bad in [
        ["--motion-compensation", "--motion-lookahead-ms", "-5"],
        ["--motion-compensation", "--motion-lookahead-ms", "150"],
    ]:
        args = _args(bad)
        with pytest.raises(SystemExit):
            _validate_live_args(args)
    args = _args(["--motion-compensation"])
    args.motion_lookahead_ms = float("nan")
    with pytest.raises(SystemExit):
        _validate_live_args(args)


def test_lookahead_invalid_rejected_before_helper():
    import pytest

    args = _args(["--motion-lookahead-ms", "10", "--max-frame-age-ms", "250"])
    with (
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.detector.Detector") as det_mock,
    ):
        with pytest.raises(SystemExit):
            cmd_start(args)
    helper_mock.assert_not_called()
    det_mock.assert_not_called()


def test_lookahead_logged_in_config(capsys):
    args = _args(
        ["--motion-compensation", "--motion-lookahead-ms", "12.5", "--max-frame-age-ms", "250"]
    )
    from bsafe.cli import _print_config

    _print_config(args)
    out = capsys.readouterr().out
    assert "motion_lookahead_ms=12.5" in out
