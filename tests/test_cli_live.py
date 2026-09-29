"""cmd_start live v2 wiring tests (fakes only, no model/network/helper)."""

import logging
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from bsafe.cli import _build_parser, _print_config, _validate_live_args, cmd_start


def _args(extra=None):
    parser, *_ = _build_parser()
    argv = ["start"]
    argv += extra or []
    return parser.parse_args(argv)


def test_parser_defaults():
    args = _args([])
    assert args.fps == 60
    assert args.detect_scale == 1.0
    assert args.stats is False
    assert args.min_padding == 0
    assert args.extra_scales is None


def test_parser_min_padding():
    args = _args(["--min-padding", "48"])
    assert args.min_padding == 48


def test_min_padding_negative_rejected_before_helper():
    args = _args(["--min-padding", "-1"])
    with (
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_mock,
        patch("bsafe.ipc.FrameServer") as server_mock,
    ):
        with pytest.raises(SystemExit):
            cmd_start(args)
    helper_mock.assert_not_called()
    det_mock.assert_not_called()
    server_mock.assert_not_called()


def test_min_padding_negative_error_message(capsys):
    from bsafe.cli import _validate_censor_args

    args = _args(["--min-padding", "-5"])
    with pytest.raises(SystemExit):
        _validate_censor_args(args)
    assert "--min-padding must be >= 0" in capsys.readouterr().err


def test_print_config_includes_min_padding(capsys):
    args = _args(["--min-padding", "48"])
    args.confidence = 0.0
    _print_config(args)
    assert "min_padding=48" in capsys.readouterr().out


def test_cmd_start_passes_min_padding_to_session():
    args = _args(["--min-padding", "48"])
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server_cls.return_value = MagicMock()
        helper = MagicMock()
        helper.poll.side_effect = [None, 0]
        helper.stderr = None
        helper_mock.return_value = helper
        det_cls.return_value = MagicMock()
        session = MagicMock()
        session_cls.return_value = session
        session.step.side_effect = KeyboardInterrupt
        cmd_start(args)
        _, kwargs = session_cls.call_args
        assert kwargs["min_padding"] == 48


def test_removed_flags_rejected():
    parser, *_ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["start", "--max-frame-age-ms", "250"])
    with pytest.raises(SystemExit):
        parser.parse_args(["start", "--inference-resolution", "320"])
    with pytest.raises(SystemExit):
        parser.parse_args(["start", "--detail-scan"])
    with pytest.raises(SystemExit):
        parser.parse_args(["start", "--motion-compensation"])
    with pytest.raises(SystemExit):
        parser.parse_args(["start", "--motion-lookahead-ms", "10"])


def test_detect_scale_invalid_rejected_before_helper():
    args = _args(["--detect-scale", "1.3"])
    with (
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_mock,
        patch("bsafe.ipc.FrameServer") as server_mock,
    ):
        with pytest.raises(SystemExit):
            cmd_start(args)
    helper_mock.assert_not_called()
    det_mock.assert_not_called()
    server_mock.assert_not_called()


def test_detect_scale_error_lists_allowed(capsys):
    args = _args(["--detect-scale", "1.3"])
    with pytest.raises(SystemExit):
        _validate_live_args(args)
    err = capsys.readouterr().err
    assert "1.0" in err and "1.25" in err and "1.5" in err and "2.0" in err


def test_erax_detect_scale_rejected_before_helper():
    args = _args(["--model", "erax-nano", "--detect-scale", "1.5"])
    with (
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.detector.Detector") as det_mock,
    ):
        with pytest.raises(SystemExit):
            cmd_start(args)
    helper_mock.assert_not_called()
    det_mock.assert_not_called()


def test_erax_scale_one_ok_validation():
    args = _args(["--model", "erax-nano", "--detect-scale", "1.0"])
    _validate_live_args(args)


def test_print_config_includes_detect_scale(capsys):
    args = _args(["--detect-scale", "1.5"])
    args.confidence = 0.0
    _print_config(args)
    out = capsys.readouterr().out
    assert "detect_scale=1.5" in out
    assert "max_frame_age_ms" not in out
    assert "inference_resolution" not in out
    assert "detail_scan" not in out
    assert "motion_compensation" not in out
    assert "motion_lookahead_ms" not in out


def test_cmd_start_wiring_scale_persist_smooth_and_shutdown(capsys):
    args = _args(
        ["--detect-scale", "1.5", "--persist-frames", "300", "--smooth-alpha", "0.7", "--stats"]
    )
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server = MagicMock()
        server_cls.return_value = server
        helper = MagicMock()
        helper.poll.side_effect = [None, 0]
        helper.stderr = None
        helper_mock.return_value = helper
        detector = MagicMock()
        det_cls.return_value = detector
        session = MagicMock()
        session_cls.return_value = session
        session.step.side_effect = KeyboardInterrupt
        cmd_start(args)
        _, kwargs = server_cls.call_args
        assert kwargs["scale_percent"] == 150
        assert kwargs["persist_passes"] == 255
        assert kwargs["smooth_percent"] == 70
        assert kwargs["stats"] is True
        assert kwargs["fps"] == 60
        assert callable(kwargs["on_stats"])
        server.shutdown.assert_called_once()
        detector.close.assert_called_once()
        out = capsys.readouterr().out
        assert "live stats (total)" in out


def test_cmd_start_erax_uses_detector():
    args = _args(["--model", "erax-nano"])
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.detector.Detector") as det_cls,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as nude_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server = MagicMock()
        server_cls.return_value = server
        helper = MagicMock()
        helper.poll.side_effect = [None, 0]
        helper.stderr = None
        helper_mock.return_value = helper
        detector = MagicMock()
        det_cls.return_value = detector
        session = MagicMock()
        session_cls.return_value = session
        session.step.side_effect = KeyboardInterrupt
        cmd_start(args)
        det_cls.assert_called_once()
        nude_cls.assert_not_called()
        _, kwargs = server_cls.call_args
        assert kwargs["scale_percent"] == 100
        assert kwargs["on_stats"] is None


def test_parser_extra_scales_values():
    assert _args(["--extra-scales", "0.5"]).extra_scales == "0.5"
    assert _args(["--extra-scales", "0.5,0.75"]).extra_scales == "0.5,0.75"
    assert _args(["--extra-scales", "none"]).extra_scales == "none"


def test_extra_scales_reject_gte_detect_scale():
    args = _args(["--extra-scales", "1.0"])
    with pytest.raises(SystemExit):
        _validate_live_args(args)
    args = _args(["--extra-scales", "1.5", "--detect-scale", "1.5"])
    with pytest.raises(SystemExit):
        _validate_live_args(args)


def test_extra_scales_reject_nonpositive():
    with pytest.raises(SystemExit):
        _validate_live_args(_args(["--extra-scales", "0"]))
    with pytest.raises(SystemExit):
        _validate_live_args(_args(["--extra-scales", "-0.5"]))


def test_extra_scales_reject_too_many_and_duplicates():
    with pytest.raises(SystemExit):
        _validate_live_args(_args(["--extra-scales", "0.1,0.2,0.3,0.4"]))
    with pytest.raises(SystemExit):
        _validate_live_args(_args(["--extra-scales", "0.5,0.5"]))


def test_extra_scales_auto_resolves():
    from bsafe.cli import _resolve_extra_scales_for_start

    assert _resolve_extra_scales_for_start(_args([]), 1.0) == (0.5,)
    erax = _args(["--model", "erax-nano"])
    assert _resolve_extra_scales_for_start(erax, 1.0) == ()


def test_extra_scales_factors_divide_by_detect_scale():
    from bsafe.cli import _resolve_extra_scales_for_start

    args = _args(["--extra-scales", "0.5", "--detect-scale", "1.5"])
    assert _resolve_extra_scales_for_start(args, 1.5) == pytest.approx((1 / 3,))


def test_extra_scales_explicit_erax_warns_and_ignored(capsys):
    from bsafe.cli import _resolve_extra_scales_for_start

    args = _args(["--model", "erax-nano", "--extra-scales", "0.5"])
    # Validation passes (EraX ignores extra scales); resolution ignores them.
    _validate_live_args(args)
    assert _resolve_extra_scales_for_start(args, 1.0) == ()
    # cmd_start warns on the EraX+explicit combination.
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.detector.Detector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server_cls.return_value = MagicMock()
        helper = MagicMock()
        helper.poll.side_effect = [None, 0]
        helper.stderr = None
        helper_mock.return_value = helper
        det_cls.return_value = MagicMock()
        session = MagicMock()
        session_cls.return_value = session
        session.step.side_effect = KeyboardInterrupt
        cmd_start(args)
    err = capsys.readouterr().err
    assert "--extra-scales" in err
    _, kwargs = session_cls.call_args
    assert kwargs["extra_factors"] == ()


def test_print_config_extra_scales(capsys):
    args = _args([])
    args.confidence = 0.0
    _print_config(args)
    assert "extra_scales=0.5" in capsys.readouterr().out
    erax = _args(["--model", "erax-nano"])
    erax.confidence = 0.2
    _print_config(erax)
    assert "extra_scales=none" in capsys.readouterr().out
    none_args = _args(["--extra-scales", "none"])
    none_args.confidence = 0.0
    _print_config(none_args)
    assert "extra_scales=none" in capsys.readouterr().out


def test_cmd_start_passes_extra_factors_to_session():
    args = _args(["--extra-scales", "0.5"])
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server_cls.return_value = MagicMock()
        helper = MagicMock()
        helper.poll.side_effect = [None, 0]
        helper.stderr = None
        helper_mock.return_value = helper
        det_cls.return_value = MagicMock()
        session = MagicMock()
        session_cls.return_value = session
        session.step.side_effect = KeyboardInterrupt
        cmd_start(args)
        _, kwargs = session_cls.call_args
        assert kwargs["extra_factors"] == (0.5,)


def test_erax_invalid_extra_scales_rejected_before_helper(capsys):
    args = _args(["--model", "erax-nano", "--extra-scales", "bogus"])
    with (
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.detector.Detector") as det_mock,
        patch("bsafe.ipc.FrameServer") as server_mock,
    ):
        with pytest.raises(SystemExit):
            cmd_start(args)
    helper_mock.assert_not_called()
    det_mock.assert_not_called()
    server_mock.assert_not_called()
    assert "Error:" in capsys.readouterr().err


def test_extra_scales_resolve_runs_before_spawn_helper():
    args = _args([])
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.cli._resolve_extra_scales_for_start") as resolve_mock,
    ):
        server_cls.return_value = MagicMock()
        det_cls.return_value = MagicMock()
        resolve_mock.side_effect = ValueError("bad user input")
        with pytest.raises(ValueError, match="bad user input"):
            cmd_start(args)
        resolve_mock.assert_called_once()
        helper_mock.assert_not_called()
        server_cls.assert_not_called()


def _write_fake_helper(directory):
    """Create an executable fake helper that floods stderr, then exits 3."""
    script = directory / "fake-helper"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "for i in range(4000):\n"
        '    sys.stderr.write(f"line {i:05d} " + "x" * 56 + "\\n")\n'
        "    sys.stderr.flush()\n"
        'sys.stderr.write("FAKE-HELPER-DONE\\n")\n'
        "sys.stderr.flush()\n"
        "sys.exit(3)\n"
    )
    script.chmod(0o755)
    return script


def test_cmd_start_reports_helper_stderr_tail_after_large_output(
    tmp_path, monkeypatch, capsys, caplog
):
    helper_path = _write_fake_helper(tmp_path)
    monkeypatch.setattr("bsafe.swift_helper.find_helper", lambda: helper_path)
    caplog.set_level(logging.INFO, logger="bsafe.swift_helper")
    deadline = time.monotonic() + 10

    def _step(_timeout):
        time.sleep(0.01)
        if time.monotonic() >= deadline:
            raise KeyboardInterrupt

    args = _args([])
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server_cls.return_value = MagicMock()
        det_cls.return_value = MagicMock()
        session = MagicMock()
        session.step.side_effect = _step
        session_cls.return_value = session
        ret = cmd_start(args)
    out = capsys.readouterr()
    assert ret == 3
    assert "Swift helper exited (code 3)" in out.err
    assert "Helper stderr:" in out.err
    assert "FAKE-HELPER-DONE" in out.err
    assert "b'" not in out.err
    assert "Shutting down" not in out.out


def test_cmd_start_helper_death_returns_exit_code():
    args = _args([])
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server_cls.return_value = MagicMock()
        helper = MagicMock()
        helper.poll.return_value = 3
        helper.returncode = 3
        helper.stderr = None
        helper_mock.return_value = helper
        det_cls.return_value = MagicMock()
        session_cls.return_value = MagicMock()
        ret = cmd_start(args)
    assert ret == 3
    helper.terminate.assert_not_called()


def test_cmd_start_helper_zero_exit_maps_to_one():
    args = _args([])
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server_cls.return_value = MagicMock()
        helper = MagicMock()
        helper.poll.return_value = 0
        helper.returncode = 0
        helper.stderr = None
        helper_mock.return_value = helper
        det_cls.return_value = MagicMock()
        session_cls.return_value = MagicMock()
        ret = cmd_start(args)
    assert ret == 1


def test_cmd_start_user_interrupt_returns_zero():
    args = _args([])
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server_cls.return_value = MagicMock()
        helper = MagicMock()
        helper.poll.return_value = None
        helper.stderr = None
        helper_mock.return_value = helper
        det_cls.return_value = MagicMock()
        session = MagicMock()
        session.step.side_effect = KeyboardInterrupt
        session_cls.return_value = session
        ret = cmd_start(args)
    assert ret == 0


def test_cmd_start_parent_death_triggers_shutdown(monkeypatch):
    import os as _os

    args = _args([])
    real_getppid = _os.getppid()
    calls = iter([real_getppid, 1, 1, 1])

    monkeypatch.setattr("os.getppid", lambda: next(calls, 1))
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server = MagicMock()
        server_cls.return_value = server
        helper = MagicMock()
        helper.poll.return_value = None
        helper.stderr = None
        helper_mock.return_value = helper
        det_cls.return_value = MagicMock()
        session_cls.return_value = MagicMock()
        ret = cmd_start(args)
    assert ret == 1
    server.shutdown.assert_called_once()
    helper.terminate.assert_called_once()


def test_cmd_start_no_watchdog_when_parent_is_init(monkeypatch):

    args = _args([])
    monkeypatch.setattr("os.getppid", lambda: 1)
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server_cls.return_value = MagicMock()
        helper = MagicMock()
        helper.poll.side_effect = [None, 0]
        helper.stderr = None
        helper_mock.return_value = helper
        det_cls.return_value = MagicMock()
        session = MagicMock()
        session.step.side_effect = KeyboardInterrupt
        session_cls.return_value = session
        ret = cmd_start(args)
    assert ret == 0


def test_parser_still_defaults():
    args = _args([])
    assert args.still_model is None
    assert args.still_period is None
    assert args.still_after is None


def test_parser_still_values():
    args = _args(["--still-model", "640m", "--still-period", "3", "--still-after", "1"])
    assert args.still_model == "640m"
    assert args.still_period == 3.0
    assert args.still_after == 1.0


def test_still_model_choices_rejected():
    with pytest.raises(SystemExit):
        _args(["--still-model", "320n"])


def test_still_period_invalid_rejected():
    for value in ("0", "-1"):
        with pytest.raises(SystemExit):
            _validate_live_args(_args(["--still-period", value]))


def test_still_after_negative_rejected():
    with pytest.raises(SystemExit):
        _validate_live_args(_args(["--still-after", "-0.5"]))


def test_still_after_zero_ok_and_defaults_filled():
    args = _args(["--still-after", "0"])
    _validate_live_args(args)
    assert args.still_after == 0
    assert args.still_period == 2.0


def test_print_config_still(capsys):
    args = _args([])
    args.confidence = 0.0
    _print_config(args)
    assert "still_model=none" in capsys.readouterr().out
    enabled = _args(["--still-model", "640m", "--still-period", "3", "--still-after", "1"])
    enabled.confidence = 0.0
    _print_config(enabled)
    out = capsys.readouterr().out
    assert "still_model=640m" in out
    assert "still_period=3.0" in out
    assert "still_after=1.0" in out
    erax = _args(["--model", "erax-nano", "--still-model", "640m"])
    erax.confidence = 0.2
    _print_config(erax)
    assert "still_model=none" in capsys.readouterr().out


def test_cmd_start_builds_and_closes_still_detector():
    args = _args(["--still-model", "640m", "--still-period", "3", "--still-after", "1"])
    with (
        patch("bsafe.detector.resolve_model") as resolve_mock,
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        resolve_mock.return_value = MagicMock(path="/tmp/640m.onnx")
        server_cls.return_value = MagicMock()
        helper = MagicMock()
        helper.poll.side_effect = [None, 0]
        helper.stderr = None
        helper_mock.return_value = helper
        detector = MagicMock()
        det_cls.return_value = detector
        session = MagicMock()
        session_cls.return_value = session
        session.step.side_effect = KeyboardInterrupt
        cmd_start(args)
        assert det_cls.call_count == 2
        _, kwargs = session_cls.call_args
        assert kwargs["still_detector"] is detector
        assert kwargs["still_period_s"] == 3.0
        assert kwargs["still_after_s"] == 1.0
        assert detector.close.call_count == 2


def test_cmd_start_no_still_by_default():
    args = _args([])
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as det_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server_cls.return_value = MagicMock()
        helper = MagicMock()
        helper.poll.side_effect = [None, 0]
        helper.stderr = None
        helper_mock.return_value = helper
        det_cls.return_value = MagicMock()
        session = MagicMock()
        session_cls.return_value = session
        session.step.side_effect = KeyboardInterrupt
        cmd_start(args)
        assert det_cls.call_count == 1
        _, kwargs = session_cls.call_args
        assert kwargs["still_detector"] is None
        assert kwargs["still_period_s"] == 2.0
        assert kwargs["still_after_s"] == 0.5


def test_cmd_start_erax_still_warns_and_disabled(capsys):
    args = _args(["--model", "erax-nano", "--still-model", "640m"])
    with (
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.detector.Detector") as det_cls,
        patch("bsafe.fastdetect.FullFrameNudeDetector") as nude_cls,
        patch("bsafe.live.LiveSession") as session_cls,
    ):
        server_cls.return_value = MagicMock()
        helper = MagicMock()
        helper.poll.side_effect = [None, 0]
        helper.stderr = None
        helper_mock.return_value = helper
        det_cls.return_value = MagicMock()
        session = MagicMock()
        session_cls.return_value = session
        session.step.side_effect = KeyboardInterrupt
        cmd_start(args)
    err = capsys.readouterr().err
    assert "--still-model has no effect with EraX models (NudeNet-only)" in err
    nude_cls.assert_not_called()
    _, kwargs = session_cls.call_args
    assert kwargs["still_detector"] is None


def test_cmd_start_missing_still_model_exits_cleanly(capsys):
    args = _args(["--still-model", "640m"])

    def _fake_resolve(name):
        if name == "640m":
            raise ValueError("model file not found: 640m")
        return MagicMock(path="/tmp/320n.onnx")

    with (
        patch("bsafe.detector.resolve_model", side_effect=_fake_resolve),
        patch("bsafe.ipc.FrameServer") as server_cls,
        patch("bsafe.swift_helper.spawn_helper") as helper_mock,
        patch("bsafe.fastdetect.FullFrameNudeDetector"),
        patch("bsafe.live.LiveSession"),
    ):
        with pytest.raises(SystemExit) as exc:
            cmd_start(args)
        assert exc.value.code == 1
    assert "Error:" in capsys.readouterr().err
    helper_mock.assert_not_called()
    server_cls.assert_not_called()


def test_still_period_nonfinite_rejected():
    for value in ("nan", "inf", "-inf"):
        with pytest.raises(SystemExit):
            _validate_live_args(_args(["--still-period", value]))


def test_still_after_nonfinite_rejected():
    for value in ("nan", "inf", "-inf"):
        with pytest.raises(SystemExit):
            _validate_live_args(_args(["--still-after", value]))


def test_still_model_config_640M_rejected(capsys):
    args = _args([])
    args.still_model = "640M"
    with pytest.raises(SystemExit):
        _validate_live_args(args)
    assert "invalid still_model" in capsys.readouterr().err


def test_still_model_config_true_rejected(capsys):
    args = _args([])
    args.still_model = True
    with pytest.raises(SystemExit):
        _validate_live_args(args)
    assert "invalid still_model" in capsys.readouterr().err
