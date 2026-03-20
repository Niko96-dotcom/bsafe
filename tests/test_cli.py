import signal
import subprocess
import sys

from bsafe.cli import _build_parser


def _parse_args(args: list[str]):
    """Parse CLI args using the real parser (no side effects)."""
    parser, *_ = _build_parser()
    return parser.parse_args(args)


def test_doctor_reports_checks():
    result = subprocess.run(
        [sys.executable, "-m", "bsafe", "doctor"], capture_output=True, text=True
    )
    # Doctor checks multiple things; Swift helper may not be built in CI
    assert "Python version:" in result.stdout
    assert "NudeNet:" in result.stdout
    assert "Swift helper:" in result.stdout


def test_start_dry_run_responds_to_interrupt():
    proc = subprocess.Popen(
        [sys.executable, "-m", "bsafe", "start", "--dry-run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait for the process to be ready before sending SIGINT
    assert proc.stdout.readline().startswith("Running")
    proc.send_signal(signal.SIGINT)
    stdout, _ = proc.communicate(timeout=5)
    assert proc.returncode == 0
    assert "Stopped" in stdout


def test_video_accepts_multiple_inputs():
    args = _parse_args(["video", "a.mp4", "b.mp4", "c.mov"])
    assert args.input == ["a.mp4", "b.mp4", "c.mov"]


def test_image_accepts_multiple_inputs():
    args = _parse_args(["image", "a.jpg", "b.png", "c.webp"])
    assert args.input == ["a.jpg", "b.png", "c.webp"]


def test_video_single_input():
    args = _parse_args(["video", "single.mp4"])
    assert args.input == ["single.mp4"]


def test_image_single_input():
    args = _parse_args(["image", "single.jpg"])
    assert args.input == ["single.jpg"]


def test_video_output_with_multiple_inputs_errors():
    result = subprocess.run(
        [sys.executable, "-m", "bsafe", "video", "a.mp4", "b.mp4", "-o", "out.mp4"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "-o/--output cannot be used with multiple input files" in result.stderr


def test_image_output_with_multiple_inputs_errors():
    result = subprocess.run(
        [sys.executable, "-m", "bsafe", "image", "a.jpg", "b.jpg", "-o", "out.jpg"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "-o/--output cannot be used with multiple input files" in result.stderr
