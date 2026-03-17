import signal
import subprocess
import sys


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
