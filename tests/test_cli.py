import signal
import subprocess
import sys


def test_doctor_exits_0():
    result = subprocess.run(
        [sys.executable, "-m", "bsafe", "doctor"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_start_responds_to_interrupt():
    proc = subprocess.Popen(
        [sys.executable, "-m", "bsafe", "start"],
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
