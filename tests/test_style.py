"""Tests for bsafe.style ANSI color helpers."""

import re
from unittest import mock

from bsafe.style import (
    bold,
    detection,
    dim,
    error,
    info,
    success,
    timestamp,
    warn,
)


class TestNoColor:
    """When NO_COLOR is set, all functions return plain text."""

    def test_dim(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert dim("hello") == "hello"

    def test_bold(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert bold("hello") == "hello"

    def test_error(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert error("Error:") == "Error:"

    def test_warn(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert warn("Warning:") == "Warning:"

    def test_success(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert success("OK") == "OK"

    def test_info(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert info("Chunk 3/5") == "Chunk 3/5"

    def test_detection(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert detection("[detection]") == "[detection]"

    def test_timestamp(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        ts = timestamp()
        assert re.match(r"^\[\d{2}:\d{2}:\d{2}\]$", ts)


class TestWithTTY:
    """When stdout is a TTY and NO_COLOR is unset, functions wrap with ANSI codes."""

    def _enable_color(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        mock_stdout = mock.MagicMock()
        mock_stdout.isatty.return_value = True
        monkeypatch.setattr("bsafe.style.sys.stdout", mock_stdout)

    def test_dim(self, monkeypatch):
        self._enable_color(monkeypatch)
        result = dim("config line")
        assert result == "\033[2mconfig line\033[0m"

    def test_bold(self, monkeypatch):
        self._enable_color(monkeypatch)
        result = bold("Running...")
        assert result == "\033[1mRunning...\033[0m"

    def test_error(self, monkeypatch):
        self._enable_color(monkeypatch)
        result = error("Error:")
        assert result == "\033[31mError:\033[0m"

    def test_warn(self, monkeypatch):
        self._enable_color(monkeypatch)
        result = warn("Warning:")
        assert result == "\033[33mWarning:\033[0m"

    def test_success(self, monkeypatch):
        self._enable_color(monkeypatch)
        result = success("OK")
        assert result == "\033[32mOK\033[0m"

    def test_info(self, monkeypatch):
        self._enable_color(monkeypatch)
        result = info("Chunk 3/5")
        assert result == "\033[36mChunk 3/5\033[0m"

    def test_detection(self, monkeypatch):
        self._enable_color(monkeypatch)
        result = detection("[detection]")
        assert result == "\033[35m[detection]\033[0m"

    def test_timestamp(self, monkeypatch):
        self._enable_color(monkeypatch)
        ts = timestamp()
        # Should contain ANSI dim codes and HH:MM:SS format
        assert "\033[2m" in ts
        assert "\033[0m" in ts
        assert re.search(r"\d{2}:\d{2}:\d{2}", ts)


class TestNotTTY:
    """When stdout is not a TTY, functions return plain text."""

    def test_plain_when_piped(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        mock_stdout = mock.MagicMock()
        mock_stdout.isatty.return_value = False
        monkeypatch.setattr("bsafe.style.sys.stdout", mock_stdout)
        assert bold("test") == "test"
        assert error("Error:") == "Error:"
        assert dim("muted") == "muted"
