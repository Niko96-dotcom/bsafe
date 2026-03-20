"""ANSI color helpers for CLI output.

Auto-disables color when stdout is not a TTY or NO_COLOR env var is set.
Pure functions only — no state, no dependencies.
"""

import os
import sys
import time


def _color_enabled() -> bool:
    """Return True if ANSI color codes should be emitted."""
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _wrap(text: str, code: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(text: str) -> str:
    """Muted text — config lines, timestamps, secondary info."""
    return _wrap(text, "2")


def bold(text: str) -> str:
    """Emphasized text — status transitions, output paths, model names."""
    return _wrap(text, "1")


def error(text: str) -> str:
    """Red text — only for 'Error:' prefix or failure markers."""
    return _wrap(text, "31")


def warn(text: str) -> str:
    """Yellow text — only for 'Warning:' or 'WARN' tokens."""
    return _wrap(text, "33")


def success(text: str) -> str:
    """Green text — 'OK', 'Saved', completion counts."""
    return _wrap(text, "32")


def info(text: str) -> str:
    """Cyan text — chunk labels, file counts, timing durations."""
    return _wrap(text, "36")


def detection(text: str) -> str:
    """Magenta text — [detection] log prefix in verbose mode."""
    return _wrap(text, "35")


def timestamp() -> str:
    """Return dim [HH:MM:SS] string."""
    t = time.strftime("%H:%M:%S")
    return dim(f"[{t}]")
