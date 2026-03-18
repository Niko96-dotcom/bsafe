"""Locate and spawn the compiled Swift helper binary."""

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

HELPER_NAME = "BsafeCapture"


def find_helper() -> Path | None:
    """Look for the compiled Swift helper binary.

    Search order:
    1. BSAFE_HELPER_PATH env var (explicit override)
    2. Relative to the project source tree (development)
    3. System PATH
    """
    # 1. Explicit env var
    env_path = os.environ.get("BSAFE_HELPER_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        logger.warning("BSAFE_HELPER_PATH set but not found: %s", env_path)

    # 2. Check relative to the project source tree (development layout)
    project_root = Path(__file__).resolve().parent.parent
    candidate = project_root / "swift" / ".build" / "release" / HELPER_NAME
    if candidate.is_file():
        return candidate

    # 3. Fall back to PATH
    on_path = shutil.which(HELPER_NAME)
    if on_path:
        return Path(on_path)

    return None


def spawn_helper(socket_path: str, fps: int) -> subprocess.Popen:
    """Spawn the Swift helper, pointing it at our socket."""
    helper = find_helper()
    if helper is None:
        raise FileNotFoundError(
            f"Swift helper '{HELPER_NAME}' not found. "
            "Build it with: cd swift && swift build -c release"
        )
    logger.info("Spawning Swift helper: %s", helper)
    verbose = logger.isEnabledFor(logging.DEBUG)
    return subprocess.Popen(
        [str(helper), "--socket", socket_path, "--fps", str(fps)],
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.PIPE,
    )
