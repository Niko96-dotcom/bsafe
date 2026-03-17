import argparse
import logging
import os
import queue
import sys
import tempfile
import time

logger = logging.getLogger(__name__)


def cmd_start(args):
    if args.dry_run:
        print("Running... press Ctrl+C to stop.", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    from bsafe.censor import CENSOR_PRESETS, build_censor_boxes
    from bsafe.detector import Detector
    from bsafe.ipc import FrameServer
    from bsafe.swift_helper import spawn_helper

    fps = args.fps
    if fps < 1 or fps > 255:
        print("Error: --fps must be between 1 and 255", file=sys.stderr)
        sys.exit(1)
    socket_path = os.path.join(tempfile.gettempdir(), f"bsafe-{os.getpid()}.sock")

    censor_classes = CENSOR_PRESETS[args.censor]
    padding = args.padding

    # Initialize detector
    print("Loading NudeNet model...", flush=True)
    detector = Detector(min_confidence=args.confidence)

    # Start IPC server
    server = FrameServer(socket_path, fps=fps)
    server.start()

    # Spawn Swift helper
    print("Starting screen capture...", flush=True)
    helper = spawn_helper(socket_path, fps)

    print("Running... press Ctrl+C to stop.", flush=True)

    try:
        while True:
            # Check if helper is still running
            if helper.poll() is not None:
                stderr_out = helper.stderr.read() if helper.stderr else ""
                print(f"\nSwift helper exited (code {helper.returncode})", file=sys.stderr)
                if stderr_out:
                    print(f"Helper stderr: {stderr_out}", file=sys.stderr)
                break

            # Dequeue and process frames
            try:
                meta, jpeg_data = server.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            detections = detector.detect(jpeg_data)
            logger.debug("Processed frame: %d detection(s)", len(detections))
            if detections:
                for d in detections:
                    logger.debug(
                        "[detection] %s (%.2f) at %s on display %d",
                        d.class_name, d.confidence, d.box, meta.display_id,
                    )
            boxes = build_censor_boxes(
                detections, censor_classes, padding, meta.width, meta.height
            )
            server.send_censor(meta.display_id, meta.width, meta.height, boxes)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.shutdown()
        detector.close()
        if helper.poll() is None:
            helper.terminate()
            helper.wait(timeout=5)
        print("Stopped.")


def cmd_bootstrap(args):
    import subprocess

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    swift_dir = os.path.join(project_dir, "swift")

    # Step 1: uv sync
    print("Installing Python dependencies...")
    result = subprocess.run(["uv", "sync"], cwd=project_dir)
    if result.returncode != 0:
        print("Error: uv sync failed", file=sys.stderr)
        sys.exit(1)
    print("Python dependencies OK\n")

    # Step 2: Build Swift helper
    print("Building Swift helper...")
    if not os.path.isdir(swift_dir):
        print(f"Error: swift directory not found at {swift_dir}", file=sys.stderr)
        sys.exit(1)
    result = subprocess.run(["swift", "build", "-c", "release"], cwd=swift_dir)
    if result.returncode != 0:
        print("Error: Swift build failed", file=sys.stderr)
        sys.exit(1)
    print("Swift helper OK\n")

    print("Bootstrap complete! Run 'bsafe doctor' to verify.")
    _print_alias_hint()


def cmd_doctor(args):
    all_ok = True

    # Python version
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {version}", end="")
    if sys.version_info >= (3, 14):
        print(" OK")
    else:
        print(" WARN: expected >= 3.14")
        all_ok = False

    # macOS platform
    print(f"Platform: {sys.platform}", end="")
    if sys.platform == "darwin":
        print(" OK")
    else:
        print(" WARN: bsafe requires macOS")
        all_ok = False

    # NudeNet importable
    print("NudeNet: ", end="")
    try:
        import nudenet  # noqa: F401

        print("OK")
    except ImportError:
        print("NOT FOUND — run: uv sync")
        all_ok = False

    # Swift helper binary
    print("Swift helper: ", end="")
    from bsafe.swift_helper import find_helper

    helper = find_helper()
    if helper:
        print(f"OK ({helper})")
    else:
        print("NOT FOUND — build with: cd swift && swift build -c release")
        all_ok = False

    if not all_ok:
        sys.exit(1)

    # Suggest shell alias
    _print_alias_hint()


def _print_alias_hint():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print()
    print("Hint: to make bsafe globally available, add this to your ~/.zshrc:")
    print(f"  alias bsafe='uv run --project {project_dir} bsafe'")


def main():
    parser = argparse.ArgumentParser(prog="bsafe", description="Censor NSFW content on screen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start the censoring process")
    # 60 FPS tested as sustainable on MacBook Pro M4 Pro (capture + inference loop)
    start_parser.add_argument("--fps", type=int, default=60, help="Capture FPS (default: 60)")
    # Aggressive default: empirical tests showed confidence 0.0 catches edge cases
    # that higher thresholds miss, with acceptable false-positive rate for censoring
    start_parser.add_argument(
        "--confidence", type=float, default=0.0, help="Min detection confidence (default: 0.0)"
    )
    start_parser.add_argument(
        "--censor", choices=["female", "male", "all"], default="all",
        help="What to censor: female, male, or all (default: all)",
    )
    start_parser.add_argument(
        "--padding", type=float, default=0.0,
        help="Box expansion fraction (default: 0.0)",
    )
    start_parser.add_argument(
        "--dry-run", action="store_true", help="Run without Swift helper or detector"
    )
    start_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers.add_parser("doctor", help="Check system requirements")
    subparsers.add_parser("bootstrap", help="Install dependencies and build Swift helper")

    args = parser.parse_args()

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    commands = {
        "start": cmd_start,
        "doctor": cmd_doctor,
        "bootstrap": cmd_bootstrap,
    }

    commands[args.command](args)
