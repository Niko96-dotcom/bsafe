import argparse
import faulthandler
import logging
import os
import queue
import signal
import subprocess
import sys
import tempfile
import time

logger = logging.getLogger(__name__)


def _add_censor_args(parser):
    """Add shared censor flags to a subcommand parser."""
    parser.add_argument(
        "--confidence", type=float, default=0.0, help="Min detection confidence (default: 0.0)"
    )
    parser.add_argument(
        "--censor",
        choices=["female", "male", "all"],
        default="all",
        help="What to censor: female, male, or all (default: all)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Box expansion fraction (default: 0.0)",
    )
    parser.add_argument(
        "--blur",
        nargs="?",
        type=float,
        const=1.0,
        default=0.0,
        help="Use blur instead of black rectangle; 1.0 = 100%% blur, >1 multiplies effect (default: 1.0)",
    )
    parser.add_argument(
        "--pixels",
        nargs="?",
        type=float,
        const=1.0,
        default=0.0,
        help="Use pixelation instead of black rectangle; 1.0 = 100%% pixelation, >1 multiplies effect (default: 1.0)",
    )
    parser.add_argument(
        "--censor-text",
        nargs="?",
        const="NSFW",
        default=None,
        help='Show text on censored region (default: "NSFW" if flag present with no value)',
    )
    parser.add_argument(
        "--full-censor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expand censor area by 3x (9x area)",
    )
    parser.add_argument(
        "--covered",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also censor covered body parts (anus, buttocks; breasts when female/all)",
    )
    parser.add_argument(
        "--face-male",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also censor male faces",
    )
    parser.add_argument(
        "--face-female",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also censor female faces",
    )
    parser.add_argument(
        "--feet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also censor exposed feet",
    )
    parser.add_argument(
        "--model",
        choices=["320n", "640m"],
        default=None,
        help="Detection model: '320n' (default, fast) or '640m' (accurate, requires download)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def _add_temporal_args(parser):
    """Add temporal tracking flags (only for live/video, not image)."""
    parser.add_argument(
        "--persist-frames",
        type=int,
        default=8,
        help="Frames a box persists after disappearing (default: 8)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="EMA weight for box position smoothing, 0.0-1.0 (default: 0.5)",
    )


def _validate_censor_args(args):
    """Validate shared censor arguments. Exits on error."""
    if args.padding < 0:
        print("Error: --padding must be >= 0", file=sys.stderr)
        sys.exit(1)
    if hasattr(args, "smooth_alpha") and not (0.0 <= args.smooth_alpha <= 1.0):
        print("Error: --smooth-alpha must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    blur = args.blur or 0.0
    pixels = args.pixels or 0.0
    if not (0.0 <= blur <= 3.0):
        print("Error: --blur must be between 0.0 and 3.0", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= pixels <= 3.0):
        print("Error: --pixels must be between 0.0 and 3.0", file=sys.stderr)
        sys.exit(1)
    if blur > 0.0 and pixels > 0.0:
        print("Error: --blur and --pixels are mutually exclusive", file=sys.stderr)
        sys.exit(1)


def cmd_start(args):
    if args.dry_run:
        stop = False

        def _handle_sigint(sig, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _handle_sigint)
        print("Running... press Ctrl+C to stop.", flush=True)
        while not stop:
            time.sleep(0.2)
        print("\nStopped.")
        return

    from bsafe.censor import (
        FULL_CENSOR_MULTIPLIER,
        build_censor_boxes,
        expand_boxes,
        resolve_censor_classes,
    )
    from bsafe.detector import Detector
    from bsafe.ipc import FrameServer
    from bsafe.swift_helper import spawn_helper
    from bsafe.tracking import BoxTracker

    fps = args.fps
    if fps < 1 or fps > 255:
        print("Error: --fps must be between 1 and 255", file=sys.stderr)
        sys.exit(1)
    _validate_censor_args(args)
    socket_path = os.path.join(tempfile.gettempdir(), f"bsafe-{os.getpid()}.sock")

    censor_classes = resolve_censor_classes(
        args.censor,
        covered=args.covered,
        face_male=args.face_male,
        face_female=args.face_female,
        feet=args.feet,
    )
    padding = args.padding
    tracker = BoxTracker(persist_frames=args.persist_frames, smooth_alpha=args.smooth_alpha)

    # Initialize detector
    print("Loading NudeNet model...", flush=True)
    detector = Detector(min_confidence=args.confidence, model=args.model)

    # Start IPC server
    server = FrameServer(
        socket_path, fps=fps, blur=args.blur, pixels=args.pixels, censor_text=args.censor_text
    )
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
                        d.class_name,
                        d.confidence,
                        d.box,
                        meta.display_id,
                    )
            boxes = build_censor_boxes(detections, censor_classes, padding, meta.width, meta.height)
            if args.full_censor:
                boxes = expand_boxes(boxes, FULL_CENSOR_MULTIPLIER, meta.width, meta.height)
            boxes = tracker.update(meta.display_id, boxes)
            server.send_censor(meta.display_id, meta.width, meta.height, boxes)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.shutdown()
        detector.close()
        if helper.poll() is None:
            helper.terminate()
            try:
                helper.wait(timeout=5)
            except subprocess.TimeoutExpired:
                helper.kill()
                helper.wait()
        print("Stopped.")


def cmd_video(args):
    _validate_censor_args(args)

    from bsafe.censor import CensorConfig
    from bsafe.video import process_video

    censor_config = CensorConfig(
        preset=args.censor,
        covered=args.covered,
        face_male=args.face_male,
        face_female=args.face_female,
        feet=args.feet,
    )

    try:
        process_video(
            args.input,
            confidence=args.confidence,
            censor_config=censor_config,
            padding=args.padding,
            persist_frames=args.persist_frames,
            smooth_alpha=args.smooth_alpha,
            blur=args.blur,
            pixels=args.pixels,
            censor_text=args.censor_text,
            full_censor=args.full_censor,
            fps_override=args.fps,
            model=args.model,
            chunk_frames=args.chunk_frames,
            verbose=args.verbose,
        )
        print("\a", end="", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_image(args):
    _validate_censor_args(args)

    from bsafe.censor import CensorConfig
    from bsafe.image import process_image

    censor_config = CensorConfig(
        preset=args.censor,
        covered=args.covered,
        face_male=args.face_male,
        face_female=args.face_female,
        feet=args.feet,
    )

    try:
        output_path = process_image(
            args.input,
            confidence=args.confidence,
            censor_config=censor_config,
            padding=args.padding,
            blur=args.blur,
            pixels=args.pixels,
            censor_text=args.censor_text,
            full_censor=args.full_censor,
            model=args.model,
            verbose=args.verbose,
        )
        print(output_path)
        print("\a", end="", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_bootstrap(args):
    import subprocess

    from bsafe.config import CONFIG_PATH, generate_default_config

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

    # Step 3: Ensure models directory exists
    models_dir = os.path.expanduser("~/.config/bsafe/models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models directory: {models_dir} OK\n")

    # Step 4: Create config file if it doesn't exist
    config_path = CONFIG_PATH
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            f.write(generate_default_config())
        print(f"Config file: {config_path} CREATED\n")
    else:
        print(f"Config file: {config_path} already exists, skipping\n")

    print("Bootstrap complete! Run 'bsafe doctor' to verify.")
    _print_alias_hint()


def cmd_doctor(args):
    from bsafe.config import CONFIG_PATH

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

    # Config file (informational only)
    print(f"Config file: {CONFIG_PATH}", end="")
    if os.path.exists(CONFIG_PATH):
        print(" OK")
    else:
        print(" NOT FOUND (optional — run bootstrap to create)")

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

    # Screen Recording permission
    print("Screen Recording: ", end="")
    if helper:
        import subprocess

        result = subprocess.run(
            [str(helper), "--check-permission"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "SCREEN_RECORDING_OK" in result.stdout:
            print("OK")
        else:
            print(
                "NOT GRANTED — open System Settings > Privacy & Security > Screen Recording "
                "and enable your terminal app"
            )
            all_ok = False
    else:
        print("SKIPPED (Swift helper not found)")
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
    from bsafe.config import load_config

    parser = argparse.ArgumentParser(prog="bsafe", description="Censor NSFW content on screen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start subcommand
    start_parser = subparsers.add_parser("start", help="Start the censoring process")
    start_parser.add_argument("--fps", type=int, default=45, help="Capture FPS (default: 45)")
    _add_censor_args(start_parser)
    _add_temporal_args(start_parser)
    start_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run without Swift helper or detector",
    )

    # video subcommand
    video_parser = subparsers.add_parser(
        "video", help="Process a video file and output a censored copy"
    )
    video_parser.add_argument("input", help="Path to video file (.mp4, .m4v, .mov)")
    video_parser.add_argument(
        "--fps", type=int, default=None, help="Detection FPS override (default: native)"
    )
    video_parser.add_argument(
        "--chunk-frames",
        type=int,
        default=None,
        help="Frames per processing chunk (default: 5000). "
        "Smaller chunks use less memory but add brief tracking gaps at boundaries.",
    )
    _add_censor_args(video_parser)
    _add_temporal_args(video_parser)

    # image subcommand
    image_parser = subparsers.add_parser(
        "image", help="Process an image file and output a censored copy"
    )
    image_parser.add_argument(
        "input", help="Path to image file (.jpg, .jpeg, .png, .bmp, .webp, .tif, .tiff)"
    )
    _add_censor_args(image_parser)

    subparsers.add_parser("doctor", help="Check system requirements")
    subparsers.add_parser("bootstrap", help="Install dependencies and build Swift helper")

    # Load user config and apply as defaults to subparsers
    config = load_config()
    if config:
        start_config = {**config.get("common", {}), **config.get("start", {})}
        video_config = {**config.get("common", {}), **config.get("video", {})}
        image_config = {**config.get("common", {}), **config.get("image", {})}
        if start_config:
            start_parser.set_defaults(**start_config)
        if video_config:
            video_parser.set_defaults(**video_config)
        if image_config:
            image_parser.set_defaults(**image_config)

    args = parser.parse_args()

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    commands = {
        "start": cmd_start,
        "doctor": cmd_doctor,
        "bootstrap": cmd_bootstrap,
        "video": cmd_video,
        "image": cmd_image,
    }

    faulthandler.enable()
    commands[args.command](args)
