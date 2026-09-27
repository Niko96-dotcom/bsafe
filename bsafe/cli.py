import argparse
import faulthandler
import gc
import logging
import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import tempfile
import time

from bsafe.style import bold, dim, error, info, success, warn

logger = logging.getLogger(__name__)


def _add_censor_args(parser):
    """Add shared censor flags to a subcommand parser."""
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Min detection confidence (default: 0.0 for NudeNet, 0.2 for EraX)",
    )
    parser.add_argument(
        "--censor",
        choices=["none", "female", "male", "all"],
        default="all",
        help="What to censor: none, female, male, or all (default: all)",
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
        choices=["320n", "640m", "erax-nano", "erax-small", "erax-medium"],
        default=None,
        help="Detection model: '320n' (default, bundled), '640m' (accurate, requires download), "
        "'erax-nano/small/medium' (EraX YOLO, requires download + uv sync --extra erax)",
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
        print(f"{error('Error:')} --padding must be >= 0", file=sys.stderr)
        sys.exit(1)
    if hasattr(args, "smooth_alpha") and not (0.0 <= args.smooth_alpha <= 1.0):
        print(f"{error('Error:')} --smooth-alpha must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    blur = args.blur or 0.0
    pixels = args.pixels or 0.0
    if not (0.0 <= blur <= 3.0):
        print(f"{error('Error:')} --blur must be between 0.0 and 3.0", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= pixels <= 3.0):
        print(f"{error('Error:')} --pixels must be between 0.0 and 3.0", file=sys.stderr)
        sys.exit(1)
    if blur > 0.0 and pixels > 0.0:
        print(f"{error('Error:')} --blur and --pixels are mutually exclusive", file=sys.stderr)
        sys.exit(1)


def _resolve_confidence(args):
    """Set confidence to a backend-appropriate default when not explicitly provided."""
    if args.confidence is not None:
        return
    from bsafe.detector import get_model_backend

    backend = get_model_backend(args.model)
    args.confidence = 0.2 if backend == "erax" else 0.0


def _warn_erax_unsupported(args):
    """Print warnings for flags that have no effect with EraX models."""
    from bsafe.detector import ERAX_UNSUPPORTED_FLAGS, get_model_backend

    if get_model_backend(args.model) != "erax":
        return
    for attr, flag in ERAX_UNSUPPORTED_FLAGS.items():
        if getattr(args, attr, False):
            print(
                f"{warn('Warning:')} {flag} has no effect with EraX models (no covered/face/feet classes)",
                file=sys.stderr,
            )


def _validate_live_args(args):
    """Validate start-only live flags. Exits on error before the helper starts."""
    from bsafe.detector import INFERENCE_RESOLUTIONS
    from bsafe.live import validate_live_options

    max_age = getattr(args, "max_frame_age_ms", 250)
    if max_age is None:
        max_age = 250
        args.max_frame_age_ms = max_age
    if isinstance(max_age, bool) or not isinstance(max_age, int) or max_age < 0:
        print(f"{error('Error:')} --max-frame-age-ms must be an int >= 0", file=sys.stderr)
        sys.exit(1)
    resolution = getattr(args, "inference_resolution", 320)
    if resolution is None:
        resolution = 320
        args.inference_resolution = resolution
    if resolution not in INFERENCE_RESOLUTIONS:
        print(
            f"{error('Error:')} --inference-resolution must be one of "
            f"{', '.join(str(r) for r in INFERENCE_RESOLUTIONS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        validate_live_options(
            getattr(args, "model", None), resolution, bool(getattr(args, "detail_scan", False))
        )
    except ValueError as e:
        print(f"{error('Error:')} {e}", file=sys.stderr)
        sys.exit(1)


def _print_config(args):
    """Print active censor configuration as a sanity check."""
    if args.censor_text:
        censor_method = f"text ({args.censor_text})"
    elif args.blur:
        censor_method = f"blur ({args.blur})"
    elif args.pixels:
        censor_method = f"pixels ({args.pixels})"
    else:
        censor_method = "bar"
    parts = [
        f"model={args.model or '320n'}",
        f"confidence={args.confidence}",
        f"censor={args.censor}",
        f"method={censor_method}",
        f"padding={args.padding}",
    ]
    extras = [
        name
        for name, enabled in [
            ("covered", args.covered),
            ("face_male", args.face_male),
            ("face_female", args.face_female),
            ("feet", args.feet),
            ("full_censor", args.full_censor),
        ]
        if enabled
    ]
    if extras:
        parts.append(f"extras={','.join(extras)}")
    if hasattr(args, "enhance") and args.enhance:
        parts.append(f"enhance={args.enhance}")
    if hasattr(args, "inference_resolution"):
        parts.append(f"inference_resolution={args.inference_resolution}")
    if getattr(args, "detail_scan", False):
        parts.append("detail_scan=true")
    if getattr(args, "motion_compensation", False):
        parts.append("motion_compensation=true")
    if hasattr(args, "max_frame_age_ms"):
        parts.append(f"max_frame_age_ms={args.max_frame_age_ms}")
    if getattr(args, "stats", False):
        parts.append("stats=true")
    print(f"{dim('Config:')} {', '.join(parts)}", flush=True)


def cmd_start(args):
    if args.dry_run:
        stop = False

        def _handle_sigint(sig, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _handle_sigint)
        print(bold("Running...") + " press Ctrl+C to stop.", flush=True)
        while not stop:
            time.sleep(0.2)
        print(f"\n{bold('Stopped.')}")
        return

    from bsafe.censor import (
        FULL_CENSOR_MULTIPLIER,
        build_censor_boxes,
        expand_boxes,
        resolve_censor_classes,
    )
    from bsafe.detector import Detector
    from bsafe.ipc import FrameServer
    from bsafe.live import DetailScanDetector, LiveStats, is_frame_stale
    from bsafe.swift_helper import spawn_helper
    from bsafe.tracking import BoxTracker

    fps = args.fps
    if fps < 1 or fps > 255:
        print(f"{error('Error:')} --fps must be between 1 and 255", file=sys.stderr)
        sys.exit(1)
    _validate_censor_args(args)
    _validate_live_args(args)
    _resolve_confidence(args)
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
    max_frame_age_ms = getattr(args, "max_frame_age_ms", 250) or 0
    show_stats = bool(getattr(args, "stats", False))
    detail_scan = bool(getattr(args, "detail_scan", False))
    motion_compensation = bool(getattr(args, "motion_compensation", False))
    inference_resolution = getattr(args, "inference_resolution", 320) or 320
    stats = LiveStats()

    # Warn about unsupported flags with EraX
    _warn_erax_unsupported(args)

    _print_config(args)

    # Initialize detector (live wrapper keeps file paths single-pass)
    print(f"Loading model {bold(args.model or '320n')}...", flush=True)
    base_detector = Detector(
        min_confidence=args.confidence,
        model=args.model,
        inference_resolution=inference_resolution,
    )
    detector = DetailScanDetector(base_detector, enabled=detail_scan)

    # Start IPC server
    server = FrameServer(
        socket_path, fps=fps, blur=args.blur, pixels=args.pixels, censor_text=args.censor_text
    )
    server.start()

    # Spawn Swift helper
    print("Starting screen capture...", flush=True)
    helper = spawn_helper(socket_path, fps, display=args.display)

    print(bold("Running...") + " press Ctrl+C to stop.", flush=True)

    try:
        while True:
            # Check if helper is still running
            if helper.poll() is not None:
                stderr_out = helper.stderr.read() if helper.stderr else ""
                print(
                    f"\n{error('Error:')} Swift helper exited (code {helper.returncode})",
                    file=sys.stderr,
                )
                if stderr_out:
                    print(f"Helper stderr: {stderr_out}", file=sys.stderr)
                break

            # Dequeue and process frames (fair per-display latest-frame mailbox)
            try:
                meta, jpeg_data, receipt_mono = server.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            dequeue_mono = time.monotonic()
            queue_age_s = dequeue_mono - receipt_mono
            infer_start = time.monotonic()
            detections = detector.detect(jpeg_data)
            inference_age_s = time.monotonic() - infer_start
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
            # Tracker stays in inference-frame coordinates; motion projects only
            # the outgoing copy and never feeds back into tracker state.
            boxes = tracker.update(meta.display_id, boxes)
            send_boxes = boxes
            if motion_compensation and send_boxes:
                pending = server.frame_queue.peek(meta.display_id)
                if pending is not None:
                    pending_meta, pending_jpeg, pending_receipt = pending
                    if (
                        pending_receipt > receipt_mono
                        and pending_meta.display_id == meta.display_id
                        and pending_meta.width == meta.width
                        and pending_meta.height == meta.height
                        and not is_frame_stale(pending_receipt, time.monotonic(), max_frame_age_ms)
                    ):
                        from bsafe.motion import compensate_boxes

                        send_boxes = compensate_boxes(
                            jpeg_data, pending_jpeg, list(send_boxes), meta.width, meta.height
                        )
            # Stale gate immediately before send (send-start timestamp covers
            # inference plus tracker/motion elapsed time).
            send_start_mono = time.monotonic()
            if max_frame_age_ms > 0 and is_frame_stale(
                receipt_mono, send_start_mono, max_frame_age_ms
            ):
                stats.record_stale()
                tracker.clear(meta.display_id)
                server.send_censor(meta.display_id, meta.width, meta.height, [])
                if show_stats:
                    line = stats.maybe_log(server.frame_queue.replaced)
                    if line is not None:
                        print(dim(line), flush=True)
                continue
            server.send_censor(meta.display_id, meta.width, meta.height, send_boxes)
            stats.record(queue_age_s, inference_age_s, send_start_mono - receipt_mono)
            if show_stats:
                line = stats.maybe_log(server.frame_queue.replaced)
                if line is not None:
                    print(dim(line), flush=True)

    except KeyboardInterrupt:
        print(f"\n{bold('Shutting down...')}")
    finally:
        server.shutdown()
        detector.close()
        if show_stats:
            print(dim(stats.summary(server.frame_queue.replaced)), flush=True)
        if helper.poll() is None:
            helper.terminate()
            try:
                helper.wait(timeout=5)
            except subprocess.TimeoutExpired:
                helper.kill()
                helper.wait()
        print(bold("Stopped."))


def _expected_output(input_path: str, model: str | None) -> str:
    """Compute the default output path for an input file."""
    from bsafe.detector import DEFAULT_MODEL

    stem, ext = os.path.splitext(input_path)
    tag = model or DEFAULT_MODEL
    return f"{stem}.{tag}.bsafe{ext}"


def _filter_inputs(
    inputs: list[str], supported_extensions: set[str], file_type: str, model: str | None
) -> tuple[list[str], int, int]:
    """Filter input paths for batch processing.

    Returns (to_process, skipped_existing, filtered_out).
    """
    filtered_out = 0
    skipped_existing = 0
    not_found = 0
    to_process = []

    for path in inputs:
        if os.path.isdir(path):
            filtered_out += 1
            continue
        if not os.path.isfile(path):
            print(f"{warn('Warning:')} file not found: {path}", file=sys.stderr)
            not_found += 1
            continue
        _, ext = os.path.splitext(path)
        if ext.lower() not in supported_extensions:
            filtered_out += 1
            continue
        if ".bsafe." in os.path.basename(path):
            filtered_out += 1
            continue
        expected = _expected_output(path, model)
        if os.path.exists(expected):
            skipped_existing += 1
            continue
        to_process.append(path)

    # Print summary
    if to_process:
        names = [os.path.basename(p) for p in to_process[:5]]
        listing = ", ".join(names)
        if len(to_process) > 5:
            listing += f", ... and {len(to_process) - 5} more"
        print(f"Found {info(str(len(to_process)))} {file_type}(s): {listing}")
    if filtered_out > 0:
        print(dim(f"Skipped {filtered_out} non-{file_type} file(s)"))
    if not_found > 0:
        print(dim(f"Skipped {not_found} missing file(s)"))
    if skipped_existing > 0:
        print(dim(f"Skipping {skipped_existing} file(s) with existing output"))
    if to_process:
        print(f"Processing {info(str(len(to_process)))} file(s)...")

    return to_process, skipped_existing, filtered_out


def _print_batch_summary(total: int, skipped: int, errors: list[tuple[str, str]]) -> None:
    """Print a summary after batch processing."""
    processed = total - len(errors)
    parts = [f"{success(str(processed))} processed"]
    if skipped:
        parts.append(f"{dim(str(skipped))} skipped")
    if errors:
        parts.append(f"{error(str(len(errors)))} failed")
    print(f"Batch complete: {', '.join(parts)}")
    if errors:
        for path, msg in errors:
            print(f"  {os.path.basename(path)}: {msg}", file=sys.stderr)


def _video_worker(
    error_queue,
    input_path: str,
    kwargs: dict,
) -> None:
    """Top-level function for subprocess video processing (must be picklable)."""
    try:
        from bsafe.video import process_video

        process_video(input_path, **kwargs)
    except Exception as e:
        error_queue.put(str(e))
        raise


def _run_file_subprocess(target, *args) -> tuple[int, str | None]:
    """Run target(*args) in a spawned subprocess. Returns (exit_code, error_message)."""
    ctx = multiprocessing.get_context("spawn")
    error_queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(error_queue, *args))
    proc.start()
    proc.join()
    error_msg = None
    if not error_queue.empty():
        error_msg = error_queue.get_nowait()
    exit_code = proc.exitcode if proc.exitcode is not None else 1
    return exit_code, error_msg


# Process tree architecture for batch mode:
#
# bsafe video *.mp4 --blur
#   main process (CLI, orchestration loop)
#     ├── file_1 subprocess (process_video → spawns chunk subprocesses)
#     │     ├── chunk_0 subprocess (loads model, processes frames)
#     │     ├── chunk_1 subprocess
#     │     └── ...
#     ├── file_2 subprocess
#     │     └── ...
#     └── gc.collect() between files
#
# bsafe image *.jpg --pixels
#   main process (CLI, orchestration loop)
#     ├── process_image(file_1) → gc.collect()  (detector created+closed internally)
#     ├── process_image(file_2) → gc.collect()
#     └── ...
#
# Video uses subprocess-per-file for full memory isolation (videos are heavy).
# Image processes in-process sequentially (images are lightweight).


def cmd_video(args):
    _validate_censor_args(args)
    _resolve_confidence(args)
    _warn_erax_unsupported(args)
    _print_config(args)

    from bsafe.censor import CensorConfig
    from bsafe.video import SUPPORTED_EXTENSIONS, process_video

    censor_config = CensorConfig(
        preset=args.censor,
        covered=args.covered,
        face_male=args.face_male,
        face_female=args.face_female,
        feet=args.feet,
    )

    inputs = args.input
    if args.output and len(inputs) > 1:
        print(
            f"{error('Error:')} -o/--output cannot be used with multiple input files",
            file=sys.stderr,
        )
        sys.exit(1)

    video_kwargs = dict(
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
        enhance=args.enhance,
    )

    if len(inputs) == 1:
        # Single-file mode: preserve existing behavior exactly
        try:
            process_video(inputs[0], output_path=args.output, **video_kwargs)
            print("\a", end="", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"\n{error('Error:')} {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Batch mode
    to_process, skipped, _ = _filter_inputs(inputs, SUPPORTED_EXTENSIONS, "video", model=args.model)
    if not to_process:
        print("Nothing to process.")
        return

    errors: list[tuple[str, str]] = []
    for i, path in enumerate(to_process, 1):
        print(f"\n{info(f'[{i}/{len(to_process)}]')} {os.path.basename(path)}")
        try:
            exit_code, error_msg = _run_file_subprocess(
                _video_worker,
                path,
                {**video_kwargs, "output_path": None},
            )
            if exit_code != 0:
                errors.append((path, error_msg or f"exit code {exit_code}"))
        except KeyboardInterrupt:
            print(f"\nInterrupted after {i - 1}/{len(to_process)} file(s).")
            sys.exit(130)
        gc.collect()

    _print_batch_summary(len(to_process), skipped, errors)
    print("\a", end="", flush=True)
    if errors:
        sys.exit(1)


def cmd_image(args):
    _validate_censor_args(args)
    _resolve_confidence(args)
    _warn_erax_unsupported(args)
    _print_config(args)

    from bsafe.censor import CensorConfig
    from bsafe.image import SUPPORTED_EXTENSIONS, process_image

    censor_config = CensorConfig(
        preset=args.censor,
        covered=args.covered,
        face_male=args.face_male,
        face_female=args.face_female,
        feet=args.feet,
    )

    inputs = args.input
    if args.output and len(inputs) > 1:
        print(
            f"{error('Error:')} -o/--output cannot be used with multiple input files",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(inputs) == 1:
        # Single-file mode: preserve existing behavior exactly
        try:
            output_path = process_image(
                inputs[0],
                output_path=args.output,
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
            print(f"\n{error('Error:')} {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Batch mode
    to_process, skipped, _ = _filter_inputs(inputs, SUPPORTED_EXTENSIONS, "image", model=args.model)
    if not to_process:
        print("Nothing to process.")
        return

    errors: list[tuple[str, str]] = []
    for i, path in enumerate(to_process, 1):
        print(f"\n{info(f'[{i}/{len(to_process)}]')} {os.path.basename(path)}")
        try:
            output_path = process_image(
                path,
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
            print(f"  → {output_path}")
        except KeyboardInterrupt:
            print(f"\nInterrupted after {i - 1}/{len(to_process)} file(s).")
            sys.exit(130)
        except Exception as e:
            print(f"  {error('Error:')} {e}", file=sys.stderr)
            errors.append((path, str(e)))
        gc.collect()

    _print_batch_summary(len(to_process), skipped, errors)
    print("\a", end="", flush=True)
    if errors:
        sys.exit(1)


def cmd_displays(args):
    from bsafe.swift_helper import list_displays

    try:
        displays = list_displays()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"{error('Error:')} {e}", file=sys.stderr)
        sys.exit(1)

    if not displays:
        print("No displays found.")
        return

    print(f"{'ID':<12} {'Resolution':<16} {'Role'}")
    print(dim(f"{'─' * 12} {'─' * 16} {'─' * 11}"))
    for d in displays:
        res = f"{d['width']}x{d['height']}"
        label = bold("(primary)") if d.get("primary") else "(secondary)"
        print(f"{d['id']:<12} {res:<16} {label}")


def cmd_bootstrap(args):
    import subprocess

    from bsafe.config import CONFIG_PATH, generate_default_config

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    swift_dir = os.path.join(project_dir, "swift")

    # Step 1: uv sync
    print("Installing Python dependencies...")
    result = subprocess.run(["uv", "sync"], cwd=project_dir)
    if result.returncode != 0:
        print(f"{error('Error:')} uv sync failed", file=sys.stderr)
        sys.exit(1)
    print(f"Python dependencies {success('OK')}\n")

    # Step 2: Build Swift helper
    print("Building Swift helper...")
    if not os.path.isdir(swift_dir):
        print(f"{error('Error:')} swift directory not found at {swift_dir}", file=sys.stderr)
        sys.exit(1)
    result = subprocess.run(["swift", "build", "-c", "release"], cwd=swift_dir)
    if result.returncode != 0:
        print(f"{error('Error:')} Swift build failed", file=sys.stderr)
        sys.exit(1)
    print(f"Swift helper {success('OK')}\n")

    # Step 3: Ensure models directory exists
    models_dir = os.path.expanduser("~/.config/bsafe/models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models directory: {models_dir} {success('OK')}\n")

    # Step 4: Create config file if it doesn't exist
    config_path = CONFIG_PATH
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            f.write(generate_default_config())
        print(f"Config file: {config_path} {success('CREATED')}\n")
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
        print(f" {success('OK')}")
    else:
        print(f" {warn('WARN')}: expected >= 3.14")
        all_ok = False

    # macOS platform
    print(f"Platform: {sys.platform}", end="")
    if sys.platform == "darwin":
        print(f" {success('OK')}")
    else:
        print(f" {warn('WARN')}: bsafe requires macOS")
        all_ok = False

    # Config file (informational only)
    print(f"Config file: {CONFIG_PATH}", end="")
    if os.path.exists(CONFIG_PATH):
        print(f" {success('OK')}")
    else:
        print(f" {warn('NOT FOUND')} (optional — run bootstrap to create)")

    # NudeNet importable
    print("NudeNet: ", end="")
    try:
        import nudenet  # noqa: F401

        print(success("OK"))
    except ImportError:
        print(f"{warn('NOT FOUND')} — run: uv sync")
        all_ok = False

    # Swift helper binary
    print("Swift helper: ", end="")
    from bsafe.swift_helper import find_helper

    helper = find_helper()
    if helper:
        print(f"{success('OK')} ({helper})")
    else:
        print(f"{warn('NOT FOUND')} — build with: cd swift && swift build -c release")
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
            print(success("OK"))
        else:
            print(
                f"{warn('NOT GRANTED')} — open System Settings > Privacy & Security > Screen Recording "
                "and enable your terminal app"
            )
            all_ok = False
    else:
        print(f"{warn('SKIPPED')} (Swift helper not found)")
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


def _build_parser():
    """Construct the CLI argument parser. Returns (parser, start_parser, video_parser, image_parser)."""
    parser = argparse.ArgumentParser(prog="bsafe", description="Censor NSFW content on screen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start subcommand
    start_parser = subparsers.add_parser("start", help="Start the censoring process")
    start_parser.add_argument("--fps", type=int, default=45, help="Capture FPS (default: 45)")
    _add_censor_args(start_parser)
    _add_temporal_args(start_parser)
    start_parser.add_argument(
        "--display",
        default="primary",
        help="Display to capture: primary, secondary, all, or numeric ID (default: primary)",
    )
    start_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run without Swift helper or detector",
    )
    start_parser.add_argument(
        "--max-frame-age-ms",
        type=int,
        default=250,
        help="Drop results older than this receipt-to-send budget in ms (default: 250, 0 disables)",
    )
    start_parser.add_argument(
        "--stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log live receive-to-send aggregate stats every ~2s (not capture-to-render)",
    )
    start_parser.add_argument(
        "--inference-resolution",
        type=int,
        default=320,
        choices=[320, 640, 960],
        help="NudeNet input resolution (default: 320). Higher recalls small regions at higher CPU cost",
    )
    start_parser.add_argument(
        "--detail-scan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="NudeNet-only: full frame plus overlapping 2x2 tiles (more recall, more CPU)",
    )
    start_parser.add_argument(
        "--motion-compensation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Map inference boxes to the newest pending frame (more responsive, more CPU)",
    )

    # video subcommand
    video_parser = subparsers.add_parser(
        "video", help="Process a video file and output a censored copy"
    )
    video_parser.add_argument("input", nargs="+", help="Path(s) to video file(s)")
    video_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: <input>.<model>.bsafe.<ext>)",
    )
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
    video_parser.add_argument(
        "--enhance",
        choices=["dim"],
        default=None,
        help="enhancement mode: 'dim' for low-light footage (default: disabled)",
    )
    _add_censor_args(video_parser)
    _add_temporal_args(video_parser)

    # image subcommand
    image_parser = subparsers.add_parser(
        "image", help="Process an image file and output a censored copy"
    )
    image_parser.add_argument("input", nargs="+", help="Path(s) to image file(s)")
    image_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: <input>.<model>.bsafe.<ext>)",
    )
    _add_censor_args(image_parser)

    subparsers.add_parser("displays", help="List connected displays")
    subparsers.add_parser("doctor", help="Check system requirements")
    subparsers.add_parser("bootstrap", help="Install dependencies and build Swift helper")

    return parser, start_parser, video_parser, image_parser


class _DetectionColorFilter(logging.Filter):
    """Color [detection] prefixes magenta in verbose log output."""

    def filter(self, record):
        from bsafe.style import detection

        if record.msg and "[detection]" in str(record.msg):
            record.msg = record.msg.replace("[detection]", detection("[detection]"), 1)
        return True


def _add_detection_filter():
    """Add detection coloring filter to the bsafe logger."""
    bsafe_logger = logging.getLogger("bsafe")
    bsafe_logger.addFilter(_DetectionColorFilter())


def main():
    from bsafe.config import load_config

    parser, start_parser, video_parser, image_parser = _build_parser()

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
        _add_detection_filter()
    else:
        logging.basicConfig(level=logging.WARNING)

    commands = {
        "start": cmd_start,
        "displays": cmd_displays,
        "doctor": cmd_doctor,
        "bootstrap": cmd_bootstrap,
        "video": cmd_video,
        "image": cmd_image,
    }

    faulthandler.enable()
    commands[args.command](args)
