"""Video processing pipeline: read a video file, run detection, write censored output."""

import ctypes
import ctypes.util
import gc
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time

import cv2

from bsafe.censor import (
    FULL_CENSOR_MULTIPLIER,
    CensorConfig,
    build_censor_boxes,
    expand_boxes,
)
from bsafe.detector import Detector
from bsafe.render import render_censors
from bsafe.tracking import BoxTracker

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".m4v", ".mov"}
_CHUNK_FRAMES = 5000  # frames per chunk — memory resets between chunks
_MEMORY_HIGH_MB = 4096  # log a warning when RSS exceeds this
_MAX_CHUNK_RETRIES = 3


def _get_rss_mb() -> float:
    """Return current RSS in MB (not peak)."""
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"))

            class _TaskVMInfo(ctypes.Structure):
                _fields_ = [
                    ("virtual_size", ctypes.c_uint64),
                    ("resident_size", ctypes.c_uint64),
                    ("resident_size_max", ctypes.c_uint64),
                ]

            TASK_VM_INFO = 22
            vm_info = _TaskVMInfo()
            count = ctypes.c_uint32(ctypes.sizeof(vm_info) // ctypes.sizeof(ctypes.c_int))
            task = libc.mach_task_self()
            kr = libc.task_info(task, TASK_VM_INFO, ctypes.byref(vm_info), ctypes.byref(count))
            if kr == 0:
                return vm_info.resident_size / (1024 * 1024)
        except OSError:
            pass
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except FileNotFoundError, OSError:
        pass
    import resource

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return maxrss / divisor


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _chunk_path(chunks_dir: str, index: int) -> str:
    return os.path.join(chunks_dir, f"chunk_{index:04d}.mp4")


def _chunk_frame_count(path: str) -> int:
    """Return frame count of a video file, or 0 if unreadable."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _find_completed_chunks(
    chunks_dir: str, expected_size: int, last_chunk_size: int, total_chunks: int
) -> int:
    """Return the number of fully completed chunks (contiguous from 0)."""
    completed = 0
    for i in range(total_chunks):
        path = _chunk_path(chunks_dir, i)
        if not os.path.isfile(path):
            break
        expected = last_chunk_size if i == total_chunks - 1 else expected_size
        actual = _chunk_frame_count(path)
        if actual < expected:
            # Partial chunk — will be re-processed
            try:
                os.unlink(path)
            except OSError:
                pass
            break
        completed += 1
    # Clean up any chunks after the break point
    for i in range(completed, total_chunks + 1):
        path = _chunk_path(chunks_dir, i)
        if os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass
    return completed


def _process_chunk(
    input_path: str,
    chunk_output: str,
    start_frame: int,
    num_frames: int,
    *,
    native_fps: float,
    width: int,
    height: int,
    detect_every: int,
    confidence: float,
    censor_config: CensorConfig,
    padding: float,
    persist_frames: int,
    smooth_alpha: float,
    blur: float,
    pixels: float,
    censor_text: str | None,
    full_censor: bool,
    model: str | None,
    total_frames: int,
    t_start: float,
) -> None:
    """Process a single chunk of frames."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")

    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detector = Detector(min_confidence=confidence, model=model)
    censor_classes = censor_config.resolve_classes()
    tracker = BoxTracker(persist_frames=persist_frames, smooth_alpha=smooth_alpha)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(chunk_output, fourcc, native_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        detector.close()
        raise RuntimeError(f"cannot create chunk writer: {chunk_output}")

    last_progress = 0.0
    memory_warned = False

    try:
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            global_frame = start_frame + i
            if global_frame % detect_every == 0:
                detections = detector.detect_frame(frame)
                detected_boxes = build_censor_boxes(
                    detections, censor_classes, padding, width, height
                )
                if full_censor:
                    detected_boxes = expand_boxes(
                        detected_boxes, FULL_CENSOR_MULTIPLIER, width, height
                    )
            else:
                detected_boxes = []

            boxes = tracker.update(0, detected_boxes)
            render_censors(frame, boxes, blur=blur, pixels=pixels, censor_text=censor_text)
            writer.write(frame)

            # Progress every ~0.5s
            now = time.monotonic()
            if now - last_progress >= 0.5 and total_frames > 0:
                elapsed = now - t_start
                done = global_frame + 1
                fps_actual = done / elapsed if elapsed > 0 else 0
                pct = done / total_frames * 100
                remaining = (total_frames - done) / fps_actual if fps_actual > 0 else 0
                eta = _fmt_duration(remaining)
                mem_mb = _get_rss_mb()

                if not memory_warned and mem_mb > _MEMORY_HIGH_MB:
                    memory_warned = True
                    print(
                        f"\n  Warning: high memory usage ({mem_mb:.0f} MB)",
                        file=sys.stderr,
                        flush=True,
                    )

                print(
                    f"\r  Frame {done}/{total_frames} ({pct:.0f}%) — "
                    f"{fps_actual:.1f} FPS — ETA {eta} — {mem_mb:.0f} MB",
                    end="",
                    flush=True,
                )
                last_progress = now
    finally:
        cap.release()
        writer.release()
        detector.close()


def _run_chunk_subprocess(**kwargs) -> int:
    """Run _process_chunk in a spawned subprocess and return its exit code.

    Returns 0 on success, negative value if killed by signal (e.g. -9 for
    SIGKILL/OOM), or 1 on any other failure.
    """
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_process_chunk, kwargs=kwargs)
    proc.start()
    proc.join()
    return proc.exitcode if proc.exitcode is not None else 1


def _concat_chunks(chunks_dir: str, chunk_count: int, output_path: str, input_path: str) -> None:
    """Concatenate chunk files and mux audio from the original."""
    ffmpeg = shutil.which("ffmpeg")

    if chunk_count == 1:
        single = _chunk_path(chunks_dir, 0)
        if ffmpeg:
            _mux_audio(single, input_path, output_path)
        else:
            os.rename(single, output_path)
        return

    if not ffmpeg:
        logger.warning("ffmpeg not found — cannot concatenate chunks or preserve audio")
        # Fall back: just use the first chunk (better than nothing)
        os.rename(_chunk_path(chunks_dir, 0), output_path)
        return

    # Write ffmpeg concat list
    list_path = os.path.join(chunks_dir, "concat.txt")
    with open(list_path, "w") as f:
        for i in range(chunk_count):
            # Use just the filename since concat.txt lives inside chunks_dir
            # and ffmpeg resolves paths relative to the concat file's location
            fname = f"chunk_{i:04d}.mp4".replace("'", "'\\''")
            f.write(f"file '{fname}'\n")

    # Concat chunks + re-encode to H.264 + mux audio
    tmp_concat = os.path.join(chunks_dir, "concat.mp4")
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
                "-y",
                tmp_concat,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg concat failed (exit %d): %s", result.returncode, result.stderr)
            # Try without audio
            result2 = subprocess.run(
                [
                    ffmpeg,
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    list_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-y",
                    tmp_concat,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result2.returncode != 0:
                logger.warning("ffmpeg concat (no audio) also failed: %s", result2.stderr)
                os.rename(_chunk_path(chunks_dir, 0), output_path)
                return
            logger.warning("audio not preserved")

        os.rename(tmp_concat, output_path)
    except subprocess.TimeoutExpired, FileNotFoundError:
        logger.warning("ffmpeg error — falling back to first chunk only")
        os.rename(_chunk_path(chunks_dir, 0), output_path)


def _mux_audio(tmp_path: str, input_path: str, output_path: str) -> None:
    """Mux audio from original into the censored video via ffmpeg.

    Falls back to renaming if ffmpeg is unavailable or fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found — audio will not be preserved")
        os.rename(tmp_path, output_path)
        return

    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-i",
                tmp_path,
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
                "-y",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            logger.warning(
                "ffmpeg failed (exit %d) — audio will not be preserved: %s",
                result.returncode,
                result.stderr,
            )
            os.rename(tmp_path, output_path)
        else:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except subprocess.TimeoutExpired, FileNotFoundError:
        logger.warning("ffmpeg error — audio will not be preserved")
        os.rename(tmp_path, output_path)


def process_video(
    input_path: str,
    *,
    output_path: str | None = None,
    confidence: float = 0.0,
    censor_config: CensorConfig = CensorConfig(),
    padding: float = 0.0,
    persist_frames: int = 8,
    smooth_alpha: float = 0.5,
    blur: float = 0.0,
    pixels: float = 0.0,
    censor_text: str | None = None,
    full_censor: bool = False,
    fps_override: int | None = None,
    model: str | None = None,
    chunk_frames: int | None = None,
    verbose: bool = False,
) -> str:
    """Process a video file and write a censored copy.

    The video is processed in chunks to limit memory usage. If the process is
    interrupted (e.g. killed by the OS due to memory pressure), completed chunks
    are preserved on disk and re-running the same command resumes from where it
    left off. Once all chunks are done, they are combined into the final output.

    Returns the output file path.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If the file format is unsupported or output already exists.
        RuntimeError: If the video cannot be opened or the writer cannot be created.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"file not found: {input_path}")

    stem, ext = os.path.splitext(input_path)
    if ext.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if output_path is None:
        output_path = f"{stem}.bsafe{ext}"
    if os.path.exists(output_path):
        raise ValueError(f"output file already exists: {output_path}")

    # Probe video metadata
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Input: {input_path}")
    print(f"  {width}x{height}, {native_fps:.1f} FPS, {total_frames} frames")

    detect_every = 1
    if fps_override and fps_override < native_fps:
        detect_every = max(1, round(native_fps / fps_override))
        print(f"  Detection every {detect_every} frames ({fps_override} detection FPS)")

    # Chunk size
    frames_per_chunk = chunk_frames if chunk_frames else _CHUNK_FRAMES

    # Set up chunks directory
    chunks_dir = f"{stem}.bsafe.chunks"
    os.makedirs(chunks_dir, exist_ok=True)

    # Calculate chunk boundaries
    total_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk
    last_chunk_size = total_frames - (frames_per_chunk * (total_chunks - 1))

    print(f"  Processing in {total_chunks} chunks of {frames_per_chunk} frames")
    print(
        f"  Chunks are saved to: {chunks_dir}\n"
        f"  If the process is killed (e.g. by the OS due to high memory usage),\n"
        f"  re-run the same command to resume. Completed chunks will be skipped.\n"
        f"  All chunks are combined into the final output at the end."
    )

    # Check for completed chunks from a previous run
    completed = _find_completed_chunks(chunks_dir, frames_per_chunk, last_chunk_size, total_chunks)
    if completed > 0:
        skipped_frames = completed * frames_per_chunk
        print(
            f"  Resuming: {completed}/{total_chunks} chunks already done ({skipped_frames} frames)"
        )

    # SIGTERM handler
    def _sigterm_handler(signum, _frame):
        print(
            f"\n  Killed by signal {signum} (possible out-of-memory).\n"
            f"  Completed chunks are saved in: {chunks_dir}\n"
            f"  Re-run the same command to resume.\n"
            f"  To reduce memory, try a smaller --chunk-frames value.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(137)

    prev_sigterm = signal.signal(signal.SIGTERM, _sigterm_handler)
    t_start = time.monotonic()

    try:
        for chunk_idx in range(completed, total_chunks):
            start_frame = chunk_idx * frames_per_chunk
            expected = last_chunk_size if chunk_idx == total_chunks - 1 else frames_per_chunk
            chunk_file = _chunk_path(chunks_dir, chunk_idx)

            chunk_kwargs = dict(
                input_path=input_path,
                chunk_output=chunk_file,
                start_frame=start_frame,
                num_frames=expected,
                native_fps=native_fps,
                width=width,
                height=height,
                detect_every=detect_every,
                confidence=confidence,
                censor_config=censor_config,
                padding=padding,
                persist_frames=persist_frames,
                smooth_alpha=smooth_alpha,
                blur=blur,
                pixels=pixels,
                censor_text=censor_text,
                full_censor=full_censor,
                model=model,
                total_frames=total_frames,
                t_start=t_start,
            )

            for attempt in range(1, _MAX_CHUNK_RETRIES + 1):
                print(
                    f"\n  Chunk {chunk_idx + 1}/{total_chunks} — loading model...",
                    flush=True,
                )

                exit_code = _run_chunk_subprocess(**chunk_kwargs)

                if exit_code == 0:
                    break

                # Clean up partial chunk file
                if os.path.isfile(chunk_file):
                    try:
                        os.unlink(chunk_file)
                    except OSError:
                        pass

                if attempt < _MAX_CHUNK_RETRIES:
                    print(
                        f"\n  Chunk {chunk_idx + 1} failed (exit code {exit_code}), "
                        f"retry {attempt + 1}/{_MAX_CHUNK_RETRIES}...",
                        file=sys.stderr,
                        flush=True,
                    )
                    gc.collect()
                else:
                    raise RuntimeError(
                        f"Chunk {chunk_idx + 1} failed after {_MAX_CHUNK_RETRIES} "
                        f"attempts (last exit code: {exit_code}). "
                        f"Try reducing --chunk-frames to lower memory usage."
                    )

            # Force memory cleanup between chunks
            gc.collect()

    except Exception as exc:
        print(
            f"\n  Error: {exc}\n"
            f"  Completed chunks saved in: {chunks_dir}\n"
            f"  Re-run the same command to resume.",
            flush=True,
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)

    elapsed = time.monotonic() - t_start
    print(f"\n  Processed in {_fmt_duration(elapsed)}")

    # Concatenate chunks + mux audio
    print("  Assembling final video (combining all chunks)...", flush=True)
    _concat_chunks(chunks_dir, total_chunks, output_path, input_path)

    # Clean up chunks directory
    shutil.rmtree(chunks_dir, ignore_errors=True)

    print(f"Output: {output_path}")
    return output_path
