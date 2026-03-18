"""Video processing pipeline: read a video file, run detection, write censored output."""

import logging
import os
import shutil
import subprocess
import time

import cv2

from bsafe.censor import (
    CENSOR_PRESETS,
    FULL_CENSOR_MULTIPLIER,
    build_censor_boxes,
    expand_boxes,
)
from bsafe.detector import Detector
from bsafe.render import render_censors
from bsafe.tracking import BoxTracker

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".m4v", ".mov"}


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


def process_video(
    input_path: str,
    *,
    confidence: float = 0.0,
    censor: str = "all",
    padding: float = 0.0,
    persist_frames: int = 8,
    smooth_alpha: float = 0.5,
    blur: float = 0.0,
    pixels: float = 0.0,
    censor_text: str | None = None,
    full_censor: bool = False,
    fps_override: int | None = None,
    verbose: bool = False,
) -> str:
    """Process a video file and write a censored copy.

    Returns the output file path.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If the file format is unsupported or output already exists.
        RuntimeError: If the video cannot be opened or the writer cannot be created.
    """
    # Validate input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"file not found: {input_path}")

    stem, ext = os.path.splitext(input_path)
    if ext.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    output_path = f"{stem}.bsafe{ext}"
    if os.path.exists(output_path):
        raise ValueError(f"output file already exists: {output_path}")

    tmp_path = f"{stem}.bsafe.tmp{ext}"

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {input_path}")
    print(f"  {width}x{height}, {native_fps:.1f} FPS, {total_frames} frames")

    # Determine detection interval
    detect_every = 1
    if fps_override and fps_override < native_fps:
        detect_every = max(1, round(native_fps / fps_override))
        print(f"  Detection every {detect_every} frames ({fps_override} detection FPS)")

    # Init detector and tracker
    print("Loading NudeNet model...", flush=True)
    detector = Detector(min_confidence=confidence)
    censor_classes = CENSOR_PRESETS[censor]
    tracker = BoxTracker(persist_frames=persist_frames, smooth_alpha=smooth_alpha)

    # Writer: mp4v codec, native FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, native_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        detector.close()
        raise RuntimeError("cannot create output video writer")

    # Process frames
    frame_idx = 0
    t_start = time.monotonic()
    last_progress = 0.0
    processing_ok = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % detect_every == 0:
                # Encode frame to JPEG for detector
                _, jpeg = cv2.imencode(".jpg", frame)
                jpeg_bytes = jpeg.tobytes()
                detections = detector.detect(jpeg_bytes)
                detected_boxes = build_censor_boxes(
                    detections, censor_classes, padding, width, height
                )
                if full_censor:
                    detected_boxes = expand_boxes(
                        detected_boxes, FULL_CENSOR_MULTIPLIER, width, height
                    )
            else:
                # No detection this frame — let tracker handle persistence/decay
                detected_boxes = []

            # Always update tracker so persistence/smoothing decay correctly
            boxes = tracker.update(0, detected_boxes)

            render_censors(frame, boxes, blur=blur, pixels=pixels, censor_text=censor_text)
            writer.write(frame)
            frame_idx += 1

            # Progress every ~0.5s
            now = time.monotonic()
            if now - last_progress >= 0.5 and total_frames > 0:
                elapsed = now - t_start
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                pct = frame_idx / total_frames * 100
                remaining = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
                eta = _fmt_duration(remaining)
                print(
                    f"\r  Frame {frame_idx}/{total_frames} ({pct:.0f}%) — "
                    f"{fps_actual:.1f} FPS — ETA {eta}",
                    end="",
                    flush=True,
                )
                last_progress = now

        processing_ok = True
    except Exception as exc:
        print(f"\n  Error at frame {frame_idx}: {exc}", flush=True)
        raise
    finally:
        print(flush=True)  # newline after \r progress
        cap.release()
        writer.release()
        detector.close()
        # Clean up partial temp file on failure
        if not processing_ok and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    elapsed = time.monotonic() - t_start
    print(f"\n  Processed {frame_idx} frames in {_fmt_duration(elapsed)}")

    # Mux audio with ffmpeg
    _mux_audio(tmp_path, input_path, output_path)

    print(f"Output: {output_path}")
    return output_path


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
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        logger.warning("ffmpeg error — audio will not be preserved")
        os.rename(tmp_path, output_path)
