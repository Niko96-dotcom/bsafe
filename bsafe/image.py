"""Image processing: read a single image, run detection, write censored output.

Privacy note: unlike live screen capture, this command intentionally writes a
censored copy of the user-provided image to disk.  This is the explicit purpose
of the command (the user asks for a file) and does not violate the "no
persistence of screen content" rule, which targets involuntary/ambient capture.
"""

import logging
import os

import cv2

from bsafe.censor import (
    FULL_CENSOR_MULTIPLIER,
    CensorConfig,
    build_censor_boxes,
    expand_boxes,
)
from bsafe.detector import Detector
from bsafe.render import render_censors

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def process_image(
    input_path: str,
    *,
    output_path: str | None = None,
    confidence: float | None = None,
    censor_config: CensorConfig | None = None,
    padding: float = 0.0,
    blur: float = 0.0,
    pixels: float = 0.0,
    censor_text: str | None = None,
    full_censor: bool = False,
    model: str | None = None,
    verbose: bool = False,
) -> str:
    """Process a single image file and write a censored copy.

    Returns the output file path.
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
    else:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.isdir(output_dir):
            raise ValueError(f"output directory does not exist: {output_dir}")
    if os.path.exists(output_path):
        raise ValueError(f"output already exists: {output_path}")

    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError(f"could not read image: {input_path}")

    h, w = frame.shape[:2]
    config = censor_config or CensorConfig()
    classes = config.resolve_classes()

    if verbose:
        print(f"Processing {input_path} ({w}x{h})...", flush=True)

    detector = Detector(min_confidence=confidence, model=model)
    detections = detector.detect_frame(frame)
    logger.debug("Detections: %d", len(detections))
    for d in detections:
        logger.debug("[detection] %s (%.2f) at %s", d.class_name, d.confidence, d.box)

    boxes = build_censor_boxes(detections, classes, padding, w, h)
    if full_censor:
        boxes = expand_boxes(boxes, FULL_CENSOR_MULTIPLIER, w, h)
    render_censors(frame, boxes, blur=blur, pixels=pixels, censor_text=censor_text)
    detector.close()

    cv2.imwrite(output_path, frame)
    if verbose:
        print(f"Saved to {output_path} ({len(detections)} detection(s))", flush=True)

    return output_path
