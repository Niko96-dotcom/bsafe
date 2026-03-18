"""NudeNet wrapper: takes JPEG bytes, returns detections."""

import logging
import os
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h


class Detector:
    """Wraps NudeDetector for JPEG bytes → detections."""

    def __init__(self, min_confidence: float = 0.5):
        from nudenet import NudeDetector

        self.min_confidence = min_confidence
        self._detector = NudeDetector()
        # Reusable temp file to avoid create/delete churn at capture FPS
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="bsafe-det-", delete=False)
        self._tmp_path = tmp.name
        tmp.close()
        logger.info("NudeNet detector initialized")

    def close(self):
        try:
            os.unlink(self._tmp_path)
        except OSError:
            pass

    def detect(self, jpeg_bytes: bytes) -> list[Detection]:
        """Run inference on JPEG bytes. Returns filtered detections."""
        # NudeNet requires a file path — reuse a single temp file
        with open(self._tmp_path, "wb") as f:
            f.write(jpeg_bytes)
        results = self._detector.detect(self._tmp_path)

        detections = []
        for r in results:
            confidence = r["score"]
            if confidence < self.min_confidence:
                continue
            box = r["box"]
            detections.append(
                Detection(
                    class_name=r["class"],
                    confidence=confidence,
                    box=(box[0], box[1], box[2], box[3]),
                )
            )
        return detections
