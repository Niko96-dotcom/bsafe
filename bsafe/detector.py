"""NudeNet wrapper: takes frames or JPEG bytes, returns detections."""

import logging
import os
import tempfile
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

KNOWN_MODELS: dict[str, str | None] = {
    "320n": None,  # bundled with nudenet
    "640m": "~/.config/bsafe/models/640m.onnx",
}

_640M_DOWNLOAD_URL = (
    "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx"
)


def resolve_model(name: str | None) -> str | None:
    """Resolve a model name to a path (or None for the bundled default).

    Raises ValueError for unknown names or missing model files.
    """
    if name is None or name == "320n":
        return None

    if name not in KNOWN_MODELS:
        raise ValueError(f"unknown model '{name}'. Known models: {', '.join(sorted(KNOWN_MODELS))}")

    raw_path = KNOWN_MODELS[name]
    assert raw_path is not None
    path = os.path.expanduser(raw_path)

    if not os.path.isfile(path):
        raise ValueError(
            f"model file not found: {path}\n"
            f"Download it with:\n"
            f"  mkdir -p ~/.config/bsafe/models && curl -Lo {path} {_640M_DOWNLOAD_URL}"
        )

    return path


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h


class Detector:
    """Wraps NudeDetector for frames or JPEG bytes → detections."""

    def __init__(self, min_confidence: float = 0.5, model: str | None = None):
        from nudenet import NudeDetector

        self.min_confidence = min_confidence
        model_path = resolve_model(model)
        self._detector = NudeDetector(model_path=model_path) if model_path else NudeDetector()
        # Reusable temp file for JPEG-bytes path (used by live capture)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="bsafe-det-", delete=False)
        self._tmp_path = tmp.name
        tmp.close()
        logger.info("NudeNet detector initialized (model=%s)", model or "320n")

    def close(self):
        try:
            os.unlink(self._tmp_path)
        except OSError:
            pass

    def _parse_results(self, results: list[dict]) -> list[Detection]:
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

    def detect(self, jpeg_bytes: bytes) -> list[Detection]:
        """Run inference on JPEG bytes. Returns filtered detections."""
        # NudeNet requires a file path — reuse a single temp file
        with open(self._tmp_path, "wb") as f:
            f.write(jpeg_bytes)
        results = self._detector.detect(self._tmp_path)
        return self._parse_results(results)

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a BGR numpy frame directly (no JPEG roundtrip)."""
        results = self._detector.detect(frame)
        return self._parse_results(results)
