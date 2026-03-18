"""Censor filtering and box padding for NSFW detections."""

from bsafe.detector import Detection

# BUTTOCKS_EXPOSED is intentionally excluded — too many false positives in practice
# (e.g. tight clothing, seated posture) and low user-reported value for censoring.
CENSOR_PRESETS: dict[str, frozenset[str]] = {
    "female": frozenset(
        {
            "FEMALE_GENITALIA_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "ANUS_EXPOSED",
        }
    ),
    "male": frozenset(
        {
            "MALE_GENITALIA_EXPOSED",
            "ANUS_EXPOSED",
        }
    ),
    "all": frozenset(
        {
            "FEMALE_GENITALIA_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "ANUS_EXPOSED",
        }
    ),
}

DEFAULT_CENSOR_CLASSES = CENSOR_PRESETS["all"]


def filter_detections(
    detections: list[Detection],
    classes: frozenset[str] = DEFAULT_CENSOR_CLASSES,
) -> list[Detection]:
    """Return only detections whose class_name is in the given set."""
    return [d for d in detections if d.class_name in classes]


def pad_box(
    box: tuple[int, int, int, int],
    padding: float,
    display_w: int,
    display_h: int,
) -> tuple[int, int, int, int]:
    """Expand (x, y, w, h) by padding fraction, clamp to bounds, return (x, y, w, h)."""
    x, y, w, h = box
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    nx = max(0, x - pad_x)
    ny = max(0, y - pad_y)
    nx2 = min(display_w, x + w + pad_x)
    ny2 = min(display_h, y + h + pad_y)

    return (nx, ny, nx2 - nx, ny2 - ny)


def build_censor_boxes(
    detections: list[Detection],
    classes: frozenset[str],
    padding: float,
    display_w: int,
    display_h: int,
) -> list[tuple[int, int, int, int]]:
    """Filter detections, pad boxes, return list of (x, y, w, h) tuples."""
    filtered = filter_detections(detections, classes)
    return [pad_box(d.box, padding, display_w, display_h) for d in filtered]
