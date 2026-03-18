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


def filter_detections(
    detections: list[Detection],
    classes: frozenset[str] = CENSOR_PRESETS["all"],
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


def merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping (x, y, w, h) boxes into their bounding unions.

    Iterates until no more merges occur. O(n^2) per pass but n is single-digit.
    """
    if len(boxes) <= 1:
        return list(boxes)

    # Convert to (x1, y1, x2, y2)
    rects = [(x, y, x + w, y + h) for x, y, w, h in boxes]

    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(rects)

        for i in range(len(rects)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = rects[i]
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = rects[j]
                # Check intersection
                if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    used[j] = True
                    changed = True
            merged.append((ax1, ay1, ax2, ay2))

        rects = merged

    # Convert back to (x, y, w, h)
    return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in rects]


def build_censor_boxes(
    detections: list[Detection],
    classes: frozenset[str],
    padding: float,
    display_w: int,
    display_h: int,
) -> list[tuple[int, int, int, int]]:
    """Filter detections, pad boxes, merge overlaps, return list of (x, y, w, h) tuples."""
    filtered = filter_detections(detections, classes)
    padded = [pad_box(d.box, padding, display_w, display_h) for d in filtered]
    return merge_overlapping_boxes(padded)
