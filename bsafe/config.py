import os
import tomllib

from bsafe.style import warn

CONFIG_PATH = os.path.expanduser("~/.config/bsafe/config.toml")

_COMMON_KEYS = {
    "confidence",
    "censor",
    "padding",
    "persist_frames",
    "smooth_alpha",
    "blur",
    "pixels",
    "censor_text",
    "full_censor",
    "covered",
    "face_male",
    "face_female",
    "feet",
    "model",
    "verbose",
}

VALID_KEYS = {
    "start": {
        "fps",
        "dry_run",
        "display",
        "max_frame_age_ms",
        "stats",
        "inference_resolution",
        "detail_scan",
        "motion_compensation",
        "motion_lookahead_ms",
    }
    | _COMMON_KEYS,
    "video": {"fps", "chunk_frames", "enhance"} | _COMMON_KEYS,
    "image": _COMMON_KEYS.copy(),
    "common": _COMMON_KEYS,
}


def generate_default_config() -> str:
    return """\
# Bsafe configuration — edit values to change defaults.
# CLI flags always override these values.
# Remove a key (or set to its zero value) to use the hardcoded default.

[start]
fps = 45
display = "primary"
dry_run = false
# model =            # override [common] model for real-time screen censoring
# max_frame_age_ms = 250  # drop results older than this receipt-to-send budget (0 = disable)
# stats = false      # log live receive-to-send aggregate stats every ~2s (not capture-to-render)
# inference_resolution = 320  # NudeNet input resolution. Options: 320, 640, 960
# detail_scan = false  # NudeNet-only: full frame plus overlapping 2x2 tiles (more CPU)
# motion_compensation = false  # map inference boxes to newest pending frame (more CPU)
# motion_lookahead_ms = 0.0  # experimental: extrapolate past pending frame in ms (0 = off, 0-100, requires motion_compensation; may overshoot/reverse)

[video]
# fps =              # unset = use native video FPS
# chunk_frames =     # unset = 5000
# enhance =          # unset = disabled. Options: dim
# model =            # override [common] model for video processing

[image]
# model =            # override [common] model for image processing
# Note: persist_frames and smooth_alpha from [common] are ignored for images.

[common]
# confidence =       # unset = 0.0 for NudeNet, 0.2 for EraX
# censor = "all"       # Options: none, female, male, all
censor = "all"
padding = 0.0
persist_frames = 8
smooth_alpha = 0.5
blur = 0.0
pixels = 0.0
# censor_text =      # unset = disabled; set to "NSFW" or custom string
full_censor = false
covered = false
face_male = false
face_female = false
feet = false
# model =            # unset = "320n". Options: 320n, 640m, erax-nano, erax-small, erax-medium
verbose = false
"""


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load config from TOML file. Returns {} if missing or malformed."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        print(f"{warn('Warning:')} could not parse {path}: {e}")
        return {}
    for section, keys in config.items():
        if section not in VALID_KEYS:
            print(f"{warn('Warning:')} unknown config section [{section}] in {path}")
            continue
        if isinstance(keys, dict):
            for key in keys:
                if key not in VALID_KEYS[section]:
                    print(f"{warn('Warning:')} unknown key '{key}' in [{section}] in {path}")
    return config
