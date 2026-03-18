import os
import tomllib

CONFIG_PATH = os.path.expanduser("~/.config/bsafe/config.toml")

VALID_KEYS = {
    "start": {"fps", "dry_run"},
    "video": {"fps", "chunk_frames"},
    "image": set(),
    "common": {
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
    },
}


def generate_default_config() -> str:
    return """\
# Bsafe configuration — edit values to change defaults.
# CLI flags always override these values.
# Remove a key (or set to its zero value) to use the hardcoded default.

[start]
fps = 45
dry_run = false

[video]
# fps =              # unset = use native video FPS
# chunk_frames =     # unset = 5000

[image]
# No image-specific keys — uses [common] settings only.
# Note: persist_frames and smooth_alpha from [common] are ignored for images.

[common]
confidence = 0.0
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
# model =            # unset = "320n"
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
        print(f"Warning: could not parse {path}: {e}")
        return {}
    for section, keys in config.items():
        if section not in VALID_KEYS:
            print(f"Warning: unknown config section [{section}] in {path}")
            continue
        if isinstance(keys, dict):
            for key in keys:
                if key not in VALID_KEYS[section]:
                    print(f"Warning: unknown key '{key}' in [{section}] in {path}")
    return config
