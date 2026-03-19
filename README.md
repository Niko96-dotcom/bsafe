# Bsafe

## About

Censor NSFW content on your MacOS screen. CLI.

## Setting up locally

Requires macOS 14+, Python 3.14, and [uv](https://docs.astral.sh/uv/):

```sh
uv run bsafe bootstrap
```

This installs Python dependencies (`uv sync`) and builds the Swift
screen-capture helper. At the end it prints an alias line you can add
to your `~/.zshrc` to make `bsafe` globally available.

You must grant **Screen Recording** permission to your terminal app
(System Settings → Privacy & Security → Screen Recording).

## Usage

Check that everything is in order:

```sh
bsafe doctor
```

Start real-time screen censoring (no daemonized support yet):

```sh
bsafe start
```

### Display selection

By default `bsafe start` captures only the **primary** display. Most setups
have one primary and one secondary display — the aliases `primary` and
`secondary` are enough to pick either without looking up IDs.

List connected displays:

```sh
bsafe displays
```

Target a specific display:

```sh
bsafe start --display secondary   # first non-primary display
bsafe start --display 1234567     # by numeric ID (from bsafe displays)
bsafe start --display all         # all displays (uses more CPU)
```

There are bottlenecks. If you try to support a lot of displays/monitors, in high resolution,
with lots of NSFW content...
the more you increase even one of these factors, the more quality will be compromised, remember you
are processing video inputs and rendering video outputs in real time, this is heavy work.

### Process a video file

Produce a censored copy of a local video (the original is never modified):

```sh
bsafe video clip.mp4                  # → clip.bsafe.mp4
bsafe video clip.mp4 --blur           # blur instead of black boxes
bsafe video clip.mp4 --pixels         # pixelation effect
bsafe video clip.mp4 --censor-text    # overlay "NSFW" text
bsafe video clip.mp4 --fps 5          # run detection at 5 FPS (output keeps native FPS)
```

Supported formats: `.mp4`, `.m4v`, `.mov`. If `ffmpeg` is installed, audio is
preserved in the output; otherwise the video is written without audio.

Videos are processed in chunks (default: 5000 frames). If the process is killed
(e.g. by the OS due to memory pressure), re-run the same command to resume —
completed chunks are preserved on disk and skipped automatically. All chunks are
combined into the final output at the end.

### Configuration file

`bsafe bootstrap` creates `~/.config/bsafe/config.toml` with all default values.
Edit this file to set your preferred defaults — CLI flags always override it.

Sections: `[start]` for start-only flags, `[video]` for video-only flags,
`[common]` for shared flags used by both commands.

### Options

Shared flags (work with both `start` and `video`):

- `--confidence N` — minimum detection confidence, 0.0–1.0 (default: 0.0 for NudeNet, 0.2 for EraX)
- `--censor {female,male,all}` — what to censor (default: all)
- `--padding N` — box expansion fraction (default: 0.0)
- `--persist-frames N` — frames a box persists after disappearing (default: 8)
- `--smooth-alpha N` — EMA smoothing weight, 0.0–1.0 (default: 0.5)
- `--blur [N]` — blur effect (default intensity: 1.0)
- `--pixels [N]` — pixelation effect (default intensity: 1.0)
- `--censor-text [TEXT]` — overlay text on censored regions (default: "NSFW")
- `--model {320n,640m,erax-nano,erax-small,erax-medium}` — detection model (default: `320n`). See model details below
- `--covered` — also censor covered body parts (anus, buttocks; breasts when `--censor` is `female` or `all`)
- `--face-male` — also censor male faces
- `--face-female` — also censor female faces
- `--feet` — also censor exposed feet
- `--full-censor` — expand censor area by 3x
- `-v` / `--verbose` — enable debug logging

`start`-only flags:

- `--fps N` — capture frames per second (default: 45)
- `--display VALUE` — display to capture: `primary`, `secondary`, `all`, or numeric ID (default: `primary`)
- `--dry-run` — run the loop without the Swift helper or detector

`image`-only flags:

- `-o` / `--output PATH` — output file path (default: `<input>.bsafe.<ext>`)

`video`-only flags:

- `-o` / `--output PATH` — output file path (default: `<input>.bsafe.<ext>`)
- `--fps N` — detection FPS override (default: native video FPS)
- `--chunk-frames N` — frames per processing chunk (default: 5000). Smaller chunks use less memory but may cause brief tracking gaps at chunk boundaries.

### Detection models

The default `320n` model is bundled with NudeNet. Other models require a manual download.

| Model | Backend | Size | Notes |
|-------|---------|------|-------|
| `320n` | NudeNet | bundled | Fast, default |
| `640m` | NudeNet | ~90 MB | More accurate |
| `erax-nano` | EraX YOLO | ~5 MB | Fastest EraX, mAP 0.438 |
| `erax-small` | EraX YOLO | ~40 MB | Balanced, mAP 0.453 |
| `erax-medium` | EraX YOLO | ~19 MB | Best EraX accuracy, mAP 0.467 |

NudeNet models have broader coverage, detecting faces, covered parts, and feet.
EraX models are more targeted/specific, for example detecting nipples specifically.

As for computational efforts, the default NudeNet (`320n`) is lightweight and best for `bsafe start`,
where keeping a decent FPS matters. EraX models are heavier and will drop frames in real time, but
work great with `bsafe image` and `bsafe video` where auto FPS removes that constraint.

#### Using the 640m model

```sh
mkdir -p ~/.config/bsafe/models
curl -Lo ~/.config/bsafe/models/640m.onnx \
  https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx
bsafe start --model 640m
```

#### Using EraX models

EraX models use [ultralytics](https://github.com/ultralytics/ultralytics) YOLO and require an extra dependency:

```sh
uv sync --extra erax
```

Download a model (e.g. `erax-nano`):

```sh
mkdir -p ~/.config/bsafe/models
curl -Lo ~/.config/bsafe/models/erax-anti-nsfw-yolo11n-v1.1.pt \
  https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1/resolve/main/erax-anti-nsfw-yolo11n-v1.1.pt
```

Other variants:

```sh
# erax-small
curl -Lo ~/.config/bsafe/models/erax-anti-nsfw-yolo11s-v1.1.pt \
  https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1/resolve/main/erax-anti-nsfw-yolo11s-v1.1.pt

# erax-medium
curl -Lo ~/.config/bsafe/models/erax-anti-nsfw-yolo11m-v1.1.pt \
  https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1/resolve/main/erax-anti-nsfw-yolo11m-v1.1.pt
```

Then use with `--model`:

```sh
bsafe start --model erax-nano
bsafe image photo.jpg --model erax-small
bsafe video clip.mp4 --model erax-medium
```

## License

Under [MIT License](./LICENSE).

Copyright (c) 2026 Marcell "Mazuh" G. C. da Silva.
