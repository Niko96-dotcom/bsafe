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

### Options

Shared flags (work with both `start` and `video`):

- `--confidence N` — minimum detection confidence, 0.0–1.0 (default: 0.0)
- `--censor {female,male,all}` — what to censor (default: all)
- `--padding N` — box expansion fraction (default: 0.0)
- `--persist-frames N` — frames a box persists after disappearing (default: 8)
- `--smooth-alpha N` — EMA smoothing weight, 0.0–1.0 (default: 0.5)
- `--blur [N]` — blur effect (default intensity: 1.0)
- `--pixels [N]` — pixelation effect (default intensity: 1.0)
- `--censor-text [TEXT]` — overlay text on censored regions (default: "NSFW")
- `--model {320n,640m}` — detection model (default: `320n`, fast). `640m` is more accurate but requires a manual download (see below)
- `--full-censor` — expand censor area by 3x
- `-v` / `--verbose` — enable debug logging

`start`-only flags:

- `--fps N` — capture frames per second (default: 45)
- `--dry-run` — run the loop without the Swift helper or detector

`video`-only flags:

- `--fps N` — detection FPS override (default: native video FPS)

### Using the 640m model

The default `320n` model is fast but less accurate. For better detection, download the `640m` model:

```sh
mkdir -p ~/.bsafe/models
curl -Lo ~/.bsafe/models/640m.onnx \
  https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx
```

Then pass `--model 640m`:

```sh
bsafe start --model 640m
bsafe video clip.mp4 --model 640m
```

## License

Under [MIT License](./LICENSE).

Copyright (c) 2026 Marcell "Mazuh" G. C. da Silva.
