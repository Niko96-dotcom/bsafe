# Bsafe

## About

Censor NSFW content on your macOS screen in real time. CLI tool, fully offline.

Supports real-time screen censoring, video file processing (with audio
preservation), and image censoring — with multiple detection models,
censor styles (black boxes, blur, pixelation, text overlay), and
per-body-part configuration.

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

Covering more displays at higher resolutions with more NSFW content increases CPU load
and may reduce quality — you are processing and rendering video in real time.

### Process video files

Produce censored copies of local videos (originals are never modified):

```sh
bsafe video clip.mp4                  # → clip.320n.bsafe.mp4
bsafe video clip.mp4 --blur           # blur instead of black boxes
bsafe video clip.mp4 --pixels         # pixelation effect
bsafe video clip.mp4 --censor-text    # overlay "NSFW" text
bsafe video clip.mp4 --fps 5          # run detection at 5 FPS (output keeps native FPS)
bsafe video clip.mp4 --enhance dim    # low-light enhancement (denoise + adaptive gamma)
```

Process multiple files at once (shell globs work):

```sh
bsafe video *.mp4 --blur              # process all .mp4 files with blur
bsafe video a.mp4 b.mov c.m4v        # explicit file list
```

Supported formats: `.mp4`, `.m4v`, `.mov`. If `ffmpeg` is installed, audio is
preserved in the output; otherwise the video is written without audio.

Videos are processed in chunks (default: 5000 frames). If interrupted, re-run the
same command to resume — completed chunks are skipped automatically. In batch mode,
files with existing output are skipped (idempotent). Non-video files and directories
are filtered out automatically.

### Process image files

Produce censored copies of local images:

```sh
bsafe image photo.jpg                 # → photo.320n.bsafe.jpg
bsafe image *.jpg --pixels            # process all .jpg files with pixelation
bsafe image a.png b.webp --blur       # explicit file list
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.
Re-running the same command skips files that already have output (idempotent).

Note: `-o/--output` cannot be used with multiple input files.

### Configuration file

`bsafe bootstrap` creates `~/.config/bsafe/config.toml` with all default values.
Edit this file to set your preferred defaults — CLI flags always override it.

Sections: `[start]` for start-only flags, `[video]` for video-only flags,
`[common]` for shared flags used by both commands.

### Options

Shared flags (work with both `start` and `video`):

- `--confidence N` — minimum detection confidence, 0.0–1.0 (default: 0.0 for NudeNet, 0.2 for EraX)
- `--censor {none,female,male,all}` — what to censor (default: all). Use `none` to skip nudity censoring while still using additive flags like `--face-male`
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
- `--max-frame-age-ms N` — drop results older than this receipt-to-send budget in ms (default: 250, `0` disables). Stale drops clear that display's overlay and tracking state
- `--stats` — log live aggregate receive-to-send/queue/inference ages, replacement and stale counts every ~2s plus a shutdown summary (no frames/images saved)
- `--inference-resolution {320,640,960}` — NudeNet input resolution (default: 320). Higher values recall smaller regions at higher CPU cost. NudeNet-only; rejected with EraX
- `--detail-scan` — NudeNet-only: full frame plus overlapping 2x2 tiles, remapped and class-aware deduped. More recall for small content at ~5x CPU cost. Rejected with EraX
- `--motion-compensation` — map inference boxes to the newest same-display pending frame via translation-only motion compensation before sending. More responsive under load at higher CPU cost. Off by default; tracker state stays in inference-frame coordinates

### Live latency vs detail

Screen overlays react after content appears — bsafe cannot guarantee prevention
before content is briefly visible.

- The live path keeps only the latest frame per display (fair across displays;
  a busy monitor cannot starve a quiet one) and drops results older than
  `--max-frame-age-ms` (default 250 ms receipt-through-processing budget).
  Under continuous capture only age-based drops apply, so results are not
  discarded merely because a newer frame exists.
- `--stats` timings are local receive-to-send-start/queue/inference ages measured
  with the local monotonic clock from frame receipt to overlay send start. This is
  not capture-to-render latency (it excludes Swift capture, JPEG encode,
  IPC transfer to Python, and overlay render). Averages are drawn-only
  (stale frames excluded); interval `replaced` is the window delta while
  shutdown totals are cumulative.
- Detail tradeoffs: `--inference-resolution 640/960` enlarges the model's
  input for smaller-region recall but slows every frame. `--detail-scan`
  adds four overlapping tiles on top (1 full + 4 tile passes, deduped), which
  further helps small content but roughly multiplies CPU cost. Prefer the
  default 320 single-pass for scrolling and general use; enable higher
  resolution/detail only when small-content recall matters more than
  responsiveness.
- `--motion-compensation` projects the current inference boxes onto the newest
  same-display pending frame before sending, so overlays track motion during
  inference. The staleness deadline is checked immediately before send and
  includes compensation time; fresh results are never dropped merely because a
  newer frame exists.

`image`-only flags:

- `-o` / `--output PATH` — output file path (default: `<input>.<model>.bsafe.<ext>`)

`video`-only flags:

- `-o` / `--output PATH` — output file path (default: `<input>.<model>.bsafe.<ext>`)
- `--fps N` — detection FPS override (default: native video FPS)
- `--chunk-frames N` — frames per processing chunk (default: 5000). Smaller chunks use less memory but may cause brief tracking gaps at chunk boundaries.
- `--enhance dim` — low-light enhancement for dim-but-visible footage. Pre-scans the video, applies FFmpeg temporal denoising (`hqdn3d`), then adaptive per-frame gamma/contrast correction. Automatically skips enhancement for already-bright footage. Requires `ffmpeg` for denoising (enhancement still works without it).

### Detection models

The default `320n` model is bundled with NudeNet. Other models require a manual download.

| Model | Backend | Size | Notes |
|-------|---------|------|-------|
| `320n` | NudeNet | bundled | Fast, default |
| `640m` | NudeNet | ~90 MB | More accurate |
| `erax-nano` | EraX YOLO | ~5 MB | Fastest EraX, mAP 0.438 |
| `erax-small` | EraX YOLO | ~40 MB | Balanced, mAP 0.453 |
| `erax-medium` | EraX YOLO | ~19 MB | Best EraX accuracy, mAP 0.467 |

NudeNet models have broader coverage (faces, covered parts, feet) and are lightweight —
best for real-time `bsafe start`. EraX models are more targeted (e.g. nipple-specific)
and heavier — better suited for `bsafe image` and `bsafe video` where FPS isn't a constraint.

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

## Privacy & censor strength

**Real-time screen mode** (`bsafe start`) draws a transparent overlay window on
top of your screen. It does not modify the underlying applications or websites
in any way — censoring is purely visual and disappears when bsafe stops.

**Video and image modes** (`bsafe video`, `bsafe image`) produce new files with
the censored pixels baked into the output. The original file is never modified.
However, not all censor styles destroy information equally:

- **Black boxes** (default) replace every pixel in the censored region with
  solid black. The original data is completely destroyed — recovery is
  impossible regardless of technique or computing power.
- **Blur** (`--blur`) applies a Gaussian blur that removes high-frequency
  detail. At default or higher intensity this is practically irreversible, but
  the exact kernel parameters are deterministic from the box dimensions and
  intensity (both visible or inferable from the output). At low intensity,
  deconvolution techniques can partially recover edges and shapes.
- **Pixelation** (`--pixels`) downscales each region and scales it back up,
  producing uniform color blocks. Each block preserves the average color of the
  original pixels, retaining more information than the other modes. Published
  machine-learning attacks have demonstrated recovering recognizable faces and
  text from pixelated images. This is the weakest censor mode.

If your priority is ensuring censored content cannot be recovered from the
output file, use black boxes (the default). If you use blur or pixelation for
aesthetic reasons, consider using higher intensity values (e.g. `--blur 3`,
`--pixels 3`) to reduce the amount of recoverable information.

When using `--fps` to skip detection frames in video mode, content that first
appears between detection frames will go uncensored for a few frames until the
next detection cycle picks it up. The default `--persist-frames` setting keeps
boxes active across gaps, but cannot predict content that hasn't been seen yet.

## License

Under [MIT License](./LICENSE).

Copyright (c) 2026 Marcell "Mazuh" G. C. da Silva.
