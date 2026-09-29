# Bsafe

Local macOS screen censor. The Python CLI (`bsafe/`, entry `bsafe/cli.py` → `main()`) runs nudity detection; the Swift package (`swift/`) captures the screen, tracks boxes and draws the overlay; a menu bar app supervises the CLI.

## Privacy

Screen data is personal data.

- No network calls. Capture, inference and overlay stay local: no HTTP clients, telemetry, analytics or other outbound connections.
- No persistence of screen content. Frames stay in memory (or short-lived temp files a library requires, cleaned up afterwards). Bench writes metadata only, never pixels.
- No logging of screen content: no raw pixels, saved-frame paths or detection crops. Detection metadata (class, confidence, box) is fine.

## Commands

Use `uv` for everything; Python >= 3.14.

```sh
uv run ruff format bsafe/ tests/
uv run ruff check bsafe/ tests/
uv run pytest
cd swift && swift build -c release   # BsafeCapture, bsafe-replay, BsafeMenuBar
cd swift && swift test               # BsafeCoreTests, BsafeMenuKitTests
scripts/install-menubar.sh           # rebuild and reinstall ~/Applications/Bsafe.app
```

Run format before committing. The pre-commit hook (`.githooks/pre-commit`, enabled with `git config core.hooksPath .githooks`) checks format, lint and tests, plus `swift test` when Swift files are staged. Don't offer to commit; wait for the user to ask.

`except A, B:` is valid Python 3.14 (PEP 758) and ruff enforces the unparenthesized form; don't add parentheses.

## Architecture

- **Live `bsafe start`** uses frame credits, at most one outstanding per display. Swift sends the newest raw frame; Python runs `FullFrameNudeDetector` (ONNX Runtime, CoreML EP) plus `--extra-scales` passes (`fastdetect.detect_multiscale`, shared with bench) and an optional `--still-model` pass (cycling `--still-scales`) while the screen is still, then replies with `CMD_CENSOR_SEQ` boxes tagged by frame `seq`. IPC lives in `bsafe/protocol.py`; Python is the socket server, Swift the client.
- **Swift `BsafeCore` owns live tracking**, persistence and smoothing (defaults in `DisplayTracker.swift`). `bsafe/tracking.py` `BoxTracker` and `--smooth-alpha` are video-only.
- **Detectors** (`bsafe/detector.py`): NudeNet ONNX, and EraX YOLO as an optional extra (`uv sync --extra erax`) whose classes map to NudeNet names so `censor.py` is unchanged.
- **Menu bar**: `BsafeMenuKit` is UI-free (settings → CLI args, quality presets in `Settings.swift`, supervised child process); `BsafeMenuBar` is the SwiftUI app. The app is the TCC-responsible process; the install script signs with the first Developer ID / Apple Development identity and falls back to ad-hoc, which resets Screen Recording grants on every reinstall.
- **Batch**: `bsafe video` runs one subprocess per file; `bsafe image` runs files in-process. `-o` is disallowed with multiple inputs; single-file mode has no batch wrapper.

## Gotchas

- Never leave a helper pipe undrained: at ~64 KB the helper blocks and the overlay freezes. Without `-v`, `StderrTail` (`bsafe/swift_helper.py`) drains stderr.
- `cv2.phaseCorrelate` mutates its inputs for some sizes; pass copies of frames you keep.
- Live capture is in points (1728x1117 on a 3456x2234 panel), not device pixels; padding and `--min-padding` are in capture pixels. Bench recordings must be scaled to the live capture size (even height) or detection won't match live.
- `kill -USR1 <BsafeCapture pid>` simulates a capture interruption; persistent failure exits with code 3.

## When changing things

- CLI flags or defaults: update `generate_default_config()` in `bsafe/config.py`. User-facing features, options or setup steps: update the README.
- Terminal output: use the `bsafe/style.py` helpers. Color short tokens only, never full lines or file paths; respect `NO_COLOR` and non-TTY; no dependencies.
- Detection or tracking defaults: measure with `bsafe bench` first ([docs/bench.md](docs/bench.md)). Record the numbers and rationale in [docs/tuning-log.md](docs/tuning-log.md) and the commit message, not here.
- Add to this file only what a future session would get wrong without it.
