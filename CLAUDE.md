# Bsafe — Project Conventions

- Use `uv` for all package management (install, sync, run).
- Python >= 3.14. Format and lint with `ruff`.
- Run tests: `uv run pytest`
- Lint: `uv run ruff check bsafe/ tests/`
- Format: `uv run ruff format bsafe/ tests/`
- CLI entry point: `bsafe/cli.py` → `main()`
- Swift helper lives in `swift/`. Build with `cd swift && swift build -c release`.
- IPC protocol defined in `bsafe/protocol.py`. Python is the socket server, Swift connects as client.
- Keep `README.md` updated when adding user-facing features, CLI options, or setup steps.
- Keep `CLAUDE.md` updated when adding conventions, entry points, or architectural decisions.
- When adding or changing CLI flags/defaults, update the config template in `bsafe/config.py` (`generate_default_config()`) to match.

## Detection backends

`bsafe/detector.py` uses a backend abstraction: `_NudeNetBackend` (NudeNet ONNX models) and `_EraXBackend` (ultralytics YOLO `.pt` models). `Detector` delegates to the correct backend based on `ModelInfo.backend`. `ultralytics` is an optional dependency — install with `uv sync --extra erax`. EraX class names are mapped to NudeNet canonical names so `censor.py` works unchanged.

## Batch processing

`bsafe video` and `bsafe image` accept multiple files (shell globs). Process tree:

- **Video batch**: each file runs in a spawned subprocess (which itself spawns chunk subprocesses). Full memory isolation per file. `gc.collect()` between files.
- **Image batch**: all files run in the main process sequentially. `gc.collect()` between files (`process_image` creates and closes the detector internally). No subprocess overhead since images are lightweight.

Single-file mode preserves existing behavior exactly (no batch wrappers). `-o/--output` is disallowed with multiple inputs.

## Before committing

Always run all CI checks locally **before** every commit — format, lint, and tests. Fix any issues before committing. Do not prompt the user to commit; wait for them to ask.

```sh
uv run ruff format bsafe/ tests/
uv run ruff check bsafe/ tests/
uv run pytest
```

## Python 3.14 syntax

This project targets Python >= 3.14. Some syntax that was invalid or had different semantics in older Python versions is now standard:

- `except A, B:` is valid and means `except (A, B):` (PEP 758). Do **not** "fix" this by adding parentheses — ruff enforces the unparenthesized form. This is **not** the Python 2 `except A as B` pattern; that was removed in Python 3.0.

Trust ruff's formatting output for 3.14-era syntax questions.

## Privacy

Screen data is personal data. Strict rules:

- **No network calls.** All processing (capture, inference, overlay) must happen locally. Never add HTTP clients, telemetry, analytics, or any outbound connections.
- **No persistence of screen content.** Frames must stay in memory (or short-lived temp files required by libraries) and never be saved to disk intentionally. Temp files must be cleaned up.
- **No logging of screen content.** Never log raw pixel data, file paths of saved frames, or detection image crops. Logging detection metadata (class, confidence, box coordinates) is acceptable.
