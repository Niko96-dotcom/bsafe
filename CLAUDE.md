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

## Privacy

Screen data is personal data. Strict rules:

- **No network calls.** All processing (capture, inference, overlay) must happen locally. Never add HTTP clients, telemetry, analytics, or any outbound connections.
- **No persistence of screen content.** Frames must stay in memory (or short-lived temp files required by libraries) and never be saved to disk intentionally. Temp files must be cleaned up.
- **No logging of screen content.** Never log raw pixel data, file paths of saved frames, or detection image crops. Logging detection metadata (class, confidence, box coordinates) is acceptable.
