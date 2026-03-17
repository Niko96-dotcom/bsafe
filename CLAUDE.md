# Bsafe — Project Conventions

- Use `uv` for all package management (install, sync, run).
- Python >= 3.14. Format and lint with `ruff`.
- Run tests: `uv run pytest`
- Lint: `uv run ruff check bsafe/ tests/`
- Format: `uv run ruff format bsafe/ tests/`
- CLI entry point: `bsafe/cli.py` → `main()`
- Swift helper lives in `swift/`. Build with `cd swift && swift build -c release`.
- IPC protocol defined in `bsafe/protocol.py`. Python is the socket server, Swift connects as client.
- See `plan.md` for the phased roadmap.
- See `idea.md` (unversioned, gitignored) for the full product vision, architecture spec, and design constraints. Consult it when working on new phases.
- Keep `README.md` updated when adding user-facing features, CLI options, or setup steps.
- Keep `CLAUDE.md` updated when adding conventions, entry points, or architectural decisions.
