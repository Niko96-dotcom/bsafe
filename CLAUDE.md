# Bsafe — Project Conventions

- Use `uv` for all package management (install, sync, run).
- Python >= 3.14. Format and lint with `ruff`.
- Run tests: `uv run pytest`
- Lint: `uv run ruff check bsafe/ tests/`
- Format: `uv run ruff format bsafe/ tests/`
- CLI entry point: `bsafe/cli.py` → `main()`
- Swift helper will live in `swift/` (future).
- See `plan.md` for the phased roadmap.
- See `idea.md` (unversioned, gitignored) for the full product vision, architecture spec, and design constraints. Consult it when working on new phases.
