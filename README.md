# Bsafe

## About

Censor NSFW content on your MacOS screen. CLI.

## Setting up locally

Requires macOS 14+, Python 3.14, and [uv](https://docs.astral.sh/uv/):

```sh
# install dependencies
uv sync

# activate the virtual environment
source .venv/bin/activate
```

### Swift helper

The Swift helper handles screen capture via ScreenCaptureKit. Build it once:

```sh
cd swift && swift build -c release
```

You must grant **Screen Recording** permission to your terminal app
(System Settings → Privacy & Security → Screen Recording).

### Shell alias (optional)

Copy the following to your `~/.zshrc` or equivalent if you wish to
make a `bsafe` alias globally available in your terminal:

```sh
alias bsafe='uv run --project /path/to/bsafe bsafe'
```

## Usage

Check that everything is in order:

```sh
bsafe doctor
```

Start the process (no daemonized support yet):

```sh
bsafe start
```

Options:

- `--fps N` — capture frames per second (default: 3)
- `--confidence N` — minimum detection confidence, 0.0–1.0 (default: 0.5)
- `--dry-run` — run the loop without the Swift helper or detector
- `-v` / `--verbose` — enable debug logging

## License

Under [MIT License](./LICENSE).

Copyright (c) 2026 Marcell "Mazuh" G. C. da Silva.
