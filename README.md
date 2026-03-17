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
