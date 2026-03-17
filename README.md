# Bsafe

## About

Censor NSFW content on your MacOS screen. CLI.

## Setting up locally

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/):

```sh
# install dependencies
uv sync

# activate the virtual environment
source .venv/bin/activate
```

Copy the following to your `~/.zshrc` or equivalent if you wish to
make a `bsafe` alias globally available in your terminal:

```sh
alias bsafe='uv run --project /path/to/bsafe bsafe'
```

## Usage

Start the process (no daemonized support yet).

```sh
bsafe start
```

And that should do it.

## License

Under [MIT License](./LICENSE).

Copyright (c) 2026 Marcell "Mazuh" G. C. da Silva.
