import argparse
import sys
import time


def cmd_start(args):
    print("Running... press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")


def cmd_doctor(args):
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {version}", end="")
    if sys.version_info >= (3, 14):
        print(" OK")
    else:
        print(" WARN: expected >= 3.14")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="bsafe", description="Censor NSFW content on screen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("start", help="Start the censoring process")
    subparsers.add_parser("doctor", help="Check system requirements")

    args = parser.parse_args()

    commands = {
        "start": cmd_start,
        "doctor": cmd_doctor,
    }

    commands[args.command](args)
