"""Command-line interface: `cavinet-ml`. Training commands are added in Phase 6."""

import argparse
from collections.abc import Sequence

from cavinet_ml import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cavinet-ml",
        description="CaviNet model training and evaluation toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"cavinet-ml {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
