"""Command-line interface: `cavinet-ml`. Training commands are added in Phase 6."""

import argparse
import os
from collections.abc import Sequence

from cavinet_ml import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cavinet-ml",
        description="CaviNet model training and evaluation toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"cavinet-ml {__version__}")
    commands = parser.add_subparsers(dest="command")

    fetch = commands.add_parser(
        "fetch-model", help="download the model named by MODEL_URL, or build the demo model"
    )
    fetch.add_argument("--out", default="models/cavinet_model.pth")
    fetch.add_argument("--url", default=os.environ.get("MODEL_URL") or None)
    fetch.add_argument("--sha256", default=os.environ.get("MODEL_SHA256") or None)
    fetch.add_argument("--force", action="store_true", help="replace an installed model")

    demo = commands.add_parser("build-demo", help="train the demo model on synthetic volumes")
    demo.add_argument("--out", default="models/cavinet_model.pth")
    demo.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch-model":
        from cavinet_ml.fetch import fetch_model

        fetch_model(args.out, url=args.url, sha256=args.sha256, force=args.force)
        return 0
    if args.command == "build-demo":
        from cavinet_ml.demo.build import build_demo_bundle
        from cavinet_ml.fetch import write_model_card

        build_demo_bundle(args.out, seed=args.seed)
        write_model_card(__import__("pathlib").Path(args.out))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
