"""Command-line interface: `cavinet-ml`.

Application helpers: fetch-model, build-demo. Training toolkit (M-10, run from the repository
root on the GPU computer; see docs/TRAINING_RUNBOOK.md): index, preprocess, split, train,
calibrate, export, evaluate, baseline, shortcut-check, compare, and synthetic-dataset for
rehearsals and tests.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from cavinet_ml import __version__, hide_library_notices

EXIT_ERROR = 2
EXIT_REFUSED = 3
_CONFIG = Path("ml/configs/train.yaml")


def default_config() -> Path:
    """ml/configs/train.yaml from the repository root, or beside the package."""
    if _CONFIG.is_file():
        return _CONFIG
    return Path(__file__).resolve().parents[1] / "configs" / "train.yaml"


def _workspace_options(
    parser: argparse.ArgumentParser, splits: bool = True, docs: bool = False
) -> None:
    parser.add_argument(
        "--work", type=Path, default=Path("work"), help="work folder (default: work)"
    )
    if splits:
        parser.add_argument(
            "--splits", type=Path, default=Path("ml/splits.json"), help="default: ml/splits.json"
        )
    if docs:
        parser.add_argument("--docs", type=Path, default=Path("docs"), help="default: docs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cavinet-ml",
        description="CaviNet model training and evaluation toolkit (docs/TRAINING_RUNBOOK.md).",
    )
    parser.add_argument("--version", action="version", version=f"cavinet-ml {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="command")

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

    synthetic = commands.add_parser(
        "synthetic-dataset", help="write a small synthetic look-alike of the Kaggle dataset"
    )
    synthetic.add_argument("--out", type=Path, required=True)
    synthetic.add_argument("--cases", type=int, default=24)
    synthetic.add_argument("--broken", type=int, default=1, help="patients without DICOM images")
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--folder", action="store_true", help="write a folder instead of a zip")

    index = commands.add_parser(
        "index", help="FR-10.1: dataset + PatientIndex.xlsx -> manifest.csv"
    )
    index.add_argument("--data", type=Path, required=True, help="the Kaggle zip or its folder")
    index.add_argument("--patient-index", type=Path, help="PatientIndex.xlsx if not in the dataset")
    index.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    _workspace_options(index, splits=False)

    pre = commands.add_parser(
        "preprocess", help="FR-10.2: section 11.1 -> float16 cache + QC report"
    )
    _workspace_options(pre, splits=False)
    pre.add_argument("--data", type=Path, help="dataset location (default: as indexed)")
    pre.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    pre.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    pre.add_argument(
        "--lungmask-weights", help="R231 weights (default: LUNGMASK_WEIGHTS or download)"
    )
    pre.add_argument("--no-download", action="store_true", help="never download the R231 weights")
    pre.add_argument(
        "--no-lungmask", action="store_true", help="TESTS ONLY: always use the fallback crop"
    )
    pre.add_argument("--cases", nargs="+", metavar="ID", help="only these patients")
    pre.add_argument("--limit", type=int, help="at most this many patients in this run")
    pre.add_argument("--retry-failed", action="store_true")

    split = commands.add_parser("split", help="FR-10.3: locked test set + folds -> splits.json")
    _workspace_options(split)
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--test-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--force", action="store_true", help="replace an existing splits.json")

    train = commands.add_parser("train", help="FR-10.4: train one fold (resumes automatically)")
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--config", type=Path, default=None, help="default: ml/configs/train.yaml")
    _workspace_options(train)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument(
        "--restart", action="store_true", help="discard this fold's run and start over"
    )
    train.add_argument("--epochs", type=int, help="override epochs.max")
    train.add_argument("--batch-size", type=int, help="override batch.size")
    train.add_argument("--num-workers", type=int, help="override data.num_workers")
    train.add_argument(
        "--pretrained", help="override model.pretrained (medicalnet, none or a path)"
    )

    calibrate = commands.add_parser("calibrate", help="FR-10.5: temperature on out-of-fold logits")
    _workspace_options(calibrate)

    export = commands.add_parser("export", help="FR-10.6: model bundle + model card")
    _workspace_options(export, docs=True)
    export.add_argument("--out", type=Path, help="default: <work>/export/cavinet_model.pth")
    export.add_argument("--name", help="model name")
    export.add_argument("--version", dest="model_version", default="model-v1")
    export.add_argument("--demo", action="store_true", help="mark the bundle as a demo")
    export.add_argument(
        "--no-model-card", action="store_true", help="do not write docs/MODEL_CARD.md"
    )

    evaluate = commands.add_parser("evaluate", help="FR-10.7: evaluation report (test split: once)")
    evaluate.add_argument("--split", choices=["dev", "test"], required=True)
    _workspace_options(evaluate, docs=True)
    evaluate.add_argument("--bundle", type=Path, help="default: <work>/export/cavinet_model.pth")
    evaluate.add_argument(
        "--force", action="store_true", help="evaluate the test set again (logged)"
    )
    evaluate.add_argument("--reason", help="required with --force")

    for name, text in (
        ("baseline", "FR-10.8: clinical-only model on the same folds"),
        ("shortcut-check", "FR-10.8: metadata-only model on the same folds"),
    ):
        command = commands.add_parser(name, help=text)
        _workspace_options(command)

    compare = commands.add_parser("compare", help="FR-10.8: DeLong test of two models' AUCs")
    _workspace_options(compare)
    compare.add_argument("--a", default="ct", help="ct, clinical, metadata or a CSV (default: ct)")
    compare.add_argument("--b", default="clinical", help="default: clinical")
    compare.add_argument("--split", choices=["dev", "test"], default="dev")
    return parser


def _run(args: argparse.Namespace) -> int:
    from cavinet_ml.workspace import Workspace

    ws = Workspace(
        work=getattr(args, "work", Path("work")),
        splits=getattr(args, "splits", Path("ml/splits.json")),
        docs=getattr(args, "docs", Path("docs")),
    )
    if args.command == "synthetic-dataset":
        from cavinet_ml.dataset.synthetic import write_synthetic_dataset

        path = write_synthetic_dataset(
            args.out, cases=args.cases, broken=args.broken, seed=args.seed, as_zip=not args.folder
        )
        print(f"Wrote {path} (synthetic, no patient data).")
    elif args.command == "index":
        from cavinet_ml.dataset.index import build_manifest

        build_manifest(
            args.data, ws.manifest, patient_index=args.patient_index, workers=args.workers
        )
    elif args.command == "preprocess":
        from cavinet_ml.dataset.cache import preprocess_dataset

        preprocess_dataset(
            ws.manifest,
            cache_dir=ws.cache,
            qc_path=ws.qc_report,
            data=args.data,
            workers=args.workers,
            lungmask_weights=args.lungmask_weights,
            use_lungmask=not args.no_lungmask,
            download_weights=not args.no_download,
            device=args.device,
            only=args.cases,
            limit=args.limit,
            retry_failed=args.retry_failed,
        )
    elif args.command == "split":
        from cavinet_ml.dataset.split import split_dataset

        split_dataset(
            ws.manifest,
            ws.qc_report,
            ws.splits,
            seed=args.seed,
            test_fraction=args.test_fraction,
            n_folds=args.folds,
            force=args.force,
        )
    elif args.command == "train":
        from cavinet_ml.training.config import load_config, with_overrides
        from cavinet_ml.training.trainer import train_fold

        config = with_overrides(
            load_config(args.config or default_config()),
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pretrained=args.pretrained,
        )
        train_fold(
            args.fold,
            config=config,
            manifest_path=ws.manifest,
            splits_path=ws.splits,
            cache_dir=ws.cache,
            runs_dir=ws.runs,
            device=args.device,
            restart=args.restart,
        )
    elif args.command == "calibrate":
        from cavinet_ml.training.calibrate import calibrate

        calibrate(runs_dir=ws.runs, splits_path=ws.splits, out=ws.calibration)
    elif args.command == "export":
        from cavinet_ml.training.export import export_bundle

        export_bundle(
            runs_dir=ws.runs,
            splits_path=ws.splits,
            manifest_path=ws.manifest,
            cache_dir=ws.cache,
            calibration_path=ws.calibration,
            out=args.out or ws.bundle,
            model_card_md=None if args.no_model_card else ws.model_card_md,
            model_name=args.name,
            model_version=args.model_version,
            demo=args.demo,
        )
    elif args.command == "evaluate":
        from cavinet_ml.evaluation.evaluate import evaluate_dev, evaluate_test

        if args.split == "dev":
            evaluate_dev(ws)
        else:
            evaluate_test(
                ws, bundle_path=args.bundle or ws.bundle, force=args.force, reason=args.reason
            )
    elif args.command in ("baseline", "shortcut-check"):
        from cavinet_ml.evaluation.baselines import run_baseline

        run_baseline(
            "clinical" if args.command == "baseline" else "metadata",
            manifest_path=ws.manifest,
            splits_path=ws.splits,
            out_dir=ws.baselines,
        )
    elif args.command == "compare":
        from cavinet_ml.evaluation.evaluate import compare

        compare(ws, args.a, args.b, split=args.split)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    hide_library_notices()
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
        write_model_card(Path(args.out))
        return 0
    if args.command is None:
        parser.print_help()
        return 0

    from cavinet_ml.dataset.source import DatasetError
    from cavinet_ml.evaluation.evaluate import EvaluationError, EvaluationRefused
    from cavinet_ml.training.config import ConfigError
    from cavinet_ml.training.trainer import TrainingError

    try:
        return _run(args)
    except EvaluationRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except (DatasetError, TrainingError, ConfigError, EvaluationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
