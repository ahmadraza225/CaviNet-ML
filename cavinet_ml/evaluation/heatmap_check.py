"""`cavinet-ml heatmap-check` (FR-11.4, hypothesis H4): the share of Grad-CAM heatmap energy
inside the lung mask, averaged over the locked test set.

For every test patient the scan is read from the dataset, prepared exactly as in the
application (section 11.1, with the lungmask segmenter), and the frozen model's fold-averaged
Grad-CAM for its predicted class is compared with the lung mask in model space. H4 is supported
if the mean share is at least 80%. Resumable: one CSV row per patient. This explains the frozen
model; it does not measure its performance again, so the locked test set rule is unaffected.

Writes <work>/evaluation/heatmap_shares.csv and heatmap_check.json, and the "Heatmap
explanation (H4)" section of docs/EVALUATION_REPORT.md.
"""

import csv
import json
import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cavinet_ml.dataset.manifest import read_manifest
from cavinet_ml.dataset.source import DatasetError, open_source
from cavinet_ml.dataset.split import load_splits
from cavinet_ml.evaluation.stats import bootstrap
from cavinet_ml.explain import H4_TARGET, grad_cam, lung_share
from cavinet_ml.inference import Ensemble
from cavinet_ml.io.dicom import load_series
from cavinet_ml.model.bundle import load_bundle
from cavinet_ml.preprocessing.pipeline import preprocess
from cavinet_ml.workspace import Workspace

COLUMNS = ["case_id", "label", "predicted_class", "lung_share", "seconds", "error"]
SECTION = "## Heatmap explanation (H4)"


def _done(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def _mean_share(labels: list[int], values: list[float]) -> dict[str, Any]:
    """Mean with a 95% CI from the bootstrap stratified by class, as for the test metrics."""
    result = bootstrap(labels, values, {"mean": lambda _y, s: float(np.mean(s))})
    return result["mean"]


def summarise(rows: list[dict[str, str]], bundle_info: dict[str, Any]) -> dict[str, Any]:
    checked = [r for r in rows if r["lung_share"]]
    shares = [float(r["lung_share"]) for r in checked]
    failed = {r["case_id"]: r["error"] for r in rows if r["error"]}
    summary: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": bundle_info.get("model_version"),
        "is_demo": bool(bundle_info.get("is_demo")),
        "n": len(shares),
        "failed": failed,
        "target": H4_TARGET,
    }
    if shares:
        mean = _mean_share([int(r["label"] == "TB") for r in checked], shares)
        summary.update(
            mean=mean["value"],
            ci=mean["ci"],
            median=float(np.median(shares)),
            minimum=float(np.min(shares)),
            at_least_target=sum(s >= H4_TARGET for s in shares),
            supported=mean["value"] >= H4_TARGET,
        )
        for label in ("TB", "NTM"):
            group = [float(r["lung_share"]) for r in checked if r["label"] == label]
            summary[f"mean_{label}"] = float(np.mean(group)) if group else None
    return summary


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.1%}"


def report_section(summary: dict[str, Any]) -> list[str]:
    """The report section for a heatmap-check summary."""
    lines = [SECTION, ""]
    if not summary.get("n"):
        return [*lines, "No test patient could be checked.", ""]
    verdict = "**supported**" if summary["supported"] else "**not supported**"
    lines += [
        f"Share of Grad-CAM heatmap energy inside the lung mask, for the predicted class, on "
        f"{summary['n']} patients of the locked test set (FR-11.4). H4 (target ≥ "
        f"{summary['target']:.0%}): {verdict}.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Mean share inside the lungs (95% CI) | {summary['mean']:.1%} "
        f"({summary['ci'][0]:.1%} to {summary['ci'][1]:.1%}) |",
        f"| Median / lowest | {summary['median']:.1%} / {summary['minimum']:.1%} |",
        f"| Patients at or above {summary['target']:.0%} | {summary['at_least_target']} of "
        f"{summary['n']} |",
        f"| Mean for TB / NTM patients | {_pct(summary['mean_TB'])} / "
        f"{_pct(summary['mean_NTM'])} |",
        "",
        "The heatmap shows where the model looked, not where disease is; it is not a lesion "
        "marker. Computed with `cavinet-ml heatmap-check` on the frozen model "
        f"({summary['model_version']}); the model's performance was not measured again.",
    ]
    if summary["failed"]:
        lines += ["", f"Not checked ({len(summary['failed'])}): " + ", ".join(summary["failed"])]
    return [*lines, ""]


def update_report(report: Path, summary: dict[str, Any]) -> Path:
    """Insert or replace the H4 section, before "## Methods" if there is one."""
    text = report.read_text(encoding="utf-8") if report.is_file() else ""
    section = "\n".join(report_section(summary))
    if SECTION in text:
        start = text.index(SECTION)
        end = text.find("\n## ", start + len(SECTION))
        text = text[:start] + section + ("\n" + text[end + 1 :] if end != -1 else "")
    elif "\n## Methods" in text:
        at = text.index("\n## Methods") + 1
        text = text[:at] + section + "\n" + text[at:]
    elif text.strip():
        text = text.rstrip("\n") + "\n\n" + section
    else:
        text = section
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    return report


def heatmap_check(
    ws: Workspace,
    data: Path | str,
    bundle_path: Path,
    *,
    lungmask_weights: str | None = None,
    use_lungmask: bool = True,
    device: str = "cpu",
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    import torch

    bundle = load_bundle(bundle_path)
    ensemble = Ensemble(bundle)
    for model in ensemble.models:
        model.to(device)
    segmenter = None
    if use_lungmask:
        from cavinet_ml.preprocessing.lungmask_segmenter import (
            BATCH_SIZE,
            GPU_BATCH_SIZE,
            LungmaskSegmenter,
        )

        logging.getLogger("lungmask").setLevel(logging.WARNING)
        on_cpu = device == "cpu"
        segmenter = LungmaskSegmenter(
            lungmask_weights, BATCH_SIZE if on_cpu else GPU_BATCH_SIZE, force_cpu=on_cpu
        )
        if not segmenter.available:
            raise DatasetError("The lungmask weights are needed (set LUNGMASK_WEIGHTS).")
    rows_by_id = {r["case_id"]: r for r in read_manifest(ws.manifest)}
    test = [c for c in load_splits(ws.splits).test if c in rows_by_id]
    out = ws.evaluation / "heatmap_shares.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    todo = [c for c in test if c not in done][: limit if limit is not None else None]
    log(f"Heatmap check: {len(done)} of {len(test)} test patients done; {len(todo)} to go.")
    new_file = not out.is_file()
    with open_source(data) as source, out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        entries = {c.case_id: c for c in source.layout().cases}
        for number, case_id in enumerate(todo, 1):
            started = time.perf_counter()
            row = {"case_id": case_id, "label": "TB" if rows_by_id[case_id]["label"] else "NTM"}
            folder = Path(tempfile.mkdtemp(prefix="cavinet-heatmap-"))
            try:
                if case_id not in entries:
                    raise DatasetError(f"{case_id} is not in the dataset")
                dicom = source.extract_case(entries[case_id], folder / "dicom")
                prepared = preprocess(load_series(dicom), segmenter, ensemble.preprocessing)
                with torch.inference_mode():
                    x = torch.from_numpy(prepared.volume.astype(np.float32))[None, None]
                    logit = float(np.mean([float(m(x.to(device))) for m in ensemble.models]))
                predicted = ensemble.label_map[1] if logit >= 0 else ensemble.label_map[0]
                sign = 1.0 if logit >= 0 else -1.0
                cam = grad_cam(ensemble.models, prepared.volume, sign)
                share = lung_share(cam, prepared.model_mask)
                row.update(
                    predicted_class=predicted,
                    lung_share="" if share is None else f"{share:.6f}",
                    error="" if share is not None else "empty heatmap",
                )
            except Exception as error:  # noqa: BLE001 - one bad scan must not stop the check
                row.update(predicted_class="", lung_share="", error=type(error).__name__)
            finally:
                shutil.rmtree(folder, ignore_errors=True)
            row["seconds"] = f"{time.perf_counter() - started:.1f}"
            writer.writerow(row)
            handle.flush()
            log(f"[{number}/{len(todo)}] {case_id}: {row['lung_share'] or row['error']}")
    rows = list(_done(out).values())
    summary = summarise([r for r in rows if r["case_id"] in set(test)], ensemble.info)
    (ws.evaluation / "heatmap_check.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    update_report(ws.report, summary)
    if summary["n"]:
        log(
            f"Mean heatmap share inside the lungs {summary['mean']:.1%} on {summary['n']} "
            f"patients: H4 {'supported' if summary['supported'] else 'not supported'}. "
            f"Report: {ws.report}"
        )
    return summary
