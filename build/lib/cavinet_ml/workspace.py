"""Where the training toolkit keeps its files. Everything under the work folder (default
`work/`, git-ignored) is local: manifest, cache, QC, runs, calibration, exports. Only
splits.json and the docs/ outputs are meant to be committed."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_WORK = Path("work")
DEFAULT_SPLITS = Path("ml/splits.json")
DEFAULT_DOCS = Path("docs")


@dataclass(frozen=True)
class Workspace:
    work: Path = DEFAULT_WORK
    splits: Path = DEFAULT_SPLITS
    docs: Path = DEFAULT_DOCS

    @property
    def manifest(self) -> Path:
        return self.work / "manifest.csv"

    @property
    def cache(self) -> Path:
        return self.work / "cache"

    @property
    def qc_report(self) -> Path:
        return self.work / "qc" / "qc_report.csv"

    @property
    def runs(self) -> Path:
        return self.work / "runs"

    @property
    def calibration(self) -> Path:
        return self.work / "calibration.json"

    @property
    def oof_predictions(self) -> Path:
        return self.work / "oof_predictions.csv"

    @property
    def baselines(self) -> Path:
        return self.work / "baselines"

    @property
    def evaluation(self) -> Path:
        return self.work / "evaluation"

    @property
    def bundle(self) -> Path:
        return self.work / "export" / "cavinet_model.pth"

    @property
    def report(self) -> Path:
        return self.docs / "EVALUATION_REPORT.md"

    @property
    def figures(self) -> Path:
        return self.docs / "figures"

    @property
    def model_card_md(self) -> Path:
        return self.docs / "MODEL_CARD.md"

    @property
    def evaluation_log(self) -> Path:
        return self.docs / "audit" / "evaluation_runs.jsonl"
