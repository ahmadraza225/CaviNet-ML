"""The application's analysis of one stored scan (M-05), in two stages so the case timeline
shows real progress:

    prepared = analyser.prepare(dicom_dir, preview_dir)   # "Preprocessing": FR-05.1, FR-05.3
    result = analyser.infer(prepared)                     # "Analysing":     FR-05.2, FR-05.4

`PipelineError` means the scan itself cannot be analysed (the message is safe to show);
any other exception is unexpected and may be transient (the worker retries once, FR-05.5).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cavinet_ml.inference.ensemble import Ensemble
from cavinet_ml.io.dicom import DicomLoadError, load_series
from cavinet_ml.preprocessing.pipeline import Preprocessed, preprocess
from cavinet_ml.preprocessing.steps import Segmenter
from cavinet_ml.previews import PreviewSet, write_previews


class PipelineError(Exception):
    """The scan cannot be analysed; the message is shown to the doctor."""


@dataclass
class Prepared:
    preprocessed: Preprocessed
    previews: PreviewSet
    seconds: dict[str, float]


@dataclass
class AnalysisResult:
    decision: dict[str, Any]
    logit: float
    fold_logits: list[float]
    warnings: list[str]
    seconds: dict[str, float]
    lung_volume_ml: float
    used_fallback_crop: bool
    previews: PreviewSet
    model: dict[str, Any] = field(default_factory=dict)


class Analyser:
    def __init__(self, ensemble: Ensemble, segmenter: Segmenter | None) -> None:
        self.ensemble = ensemble
        self.segmenter = segmenter

    def prepare(self, dicom_dir: Path, preview_dir: Path) -> Prepared:
        seconds: dict[str, float] = {}
        started = time.perf_counter()
        try:
            image = load_series(dicom_dir)
        except DicomLoadError as error:
            raise PipelineError(str(error)) from error
        seconds["load"] = round(time.perf_counter() - started, 3)

        config = self.ensemble.preprocessing
        preprocessed = preprocess(image, self.segmenter, config)
        seconds.update(preprocessed.seconds)

        started = time.perf_counter()
        previews = write_previews(
            preprocessed.hu_image, preprocessed.lung.mask, preview_dir, config
        )
        seconds["previews"] = round(time.perf_counter() - started, 3)
        return Prepared(preprocessed, previews, seconds)

    def infer(self, prepared: Prepared) -> AnalysisResult:
        started = time.perf_counter()
        prediction = self.ensemble.predict(prepared.preprocessed.volume)
        seconds = {**prepared.seconds, "inference": round(time.perf_counter() - started, 3)}
        seconds["total"] = round(sum(seconds.values()), 3)
        info = self.ensemble.info
        return AnalysisResult(
            decision=prediction.decision.to_dict(),
            logit=prediction.logit,
            fold_logits=prediction.fold_logits,
            warnings=prepared.preprocessed.warnings,
            seconds=seconds,
            lung_volume_ml=prepared.preprocessed.lung.volume_ml,
            used_fallback_crop=prepared.preprocessed.lung.used_fallback,
            previews=prepared.previews,
            model={
                "name": info["model_name"],
                "version": info.get("model_version") or info["created_at"],
                "created_at": info["created_at"],
                "is_demo": bool(info["is_demo"]),
                "folds": info["folds"],
                "metrics": info["metrics"],
            },
        )
