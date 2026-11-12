"""Symptom-assisted prediction (M-12, FR-12.2): the fusion model used by the application.

A logistic regression combines the CT model's ensemble logit (before temperature scaling)
with age, sex and the 12 dataset symptoms. `cavinet-ml fusion` fits it on the out-of-fold CT
logits of the development set and writes its parameters to a small versioned JSON file
(`cavinet_fusion.json`), which `fetch-model` places next to the model bundle. This module
only reads that file and applies it, with numpy, so the application does not need
scikit-learn.

Inputs are prepared exactly as in training:

    ct_logit, age     -> missing value replaced by the training median, then standardised
    sex_male, symptom -> 1 present / 0 absent; a missing value takes the training mode

    p(TB) = sigmoid(intercept + sum(coefficient_i * x_i))

The fusion file names the model weights it was fitted for; it is used only with that model.
"""

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cavinet_ml.dataset.patient_index import SYMPTOM_COLUMNS, SYMPTOMS

FUSION_FILE = "cavinet_fusion.json"
FORMAT = 1
CONTINUOUS = ("ct_logit", "age")
BINARY = ("sex_male", *SYMPTOM_COLUMNS)
FEATURES = (*CONTINUOUS, *BINARY)
# Display names, in the dataset's order (PatientIndex.xlsx), for the upload form and the PDF.
SYMPTOM_NAMES = dict(zip(SYMPTOM_COLUMNS, SYMPTOMS, strict=True))


class FusionError(Exception):
    """The fusion parameters are missing, invalid or for another model."""


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def symptom_vector(symptoms: Iterable[str]) -> dict[str, float]:
    """{column: 1.0 or 0.0} for the 12 symptoms from the ticked columns (FR-12.1)."""
    ticked = set(symptoms)
    unknown = sorted(ticked - set(SYMPTOM_COLUMNS))
    if unknown:
        raise FusionError(f"unknown symptoms: {', '.join(unknown)}")
    return {c: 1.0 if c in ticked else 0.0 for c in SYMPTOM_COLUMNS}


@dataclass(frozen=True)
class FusionModel:
    params: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.params["version"])

    @property
    def is_demo(self) -> bool:
        return bool(self.params.get("is_demo"))

    @property
    def weights_sha256(self) -> str | None:
        return self.params.get("bundle_weights_sha256")

    def matches(self, weights_sha256: str | None) -> bool:
        """True if these parameters were fitted for the model with these weights."""
        return bool(weights_sha256) and weights_sha256 == self.weights_sha256

    def prepared(self, values: Mapping[str, float | None]) -> list[float]:
        """The model's inputs after imputation and scaling, in FEATURES order."""
        p = self.params["preprocessing"]
        out = []
        for name in FEATURES:
            value = values.get(name)
            missing = value is None or (isinstance(value, float) and math.isnan(value))
            if name in CONTINUOUS:
                x = p["median"][name] if missing else float(value)
                out.append((x - p["mean"][name]) / p["scale"][name])
            else:
                out.append(float(p["mode"][name]) if missing else float(value))
        return out

    def probability(self, values: Mapping[str, float | None]) -> float:
        z = float(self.params["intercept"]) + sum(
            float(self.params["coefficients"][name]) * x
            for name, x in zip(FEATURES, self.prepared(values), strict=True)
        )
        return _sigmoid(z)

    def predict(
        self, *, ct_logit: float, age: float | None, sex: str | None, symptoms: Iterable[str]
    ) -> float:
        """p(TB) for one patient. `sex` is "male" or "female" (anything else: unknown)."""
        values: dict[str, float | None] = {
            "ct_logit": float(ct_logit),
            "age": None if age is None else float(age),
            "sex_male": 1.0 if sex == "male" else 0.0 if sex == "female" else None,
            **symptom_vector(symptoms),
        }
        return self.probability(values)


def validate(params: dict[str, Any]) -> dict[str, Any]:
    if params.get("format") != FORMAT:
        raise FusionError(f"unsupported fusion file format {params.get('format')!r}")
    if tuple(params.get("features", ())) != FEATURES:
        raise FusionError("the fusion file lists different features")
    try:
        float(params["intercept"])
        for name in FEATURES:
            float(params["coefficients"][name])
        for name in CONTINUOUS:
            for key in ("median", "mean", "scale"):
                float(params["preprocessing"][key][name])
            if float(params["preprocessing"]["scale"][name]) <= 0:
                raise FusionError(f"scale of {name} must be positive")
        for name in BINARY:
            float(params["preprocessing"]["mode"][name])
        str(params["version"])
    except (KeyError, TypeError, ValueError) as error:
        raise FusionError(f"invalid fusion file: {error}") from error
    return params


def load_fusion(path: Path | str) -> FusionModel:
    path = Path(path)
    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FusionError(f"{path} not found") from error
    except (OSError, json.JSONDecodeError) as error:
        raise FusionError(f"{path} is not a valid fusion file: {error}") from error
    if not isinstance(params, dict):
        raise FusionError(f"{path} is not a valid fusion file")
    return FusionModel(validate(params))


def save_fusion(params: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    validate(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    return path


def demo_fusion(weights_sha256: str | None) -> dict[str, Any]:
    """Fixed parameters for the demo model (synthetic data, NOT FOR CLINICAL USE): the CT
    logit dominates and a few symptoms nudge the result, so the combined result can be shown
    and tested without the real dataset."""
    coefficients = {name: 0.0 for name in FEATURES}
    coefficients.update(
        ct_logit=1.5,
        age=0.2,
        sex_male=0.1,
        fever=0.4,
        night_sweats=0.4,
        weight_loss=0.3,
        haemoptysis=0.3,
        fatigue=-0.2,
        dyspnoea=-0.3,
    )
    return {
        "format": FORMAT,
        "version": "demo",
        "is_demo": True,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bundle_weights_sha256": weights_sha256,
        "features": list(FEATURES),
        "preprocessing": {
            "median": {"ct_logit": 0.0, "age": 50.0},
            "mean": {"ct_logit": 0.0, "age": 50.0},
            "scale": {"ct_logit": 1.0, "age": 15.0},
            "mode": {name: 0.0 for name in BINARY},
        },
        "coefficients": coefficients,
        "intercept": -0.3,
        "threshold": 0.5,
        "trained_on": "fixed demo values (no training)",
    }
