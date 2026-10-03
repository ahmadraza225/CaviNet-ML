"""Section 11.5: ensemble inference, temperature scaling, decision rule and confidence."""

from cavinet_ml.inference.calibration import calibrated_probability, fit_temperature
from cavinet_ml.inference.decision import DISCLAIMER, Decision, decide
from cavinet_ml.inference.ensemble import Ensemble, Prediction

__all__ = [
    "DISCLAIMER",
    "Decision",
    "Ensemble",
    "Prediction",
    "calibrated_probability",
    "decide",
    "fit_temperature",
]
