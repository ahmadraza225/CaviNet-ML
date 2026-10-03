"""Section 11.5: decision rule, confidence, bands and temperature scaling."""

import math

import numpy as np
import pytest

from cavinet_ml.inference import DISCLAIMER, calibrated_probability, decide, fit_temperature
from cavinet_ml.inference.decision import band_for


@pytest.mark.parametrize(
    ("p", "predicted"),
    [(0.50, "TB"), (0.4999, "NTM"), (0.99, "TB"), (0.0, "NTM"), (1.0, "TB")],
)
def test_predicted_class_is_tb_if_p_at_least_0_50(p, predicted):
    assert decide(p).predicted_class == predicted


def test_confidence_is_p_for_tb_and_1_minus_p_for_ntm():
    tb, ntm = decide(0.86), decide(0.14)
    assert (tb.predicted_class, tb.confidence_pct) == ("TB", 86.0)
    assert (ntm.predicted_class, ntm.confidence_pct) == ("NTM", 86.0)
    assert decide(0.5).confidence_pct == 50.0


@pytest.mark.parametrize(
    ("confidence_pct", "band"),
    [
        (64.9, "Low"),
        (65.0, "Moderate"),
        (79.9, "Moderate"),
        (80.0, "High"),
        (50.0, "Low"),
        (100.0, "High"),
    ],
)
def test_band_boundaries(confidence_pct, band):
    assert band_for(confidence_pct) == band


@pytest.mark.parametrize(
    ("p", "band"),
    [
        (0.649, "Low"),
        (0.65, "Moderate"),
        (0.799, "Moderate"),
        (0.80, "High"),  # TB side
        (0.351, "Low"),
        (0.35, "Moderate"),
        (0.201, "Moderate"),
        (0.20, "High"),  # NTM side
    ],
)
def test_band_boundaries_from_probabilities(p, band):
    assert decide(p).band == band


def test_explanation_wording():
    assert decide(0.86).explanation == (
        "The scan pattern is more consistent with TB (confidence: High, 86.0%). "
        "Confirm with laboratory testing."
    )
    assert decide(0.30).explanation == (
        "The scan pattern is more consistent with NTM (confidence: Moderate, 70.0%). "
        "Confirm with laboratory testing."
    )
    low = decide(0.58)
    assert low.inconclusive and low.band == "Low"
    assert low.explanation.startswith("Inconclusive: the model is not confident")
    assert DISCLAIMER == "Decision support only. Not a diagnosis. Confirm with laboratory tests."


def test_probability_must_be_a_probability():
    with pytest.raises(ValueError):
        decide(1.2)


def test_temperature_scaling_divides_the_logit():
    assert calibrated_probability(2.0, 2.0) == pytest.approx(1 / (1 + math.exp(-1)))
    assert calibrated_probability(-800.0, 1.0) == pytest.approx(0.0)


def test_fit_temperature_recovers_a_known_temperature():
    rng = np.random.default_rng(0)
    true_t = 2.5
    logits = rng.normal(0, 4, 20000)
    labels = rng.random(20000) < 1 / (1 + np.exp(-logits / true_t))
    assert fit_temperature(logits, labels) == pytest.approx(true_t, rel=0.05)


def test_fit_temperature_handles_degenerate_input():
    assert fit_temperature([], []) == 1.0
    assert fit_temperature([1.0, 2.0], [1, 1]) == 1.0
