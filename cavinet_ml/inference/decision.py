"""Section 11.5 decision rule and FR-06.1/FR-06.2 wording.

    predicted class = TB if p >= 0.50, otherwise NTM
    confidence      = p for TB, 1 - p for NTM (always 50-100%)
    band            = High >= 80%; Moderate 65-79.9%; Low < 65% ("Inconclusive")

Percentages are rounded to one decimal before the band is chosen, so the band always agrees
with the number shown (64.9% Low, 65.0% Moderate, 79.9% Moderate, 80.0% High).
"""

from dataclasses import asdict, dataclass

from cavinet_ml.model.bundle import DEFAULT_BANDS, DEFAULT_THRESHOLD, LABEL_MAP

BAND_HIGH = "High"
BAND_MODERATE = "Moderate"
BAND_LOW = "Low"
DISCLAIMER = "Decision support only. Not a diagnosis. Confirm with laboratory tests."


@dataclass(frozen=True)
class Decision:
    predicted_class: str  # "TB" or "NTM"
    probability_tb: float  # 0-1
    probability_tb_pct: float  # 0-100, one decimal
    confidence: float  # 0.5-1
    confidence_pct: float  # 50-100, one decimal
    band: str  # High / Moderate / Low
    inconclusive: bool
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


def band_for(confidence_pct: float, bands: dict[str, float] = DEFAULT_BANDS) -> str:
    if confidence_pct >= round(bands["high"] * 100, 1):
        return BAND_HIGH
    if confidence_pct >= round(bands["moderate"] * 100, 1):
        return BAND_MODERATE
    return BAND_LOW


def explanation_for(predicted_class: str, band: str, confidence_pct: float) -> str:
    """FR-06.2 plain-language explanation."""
    if band == BAND_LOW:
        return (
            "Inconclusive: the model is not confident "
            f"(it leans towards {predicted_class} with {confidence_pct:.1f}% confidence). "
            "Confirm with laboratory testing."
        )
    return (
        f"The scan pattern is more consistent with {predicted_class} "
        f"(confidence: {band}, {confidence_pct:.1f}%). Confirm with laboratory testing."
    )


def decide(
    probability_tb: float,
    threshold: float = DEFAULT_THRESHOLD,
    bands: dict[str, float] = DEFAULT_BANDS,
    label_map: dict[int, str] = LABEL_MAP,
) -> Decision:
    if not 0.0 <= probability_tb <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    is_tb = probability_tb >= threshold
    predicted = label_map[1] if is_tb else label_map[0]
    confidence = probability_tb if is_tb else 1.0 - probability_tb
    confidence_pct = round(confidence * 100, 1)
    band = band_for(confidence_pct, bands)
    return Decision(
        predicted_class=predicted,
        probability_tb=probability_tb,
        probability_tb_pct=round(probability_tb * 100, 1),
        confidence=confidence,
        confidence_pct=confidence_pct,
        band=band,
        inconclusive=band == BAND_LOW,
        explanation=explanation_for(predicted, band, confidence_pct),
    )
