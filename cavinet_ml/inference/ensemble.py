"""Section 11.2/11.5 inference: the fold models' logits are averaged, divided by the
temperature and turned into the TB probability."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cavinet_ml.config import PreprocessingConfig
from cavinet_ml.inference.calibration import calibrated_probability
from cavinet_ml.inference.decision import Decision, decide
from cavinet_ml.model.bundle import metadata
from cavinet_ml.model.network import CaviNetResNet, build_model


@dataclass
class Prediction:
    logit: float  # ensemble mean, before temperature
    fold_logits: list[float]
    decision: Decision


class Ensemble:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.info = metadata(bundle)
        self.temperature = float(bundle["temperature"])
        self.threshold = float(bundle["decision_threshold"])
        self.bands = dict(bundle["confidence_bands"])
        self.label_map = {int(k): v for k, v in bundle["label_map"].items()}
        self.preprocessing = PreprocessingConfig.from_dict(bundle["preprocessing"])
        self.models: list[CaviNetResNet] = []
        for state in bundle["fold_state_dicts"]:
            model = build_model(bundle["architecture"])
            model.load_state_dict(
                {name: t.float() if t.is_floating_point() else t for name, t in state.items()}
            )
            model.eval()
            self.models.append(model)

    @property
    def is_demo(self) -> bool:
        return bool(self.info["is_demo"])

    def logits(self, volume: np.ndarray) -> list[float]:
        x = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None]
        with torch.inference_mode():
            return [float(model(x).reshape(-1)[0]) for model in self.models]

    def predict(self, volume: np.ndarray) -> Prediction:
        fold_logits = self.logits(volume)
        mean = float(np.mean(fold_logits))
        probability = calibrated_probability(mean, self.temperature)
        return Prediction(
            logit=mean,
            fold_logits=fold_logits,
            decision=decide(probability, self.threshold, self.bands, self.label_map),
        )
