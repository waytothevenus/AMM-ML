"""
Impact Severity Adjuster (ISA) – Layer 2 ML Model 2/3.

Regression model that adjusts raw CVSS impact scores based on the
*actual deployment context* (asset criticality, network topology,
business-unit exposure, threat-intel signals).

Output: adjusted_impact ∈ [0, 10] (same scale as CVSS).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from config import get_config

logger = logging.getLogger(__name__)


class ImpactSeverityAdjuster:
    """
    Context-aware CVSS impact adjustment.
    Corrects the "one-size-fits-all" CVSS impact score using local context.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._model_dir = Path(cfg.models.isa_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any = None
        self._feature_names: list[str] = []
        self._version: str = "0.0.0"

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        version: str = "1.0.0",
        **kwargs,
    ) -> dict[str, float]:
        from sklearn.model_selection import cross_val_score

        params = {
            "n_estimators": kwargs.get("n_estimators", 300),
            "max_depth": kwargs.get("max_depth", 5),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "subsample": kwargs.get("subsample", 0.8),
            "random_state": 42,
        }
        self._model = GradientBoostingRegressor(**params)
        self._model.fit(X, y)
        self._feature_names = list(feature_names)
        self._version = version

        scores = cross_val_score(
            GradientBoostingRegressor(**params),
            X, y, cv=5, scoring="neg_mean_absolute_error",
        )
        metrics = {
            "mae_mean": float(-scores.mean()),
            "mae_std": float(scores.std()),
        }
        logger.info("ISA trained v%s – MAE=%.4f±%.4f", version,
                     metrics["mae_mean"], metrics["mae_std"])
        return metrics

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------
    def save(self, tag: str | None = None) -> Path:
        tag = tag or self._version
        path = self._model_dir / f"isa_{tag}.joblib"
        joblib.dump({
            "model": self._model,
            "feature_names": self._feature_names,
            "version": self._version,
        }, path)
        logger.info("ISA model saved → %s", path)
        return path

    def load(self, tag: str | None = None) -> None:
        if tag:
            path = self._model_dir / f"isa_{tag}.joblib"
        else:
            candidates = sorted(self._model_dir.glob("isa_*.joblib"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise FileNotFoundError(f"No ISA model found in {self._model_dir}")
            path = candidates[0]

        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._feature_names = bundle["feature_names"]
        self._version = bundle["version"]
        logger.info("ISA model loaded v%s from %s", self._version, path)

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------
    def predict(self, features: dict[str, float]) -> float:
        """Return adjusted impact score ∈ [0, 10]."""
        self._ensure_loaded()
        x = self._align(features)
        raw = float(self._model.predict(x.reshape(1, -1))[0])
        return max(0.0, min(10.0, raw))

    def predict_batch(self, feature_list: list[dict[str, float]]) -> np.ndarray:
        self._ensure_loaded()
        X = np.array([self._align(f) for f in feature_list], dtype=np.float32)
        raw = self._model.predict(X)
        return np.clip(raw, 0.0, 10.0)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def _align(self, features: dict[str, float]) -> np.ndarray:
        return np.array(
            [features.get(k, 0.0) for k in self._feature_names],
            dtype=np.float32,
        )

    @property
    def version(self) -> str:
        return self._version
