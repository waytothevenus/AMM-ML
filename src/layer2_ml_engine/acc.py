"""
Asset Criticality Classifier (ACC) – Layer 2 ML Model 3/3.

Multi-class classifier that predicts the operational criticality tier
of an asset based on its graph neighbourhood, connectivity, business
unit, and historical incident data.

Classes: critical | high | medium | low
Output : probability distribution over the 4 tiers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config import get_config

logger = logging.getLogger(__name__)

TIERS = ["critical", "high", "medium", "low"]


class AssetCriticalityClassifier:
    """
    Predicts the criticality tier of an asset.
    Useful when CMDB metadata is missing or outdated.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._model_dir = Path(cfg.models.acc_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any = None
        self._feature_names: list[str] = []
        self._version: str = "0.0.0"
        self._classes: list[str] = TIERS

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
            "n_estimators": kwargs.get("n_estimators", 200),
            "max_depth": kwargs.get("max_depth", 8),
            "class_weight": "balanced",
            "random_state": 42,
        }
        self._model = RandomForestClassifier(**params)
        self._model.fit(X, y)
        self._feature_names = list(feature_names)
        self._version = version
        self._classes = list(self._model.classes_)

        scores = cross_val_score(
            RandomForestClassifier(**params),
            X, y, cv=5, scoring="f1_weighted",
        )
        metrics = {
            "f1_weighted_mean": float(scores.mean()),
            "f1_weighted_std": float(scores.std()),
        }
        logger.info("ACC trained v%s – F1=%.4f±%.4f", version,
                     metrics["f1_weighted_mean"], metrics["f1_weighted_std"])
        return metrics

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------
    def save(self, tag: str | None = None) -> Path:
        tag = tag or self._version
        path = self._model_dir / f"acc_{tag}.joblib"
        joblib.dump({
            "model": self._model,
            "feature_names": self._feature_names,
            "version": self._version,
            "classes": self._classes,
        }, path)
        logger.info("ACC model saved → %s", path)
        return path

    def load(self, tag: str | None = None) -> None:
        if tag:
            path = self._model_dir / f"acc_{tag}.joblib"
        else:
            candidates = sorted(self._model_dir.glob("acc_*.joblib"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise FileNotFoundError(f"No ACC model found in {self._model_dir}")
            path = candidates[0]

        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._feature_names = bundle["feature_names"]
        self._version = bundle["version"]
        self._classes = bundle["classes"]
        logger.info("ACC model loaded v%s from %s", self._version, path)

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------
    def predict(self, features: dict[str, float]) -> dict[str, float]:
        """Return {tier: probability} for a single asset."""
        self._ensure_loaded()
        x = self._align(features)
        probas = self._model.predict_proba(x.reshape(1, -1))[0]
        return {cls: float(p) for cls, p in zip(self._classes, probas)}

    def predict_tier(self, features: dict[str, float]) -> str:
        """Return the most likely tier."""
        dist = self.predict(features)
        return max(dist, key=dist.get)  # type: ignore[arg-type]

    def predict_batch(self, feature_list: list[dict[str, float]]) -> list[dict[str, float]]:
        self._ensure_loaded()
        X = np.array([self._align(f) for f in feature_list], dtype=np.float32)
        probas = self._model.predict_proba(X)
        return [
            {cls: float(p) for cls, p in zip(self._classes, row)}
            for row in probas
        ]

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
