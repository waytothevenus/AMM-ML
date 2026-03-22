"""
Exploit Likelihood Predictor (ELP) – Layer 2 ML Model 1/3.

Binary classifier predicting the probability that a vulnerability
will have a working exploit developed within a configurable time
horizon (default 30 days).

Model: XGBoost or RandomForest (offline-friendly, no GPU required).
Output: P(exploit | features) ∈ [0, 1].
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from config import get_config

logger = logging.getLogger(__name__)


class ExploitLikelihoodPredictor:
    """
    Binary classifier: will an exploit appear for this vuln?

    Lifecycle:
        1. Train on staging machine with labelled data  (train / save)
        2. Load pre-trained model on air-gapped host    (load)
        3. Predict during daily batch                   (predict / predict_batch)
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._model_dir = Path(cfg.models.elp_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any = None
        self._feature_names: list[str] = []
        self._version: str = "0.0.0"

    # ------------------------------------------------------------------
    #  Training (run on staging machine)
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        version: str = "1.0.0",
        **kwargs,
    ) -> dict[str, float]:
        """Train the ELP model. Returns metric dict."""
        from sklearn.model_selection import cross_val_score

        params = {
            "n_estimators": kwargs.get("n_estimators", 300),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "subsample": kwargs.get("subsample", 0.8),
            "random_state": 42,
        }
        self._model = GradientBoostingClassifier(**params)
        self._model.fit(X, y)
        self._feature_names = list(feature_names)
        self._version = version

        scores = cross_val_score(
            GradientBoostingClassifier(**params),
            X, y, cv=5, scoring="roc_auc",
        )
        metrics = {
            "roc_auc_mean": float(scores.mean()),
            "roc_auc_std": float(scores.std()),
        }
        logger.info("ELP trained v%s – AUC=%.4f±%.4f", version,
                     metrics["roc_auc_mean"], metrics["roc_auc_std"])
        return metrics

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------
    def save(self, tag: str | None = None) -> Path:
        tag = tag or self._version
        path = self._model_dir / f"elp_{tag}.joblib"
        joblib.dump({
            "model": self._model,
            "feature_names": self._feature_names,
            "version": self._version,
        }, path)
        logger.info("ELP model saved → %s", path)
        return path

    def load(self, tag: str | None = None) -> None:
        if tag:
            path = self._model_dir / f"elp_{tag}.joblib"
        else:
            # Load latest by modification time
            candidates = sorted(self._model_dir.glob("elp_*.joblib"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise FileNotFoundError(f"No ELP model found in {self._model_dir}")
            path = candidates[0]

        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._feature_names = bundle["feature_names"]
        self._version = bundle["version"]
        logger.info("ELP model loaded v%s from %s", self._version, path)

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------
    def predict(self, features: dict[str, float]) -> float:
        """Return P(exploit) for a single feature vector."""
        self._ensure_loaded()
        x = self._align(features)
        proba = self._model.predict_proba(x.reshape(1, -1))[0, 1]
        return float(proba)

    def predict_batch(self, feature_list: list[dict[str, float]]) -> np.ndarray:
        """Return P(exploit) array for a batch."""
        self._ensure_loaded()
        X = np.array([self._align(f) for f in feature_list], dtype=np.float32)
        return self._model.predict_proba(X)[:, 1].astype(np.float64)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def _align(self, features: dict[str, float]) -> np.ndarray:
        """Align feature dict to training column order, filling 0 for missing."""
        return np.array(
            [features.get(k, 0.0) for k in self._feature_names],
            dtype=np.float32,
        )

    @property
    def version(self) -> str:
        return self._version
