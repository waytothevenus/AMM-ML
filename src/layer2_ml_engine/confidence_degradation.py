"""
Confidence Degradation – RECOMMENDED ADDITION #2.

Reduces ML prediction confidence as data freshness drops and models
age without retraining.  Prevents stale predictions from being trusted
at face value.

Two degradation axes:
  1. Data freshness  – from DataFreshnessMonitor scores
  2. Model age       – days since model was last trained
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from config import get_config

logger = logging.getLogger(__name__)


class ConfidenceDegradation:
    """
    Applies a multiplicative confidence modifier to ML outputs.

    final_confidence = raw_confidence × data_factor × model_factor

    Both factors are in (0, 1]:
      - data_factor  = freshness_score  (from DataFreshnessMonitor)
      - model_factor = exp(-λ × model_age_days)  where λ = decay_rate
    """

    def __init__(
        self,
        model_decay_rate: float | None = None,
        min_confidence: float = 0.05,
    ) -> None:
        cfg = get_config()
        if model_decay_rate:
            self._decay_rate = model_decay_rate
        else:
            halflife = cfg.layer2.confidence_decay_halflife_days
            self._decay_rate = math.log(2) / halflife
        self._min_conf = min_confidence

    def adjust(
        self,
        raw_confidence: float,
        data_freshness_score: float,
        model_trained_at: datetime | str | None = None,
        reference_time: datetime | None = None,
    ) -> float:
        """
        Return degraded confidence value.

        Parameters
        ----------
        raw_confidence : float
            Original model output (probability or score).
        data_freshness_score : float
            Overall freshness score from DataFreshnessMonitor [0,1].
        model_trained_at : datetime
            When the predicting model was last trained.
        reference_time : datetime
            Current time (default: now).
        """
        now = reference_time or datetime.utcnow()

        # Data factor
        data_factor = max(self._min_conf, min(1.0, data_freshness_score))

        # Model-age factor
        model_factor = 1.0
        if model_trained_at is not None:
            if isinstance(model_trained_at, str):
                model_trained_at = datetime.fromisoformat(model_trained_at)
            age_days = (now - model_trained_at).total_seconds() / 86400.0
            model_factor = math.exp(-self._decay_rate * max(0.0, age_days))
            model_factor = max(self._min_conf, model_factor)

        degraded = raw_confidence * data_factor * model_factor
        return max(self._min_conf, min(1.0, degraded))

    def adjust_batch(
        self,
        raw_scores: list[float],
        data_freshness_score: float,
        model_trained_at: datetime | str | None = None,
        reference_time: datetime | None = None,
    ) -> list[float]:
        """Apply degradation to a list of raw scores."""
        return [
            self.adjust(s, data_freshness_score, model_trained_at, reference_time)
            for s in raw_scores
        ]
