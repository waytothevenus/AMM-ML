"""
Temporal Risk Forecasting – projects risk scores into the future
using Markov chain evolution and trend analysis.
"""

from __future__ import annotations

import logging

import numpy as np

from layer3_markov_engine.chapman_kolmogorov import ChapmanKolmogorovSolver
from models import MarkovState

logger = logging.getLogger(__name__)


class TemporalRiskForecasting:
    """
    Forecasts risk trajectories at multiple time horizons.

    Uses the Chapman-Kolmogorov solver to evolve state distributions
    forward and converts future distributions to risk scores.
    """

    # Risk weight per state: higher for exploited states
    STATE_RISK_WEIGHTS = np.array([
        0.05,  # Unknown
        0.20,  # Disclosed
        0.60,  # ExploitAvailable
        0.95,  # ActivelyExploited
        0.15,  # Mitigated
        0.00,  # Remediated
    ], dtype=np.float64)

    def __init__(self, horizons: list[int] | None = None) -> None:
        self.solver = ChapmanKolmogorovSolver()
        self.horizons = horizons or [1, 7, 14, 30, 90]

    def forecast_pair(
        self,
        distribution: np.ndarray,
        tpm: np.ndarray,
    ) -> dict[int, float]:
        """
        Forecast risk score at each horizon for one pair.

        Returns {horizon_days: risk_score}.
        """
        future_dists = self.solver.forecast(distribution, tpm, self.horizons)
        return {
            h: float(np.dot(dist, self.STATE_RISK_WEIGHTS))
            for h, dist in future_dists.items()
        }

    def forecast_batch(
        self,
        states_and_tpms: list[tuple[str, np.ndarray, np.ndarray]],
    ) -> dict[str, dict[int, float]]:
        """
        Forecast risk for multiple pairs.

        Parameters
        ----------
        states_and_tpms : list of (pair_key, distribution, tpm)

        Returns
        -------
        dict mapping pair_key → {horizon: risk_score}
        """
        results = {}
        for pair_key, dist, tpm in states_and_tpms:
            results[pair_key] = self.forecast_pair(dist, tpm)
        return results

    def risk_trend(
        self,
        history: list[MarkovState],
        current: MarkovState,
    ) -> dict[str, float]:
        """
        Analyze the risk trend from historical states.

        Returns trend metrics: current_risk, avg_risk_7d, trend_direction,
        acceleration.
        """
        dists = [np.asarray(s.distribution) for s in history] + [np.asarray(current.distribution)]
        scores = [float(np.dot(d, self.STATE_RISK_WEIGHTS)) for d in dists]

        current_risk = scores[-1]
        avg_7d = np.mean(scores[-7:]) if len(scores) >= 7 else np.mean(scores)

        # Trend direction: slope over recent scores
        if len(scores) >= 3:
            x = np.arange(len(scores[-5:]), dtype=np.float64)
            y = np.array(scores[-5:], dtype=np.float64)
            n = len(x)
            slope = (n * np.dot(x, y) - x.sum() * y.sum()) / (n * np.dot(x, x) - x.sum() ** 2 + 1e-12)
            direction = float(slope)
        else:
            direction = 0.0

        # Acceleration (change in slope)
        if len(scores) >= 5:
            first_half = scores[-5:-3]
            second_half = scores[-3:]
            slope1 = np.mean(np.diff(first_half)) if len(first_half) > 1 else 0.0
            slope2 = np.mean(np.diff(second_half)) if len(second_half) > 1 else 0.0
            accel = float(slope2 - slope1)
        else:
            accel = 0.0

        return {
            "current_risk": current_risk,
            "avg_risk_7d": float(avg_7d),
            "trend_direction": direction,
            "acceleration": accel,
        }
