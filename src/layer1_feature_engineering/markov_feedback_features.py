"""
Markov Feedback Features – Feature Family 6/6.

Extracts features from the *previous* Markov engine cycle so the ML
models can condition on the current risk-state distribution:
  - state_prob_0 … state_prob_5   (π(t) for each Markov state)
  - entropy                       (Shannon entropy of π(t))
  - time_in_current_state         (days)
  - expected_absorption_time      (days to Remediated)
  - dominant_state                (index of argmax π(t))
  - risk_trajectory_slope         (Δ risk over last 3 cycles)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from models import MarkovState


def compute_markov_feedback_features(
    markov_state: MarkovState | None,
    history: list[MarkovState] | None = None,
) -> dict[str, float]:
    """
    Given the latest Markov state (and optionally a history of past
    states), produce feedback features for the ML models.
    """
    features: dict[str, float] = {}

    if markov_state is None:
        # No Markov data yet → all features are defaults
        for i in range(6):
            features[f"state_prob_{i}"] = 0.0
        features["entropy"] = 0.0
        features["time_in_current_state"] = 0.0
        features["expected_absorption_time"] = -1.0
        features["dominant_state"] = 0.0
        features["risk_trajectory_slope"] = 0.0
        return features

    dist = np.asarray(markov_state.distribution, dtype=np.float64)
    # Ensure valid distribution
    dist = np.clip(dist, 0.0, 1.0)
    s = dist.sum()
    if s > 0:
        dist /= s

    for i in range(min(6, len(dist))):
        features[f"state_prob_{i}"] = float(dist[i])
    # Pad if fewer than 6 states
    for i in range(len(dist), 6):
        features[f"state_prob_{i}"] = 0.0

    # Shannon entropy
    features["entropy"] = float(markov_state.entropy) if markov_state.entropy is not None else _entropy(dist)

    # Time in current state
    features["time_in_current_state"] = float(
        markov_state.time_in_state
        if markov_state.time_in_state is not None
        else 0.0
    )

    # Expected absorption time
    features["expected_absorption_time"] = (
        float(markov_state.absorption_time) if markov_state.absorption_time is not None else -1.0
    )

    # Dominant state (argmax)
    features["dominant_state"] = float(np.argmax(dist))

    # Risk trajectory slope over last 3 cycles
    features["risk_trajectory_slope"] = _trajectory_slope(history, dist)

    return features


def _entropy(dist: np.ndarray) -> float:
    """Shannon entropy (nats)."""
    h = 0.0
    for p in dist:
        if p > 1e-12:
            h -= p * math.log(p)
    return h


def _trajectory_slope(
    history: list[MarkovState] | None,
    current_dist: np.ndarray,
) -> float:
    """
    Fit a simple linear slope to the "risk mass" (probability of being in
    states 2,3 – ExploitAvailable or ActivelyExploited) over the last few cycles.
    """
    if not history or len(history) < 2:
        return 0.0

    # Collect risk masses (prob of being in state 2 or 3)
    masses: list[float] = []
    for ms in history[-3:]:
        d = np.asarray(ms.distribution, dtype=np.float64)
        if len(d) >= 4:
            masses.append(float(d[2] + d[3]))
        else:
            masses.append(0.0)
    masses.append(float(current_dist[2] + current_dist[3]) if len(current_dist) >= 4 else 0.0)

    if len(masses) < 2:
        return 0.0

    # Simple finite-difference slope
    x = np.arange(len(masses), dtype=np.float64)
    y = np.array(masses, dtype=np.float64)
    # Least-squares slope
    n = len(x)
    slope = (n * np.dot(x, y) - x.sum() * y.sum()) / (n * np.dot(x, x) - x.sum() ** 2 + 1e-12)
    return float(slope)
