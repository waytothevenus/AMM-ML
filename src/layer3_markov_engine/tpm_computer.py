"""
TPM Computer – builds Transition Probability Matrices from ML outputs.

This is the FORWARD link of the bidirectional AMM↔ML feedback loop:
ML predictions (ELP, ISA, ACC) parameterize the Markov transition matrix
for each (vuln, asset) pair.

Matrix layout (6×6):
    Rows/cols: Unknown(0), Disclosed(1), ExploitAvailable(2),
               ActivelyExploited(3), Mitigated(4), Remediated(5-absorbing)
"""

from __future__ import annotations

import logging

import numpy as np
import yaml

from config import get_config, CONFIG_DIR
from models import MLPredictions

logger = logging.getLogger(__name__)

NUM_STATES = 6


class TPMComputer:
    """
    Builds per-pair transition probability matrices.

    The base TPM (from config/risk_states.yaml) encodes structural
    constraints (e.g., "Remediated" is absorbing). ML outputs then
    *modulate* specific transition probabilities.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._base_tpm = self._load_base_tpm(str(CONFIG_DIR / "risk_states.yaml"))

    @staticmethod
    def _load_base_tpm(path: str) -> np.ndarray:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        raw = data.get("base_tpm", [])
        tpm = np.array(raw, dtype=np.float64)
        assert tpm.shape == (NUM_STATES, NUM_STATES), f"Expected {NUM_STATES}×{NUM_STATES}, got {tpm.shape}"
        return _normalize_rows(tpm)

    def compute(self, predictions: MLPredictions) -> np.ndarray:
        """
        Build a customized TPM for one (vuln, asset) pair.

        Modulation rules:
          - ELP (exploit likelihood) increases transitions toward
            ExploitAvailable and ActivelyExploited.
          - ISA (adjusted impact) modulates dwell time in exploited states.
          - ACC (asset criticality) affects mitigation/remediation rates.
        """
        tpm = self._base_tpm.copy()

        elp = predictions.exploit_probability  # [0, 1]
        impact = predictions.impact_adjustment / 10.0  # normalize to [0, 1]
        crit_weight = predictions.asset_criticality_score  # [0, 1]

        # --- Modulate: Disclosed → ExploitAvailable (row 1, col 2) ---
        tpm[1, 2] = _blend(tpm[1, 2], elp, alpha=0.6)

        # --- Modulate: ExploitAvailable → ActivelyExploited (row 2, col 3) ---
        tpm[2, 3] = _blend(tpm[2, 3], elp * impact, alpha=0.5)

        # --- Modulate: ActivelyExploited → Mitigated (row 3, col 4) ---
        # Higher criticality → faster mitigation response
        mitigation_boost = 0.3 * crit_weight
        tpm[3, 4] = min(1.0, tpm[3, 4] + mitigation_boost)

        # --- Modulate: Mitigated → Remediated (row 4, col 5) ---
        # Higher impact → slower remediation (more complex fixes)
        slowdown = 0.2 * impact
        tpm[4, 5] = max(0.01, tpm[4, 5] - slowdown)

        # --- Modulate: Unknown → Disclosed (row 0, col 1) ---
        # If we have intel, disclosure probability increases
        if predictions.exploit_probability > 0.5:
            tpm[0, 1] = min(1.0, tpm[0, 1] * 1.3)

        # Ensure Remediated remains absorbing
        tpm[5, :] = 0.0
        tpm[5, 5] = 1.0

        return _normalize_rows(tpm)

    def compute_batch(
        self,
        predictions_list: list[MLPredictions],
    ) -> list[np.ndarray]:
        return [self.compute(p) for p in predictions_list]


def _normalize_rows(tpm: np.ndarray) -> np.ndarray:
    """Ensure each row sums to 1."""
    row_sums = tpm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return tpm / row_sums


def _blend(base: float, signal: float, alpha: float) -> float:
    """Convex combination: blend = (1-α)*base + α*signal, clamped [0,1]."""
    return max(0.0, min(1.0, (1 - alpha) * base + alpha * signal))
