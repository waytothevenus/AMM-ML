"""
Absorption Time Analyzer – computes expected time to reach the absorbing
state (Remediated) from the current state distribution.

Uses the fundamental matrix N = (I - Q)^{-1} of the absorbing Markov chain,
where Q is the sub-matrix of transient states.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# State 5 (Remediated) is the absorbing state
ABSORBING_STATE = 5
NUM_STATES = 6


class AbsorptionTimeAnalyzer:
    """
    Computes how many expected time-steps until a vulnerability is
    fully remediated, given its current state distribution and TPM.
    """

    def expected_absorption_time(
        self,
        distribution: np.ndarray,
        tpm: np.ndarray,
    ) -> float:
        """
        Compute the expected time steps to absorption.

        E[T_absorb | π(t)] = π_transient · N · 1

        where N = (I - Q)^{-1} is the fundamental matrix,
        Q is the transient sub-matrix of the TPM.
        """
        P = np.asarray(tpm, dtype=np.float64)
        pi = np.asarray(distribution, dtype=np.float64)

        # Identify transient states (all except absorbing)
        transient_idx = [i for i in range(NUM_STATES) if i != ABSORBING_STATE]
        n_transient = len(transient_idx)

        if n_transient == 0:
            return 0.0

        # Sub-matrix Q (transient → transient transitions)
        Q = P[np.ix_(transient_idx, transient_idx)]

        # Fundamental matrix N = (I - Q)^{-1}
        I = np.eye(n_transient)
        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            logger.warning("Singular (I-Q) matrix; returning inf absorption time")
            return float("inf")

        # Expected absorption times from each transient state
        # t_i = sum_j N[i,j]  (row sums of N)
        absorption_times = N.sum(axis=1)  # shape (n_transient,)

        # Weighted by current distribution on transient states
        pi_transient = pi[transient_idx]
        s = pi_transient.sum()
        if s < 1e-12:
            # Already absorbed
            return 0.0

        pi_transient /= s
        expected = float(np.dot(pi_transient, absorption_times))
        return max(0.0, expected)

    def absorption_probabilities(
        self,
        tpm: np.ndarray,
    ) -> np.ndarray:
        """
        For each transient state, compute the probability of eventually
        being absorbed (should be 1.0 for a valid absorbing chain).
        """
        P = np.asarray(tpm, dtype=np.float64)
        transient_idx = [i for i in range(NUM_STATES) if i != ABSORBING_STATE]
        n_transient = len(transient_idx)

        Q = P[np.ix_(transient_idx, transient_idx)]
        R = P[np.ix_(transient_idx, [ABSORBING_STATE])]

        I = np.eye(n_transient)
        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            return np.ones(n_transient)

        B = N @ R  # Absorption probabilities
        return B.flatten()

    def conditional_absorption_times(
        self,
        tpm: np.ndarray,
    ) -> dict[str, float]:
        """
        Return expected absorption time conditional on starting in each
        transient state.
        """
        from models import RiskState

        P = np.asarray(tpm, dtype=np.float64)
        transient_idx = [i for i in range(NUM_STATES) if i != ABSORBING_STATE]

        Q = P[np.ix_(transient_idx, transient_idx)]
        I = np.eye(len(transient_idx))

        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            return {RiskState(i).name: float("inf") for i in transient_idx}

        times = N.sum(axis=1)
        return {RiskState(idx).name: float(t) for idx, t in zip(transient_idx, times)}
