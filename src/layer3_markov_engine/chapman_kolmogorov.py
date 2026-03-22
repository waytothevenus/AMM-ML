"""
Chapman-Kolmogorov Solver – evolves state distributions forward in time.

Given π(t) and transition matrix P, computes:
    π(t+k) = π(t) · P^k

Also supports continuous-time approximation for fractional steps.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import expm

logger = logging.getLogger(__name__)


class ChapmanKolmogorovSolver:
    """
    Discrete-time and continuous-time Markov chain solver.

    Discrete:  π(t+k) = π(t) · P^k
    Continuous: π(t+Δ) = π(t) · exp(Q·Δ)  where Q = (P - I) / Δt
    """

    def evolve_discrete(
        self,
        distribution: np.ndarray,
        tpm: np.ndarray,
        steps: int = 1,
    ) -> np.ndarray:
        """
        Evolve a state distribution by `steps` discrete time steps.

        Parameters
        ----------
        distribution : (S,) array – current π(t)
        tpm : (S,S) array – one-step transition probability matrix
        steps : int – number of steps to advance

        Returns
        -------
        (S,) array – π(t+steps)
        """
        pi = np.asarray(distribution, dtype=np.float64)
        P = np.asarray(tpm, dtype=np.float64)

        if steps == 1:
            pi_new = pi @ P
        else:
            Pk = np.linalg.matrix_power(P, steps)
            pi_new = pi @ Pk

        # Ensure valid distribution
        pi_new = np.clip(pi_new, 0.0, None)
        s = pi_new.sum()
        if s > 0:
            pi_new /= s
        return pi_new

    def evolve_continuous(
        self,
        distribution: np.ndarray,
        tpm: np.ndarray,
        delta_t: float = 1.0,
    ) -> np.ndarray:
        """
        Continuous-time evolution using matrix exponential.

        Converts the discrete TPM to a rate matrix Q and uses
        exp(Q·Δt) for the transition kernel.
        """
        pi = np.asarray(distribution, dtype=np.float64)
        P = np.asarray(tpm, dtype=np.float64)
        S = P.shape[0]

        # Rate matrix Q = P - I (assuming unit time step for the TPM)
        Q = P - np.eye(S)
        kernel = expm(Q * delta_t)

        pi_new = pi @ kernel
        pi_new = np.clip(pi_new, 0.0, None)
        s = pi_new.sum()
        if s > 0:
            pi_new /= s
        return pi_new

    def forecast(
        self,
        distribution: np.ndarray,
        tpm: np.ndarray,
        horizons: list[int],
    ) -> dict[int, np.ndarray]:
        """
        Forecast state distributions at multiple future time horizons.

        Returns {horizon_steps: π(t+horizon)} dict.
        """
        results = {}
        pi = np.asarray(distribution, dtype=np.float64)
        P = np.asarray(tpm, dtype=np.float64)

        sorted_horizons = sorted(horizons)
        prev_step = 0
        current_pi = pi.copy()

        for h in sorted_horizons:
            diff = h - prev_step
            if diff > 0:
                current_pi = self.evolve_discrete(current_pi, P, steps=diff)
            results[h] = current_pi.copy()
            prev_step = h

        return results

    def steady_state(self, tpm: np.ndarray) -> np.ndarray:
        """
        Compute the stationary distribution π∞ satisfying π∞ = π∞ · P.
        Uses eigenvalue decomposition.
        """
        P = np.asarray(tpm, dtype=np.float64)
        S = P.shape[0]

        # Left eigenvector for eigenvalue 1: π P = π  ⟹  Pᵀ π = π
        eigenvalues, eigenvectors = np.linalg.eig(P.T)

        # Find eigenvector closest to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])

        # Normalize to probability distribution
        pi = np.abs(pi)
        s = pi.sum()
        if s > 0:
            pi /= s
        return pi
