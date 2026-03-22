"""
Markov Engine – top-level orchestrator for Layer 3.

Runs the complete Markov cycle for all (vuln, asset) pairs:
    1. Load current states
    2. Build per-pair TPMs (from ML outputs via TPMComputer)
    3. Apply coupling adjustments
    4. Evolve states via Chapman-Kolmogorov
    5. Compute absorption times and risk decay
    6. Persist updated states
    7. Return Markov feedback features for next ML pass
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np

from layer0_knowledge_graph.graph_store import GraphStore
from layer3_markov_engine.state_manager import StateManager
from layer3_markov_engine.tpm_computer import TPMComputer
from layer3_markov_engine.chapman_kolmogorov import ChapmanKolmogorovSolver
from layer3_markov_engine.coupled_markov_networks import CoupledMarkovNetworks
from layer3_markov_engine.absorption_time_analyzer import AbsorptionTimeAnalyzer
from layer3_markov_engine.risk_decay_calculator import RiskDecayCalculator
from layer3_markov_engine.warm_start_estimator import WarmStartEstimator
from models import MLPredictions, MarkovState, RiskState

logger = logging.getLogger(__name__)

NUM_STATES = 6


class MarkovEngine:
    """Orchestrates the full Markov cycle."""

    def __init__(
        self,
        graph: GraphStore,
        state_manager: StateManager | None = None,
        tpm_computer: TPMComputer | None = None,
    ) -> None:
        self.graph = graph
        self.states = state_manager or StateManager()
        self.tpm = tpm_computer or TPMComputer()
        self.solver = ChapmanKolmogorovSolver()
        self.coupling = CoupledMarkovNetworks(graph, self.states)
        self.absorption = AbsorptionTimeAnalyzer()
        self.decay = RiskDecayCalculator()
        self.warm_start = WarmStartEstimator(graph, self.states)

    def run_cycle(
        self,
        predictions: list[MLPredictions],
        cycle_id: str,
    ) -> dict[str, MarkovState]:
        """
        Execute one full Markov cycle.

        Returns
        -------
        dict mapping "vuln_id::asset_id" → updated MarkovState
        """
        logger.info("Markov cycle %s: processing %d pairs", cycle_id, len(predictions))
        results: dict[str, MarkovState] = {}

        for pred in predictions:
            vid, aid = pred.vuln_id, pred.asset_id
            pair_key = f"{vid}::{aid}"

            # 1. Get or warm-start current state
            current = self.states.get_state(vid, aid)
            if current is None:
                pi0 = self.warm_start.estimate_initial_distribution(vid, aid)
                current = MarkovState(
                    vuln_id=vid, asset_id=aid,
                    distribution=pi0.tolist(),
                    current_state=RiskState(int(np.argmax(pi0))),
                    entropy=_entropy(pi0),
                    absorption_time=None,
                )

            pi = np.asarray(current.distribution, dtype=np.float64)

            # 2. Compute per-pair TPM from ML predictions
            tpm = self.tpm.compute(pred)

            # 3. Apply coupling adjustments
            tpm = self.coupling.adjust_tpm(vid, aid, tpm)

            # 4. Evolve state one step
            pi_new = self.solver.evolve_discrete(pi, tpm, steps=1)

            # 5. Compute absorption time
            abs_time = self.absorption.expected_absorption_time(pi_new, tpm)

            # 5b. Compute risk decay half-life based on dominant state
            dominant = int(np.argmax(pi_new))
            decay_halflife = None
            if dominant == RiskState.MITIGATED:
                decay_halflife = self.decay._hl_mit
            elif dominant == RiskState.REMEDIATED:
                decay_halflife = self.decay._hl_rem

            # 6. Compute entropy
            ent = _entropy(pi_new)

            # 7. Persist
            self.states.update_state(
                vid, aid, pi_new, ent, abs_time, cycle_id
            )

            state = MarkovState(
                vuln_id=vid, asset_id=aid,
                distribution=pi_new.tolist(),
                current_state=RiskState(dominant),
                entropy=ent,
                absorption_time=abs_time,
                risk_decay_halflife=decay_halflife,
            )
            results[pair_key] = state

        logger.info("Markov cycle %s complete: %d states updated", cycle_id, len(results))
        return results

    def get_feedback_states(self) -> dict[str, MarkovState]:
        """Return all current states for ML feedback features."""
        return self.states.get_all_states()

    def get_feedback_history(
        self,
        vuln_id: str,
        asset_id: str,
        limit: int = 5,
    ) -> list[MarkovState]:
        return self.states.get_history(vuln_id, asset_id, limit)


def _entropy(dist: np.ndarray) -> float:
    h = 0.0
    for p in dist:
        if p > 1e-12:
            h -= p * math.log(p)
    return h
