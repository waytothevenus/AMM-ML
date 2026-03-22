"""
Feedback Loop – implements the bidirectional ML ↔ Markov coupling.

This module encapsulates the CORE NOVELTY of the system:
  Forward:  ML predictions → Markov TPM parameterization
  Feedback: Markov states  → Feature Store → next ML inference round

It can be run as a standalone sub-pipeline within daily_batch
or called iteratively for convergence testing.
"""

from __future__ import annotations

import logging

import numpy as np

from layer0_knowledge_graph.graph_store import GraphStore
from layer1_feature_engineering.feature_store import FeatureStore
from layer1_feature_engineering.feature_assembler import FeatureAssembler
from layer2_ml_engine.inference_engine import InferenceEngine
from layer3_markov_engine.markov_engine import MarkovEngine
from layer3_markov_engine.state_manager import StateManager
from models import MLPredictions, MarkovState

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    Iterative ML ↔ Markov feedback loop.

    Can run multiple passes until convergence (distribution changes
    fall below a threshold) or a maximum number of iterations.
    """

    def __init__(
        self,
        graph: GraphStore,
        feature_store: FeatureStore,
        inference_engine: InferenceEngine,
        markov_engine: MarkovEngine,
        max_iterations: int = 2,
        convergence_threshold: float = 0.01,
    ) -> None:
        self.graph = graph
        self.feature_store = feature_store
        self.inference = inference_engine
        self.markov = markov_engine
        self.max_iter = max_iterations
        self.conv_threshold = convergence_threshold

    def run(
        self,
        initial_feature_rows: list[dict],
        cycle_id: str,
        data_freshness: float = 1.0,
    ) -> tuple[list[MLPredictions], dict[str, MarkovState]]:
        """
        Run the feedback loop until convergence or max iterations.

        Returns
        -------
        (final_predictions, final_markov_states)
        """
        prev_distributions: dict[str, np.ndarray] = {}
        current_rows = initial_feature_rows
        predictions = []
        markov_states: dict[str, MarkovState] = {}

        for iteration in range(1, self.max_iter + 1):
            logger.info("Feedback loop iteration %d/%d", iteration, self.max_iter)

            # Forward: ML inference
            predictions = self.inference.predict_batch(current_rows)

            # Forward: Markov evolution
            iter_cycle = f"{cycle_id}_iter{iteration}"
            markov_states = self.markov.run_cycle(predictions, iter_cycle)

            # Check convergence
            if iteration > 1:
                max_delta = self._compute_delta(prev_distributions, markov_states)
                logger.info("Max distribution delta: %.6f (threshold: %.6f)",
                            max_delta, self.conv_threshold)
                if max_delta < self.conv_threshold:
                    logger.info("Converged after %d iterations", iteration)
                    break

            # Save current distributions for convergence check
            prev_distributions = {
                k: np.asarray(v.distribution)
                for k, v in markov_states.items()
            }

            # Feedback: Update features with new Markov states
            if iteration < self.max_iter:
                history_map = {}
                for key in markov_states:
                    parts = key.split("::", 1)
                    if len(parts) == 2:
                        history_map[key] = self.markov.get_feedback_history(parts[0], parts[1])

                assembler = FeatureAssembler(
                    self.graph, self.feature_store,
                    markov_states=markov_states,
                    markov_history=history_map,
                )
                current_rows = assembler.compute_all_pairs(
                    cycle_id=f"{cycle_id}_fb{iteration}",
                    persist=True,
                )

        return predictions, markov_states

    def _compute_delta(
        self,
        prev: dict[str, np.ndarray],
        current: dict[str, MarkovState],
    ) -> float:
        """Maximum L1 distance between previous and current distributions."""
        max_delta = 0.0
        for key, ms in current.items():
            curr_dist = np.asarray(ms.distribution)
            prev_dist = prev.get(key)
            if prev_dist is not None:
                delta = np.abs(curr_dist - prev_dist).sum()
                max_delta = max(max_delta, delta)
        return max_delta
