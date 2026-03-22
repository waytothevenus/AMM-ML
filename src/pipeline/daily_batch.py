"""
Daily Batch Pipeline – the main orchestration cycle.

Runs the full 2-pass bidirectional feedback loop:

  PASS 1 (Forward):
    Layer 0 → Layer 1 → Layer 2 → Layer 3

  PASS 2 (Feedback):
    Layer 3 → Layer 1 (Markov feedback features) → Layer 2 (re-infer) → Layer 3 (update)

  Then:
    Layer 4 (aggregate + prioritize) → Layer 5 (update caches)
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from config import get_config
from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.data_freshness_monitor import DataFreshnessMonitor
from layer1_feature_engineering.feature_store import FeatureStore
from layer1_feature_engineering.feature_assembler import FeatureAssembler
from layer2_ml_engine.inference_engine import InferenceEngine
from layer2_ml_engine.model_version_manager import ModelVersionManager
from layer3_markov_engine.markov_engine import MarkovEngine
from layer3_markov_engine.state_manager import StateManager
from layer3_markov_engine.tpm_computer import TPMComputer
from layer4_risk_aggregation.risk_aggregation_engine import RiskAggregationEngine
from models import MarkovState

logger = logging.getLogger(__name__)


class DailyBatchPipeline:
    """
    Orchestrates the complete daily risk assessment cycle.
    """

    def __init__(self) -> None:
        self.graph = GraphStore()
        self.graph.load()
        self.feature_store = FeatureStore()
        self.state_mgr = StateManager()
        self.tpm_computer = TPMComputer()
        self.vm = ModelVersionManager()
        self.markov_engine = MarkovEngine(self.graph, self.state_mgr, self.tpm_computer)
        self.risk_engine = RiskAggregationEngine(self.graph)
        self.freshness_monitor = DataFreshnessMonitor(self.graph)

    def run(self, cycle_id: str | None = None) -> dict:
        """Execute the full daily batch pipeline."""
        cycle_id = cycle_id or datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
        logger.info("=" * 60)
        logger.info("DAILY BATCH PIPELINE – Cycle %s", cycle_id)
        logger.info("=" * 60)

        # --- Step 0: Check freshness ---
        freshness_score = self.freshness_monitor.get_overall_freshness()
        logger.info("Data freshness: %.1f%%", freshness_score * 100)

        # --- PASS 1: Forward ---
        logger.info("--- PASS 1: Forward (L0→L1→L2→L3) ---")

        # L1: Compute features (without Markov feedback initially)
        assembler = FeatureAssembler(
            self.graph, self.feature_store,
            markov_states={}, markov_history={},
        )
        feature_rows = assembler.compute_all_pairs(
            cycle_id=f"{cycle_id}_pass1", persist=True
        )

        if not feature_rows:
            logger.warning("No vuln-asset pairs found. Pipeline complete (empty).")
            return {"cycle_id": cycle_id, "pairs": 0, "status": "empty"}

        # L2: ML inference (Pass 1)
        inference = InferenceEngine(
            version_manager=self.vm,
            data_freshness_score=freshness_score,
        )
        predictions_p1 = inference.predict_batch(feature_rows)

        # L3: Markov evolution (Pass 1)
        markov_results_p1 = self.markov_engine.run_cycle(predictions_p1, f"{cycle_id}_p1")

        # --- PASS 2: Feedback ---
        logger.info("--- PASS 2: Feedback (L3→L1→L2→L3) ---")

        # Collect Markov states and history for feedback features
        all_states = self.markov_engine.get_feedback_states()
        history_map: dict[str, list[MarkovState]] = {}
        for key in all_states:
            parts = key.split("::", 1)
            if len(parts) == 2:
                history_map[key] = self.markov_engine.get_feedback_history(parts[0], parts[1])

        # L1: Re-compute features WITH Markov feedback
        assembler_p2 = FeatureAssembler(
            self.graph, self.feature_store,
            markov_states=all_states,
            markov_history=history_map,
        )
        feature_rows_p2 = assembler_p2.compute_all_pairs(
            cycle_id=f"{cycle_id}_pass2", persist=True
        )

        # L2: Re-infer with enriched features
        predictions_p2 = inference.predict_batch(feature_rows_p2)

        # L3: Update Markov with refined predictions
        markov_results_p2 = self.markov_engine.run_cycle(predictions_p2, f"{cycle_id}_p2")

        # --- Layer 4: Aggregate & Prioritize ---
        logger.info("--- Layer 4: Risk Aggregation ---")

        # Collect TPMs for forecasting
        tpms: dict[str, np.ndarray] = {}
        for pred in predictions_p2:
            key = f"{pred.vuln_id}::{pred.asset_id}"
            tpms[key] = self.tpm_computer.compute(pred)

        aggregated = self.risk_engine.aggregate(
            predictions_p2, markov_results_p2, tpms
        )

        # Build cache for Layer 5
        risks_cache: dict[str, dict] = {}
        prio_cache: list[dict] = []

        for agg in aggregated:
            key = f"{agg.vuln_id}::{agg.asset_id}"
            risks_cache[key] = {
                "composite_risk": agg.composite_risk,
                "propagated_risk": agg.propagated_risk,
                "final_rank": agg.final_rank,
                "exploit_likelihood": next(
                    (p.exploit_likelihood for p in predictions_p2
                     if p.vuln_id == agg.vuln_id and p.asset_id == agg.asset_id),
                    0.0,
                ),
                "adjusted_impact": next(
                    (p.adjusted_impact for p in predictions_p2
                     if p.vuln_id == agg.vuln_id and p.asset_id == agg.asset_id),
                    0.0,
                ),
            }
            prio_cache.append({
                "vuln_id": agg.vuln_id,
                "asset_id": agg.asset_id,
                "composite_risk": agg.composite_risk,
                "ranks": agg.algorithm_ranks,
                "final_rank": agg.final_rank,
            })

        prio_cache.sort(key=lambda x: x["final_rank"])

        # Forecasts
        forecasts_cache: dict[str, dict] = {}
        for agg in aggregated:
            key = f"{agg.vuln_id}::{agg.asset_id}"
            if agg.forecast:
                forecasts_cache[key] = agg.forecast

        # BU rollup
        from layer4_risk_aggregation.business_unit_rollup import BusinessUnitRollup
        bu_engine = BusinessUnitRollup(self.graph)
        asset_risks = {}
        for agg in aggregated:
            if agg.asset_id not in asset_risks or agg.propagated_risk > asset_risks[agg.asset_id]:
                asset_risks[agg.asset_id] = agg.propagated_risk
        bu_rollup = bu_engine.rollup(asset_risks)

        # Save graph
        self.graph.save()

        logger.info("=" * 60)
        logger.info(
            "PIPELINE COMPLETE: %d pairs, freshness=%.0f%%, cycle=%s",
            len(aggregated), freshness_score * 100, cycle_id,
        )
        logger.info("=" * 60)

        return {
            "cycle_id": cycle_id,
            "pairs": len(aggregated),
            "freshness": freshness_score,
            "status": "complete",
            "risks_cache": risks_cache,
            "prio_cache": prio_cache,
            "bu_rollup": bu_rollup,
            "forecasts_cache": forecasts_cache,
        }
