"""
On-Demand Pipeline – re-assess a single (vuln, asset) pair
or a small set of pairs without running the full batch.
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
from layer3_markov_engine.markov_engine import MarkovEngine
from layer3_markov_engine.state_manager import StateManager
from layer3_markov_engine.tpm_computer import TPMComputer

logger = logging.getLogger(__name__)


class OnDemandPipeline:
    """Run assessment for a specific vuln-asset pair."""

    def __init__(
        self,
        graph: GraphStore | None = None,
        state_mgr: StateManager | None = None,
    ) -> None:
        self.graph = graph or GraphStore()
        if graph is None:
            self.graph.load()
        self.feature_store = FeatureStore()
        self.state_mgr = state_mgr or StateManager()
        self.tpm_computer = TPMComputer()
        self.markov_engine = MarkovEngine(self.graph, self.state_mgr, self.tpm_computer)
        self.freshness = DataFreshnessMonitor(self.graph)

    def assess_pair(self, vuln_id: str, asset_id: str) -> dict:
        """Full assessment for one pair (2-pass)."""
        cycle_id = f"ondemand_{datetime.utcnow().strftime('%H%M%S')}"
        freshness_score = self.freshness.get_overall_freshness()

        # Get Markov state for feedback
        all_states = self.markov_engine.get_feedback_states()
        pair_key = f"{vuln_id}::{asset_id}"
        history = {pair_key: self.markov_engine.get_feedback_history(vuln_id, asset_id)}

        # Build features with Markov feedback
        assembler = FeatureAssembler(
            self.graph, self.feature_store,
            markov_states=all_states,
            markov_history=history,
        )
        features = assembler.compute_pair(vuln_id, asset_id)

        # ML inference
        inference = InferenceEngine(data_freshness_score=freshness_score)
        prediction = inference.predict(vuln_id, asset_id, features)

        # Markov update
        markov_results = self.markov_engine.run_cycle([prediction], cycle_id)
        ms = markov_results.get(pair_key)

        # Composite risk
        from layer4_risk_aggregation.risk_aggregation_engine import RiskAggregationEngine
        risk_engine = RiskAggregationEngine(self.graph)
        composite = risk_engine._composite_risk(prediction, ms)

        return {
            "vuln_id": vuln_id,
            "asset_id": asset_id,
            "exploit_likelihood": prediction.exploit_likelihood,
            "adjusted_impact": prediction.adjusted_impact,
            "asset_criticality_tier": prediction.asset_criticality_tier,
            "confidence": prediction.confidence,
            "composite_risk": composite,
            "markov_distribution": ms.distribution if ms else None,
            "markov_entropy": ms.entropy if ms else None,
            "absorption_time": ms.absorption_time if ms else None,
            "freshness": freshness_score,
        }
