"""
Risk Aggregation Engine – Layer 4 orchestrator.

Combines ML predictions, Markov states, attack path propagation,
BU rollup, temporal forecasting, and prioritization into a single
aggregated risk picture.
"""

from __future__ import annotations

import logging

import numpy as np

from layer0_knowledge_graph.graph_store import GraphStore
from layer4_risk_aggregation.attack_path_propagation import AttackPathPropagation
from layer4_risk_aggregation.business_unit_rollup import BusinessUnitRollup
from layer4_risk_aggregation.temporal_risk_forecasting import TemporalRiskForecasting
from layer4_risk_aggregation.prioritization import run_prioritization_pipeline, PrioritizedItem
from models import MLPredictions, MarkovState, AggregatedRisk

logger = logging.getLogger(__name__)

# Weights for composite risk formula
W_EXPLOIT = 0.30
W_IMPACT = 0.25
W_MARKOV = 0.25
W_CRITICALITY = 0.20


class RiskAggregationEngine:
    """Orchestrates Layer 4: composite scoring + prioritization."""

    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph
        self.attack_paths = AttackPathPropagation(graph)
        self.bu_rollup = BusinessUnitRollup(graph)
        self.forecasting = TemporalRiskForecasting()

    def aggregate(
        self,
        ml_predictions: list[MLPredictions],
        markov_states: dict[str, MarkovState],
        tpms: dict[str, np.ndarray] | None = None,
    ) -> list[AggregatedRisk]:
        """
        Full aggregation pipeline.

        1. Compute composite risk per pair
        2. Run attack-path propagation
        3. Run temporal forecasting
        4. Run prioritization pipeline
        5. Compute BU rollup
        """
        # Step 1: Composite risk per pair
        pair_items = []
        pair_risks: dict[str, float] = {}

        for pred in ml_predictions:
            pair_key = f"{pred.vuln_id}::{pred.asset_id}"
            ms = markov_states.get(pair_key)

            composite = self._composite_risk(pred, ms)
            pair_risks[pair_key] = composite

            item = {
                "pair_key": pair_key,
                "vuln_id": pred.vuln_id,
                "asset_id": pred.asset_id,
                "composite_risk": composite,
                "exploit_likelihood": pred.exploit_likelihood,
                "adjusted_impact": pred.adjusted_impact,
                "asset_criticality_score": self._criticality_score(pred),
                "remediation_cost": 1.0,  # default; updated from cost model
                "has_public_exploit": pred.exploit_likelihood > 0.7,
                "is_in_kev": False,  # enriched from graph
            }

            # Enrich from graph
            vuln_node = self.graph.get_node(pred.vuln_id)
            if vuln_node:
                item["is_in_kev"] = bool(vuln_node.get("is_in_kev"))
                item["has_public_exploit"] = bool(vuln_node.get("has_public_exploit"))
                item["exploit_velocity_days"] = vuln_node.get("exploit_velocity_days", -1)

            pair_items.append(item)

        # Step 2: Attack-path propagation
        asset_propagated = self.attack_paths.compute_propagated_risk(
            pair_risks, markov_states
        )

        # Step 3: Temporal forecasting
        forecast_data = {}
        if tpms:
            states_tpms = []
            for key, ms in markov_states.items():
                if key in tpms:
                    states_tpms.append((key, np.asarray(ms.distribution), tpms[key]))
            forecast_data = self.forecasting.forecast_batch(states_tpms)

        # Step 4: Prioritization
        prioritized = run_prioritization_pipeline(pair_items)

        # Step 5: BU rollup
        bu_scores = self.bu_rollup.rollup(asset_propagated)

        # Build results
        results = []
        prio_map = {p.pair_key: p for p in prioritized}

        for pred in ml_predictions:
            pair_key = f"{pred.vuln_id}::{pred.asset_id}"
            p = prio_map.get(pair_key)

            result = AggregatedRisk(
                vuln_id=pred.vuln_id,
                asset_id=pred.asset_id,
                composite_risk=pair_risks.get(pair_key, 0.0),
                propagated_risk=asset_propagated.get(pred.asset_id, 0.0),
                forecast=forecast_data.get(pair_key, {}),
                algorithm_ranks=p.ranks if p else {},
                final_rank=p.final_rank if p else 999,
            )
            results.append(result)

        logger.info(
            "Risk aggregation complete: %d pairs, %d BUs",
            len(results), len(bu_scores),
        )
        return results

    def _composite_risk(
        self,
        pred: MLPredictions,
        ms: MarkovState | None,
    ) -> float:
        """
        Compute composite risk score [0, 1] from ML + Markov.

        composite = w_exploit × ELP
                   + w_impact  × (ISA / 10)
                   + w_markov  × markov_risk
                   + w_crit    × criticality_score
        """
        elp = pred.exploit_likelihood
        isa_norm = pred.adjusted_impact / 10.0
        crit = self._criticality_score(pred)

        # Markov risk: probability of being in states 2 or 3
        markov_risk = 0.5  # default if no Markov data
        if ms:
            dist = np.asarray(ms.distribution)
            if len(dist) >= 4:
                markov_risk = float(dist[2] + dist[3])

        score = (
            W_EXPLOIT * elp
            + W_IMPACT * isa_norm
            + W_MARKOV * markov_risk
            + W_CRITICALITY * crit
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def _criticality_score(pred: MLPredictions) -> float:
        """Convert tier distribution to a scalar [0, 1]."""
        dist = pred.asset_criticality_distribution or {}
        if not dist:
            return pred.asset_criticality_score
        return (
            dist.get("critical", 0.0) * 1.0
            + dist.get("high", 0.0) * 0.75
            + dist.get("medium", 0.0) * 0.5
            + dist.get("low", 0.0) * 0.25
        )
