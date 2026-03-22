"""
Feature Assembler – orchestrates all 6 feature families into a unified vector.

Usage:
    assembler = FeatureAssembler(graph, feature_store)
    vectors   = assembler.compute_all_pairs(cycle_id="2024-06-01")
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import NodeType, RelationType
from layer1_feature_engineering.temporal_features import compute_temporal_features
from layer1_feature_engineering.topological_features import compute_topological_features
from layer1_feature_engineering.threat_intel_features import compute_threat_intel_features
from layer1_feature_engineering.textual_embeddings import compute_textual_embeddings
from layer1_feature_engineering.historical_statistics import compute_historical_features
from layer1_feature_engineering.markov_feedback_features import compute_markov_feedback_features
from layer1_feature_engineering.feature_store import FeatureStore
from models import MarkovState

logger = logging.getLogger(__name__)


class FeatureAssembler:
    """Combines all 6 feature families into a single flat feature vector."""

    def __init__(
        self,
        graph: GraphStore,
        feature_store: FeatureStore,
        markov_states: dict[str, MarkovState] | None = None,
        markov_history: dict[str, list[MarkovState]] | None = None,
    ) -> None:
        self.graph = graph
        self.store = feature_store
        self._markov_states = markov_states or {}
        self._markov_history = markov_history or {}

    def compute_pair(
        self,
        vuln_id: str,
        asset_id: str,
        reference_time: datetime | None = None,
    ) -> dict[str, float]:
        """Compute the full feature vector for a (vuln, asset) pair."""
        now = reference_time or datetime.utcnow()
        features: dict[str, float] = {}

        # Family 1 – Temporal
        features.update(compute_temporal_features(vuln_id, asset_id, self.graph, now))

        # Family 2 – Topological
        features.update(compute_topological_features(vuln_id, asset_id, self.graph))

        # Family 3 – Threat Intel
        features.update(compute_threat_intel_features(vuln_id, self.graph, now))

        # Family 4 – Textual Embeddings
        features.update(compute_textual_embeddings(vuln_id, self.graph))

        # Family 5 – Historical Statistics
        features.update(compute_historical_features(vuln_id, self.graph))

        # Family 6 – Markov Feedback
        pair_key = f"{vuln_id}::{asset_id}"
        ms = self._markov_states.get(pair_key)
        hist = self._markov_history.get(pair_key)
        features.update(compute_markov_feedback_features(ms, hist))

        return features

    def compute_all_pairs(
        self,
        cycle_id: str,
        reference_time: datetime | None = None,
        persist: bool = True,
    ) -> list[dict]:
        """
        Compute features for every (vuln → asset) pair in the graph
        and optionally persist to the Feature Store.
        """
        pairs = self._enumerate_pairs()
        logger.info("Computing features for %d vuln-asset pairs", len(pairs))

        rows = []
        for vuln_id, asset_id in pairs:
            features = self.compute_pair(vuln_id, asset_id, reference_time)
            rows.append({
                "vuln_id": vuln_id,
                "asset_id": asset_id,
                "features": features,
            })

        if persist and rows:
            self.store.persist_features(rows, cycle_id)

        logger.info("Feature assembly complete: %d vectors", len(rows))
        return rows

    def _enumerate_pairs(self) -> list[tuple[str, str]]:
        """Find all (vuln_id, asset_id) connected via VULN_PRESENT_ON."""
        pairs = []
        G = self.graph.graph
        for u, v, data in G.edges(data=True):
            if data.get("relation") == RelationType.VULN_PRESENT_ON.value:
                pairs.append((u, v))
        return pairs
