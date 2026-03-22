"""
Coupled Markov Networks – models interdependencies between assets.

When two assets are network-adjacent or share vulnerabilities, their
Markov chains are *coupled*: a state change in one increases the
transition probabilities of the other.

Uses a lightweight message-passing scheme on the Knowledge Graph topology.
"""

from __future__ import annotations

import logging

import numpy as np

from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import RelationType
from layer3_markov_engine.state_manager import StateManager

logger = logging.getLogger(__name__)

NUM_STATES = 6


class CoupledMarkovNetworks:
    """
    Adjusts per-pair TPMs based on the Markov states of adjacent assets.

    Coupling mechanism:
        For each (vuln, asset_i), find all adjacent assets asset_j that
        share the same vulnerability.  If asset_j has high probability
        of being in an exploited state, increase asset_i's transition
        to exploited states (lateral movement / contagion).
    """

    def __init__(
        self,
        graph: GraphStore,
        state_manager: StateManager,
        coupling_strength: float = 0.15,
    ) -> None:
        self.graph = graph
        self.states = state_manager
        self.coupling = coupling_strength

    def adjust_tpm(
        self,
        vuln_id: str,
        asset_id: str,
        base_tpm: np.ndarray,
    ) -> np.ndarray:
        """
        Apply coupling adjustments to a base TPM.

        Looks at neighboring assets that share the same vuln and
        are network-adjacent.  High exploit-state probability in
        neighbors increases the local exploit transition.
        """
        G = self.graph.graph
        tpm = base_tpm.copy()

        # Find adjacent assets with the same vuln
        neighbor_exploit_pressure = 0.0
        neighbor_count = 0

        if not G.has_node(asset_id):
            return tpm

        for _, adj, edata in G.edges(asset_id, data=True):
            if edata.get("relation") != RelationType.ASSET_CONNECTS_TO.value:
                continue
            # Check if adj also has this vuln
            adj_pair_key = f"{vuln_id}::{adj}"
            adj_state = self.states.get_state(vuln_id, adj)
            if adj_state is None:
                continue
            dist = np.asarray(adj_state.distribution, dtype=np.float64)
            # Exploit pressure = P(ExploitAvailable) + P(ActivelyExploited)
            if len(dist) >= 4:
                neighbor_exploit_pressure += dist[2] + dist[3]
                neighbor_count += 1

        if neighbor_count == 0:
            return tpm

        avg_pressure = neighbor_exploit_pressure / neighbor_count

        # Boost transitions toward exploited states
        boost = self.coupling * avg_pressure

        # Row 1 (Disclosed) → Col 2 (ExploitAvailable)
        tpm[1, 2] = min(1.0, tpm[1, 2] + boost)
        # Row 2 (ExploitAvailable) → Col 3 (ActivelyExploited)
        tpm[2, 3] = min(1.0, tpm[2, 3] + boost * 0.5)

        # Preserve absorbing state
        tpm[5, :] = 0.0
        tpm[5, 5] = 1.0

        # Re-normalize
        return _normalize_rows(tpm)

    def adjust_batch(
        self,
        pairs_and_tpms: list[tuple[str, str, np.ndarray]],
    ) -> list[np.ndarray]:
        """Adjust a batch of (vuln_id, asset_id, tpm) tuples."""
        return [
            self.adjust_tpm(vid, aid, tpm)
            for vid, aid, tpm in pairs_and_tpms
        ]


def _normalize_rows(tpm: np.ndarray) -> np.ndarray:
    row_sums = tpm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return tpm / row_sums
