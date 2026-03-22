"""
Warm-Start Estimator – RECOMMENDED ADDITION #3.

When a brand-new vulnerability appears, there is no Markov history
to draw on.  The WarmStartEstimator bootstraps an initial state
distribution from similar historical vulnerabilities.

Approach:
  1. Find k nearest neighbors in the Feature Store (by CVSS, CWE, vendor).
  2. Average their current Markov distributions to get π_warm(0).
  3. Optionally weight by recency and feature similarity.
"""

from __future__ import annotations

import logging

import numpy as np

from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import NodeType
from layer3_markov_engine.state_manager import StateManager

logger = logging.getLogger(__name__)

NUM_STATES = 6


class WarmStartEstimator:
    """
    Bootstrap initial Markov distribution for new vulnerabilities
    by interpolating distributions of similar known vulns.
    """

    def __init__(
        self,
        graph: GraphStore,
        state_manager: StateManager,
        k: int = 5,
    ) -> None:
        self.graph = graph
        self.states = state_manager
        self.k = k

    def estimate_initial_distribution(
        self,
        vuln_id: str,
        asset_id: str,
    ) -> np.ndarray:
        """
        Return a warm-start π(0) for a new (vuln, asset) pair.
        Falls back to the "Unknown" deterministic state if no
        similar vulns exist.
        """
        vuln_node = self.graph.get_node(vuln_id)
        if vuln_node is None:
            return _default_distribution()

        # Gather candidate similar vulns
        candidates = self._find_similar_vulns(vuln_id, vuln_node)
        if not candidates:
            return _default_distribution()

        # Compute similarity scores and select top-k
        scored = [
            (cand_id, self._similarity(vuln_node, cand_data))
            for cand_id, cand_data in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[: self.k]

        # Weighted average of their Markov distributions
        weighted_sum = np.zeros(NUM_STATES, dtype=np.float64)
        total_weight = 0.0

        for cand_id, sim_score in top_k:
            # Try to find this vuln paired with *any* asset
            state = self._find_any_state(cand_id)
            if state is None:
                continue
            dist = np.asarray(state.distribution, dtype=np.float64)
            if len(dist) != NUM_STATES:
                continue
            weighted_sum += sim_score * dist
            total_weight += sim_score

        if total_weight < 1e-12:
            return _default_distribution()

        pi = weighted_sum / total_weight
        pi = np.clip(pi, 0.0, None)
        pi /= pi.sum()
        logger.info(
            "Warm-start for %s::%s from %d neighbors → dominant=%d",
            vuln_id, asset_id, len(top_k), int(np.argmax(pi)),
        )
        return pi

    def _find_similar_vulns(self, vuln_id: str, target: dict) -> list[tuple[str, dict]]:
        """Find vulns with matching CWE and/or similar CVSS."""
        target_cwes_raw = target.get("cwe_ids", "")
        if isinstance(target_cwes_raw, str):
            target_cwes = set(c.strip() for c in target_cwes_raw.split(";") if c.strip())
        else:
            target_cwes = set(target_cwes_raw)
        target_cvss = target.get("cvss_base_score")
        candidates = []

        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("node_type") != NodeType.VULNERABILITY.value:
                continue
            if nid == vuln_id:
                continue
            # Must share at least one CWE or be within CVSS range
            their_cwes_raw = data.get("cwe_ids", "")
            if isinstance(their_cwes_raw, str):
                their_cwes = set(c.strip() for c in their_cwes_raw.split(";") if c.strip())
            else:
                their_cwes = set(their_cwes_raw)
            cwe_overlap = len(target_cwes & their_cwes) if target_cwes else 0
            their_cvss = data.get("cvss_base_score")

            cvss_close = False
            if target_cvss is not None and their_cvss is not None:
                cvss_close = abs(float(target_cvss) - float(their_cvss)) < 2.0

            if cwe_overlap > 0 or cvss_close:
                candidates.append((nid, data))

        return candidates

    def _similarity(self, target: dict, candidate: dict) -> float:
        """Simple similarity score [0, 1] based on CWE overlap + CVSS proximity."""
        score = 0.0

        # CWE overlap
        t_cwes_raw = target.get("cwe_ids", "")
        if isinstance(t_cwes_raw, str):
            t_cwes = set(c.strip() for c in t_cwes_raw.split(";") if c.strip())
        else:
            t_cwes = set(t_cwes_raw)
        c_cwes_raw = candidate.get("cwe_ids", "")
        if isinstance(c_cwes_raw, str):
            c_cwes = set(c.strip() for c in c_cwes_raw.split(";") if c.strip())
        else:
            c_cwes = set(c_cwes_raw)
        if t_cwes and c_cwes:
            jaccard = len(t_cwes & c_cwes) / len(t_cwes | c_cwes)
            score += 0.5 * jaccard

        # CVSS proximity
        t_cvss = target.get("cvss_base_score")
        c_cvss = candidate.get("cvss_base_score")
        if t_cvss is not None and c_cvss is not None:
            diff = abs(float(t_cvss) - float(c_cvss))
            score += 0.3 * max(0.0, 1.0 - diff / 10.0)

        # Vendor match
        if target.get("vendor") and target.get("vendor") == candidate.get("vendor"):
            score += 0.2

        return score

    def _find_any_state(self, vuln_id: str) -> object | None:
        """Find any Markov state for this vuln paired with any asset."""
        all_states = self.states.get_all_states()
        for key, state in all_states.items():
            if key.startswith(vuln_id + "::"):
                return state
        return None


def _default_distribution() -> np.ndarray:
    """All probability mass on Unknown(0)."""
    dist = np.zeros(NUM_STATES, dtype=np.float64)
    dist[0] = 1.0
    return dist
