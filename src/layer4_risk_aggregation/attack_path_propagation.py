"""
Attack Path Propagation – propagates risk scores along attack paths
in the Knowledge Graph.

An asset that is reachable via a chain of exploitable vulns accumulates
*propagated risk* from upstream nodes.  Uses a damped BFS model inspired
by PageRank-style propagation.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx
import numpy as np

from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import NodeType, RelationType
from models import MarkovState

logger = logging.getLogger(__name__)


class AttackPathPropagation:
    """
    Propagate risk scores from exploitable vulns through network adjacency.

    Algorithm:
        For each asset, find all vulns present on reachable upstream assets.
        Accumulate risk with exponential damping per hop:
            propagated_risk[asset] += Σ risk(vuln, upstream) × damping^hops
    """

    def __init__(
        self,
        graph: GraphStore,
        damping: float = 0.7,
        max_hops: int = 4,
    ) -> None:
        self.graph = graph
        self.damping = damping
        self.max_hops = max_hops

    def compute_propagated_risk(
        self,
        pair_risks: dict[str, float],
        markov_states: dict[str, MarkovState] | None = None,
    ) -> dict[str, float]:
        """
        Compute propagated risk for every asset.

        Parameters
        ----------
        pair_risks : dict
            Mapping "vuln_id::asset_id" → base risk score.
        markov_states : dict, optional
            Current Markov states for exploitability weighting.

        Returns
        -------
        dict mapping asset_id → total propagated risk.
        """
        G = self.graph.graph
        asset_risk: dict[str, float] = defaultdict(float)

        # Pre-index: asset → list of (vuln_id, base_risk)
        asset_vulns: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for key, risk in pair_risks.items():
            parts = key.split("::")
            if len(parts) == 2:
                vid, aid = parts
                asset_vulns[aid].append((vid, risk))
                asset_risk[aid] += risk  # direct risk

        # Build an adjacency sub-graph of assets only
        asset_nodes = [
            n for n, d in G.nodes(data=True)
            if d.get("node_type") == NodeType.ASSET.value
        ]

        for target_asset in asset_nodes:
            # BFS from target backwards along NETWORK_ADJACENT edges
            visited = {target_asset}
            frontier = [(target_asset, 0)]

            while frontier:
                current, depth = frontier.pop(0)
                if depth >= self.max_hops:
                    continue

                for pred in G.predecessors(current):
                    edge_data = G.edges[pred, current]
                    if edge_data.get("relation") != RelationType.ASSET_CONNECTS_TO.value:
                        continue
                    if pred in visited:
                        continue
                    visited.add(pred)

                    # Propagate risk from this upstream asset
                    hop_damping = self.damping ** (depth + 1)
                    for vid, base_risk in asset_vulns.get(pred, []):
                        # Weight by exploit probability if state available
                        exploit_weight = 1.0
                        if markov_states:
                            key = f"{vid}::{pred}"
                            ms = markov_states.get(key)
                            if ms:
                                dist = ms.distribution
                                if len(dist) >= 4:
                                    exploit_weight = dist[2] + dist[3]

                        propagated = base_risk * hop_damping * exploit_weight
                        asset_risk[target_asset] += propagated

                    frontier.append((pred, depth + 1))

        return dict(asset_risk)

    def find_critical_paths(
        self,
        pair_risks: dict[str, float],
        top_n: int = 10,
    ) -> list[dict]:
        """
        Identify the most critical attack paths (chains of
        exploitable assets leading to high-value targets).
        """
        G = self.graph.graph
        # Find entry points: assets with external-facing services
        entry_points = [
            n for n, d in G.nodes(data=True)
            if d.get("node_type") == NodeType.ASSET.value
            and d.get("zone", "").lower() in ("dmz", "external", "internet-facing")
        ]

        # Find high-value targets
        targets = [
            n for n, d in G.nodes(data=True)
            if d.get("node_type") == NodeType.ASSET.value
            and d.get("criticality", "").lower() in ("critical", "high")
        ]

        paths = []
        for entry in entry_points:
            for target in targets:
                if entry == target:
                    continue
                try:
                    path = nx.shortest_path(G, entry, target)
                    # Compute path risk = sum of pair risks along the path
                    path_risk = 0.0
                    for node in path:
                        for key, risk in pair_risks.items():
                            if key.endswith(f"::{node}"):
                                path_risk += risk
                    paths.append({
                        "entry": entry,
                        "target": target,
                        "path": path,
                        "hops": len(path) - 1,
                        "path_risk": path_risk,
                    })
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        paths.sort(key=lambda p: p["path_risk"], reverse=True)
        return paths[:top_n]
