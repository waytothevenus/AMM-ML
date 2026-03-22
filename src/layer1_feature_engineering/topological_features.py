"""
Topological Features – Feature Family 2/6.

Computes graph-structural features for each vulnerability–asset pair
using the Knowledge Graph topology:
  - asset_degree            (direct connections in network graph)
  - vuln_reach              (how many assets share this vuln via CPE)
  - attack_path_depth       (shortest exploit→asset path length)
  - network_exposure_score  (weighted betweenness of asset)
  - vuln_clustering         (local clustering coefficient of vuln node)
  - connected_critical_assets (# of critical-tier neighbors)
"""

from __future__ import annotations

import networkx as nx

from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import NodeType, RelationType


def compute_topological_features(
    vuln_id: str,
    asset_id: str,
    graph: GraphStore,
) -> dict[str, float]:
    """Return topological feature dict for a (vuln, asset) pair."""
    G = graph.graph
    features: dict[str, float] = {}

    # --- asset degree (only NETWORK_ADJACENT edges) ---
    asset_deg = 0
    if G.has_node(asset_id):
        for _, _, data in G.edges(asset_id, data=True):
            if data.get("relation") == RelationType.ASSET_CONNECTS_TO.value:
                asset_deg += 1
    features["asset_degree"] = float(asset_deg)

    # --- vuln reach (how many assets this vuln is present on) ---
    vuln_reach = 0
    if G.has_node(vuln_id):
        for _, target, data in G.edges(vuln_id, data=True):
            if data.get("relation") == RelationType.VULN_PRESENT_ON.value:
                vuln_reach += 1
    features["vuln_reach"] = float(vuln_reach)

    # --- attack path depth (shortest path from any exploit node to this asset) ---
    exploit_nodes = [
        n for n in G.predecessors(vuln_id)
        if G.nodes[n].get("node_type") == NodeType.INDICATOR.value
    ] if G.has_node(vuln_id) else []

    shortest = float("inf")
    for en in exploit_nodes:
        try:
            length = nx.shortest_path_length(G, en, asset_id)
            shortest = min(shortest, length)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    features["attack_path_depth"] = shortest if shortest != float("inf") else -1.0

    # --- network exposure (betweenness centrality of asset in its subgraph) ---
    features["network_exposure_score"] = _local_betweenness(G, asset_id)

    # --- vuln clustering coefficient (on undirected projection) ---
    features["vuln_clustering"] = _local_clustering(G, vuln_id)

    # --- connected critical assets ---
    crit_count = 0
    if G.has_node(asset_id):
        for neighbor in G.successors(asset_id):
            nd = G.nodes[neighbor]
            if nd.get("node_type") == NodeType.ASSET.value and nd.get("criticality") in ("critical", "high"):
                crit_count += 1
        for neighbor in G.predecessors(asset_id):
            nd = G.nodes[neighbor]
            if nd.get("node_type") == NodeType.ASSET.value and nd.get("criticality") in ("critical", "high"):
                crit_count += 1
    features["connected_critical_assets"] = float(crit_count)

    return features


def _local_betweenness(G: nx.DiGraph, node: str, radius: int = 3) -> float:
    """Approximate betweenness in the ego-graph around *node*."""
    if not G.has_node(node):
        return 0.0
    ego = nx.ego_graph(G, node, radius=radius, undirected=True)
    if ego.number_of_nodes() < 3:
        return 0.0
    bc = nx.betweenness_centrality(ego.to_undirected())
    return bc.get(node, 0.0)


def _local_clustering(G: nx.DiGraph, node: str) -> float:
    if not G.has_node(node):
        return 0.0
    U = G.to_undirected(as_view=True)
    return nx.clustering(U, node)
