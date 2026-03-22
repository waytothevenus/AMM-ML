"""
Threat Intelligence Features – Feature Family 3/6.

Computes threat-intel-derived features for each vulnerability:
  - has_public_exploit           (binary)
  - exploit_maturity_score       (0-1 ordinal)
  - is_in_kev                   (binary – CISA Known Exploited Vulns)
  - threat_actor_interest        (# of linked threat actors)
  - campaign_count               (# of campaigns referencing this vuln)
  - ioc_count                   (# of IoCs linked)
  - ttp_count                   (# of ATT&CK techniques linked)
  - intel_freshness              (days since newest intel update)
"""

from __future__ import annotations

from datetime import datetime

from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import NodeType, RelationType

_MATURITY = {
    "unproven": 0.1,
    "poc": 0.3,
    "proof-of-concept": 0.3,
    "functional": 0.6,
    "weaponized": 0.9,
    "high": 0.8,
}


def compute_threat_intel_features(
    vuln_id: str,
    graph: GraphStore,
    reference_time: datetime | None = None,
) -> dict[str, float]:
    """Return threat-intel feature dict for a vulnerability."""
    now = reference_time or datetime.utcnow()
    G = graph.graph
    vuln = G.nodes.get(vuln_id, {})
    features: dict[str, float] = {}

    # --- exploit flags ---
    features["has_public_exploit"] = 1.0 if vuln.get("has_public_exploit") else 0.0
    maturity_raw = str(vuln.get("exploit_maturity", "")).lower()
    features["exploit_maturity_score"] = _MATURITY.get(maturity_raw, 0.0)
    features["is_in_kev"] = 1.0 if vuln.get("is_in_kev") else 0.0

    # --- counts of linked intel entities ---
    ta_count = 0
    campaign_count = 0
    ioc_count = 0
    ttp_count = 0
    newest_intel_ts: datetime | None = None

    if G.has_node(vuln_id):
        for neighbor in _all_neighbors(G, vuln_id):
            nd = G.nodes[neighbor]
            nt = nd.get("node_type")
            if nt == NodeType.THREAT_ACTOR.value:
                ta_count += 1
            elif nt == NodeType.CAMPAIGN.value:
                campaign_count += 1
            elif nt == NodeType.INDICATOR.value:
                ioc_count += 1
            elif nt == NodeType.TTP.value:
                ttp_count += 1

            ts = _try_parse_ts(nd.get("modified") or nd.get("created"))
            if ts and (newest_intel_ts is None or ts > newest_intel_ts):
                newest_intel_ts = ts

    features["threat_actor_interest"] = float(ta_count)
    features["campaign_count"] = float(campaign_count)
    features["ioc_count"] = float(ioc_count)
    features["ttp_count"] = float(ttp_count)

    if newest_intel_ts:
        features["intel_freshness_days"] = max(0.0, (now - newest_intel_ts).total_seconds() / 86400.0)
    else:
        features["intel_freshness_days"] = -1.0

    return features


def _all_neighbors(G, node):
    """Yield unique neighbors (successors + predecessors) in a DiGraph."""
    seen = set()
    for n in G.successors(node):
        if n not in seen:
            seen.add(n)
            yield n
    for n in G.predecessors(node):
        if n not in seen:
            seen.add(n)
            yield n


def _try_parse_ts(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
