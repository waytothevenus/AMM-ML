"""
Historical Statistics Features – Feature Family 5/6.

Computes historically-derived features for each vulnerability:
  - cvss_base_score
  - cvss_exploitability_subscore
  - cvss_impact_subscore
  - epss_score_estimate       (local approximation since no internet)
  - historical_exploit_rate   (fraction of similar CWEs that had exploits)
  - vendor_patch_rate         (vendor-specific average patch lag)
  - sibling_vuln_count        (# vulns sharing same CWE in the graph)
"""

from __future__ import annotations

import logging
from collections import defaultdict

from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import NodeType

logger = logging.getLogger(__name__)


def compute_historical_features(
    vuln_id: str,
    graph: GraphStore,
) -> dict[str, float]:
    """Return historical/statistical features for a vulnerability."""
    G = graph.graph
    vuln = G.nodes.get(vuln_id, {})
    features: dict[str, float] = {}

    # --- CVSS scores ---
    features["cvss_base_score"] = _safe_float(vuln.get("cvss_base_score"), -1.0)
    features["cvss_exploitability"] = _safe_float(vuln.get("cvss_exploitability"), -1.0)
    features["cvss_impact"] = _safe_float(vuln.get("cvss_impact"), -1.0)

    # --- local EPSS approximation ---
    # Simple heuristic: baseline from CVSS + exploit presence + KEV
    features["epss_score_estimate"] = _local_epss_estimate(vuln)

    # --- CWE-based historical stats ---
    cwe_ids = vuln.get("cwe_ids", "")
    if isinstance(cwe_ids, str):
        cwe_ids = [c.strip() for c in cwe_ids.split(";") if c.strip()]

    exploit_rates, sibling_counts = _cwe_statistics(graph, cwe_ids)
    features["historical_exploit_rate"] = exploit_rates
    features["sibling_vuln_count"] = float(sibling_counts)

    # --- Vendor patch rate ---
    vendor = vuln.get("vendor", "")
    features["vendor_patch_rate"] = _vendor_patch_rate(graph, vendor)

    return features


def _local_epss_estimate(vuln: dict) -> float:
    """
    Heuristic EPSS-like score when real EPSS data unavailable.
    Range [0, 1].
    """
    base = _safe_float(vuln.get("cvss_base_score"), 5.0) / 10.0

    modifiers = 0.0
    if vuln.get("has_public_exploit"):
        modifiers += 0.25
    if vuln.get("is_in_kev"):
        modifiers += 0.30

    score = base * 0.5 + modifiers
    return min(1.0, max(0.0, score))


def _cwe_statistics(
    graph: GraphStore,
    cwe_ids: list[str],
) -> tuple[float, int]:
    """
    Scan the graph for vulns sharing the same CWEs.
    Returns (exploit_rate, sibling_count).
    """
    if not cwe_ids:
        return 0.0, 0

    cwe_set = set(cwe_ids)
    total_siblings = 0
    exploited = 0

    for node_id, data in graph.graph.nodes(data=True):
        if data.get("node_type") != NodeType.VULNERABILITY.value:
            continue
        their_cwes = data.get("cwe_ids", "")
        if isinstance(their_cwes, str):
            their_cwes = [c.strip() for c in their_cwes.split(";") if c.strip()]
        if set(their_cwes) & cwe_set:
            total_siblings += 1
            if data.get("has_public_exploit"):
                exploited += 1

    rate = exploited / total_siblings if total_siblings > 0 else 0.0
    return rate, total_siblings


def _vendor_patch_rate(graph: GraphStore, vendor: str) -> float:
    """
    Average patch-lag (days) for vulns from same vendor.
    Returns -1 if unknown.
    """
    if not vendor:
        return -1.0

    total_lag = 0.0
    count = 0
    vendor_lower = vendor.lower()

    for _, data in graph.graph.nodes(data=True):
        if data.get("node_type") != NodeType.VULNERABILITY.value:
            continue
        if str(data.get("vendor", "")).lower() != vendor_lower:
            continue
        lag = data.get("patch_lag_days")
        if lag is not None and lag >= 0:
            total_lag += lag
            count += 1

    return total_lag / count if count > 0 else -1.0


def _safe_float(value, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
