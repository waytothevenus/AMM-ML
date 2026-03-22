"""
Temporal Features – Feature Family 1/6.

Computes time-based features for each vulnerability–asset pair:
  - days_since_disclosure
  - days_since_exploit_published
  - days_since_last_scan
  - days_since_patch_available
  - exploit_velocity (time from disclosure to public exploit)
  - patch_lag (time from disclosure to patch release)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import NodeType

_SECONDS_PER_DAY = 86_400.0


def compute_temporal_features(
    vuln_id: str,
    asset_id: str,
    graph: GraphStore,
    reference_time: datetime | None = None,
) -> dict[str, float]:
    """Return temporal feature dict for a (vuln, asset) pair."""
    now = reference_time or datetime.utcnow()
    vuln = graph.get_node(vuln_id) or {}
    asset = graph.get_node(asset_id) or {}

    features: dict[str, float] = {}

    # days since disclosure
    disc = _parse_ts(vuln.get("published_date"))
    features["days_since_disclosure"] = _days_between(disc, now) if disc else -1.0

    # days since exploit published (look up connected exploit nodes)
    exploit_dates: list[datetime] = []
    for neighbor in graph.graph.successors(vuln_id):
        ndata = graph.graph.nodes[neighbor]
        if ndata.get("node_type") == NodeType.INDICATOR.value:
            edt = _parse_ts(ndata.get("date_published") or ndata.get("published"))
            if edt:
                exploit_dates.append(edt)
    if exploit_dates:
        earliest_exploit = min(exploit_dates)
        features["days_since_exploit_published"] = _days_between(earliest_exploit, now)
    else:
        features["days_since_exploit_published"] = -1.0

    # exploit velocity (days from disclosure to first exploit)
    if disc and exploit_dates:
        features["exploit_velocity"] = _days_between(disc, min(exploit_dates))
    else:
        features["exploit_velocity"] = -1.0

    # days since patch available
    patch_ts = _parse_ts(vuln.get("patch_available_date"))
    features["days_since_patch_available"] = _days_between(patch_ts, now) if patch_ts else -1.0

    # patch lag (days from disclosure to patch)
    if disc and patch_ts:
        features["patch_lag"] = _days_between(disc, patch_ts)
    else:
        features["patch_lag"] = -1.0

    # days since last scan for this asset
    scan_ts = _parse_ts(asset.get("last_scan_date"))
    features["days_since_last_scan"] = _days_between(scan_ts, now) if scan_ts else -1.0

    return features


# ---- helpers ----

def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _days_between(earlier: datetime, later: datetime) -> float:
    delta = later - earlier
    return max(0.0, delta.total_seconds() / _SECONDS_PER_DAY)
