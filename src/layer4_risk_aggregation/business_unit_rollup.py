"""
Business Unit Rollup – aggregates risk scores up the organizational
hierarchy defined in config/business_units.yaml.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import yaml

from config import get_config, CONFIG_DIR
from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import NodeType

logger = logging.getLogger(__name__)


class BusinessUnitRollup:
    """
    Aggregates per-asset risk into BU-level and organization-level scores.

    Hierarchy loaded from business_units.yaml.
    Aggregation: weighted sum where weight = asset criticality tier multiplier.
    """

    TIER_WEIGHTS = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}

    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph
        cfg = get_config()
        self._hierarchy = self._load_hierarchy(str(CONFIG_DIR / "business_units.yaml"))

    @staticmethod
    def _load_hierarchy(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def rollup(
        self,
        asset_risks: dict[str, float],
    ) -> dict[str, dict]:
        """
        Aggregate asset-level risks into BU scores.

        Returns
        -------
        dict of BU name → {
            "total_risk": float,
            "weighted_risk": float,
            "asset_count": int,
            "mean_risk": float,
            "max_risk": float,
            "children": {child_bu: {...}},
        }
        """
        G = self.graph.graph

        # Map asset → BU
        asset_bu: dict[str, str] = {}
        for nid, data in G.nodes(data=True):
            if data.get("node_type") == NodeType.ASSET.value:
                bu = data.get("business_unit", "Unassigned")
                asset_bu[nid] = bu

        # Aggregate per BU
        bu_data: dict[str, dict] = defaultdict(lambda: {
            "total_risk": 0.0,
            "weighted_risk": 0.0,
            "asset_count": 0,
            "max_risk": 0.0,
            "risks": [],
        })

        for asset_id, risk in asset_risks.items():
            bu = asset_bu.get(asset_id, "Unassigned")
            tier = G.nodes.get(asset_id, {}).get("criticality", "medium")
            w = self.TIER_WEIGHTS.get(tier, 1.0)

            entry = bu_data[bu]
            entry["total_risk"] += risk
            entry["weighted_risk"] += risk * w
            entry["asset_count"] += 1
            entry["max_risk"] = max(entry["max_risk"], risk)
            entry["risks"].append(risk)

        # Compute means and build result
        result = {}
        for bu, d in bu_data.items():
            n = d["asset_count"]
            result[bu] = {
                "total_risk": d["total_risk"],
                "weighted_risk": d["weighted_risk"],
                "asset_count": n,
                "mean_risk": d["total_risk"] / n if n > 0 else 0.0,
                "max_risk": d["max_risk"],
            }

        # Rollup hierarchy (parent = sum of children)
        result = self._rollup_hierarchy(result, self._hierarchy)
        return result

    def _rollup_hierarchy(
        self,
        bu_scores: dict[str, dict],
        hierarchy: dict,
    ) -> dict[str, dict]:
        """Recursively aggregate children into parents."""
        bus = hierarchy.get("business_units", [])
        for bu in bus:
            self._rollup_node(bu_scores, bu)
        return bu_scores

    def _rollup_node(
        self,
        bu_scores: dict[str, dict],
        bu: dict,
    ) -> None:
        """Recursively aggregate a single node and its descendants."""
        name = bu.get("name", "")
        children = bu.get("children", [])

        if children:
            # Recurse into children first (depth-first)
            for child in children:
                self._rollup_node(bu_scores, child)

            child_names = [c.get("name", "") for c in children]
            for cn in child_names:
                if cn not in bu_scores:
                    bu_scores[cn] = {
                        "total_risk": 0.0, "weighted_risk": 0.0,
                        "asset_count": 0, "mean_risk": 0.0, "max_risk": 0.0,
                    }

            if name not in bu_scores:
                bu_scores[name] = {
                    "total_risk": 0.0, "weighted_risk": 0.0,
                    "asset_count": 0, "mean_risk": 0.0, "max_risk": 0.0,
                }

            for cn in child_names:
                cs = bu_scores[cn]
                bu_scores[name]["total_risk"] += cs["total_risk"]
                bu_scores[name]["weighted_risk"] += cs["weighted_risk"]
                bu_scores[name]["asset_count"] += cs["asset_count"]
                bu_scores[name]["max_risk"] = max(
                    bu_scores[name]["max_risk"], cs["max_risk"]
                )

            total = bu_scores[name]["asset_count"]
            if total > 0:
                bu_scores[name]["mean_risk"] = bu_scores[name]["total_risk"] / total

            bu_scores[name]["children"] = {
                cn: bu_scores[cn] for cn in child_names
            }

        return bu_scores
