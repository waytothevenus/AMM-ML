"""
Service Layer – bridges the API endpoints with the core computation layers.

Thin adapter that retrieves data from the graph store, feature store,
model outputs, and Markov states to serve API responses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import get_config, resolve_path
from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import NodeType
from layer0_knowledge_graph.data_freshness_monitor import DataFreshnessMonitor
from layer0_knowledge_graph.import_manager import ImportManager
from layer3_markov_engine.state_manager import StateManager
from layer3_markov_engine.chapman_kolmogorov import ChapmanKolmogorovSolver

logger = logging.getLogger(__name__)


class ServiceLayer:
    """Singleton-ish service that lazily initializes stores on first access."""

    def __init__(self) -> None:
        self._graph: GraphStore | None = None
        self._state_mgr: StateManager | None = None
        self._freshness: DataFreshnessMonitor | None = None
        self._import_mgr: ImportManager | None = None
        # Cached aggregation results (populated by daily_batch pipeline)
        self._cached_risks: dict[str, dict] = {}
        self._cached_prioritization: list[dict] = []
        self._cached_bu_rollup: dict[str, dict] = {}
        self._cached_forecasts: dict[str, dict] = {}

    @property
    def graph(self) -> GraphStore:
        if self._graph is None:
            self._graph = GraphStore()
            self._graph.load()
        return self._graph

    @property
    def state_mgr(self) -> StateManager:
        if self._state_mgr is None:
            self._state_mgr = StateManager()
        return self._state_mgr

    @property
    def freshness(self) -> DataFreshnessMonitor:
        if self._freshness is None:
            self._freshness = DataFreshnessMonitor(self.graph)
        return self._freshness

    # ------------------------------------------------------------------
    #  Vulnerabilities
    # ------------------------------------------------------------------
    def list_vulnerabilities(
        self, limit: int = 100, offset: int = 0, sort_by: str = "composite_risk"
    ) -> dict:
        vulns = []
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("node_type") != NodeType.VULNERABILITY.value:
                continue
            risk_data = self._cached_risks.get(nid, {})
            vulns.append({
                "id": nid,
                "cvss_base_score": data.get("cvss_base_score"),
                "has_public_exploit": bool(data.get("has_public_exploit")),
                "is_in_kev": bool(data.get("is_in_kev")),
                "exploit_likelihood": risk_data.get("exploit_likelihood"),
                "adjusted_impact": risk_data.get("adjusted_impact"),
                "composite_risk": risk_data.get("composite_risk", 0.0),
            })

        vulns.sort(key=lambda v: v.get(sort_by, 0.0) or 0.0, reverse=True)
        return {"total": len(vulns), "items": vulns[offset:offset + limit]}

    def get_vulnerability(self, vuln_id: str) -> dict | None:
        node = self.graph.get_node(vuln_id)
        if node is None or node.get("node_type") != NodeType.VULNERABILITY.value:
            return None
        node["id"] = vuln_id
        risk_data = self._cached_risks.get(vuln_id, {})
        node.update(risk_data)
        # Add affected assets
        assets = self.graph.get_assets_for_vuln(vuln_id)
        node["affected_assets"] = assets
        return node

    # ------------------------------------------------------------------
    #  Assets
    # ------------------------------------------------------------------
    def list_assets(self, limit: int = 100, offset: int = 0) -> dict:
        assets = []
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("node_type") != NodeType.ASSET.value:
                continue
            assets.append({
                "id": nid,
                "hostname": data.get("hostname"),
                "business_unit": data.get("business_unit"),
                "criticality": data.get("criticality"),
                "propagated_risk": self._cached_risks.get(nid, {}).get("propagated_risk"),
            })
        assets.sort(key=lambda a: a.get("propagated_risk") or 0.0, reverse=True)
        return {"total": len(assets), "items": assets[offset:offset + limit]}

    def get_asset(self, asset_id: str) -> dict | None:
        node = self.graph.get_node(asset_id)
        if node is None or node.get("node_type") != NodeType.ASSET.value:
            return None
        vulns = self.graph.get_vulns_for_asset(asset_id)
        node["vulnerabilities"] = vulns
        return node

    # ------------------------------------------------------------------
    #  Risk
    # ------------------------------------------------------------------
    def get_risk_pairs(self, limit: int = 100, min_risk: float = 0.0) -> dict:
        pairs = []
        for key, data in self._cached_risks.items():
            if "::" not in key:
                continue
            risk = data.get("composite_risk", 0.0)
            if risk < min_risk:
                continue
            vid, aid = key.split("::", 1)
            pairs.append({
                "vuln_id": vid,
                "asset_id": aid,
                "composite_risk": risk,
                "exploit_likelihood": data.get("exploit_likelihood", 0.0),
                "adjusted_impact": data.get("adjusted_impact", 0.0),
                "final_rank": data.get("final_rank", 999),
            })
        pairs.sort(key=lambda p: p["composite_risk"], reverse=True)
        return {"total": len(pairs), "items": pairs[:limit]}

    def get_risk_summary(self) -> dict:
        all_risks = [
            d.get("composite_risk", 0.0)
            for k, d in self._cached_risks.items() if "::" in k
        ]
        if not all_risks:
            return {"total_pairs": 0, "mean_risk": 0.0, "max_risk": 0.0, "median_risk": 0.0}
        arr = np.array(all_risks)
        return {
            "total_pairs": len(arr),
            "mean_risk": float(arr.mean()),
            "max_risk": float(arr.max()),
            "median_risk": float(np.median(arr)),
            "p95_risk": float(np.percentile(arr, 95)),
            "overall_freshness": self.freshness.get_overall_freshness(),
        }

    def get_bu_rollup(self) -> dict:
        return self._cached_bu_rollup

    def get_forecast(self, pair_key: str) -> dict | None:
        f = self._cached_forecasts.get(pair_key)
        if f is None:
            return None
        return {"pair_key": pair_key, "points": f}

    def get_prioritization(self, limit: int = 50, algorithm: str = "ensemble") -> dict:
        items = self._cached_prioritization[:limit]
        return {"total": len(self._cached_prioritization), "items": items}

    # ------------------------------------------------------------------
    #  What-If Simulator
    # ------------------------------------------------------------------
    def run_what_if(self, pair_key: str, action: str, parameters: dict) -> dict:
        """
        Simulate the effect of a remediation action on a single pair.
        Actions: "patch" (move to Remediated), "mitigate" (move to Mitigated),
                 "isolate" (reduce network exposure).
        """
        parts = pair_key.split("::")
        if len(parts) != 2:
            return {"error": "Invalid pair_key format"}
        vid, aid = parts

        current = self.state_mgr.get_state(vid, aid)
        if current is None:
            return {"error": "No Markov state for this pair"}

        dist = np.asarray(current.distribution, dtype=np.float64)

        if action == "patch":
            # Move all mass to Remediated (state 5)
            new_dist = np.zeros(6)
            new_dist[5] = 1.0
        elif action == "mitigate":
            # Shift mass from states 2,3 to state 4
            new_dist = dist.copy()
            shift = new_dist[2] + new_dist[3]
            new_dist[4] += shift
            new_dist[2] = 0.0
            new_dist[3] = 0.0
        elif action == "isolate":
            # Reduce exploit propagation by dampening states 2,3
            new_dist = dist.copy()
            factor = max(0.0, min(1.0, parameters.get("isolation_factor", 0.5)))
            new_dist[2] *= factor
            new_dist[3] *= factor
            residual = 1.0 - new_dist.sum()
            new_dist[4] += max(0, residual)
        else:
            return {"error": f"Unknown action: {action}"}

        # Normalize
        s = new_dist.sum()
        if s > 0:
            new_dist /= s

        # Compute new risk
        state_weights = np.array([0.05, 0.20, 0.60, 0.95, 0.15, 0.00])
        current_risk = float(np.dot(dist, state_weights))
        new_risk = float(np.dot(new_dist, state_weights))

        # Forecast with new distribution
        solver = ChapmanKolmogorovSolver()
        # Use base TPM from config as approximation
        from layer3_markov_engine.tpm_computer import TPMComputer
        tpm_comp = TPMComputer()
        base_tpm = tpm_comp._base_tpm
        future = solver.forecast(new_dist, base_tpm, [7, 30, 90])

        return {
            "pair_key": pair_key,
            "action": action,
            "current_risk": current_risk,
            "projected_risk": new_risk,
            "risk_reduction": current_risk - new_risk,
            "risk_reduction_pct": (current_risk - new_risk) / max(current_risk, 1e-6) * 100,
            "forecast": {h: float(np.dot(d, state_weights)) for h, d in future.items()},
            "new_distribution": new_dist.tolist(),
        }

    # ------------------------------------------------------------------
    #  Freshness
    # ------------------------------------------------------------------
    def get_freshness_report(self) -> dict:
        reports = self.freshness.check_all()
        overall = self.freshness.get_overall_freshness()
        return {
            "overall_freshness": overall,
            "sources": [
                {
                    "source": r.source,
                    "age_days": r.age_days,
                    "is_stale": r.is_stale,
                    "freshness_score": r.freshness_score,
                    "threshold_days": r.threshold_days,
                }
                for r in reports
            ],
        }

    # ------------------------------------------------------------------
    #  Import
    # ------------------------------------------------------------------
    def trigger_import(self, file_path: str) -> dict:
        cfg = get_config()
        imports_dir = resolve_path(cfg.data.imports_dir).resolve()
        target = Path(file_path).resolve()
        if not target.is_relative_to(imports_dir):
            return {
                "file": file_path,
                "connector": "unknown",
                "nodes_added": 0,
                "edges_added": 0,
                "success": False,
                "error": "Path must be within the configured imports directory",
            }
        if self._import_mgr is None:
            self._import_mgr = ImportManager(self.graph)
        result = self._import_mgr.import_file(target)
        return {
            "file": str(result.file),
            "connector": result.connector,
            "nodes_added": result.nodes_added,
            "edges_added": result.edges_added,
            "success": result.success,
            "error": result.error,
        }

    # ------------------------------------------------------------------
    #  Cache update (called by pipeline)
    # ------------------------------------------------------------------
    def update_caches(
        self,
        risks: dict[str, dict],
        prioritization: list[dict],
        bu_rollup: dict[str, dict],
        forecasts: dict[str, dict],
    ) -> None:
        self._cached_risks = risks
        self._cached_prioritization = prioritization
        self._cached_bu_rollup = bu_rollup
        self._cached_forecasts = forecasts
        logger.info("Service layer caches updated")
