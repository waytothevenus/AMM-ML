"""
Prioritization Algorithms – 5 methods for ranking remediation actions.

1. Pareto Optimal       – non-dominated front across risk × cost
2. Cost-Benefit         – risk reduction per unit cost
3. Time-Sensitive       – prioritizes based on exploit velocity / deadline
4. Multi-Criteria (TOPSIS) – MCDM with configurable weights
5. Ensemble             – aggregates ranks from all 4 methods
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PrioritizedItem:
    pair_key: str
    vuln_id: str
    asset_id: str
    composite_risk: float
    ranks: dict[str, int]  # algorithm → rank (1-based)
    final_rank: int


# ============================================================
#  1. Pareto Optimal Prioritization
# ============================================================

def pareto_prioritize(
    items: list[dict],
) -> list[dict]:
    """
    Pareto-optimal ranking.  An item dominates another if it has
    higher risk AND lower remediation cost.

    Items on the Pareto front get rank 1; second front gets rank 2, etc.
    """
    n = len(items)
    if n == 0:
        return []

    # Extract two objectives: risk (maximize) and cost (minimize)
    risks = np.array([it["composite_risk"] for it in items])
    costs = np.array([it.get("remediation_cost", 1.0) for it in items])

    remaining = set(range(n))
    ranks = np.zeros(n, dtype=int)
    front = 0

    while remaining:
        front += 1
        front_members = []
        indices = list(remaining)

        for i in indices:
            dominated = False
            for j in indices:
                if i == j:
                    continue
                # j dominates i if j has higher risk AND lower cost
                if risks[j] >= risks[i] and costs[j] <= costs[i] and (
                    risks[j] > risks[i] or costs[j] < costs[i]
                ):
                    dominated = True
                    break
            if not dominated:
                front_members.append(i)

        for m in front_members:
            ranks[m] = front
            remaining.discard(m)

    # Sort by front, then by risk within front
    order = sorted(range(n), key=lambda i: (ranks[i], -risks[i]))
    result = []
    for rank_pos, idx in enumerate(order, 1):
        it = items[idx].copy()
        it["pareto_rank"] = rank_pos
        it["pareto_front"] = int(ranks[idx])
        result.append(it)
    return result


# ============================================================
#  2. Cost-Benefit Prioritization
# ============================================================

def cost_benefit_prioritize(
    items: list[dict],
) -> list[dict]:
    """
    Rank by ROI = risk_reduction / remediation_cost.
    Higher ROI → higher priority.
    """
    for it in items:
        cost = max(it.get("remediation_cost", 1.0), 0.01)
        it["cost_benefit_ratio"] = it["composite_risk"] / cost

    sorted_items = sorted(items, key=lambda x: x["cost_benefit_ratio"], reverse=True)
    for rank, it in enumerate(sorted_items, 1):
        it["cost_benefit_rank"] = rank
    return sorted_items


# ============================================================
#  3. Time-Sensitive Prioritization
# ============================================================

def time_sensitive_prioritize(
    items: list[dict],
) -> list[dict]:
    """
    Prioritizes items with:
      - Active exploits or KEV entries
      - Short time-to-exploit (high exploit velocity)
      - Approaching compliance deadlines

    Urgency = risk × time_pressure_factor
    """
    for it in items:
        kev = 1.5 if it.get("is_in_kev") else 1.0
        exploit_active = 1.3 if it.get("has_public_exploit") else 1.0
        velocity = it.get("exploit_velocity_days", -1)
        speed_factor = 1.0
        if velocity is not None and velocity >= 0:
            speed_factor = max(1.0, 2.0 - velocity / 30.0)  # faster = more urgent

        it["time_urgency"] = it["composite_risk"] * kev * exploit_active * speed_factor

    sorted_items = sorted(items, key=lambda x: x["time_urgency"], reverse=True)
    for rank, it in enumerate(sorted_items, 1):
        it["time_sensitive_rank"] = rank
    return sorted_items


# ============================================================
#  4. Multi-Criteria Decision Making (TOPSIS)
# ============================================================

def topsis_prioritize(
    items: list[dict],
    criteria: list[str] | None = None,
    weights: list[float] | None = None,
    is_benefit: list[bool] | None = None,
) -> list[dict]:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

    Default criteria: [composite_risk, exploit_likelihood, adjusted_impact, asset_criticality_score]
    """
    if not items:
        return []

    prio_cfg = _load_prioritization_config()
    mc_cfg = prio_cfg.get("algorithms", {}).get("multi_criteria", {})

    if criteria is None:
        cw = mc_cfg.get("criteria_weights", {})
        if cw:
            criteria = list(cw.keys())
            weights = list(cw.values())
        else:
            criteria = ["composite_risk", "exploit_likelihood", "adjusted_impact", "remediation_cost"]
    if weights is None:
        weights = [0.35, 0.25, 0.25, 0.15]
    if is_benefit is None:
        loaded = mc_cfg.get("is_benefit")
        if loaded and len(loaded) >= len(criteria):
            is_benefit = loaded[:len(criteria)]
        else:
            # Default: all are benefits except "cost" and "time" criteria
            cost_keywords = ("cost", "time_exposure")
            is_benefit = [not any(kw in c for kw in cost_keywords) for c in criteria]

    # Build decision matrix
    n = len(items)
    m = len(criteria)
    D = np.zeros((n, m), dtype=np.float64)
    for i, it in enumerate(items):
        for j, c in enumerate(criteria):
            D[i, j] = float(it.get(c, 0.0))

    # Normalize columns
    norms = np.linalg.norm(D, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    N = D / norms

    # Weighted normalized
    w = np.array(weights[:m], dtype=np.float64)
    w /= w.sum()
    V = N * w

    # Ideal best and worst
    ideal_best = np.zeros(m)
    ideal_worst = np.zeros(m)
    for j in range(m):
        if is_benefit[j]:
            ideal_best[j] = V[:, j].max()
            ideal_worst[j] = V[:, j].min()
        else:
            ideal_best[j] = V[:, j].min()
            ideal_worst[j] = V[:, j].max()

    # Distances
    d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    # Closeness coefficient
    closeness = d_worst / (d_best + d_worst + 1e-12)

    order = np.argsort(-closeness)
    for rank_pos, idx in enumerate(order, 1):
        items[idx]["topsis_score"] = float(closeness[idx])
        items[idx]["topsis_rank"] = rank_pos

    return sorted(items, key=lambda x: x.get("topsis_rank", 999))


# ============================================================
#  5. Ensemble Prioritization
# ============================================================

def ensemble_prioritize(
    items: list[dict],
    method: str = "rank_average",
) -> list[PrioritizedItem]:
    """
    Combine ranks from all 4 algorithms into a final ranking.

    Methods:
      - "rank_average": average of all ranks
      - "rank_min":     take the best (minimum) rank
      - "weighted":     weighted average using config weights
    """
    rank_keys = ["pareto_rank", "cost_benefit_rank", "time_sensitive_rank", "topsis_rank"]

    pcfg = _load_prioritization_config() if method == "weighted" else {}
    ensemble_cfg = pcfg.get("algorithms", {}).get("ensemble", {}) if pcfg else {}

    for it in items:
        ranks = [it.get(k, len(items)) for k in rank_keys]
        if method == "rank_min":
            it["ensemble_score"] = min(ranks)
        elif method == "weighted":
            w = ensemble_cfg.get("weights", [0.25, 0.25, 0.25, 0.25])
            it["ensemble_score"] = sum(r * wi for r, wi in zip(ranks, w))
        else:  # rank_average
            it["ensemble_score"] = sum(ranks) / len(ranks)

    sorted_items = sorted(items, key=lambda x: x["ensemble_score"])

    results = []
    for rank, it in enumerate(sorted_items, 1):
        results.append(PrioritizedItem(
            pair_key=it.get("pair_key", f"{it.get('vuln_id', '')}::{it.get('asset_id', '')}"),
            vuln_id=it.get("vuln_id", ""),
            asset_id=it.get("asset_id", ""),
            composite_risk=it.get("composite_risk", 0.0),
            ranks={k: it.get(k, 999) for k in rank_keys},
            final_rank=rank,
        ))
    return results


# ============================================================
#  Full Pipeline
# ============================================================

def run_prioritization_pipeline(
    items: list[dict],
) -> list[PrioritizedItem]:
    """
    Run all 5 prioritization algorithms sequentially,
    then apply ensemble ranking.
    """
    items = pareto_prioritize(items)
    items = cost_benefit_prioritize(items)
    items = time_sensitive_prioritize(items)
    items = topsis_prioritize(items)
    return ensemble_prioritize(items)


def _load_prioritization_config(path: str | None = None) -> dict:
    try:
        if path:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        else:
            from config import load_yaml
            return load_yaml("prioritization.yaml")
    except (FileNotFoundError, yaml.YAMLError):
        return {}
