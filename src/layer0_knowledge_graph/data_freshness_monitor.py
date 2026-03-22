"""
Data Freshness Monitor – RECOMMENDED ADDITION #1.

Tracks how old each data source is and computes confidence degradation
factors.  Alerts when any source exceeds its acceptable staleness threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from config import get_config
from layer0_knowledge_graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class FreshnessReport:
    source: str
    last_import_date: datetime | None
    threshold_days: int
    age_days: float
    is_stale: bool
    freshness_score: float  # 1.0 = perfectly fresh, 0.0 = completely stale


class DataFreshnessMonitor:
    """
    Monitors data source staleness and computes freshness scores.

    Freshness score formula:
        If age <= threshold: score = 1.0 - 0.5 * (age / threshold)
        If age > threshold:  score = max(0, 0.5 * exp(-(age - threshold) / threshold))

    This gives a smooth degradation: fresh data scores ~1.0, data at the
    threshold scores ~0.5, and very stale data approaches 0.
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self.graph = graph_store
        cfg = get_config()
        self.thresholds = cfg.layer0.freshness_thresholds_days

    def check_all(self, reference_time: datetime | None = None) -> list[FreshnessReport]:
        """Check freshness of all configured data sources."""
        now = reference_time or datetime.utcnow()
        reports = []

        for source, threshold_days in self.thresholds.items():
            report = self._check_source(source, threshold_days, now)
            reports.append(report)
            if report.is_stale:
                logger.warning(
                    "DATA STALE: %s is %.1f days old (threshold: %d days, score: %.2f)",
                    source, report.age_days, threshold_days, report.freshness_score,
                )

        return reports

    def get_overall_freshness(self, reference_time: datetime | None = None) -> float:
        """
        Compute a single overall freshness score (weighted average).
        Critical sources (NVD, KEV) are weighted more heavily.
        """
        reports = self.check_all(reference_time)
        if not reports:
            return 0.0

        # Weight critical sources higher
        weights = {
            "nvd": 3.0,
            "cisa_kev": 2.5,
            "exploitdb": 2.0,
            "otx": 1.5,
            "cmdb": 2.0,
            "vendor_advisories": 1.5,
            "network_scan": 2.0,
            "siem_alerts": 1.0,
        }

        total_weight = 0.0
        weighted_sum = 0.0
        for r in reports:
            w = weights.get(r.source, 1.0)
            weighted_sum += w * r.freshness_score
            total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _check_source(self, source: str, threshold_days: int,
                      now: datetime) -> FreshnessReport:
        latest = self.graph.get_latest_import(source)

        if latest is None:
            return FreshnessReport(
                source=source,
                last_import_date=None,
                threshold_days=threshold_days,
                age_days=float("inf"),
                is_stale=True,
                freshness_score=0.0,
            )

        import_ts_str = latest.get("import_ts", "")
        try:
            import_ts = datetime.fromisoformat(import_ts_str)
        except (ValueError, TypeError):
            return FreshnessReport(
                source=source,
                last_import_date=None,
                threshold_days=threshold_days,
                age_days=float("inf"),
                is_stale=True,
                freshness_score=0.0,
            )

        age = now - import_ts
        age_days = age.total_seconds() / 86400.0
        is_stale = age_days > threshold_days
        score = self._compute_score(age_days, threshold_days)

        return FreshnessReport(
            source=source,
            last_import_date=import_ts,
            threshold_days=threshold_days,
            age_days=age_days,
            is_stale=is_stale,
            freshness_score=score,
        )

    @staticmethod
    def _compute_score(age_days: float, threshold_days: int) -> float:
        """Smooth freshness score degradation."""
        import math
        if threshold_days <= 0:
            return 0.0
        if age_days <= threshold_days:
            return 1.0 - 0.5 * (age_days / threshold_days)
        # Exponential decay beyond threshold
        overage = age_days - threshold_days
        return max(0.0, 0.5 * math.exp(-overage / threshold_days))
