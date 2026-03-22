"""
Inference Engine – orchestrates ELP, ISA, ACC predictions for a batch
of (vuln, asset) pairs, applying A/B routing and confidence degradation.
"""

from __future__ import annotations

import logging
from datetime import datetime

from models import MLPredictions
from layer2_ml_engine.elp import ExploitLikelihoodPredictor
from layer2_ml_engine.isa import ImpactSeverityAdjuster
from layer2_ml_engine.acc import AssetCriticalityClassifier
from layer2_ml_engine.confidence_degradation import ConfidenceDegradation
from layer2_ml_engine.model_version_manager import ModelVersionManager

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Runs all three ML models for each (vuln, asset) pair and returns
    an MLPredictions bundle with degraded confidence.
    """

    def __init__(
        self,
        elp: ExploitLikelihoodPredictor | None = None,
        isa: ImpactSeverityAdjuster | None = None,
        acc: AssetCriticalityClassifier | None = None,
        version_manager: ModelVersionManager | None = None,
        data_freshness_score: float = 1.0,
    ) -> None:
        self.elp = elp or ExploitLikelihoodPredictor()
        self.isa = isa or ImpactSeverityAdjuster()
        self.acc = acc or AssetCriticalityClassifier()
        self.vm = version_manager
        self.degrader = ConfidenceDegradation()
        self._data_freshness = data_freshness_score

    def predict(
        self,
        vuln_id: str,
        asset_id: str,
        features: dict[str, float],
        reference_time: datetime | None = None,
    ) -> MLPredictions:
        """Run ELP + ISA + ACC for one pair."""
        now = reference_time or datetime.utcnow()

        # ELP
        raw_elp = self.elp.predict(features)
        elp_trained = self._trained_at("elp")
        adj_elp = self.degrader.adjust(raw_elp, self._data_freshness, elp_trained, now)
        elp_conf = self.degrader.adjust(1.0, self._data_freshness, elp_trained, now)

        # ISA
        raw_isa = self.isa.predict(features)
        isa_trained = self._trained_at("isa")
        isa_conf = self.degrader.adjust(1.0, self._data_freshness, isa_trained, now)

        # ACC
        acc_dist = self.acc.predict(features)
        acc_tier = max(acc_dist, key=acc_dist.get)  # type: ignore[arg-type]
        acc_trained = self._trained_at("acc")
        acc_conf = self.degrader.adjust(1.0, self._data_freshness, acc_trained, now)

        # Asset criticality score from distribution
        crit_score = (
            acc_dist.get("critical", 0.0) * 1.0
            + acc_dist.get("high", 0.0) * 0.75
            + acc_dist.get("medium", 0.0) * 0.5
            + acc_dist.get("low", 0.0) * 0.25
        )

        # Overall confidence: geometric mean of individual factors
        overall_conf = (elp_conf * isa_conf * acc_conf) ** (1.0 / 3.0)

        # Model age in days (use max across models)
        ages = []
        for trained in (elp_trained, isa_trained, acc_trained):
            if trained:
                ages.append((now - trained).total_seconds() / 86400.0)
        max_age = max(ages) if ages else 0.0

        return MLPredictions(
            cve_id=vuln_id,
            asset_id=asset_id,
            exploit_probability=adj_elp,
            impact_adjustment=raw_isa,
            asset_criticality_score=crit_score,
            asset_criticality_tier=acc_tier,
            asset_criticality_distribution=acc_dist,
            confidence=overall_conf,
            elp_confidence=elp_conf,
            isa_confidence=isa_conf,
            acc_confidence=acc_conf,
            model_age_days=max_age,
            model_versions={
                "elp": self.elp.version,
                "isa": self.isa.version,
                "acc": self.acc.version,
            },
        )

    def predict_batch(
        self,
        pairs: list[dict],
        reference_time: datetime | None = None,
    ) -> list[MLPredictions]:
        """
        Run inference for a batch of pairs.
        Each dict must have: vuln_id, asset_id, features.
        """
        results = []
        for p in pairs:
            pred = self.predict(
                p["vuln_id"], p["asset_id"], p["features"], reference_time
            )
            results.append(pred)
        return results

    def _trained_at(self, model_name: str) -> datetime | None:
        """Look up when a model was last trained from the version manager."""
        if self.vm is None:
            return None
        info = self.vm.get_production_version(model_name)
        if info and info.get("trained_at"):
            try:
                return datetime.fromisoformat(info["trained_at"])
            except (ValueError, TypeError):
                return None
        return None
