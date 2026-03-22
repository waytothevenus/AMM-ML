"""
Test suite for the Hybrid AMM+ML Vulnerability Risk Assessment System.

These tests validate core components in isolation without requiring
external data, databases, or a running config environment.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src/ on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config singleton between tests."""
    import config
    config._config = None
    yield
    config._config = None


# ═══════════════════════════════════════════════════════════════════════════
# Test: Domain Models
# ═══════════════════════════════════════════════════════════════════════════

class TestDomainModels:
    def test_risk_state_enum(self):
        from models import RiskState
        assert RiskState.UNKNOWN == 0
        assert RiskState.REMEDIATED == 5
        assert len(RiskState) == 6

    def test_vulnerability_model(self):
        from models import Vulnerability
        v = Vulnerability(cve_id="CVE-2024-0001", cvss_base_score=9.8)
        assert v.cve_id == "CVE-2024-0001"
        assert v.cvss_base_score == 9.8
        assert v.is_in_kev is False

    def test_ml_predictions_aliases(self):
        from models import MLPredictions
        p = MLPredictions(
            cve_id="CVE-2024-0001",
            asset_id="host-1",
            exploit_probability=0.85,
            impact_adjustment=7.5,
        )
        # Properties should alias to field values
        assert p.exploit_likelihood == 0.85
        assert p.adjusted_impact == 7.5
        assert p.vuln_id == "CVE-2024-0001"

    def test_markov_state_distribution_alias(self):
        from models import MarkovState, RiskState
        ms = MarkovState(
            cve_id="CVE-2024-0001",
            asset_id="host-1",
            state_distribution=[0.0, 0.3, 0.4, 0.2, 0.1, 0.0],
            current_state=RiskState.EXPLOIT_AVAILABLE,
        )
        assert ms.distribution == [0.0, 0.3, 0.4, 0.2, 0.1, 0.0]
        assert ms.current_state == RiskState.EXPLOIT_AVAILABLE

    def test_markov_state_alias_constructor(self):
        """Test that vuln_id/distribution aliases work in constructor."""
        from models import MarkovState
        ms = MarkovState(
            vuln_id="CVE-2024-0001",
            asset_id="host-1",
            distribution=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            entropy=0.5,
        )
        assert ms.cve_id == "CVE-2024-0001"
        assert ms.state_distribution == [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        assert ms.state_entropy == 0.5

    def test_aggregated_risk(self):
        from models import AggregatedRisk
        r = AggregatedRisk(
            vuln_id="CVE-2024-0001",
            asset_id="host-1",
            composite_risk=0.78,
            final_rank=3,
        )
        assert r.composite_risk == 0.78
        assert r.final_rank == 3


# ═══════════════════════════════════════════════════════════════════════════
# Test: Layer 0 – Ontologies
# ═══════════════════════════════════════════════════════════════════════════

class TestOntologies:
    def test_node_and_relation_types(self):
        from layer0_knowledge_graph.ontologies import NodeType, RelationType
        assert NodeType.VULNERABILITY.value == "Vulnerability"
        assert RelationType.VULN_PRESENT_ON.value == "PRESENT_ON"
        assert len(NodeType) >= 8
        assert len(RelationType) >= 10


# ═══════════════════════════════════════════════════════════════════════════
# Test: Layer 0 – Data Freshness Monitor
# ═══════════════════════════════════════════════════════════════════════════

class TestDataFreshnessMonitor:
    def test_freshness_report(self):
        from layer0_knowledge_graph.data_freshness_monitor import DataFreshnessMonitor
        from layer0_knowledge_graph.graph_store import GraphStore
        gs = GraphStore()
        monitor = DataFreshnessMonitor(gs)
        # Should work even with empty graph
        reports = monitor.check_all()
        assert isinstance(reports, list)
        overall = monitor.get_overall_freshness()
        assert 0.0 <= overall <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Test: Layer 1 – Feature Engineering (unit-level)
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalFeatures:
    def test_compute_temporal_features(self):
        from layer0_knowledge_graph.graph_store import GraphStore
        from layer0_knowledge_graph.ontologies import NodeType, RelationType
        from layer1_feature_engineering.temporal_features import compute_temporal_features

        gs = GraphStore()
        now = datetime.utcnow()
        # Add a vulnerability node with published_date
        gs.add_node("CVE-2024-0001", NodeType.VULNERABILITY, {
            "published_date": (now - timedelta(days=10)).isoformat(),
        })
        gs.add_node("host-1", NodeType.ASSET, {})
        gs.add_edge("CVE-2024-0001", "host-1", RelationType.VULN_PRESENT_ON, {})

        feats = compute_temporal_features("CVE-2024-0001", "host-1", gs, now)
        assert isinstance(feats, dict)
        assert "days_since_disclosure" in feats


class TestFeatureStore:
    def test_persist_and_retrieve(self, tmp_path):
        from layer1_feature_engineering.feature_store import FeatureStore
        db_path = tmp_path / "test_features.duckdb"
        store = FeatureStore(db_path=str(db_path))

        rows = [
            {"vuln_id": "CVE-2024-0001", "asset_id": "host-1",
             "features": {"f1": 1.0, "f2": 2.0, "f3": 3.0}},
        ]
        n = store.persist_features(rows, cycle_id="cycle_001")
        assert n >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Test: Layer 2 – Confidence Degradation
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceDegradation:
    def test_decay_reduces_confidence(self):
        from layer2_ml_engine.confidence_degradation import ConfidenceDegradation

        degrader = ConfidenceDegradation()
        now = datetime.utcnow()
        old_model_time = now - timedelta(days=60)

        conf = degrader.adjust(1.0, 1.0, old_model_time, now)
        assert 0.0 < conf < 1.0  # should have decayed

    def test_fresh_model_no_decay(self):
        from layer2_ml_engine.confidence_degradation import ConfidenceDegradation

        degrader = ConfidenceDegradation()
        now = datetime.utcnow()

        conf = degrader.adjust(1.0, 1.0, now, now)
        assert conf == pytest.approx(1.0, abs=0.05)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Layer 3 – Chapman-Kolmogorov Solver
# ═══════════════════════════════════════════════════════════════════════════

class TestChapmanKolmogorov:
    def test_discrete_evolution_preserves_probability(self):
        from layer3_markov_engine.chapman_kolmogorov import ChapmanKolmogorovSolver

        solver = ChapmanKolmogorovSolver()
        pi = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Simple valid stochastic matrix
        tpm = np.array([
            [0.5, 0.3, 0.1, 0.05, 0.05, 0.0],
            [0.0, 0.4, 0.3, 0.1,  0.1,  0.1],
            [0.0, 0.0, 0.3, 0.4,  0.2,  0.1],
            [0.0, 0.0, 0.0, 0.3,  0.4,  0.3],
            [0.0, 0.0, 0.0, 0.0,  0.3,  0.7],
            [0.0, 0.0, 0.0, 0.0,  0.0,  1.0],  # absorbing
        ])

        pi_evolved = solver.evolve_discrete(pi, tpm, steps=10)
        assert pi_evolved.sum() == pytest.approx(1.0, abs=1e-10)
        assert all(p >= 0 for p in pi_evolved)

    def test_absorbing_state_converges(self):
        from layer3_markov_engine.chapman_kolmogorov import ChapmanKolmogorovSolver

        solver = ChapmanKolmogorovSolver()
        pi = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tpm = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        pi_final = solver.evolve_discrete(pi, tpm, steps=100)
        # Should be fully absorbed into state 5
        assert pi_final[5] == pytest.approx(1.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Layer 3 – Absorption Time Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class TestAbsorptionTimeAnalyzer:
    def test_expected_absorption_time(self):
        from layer3_markov_engine.absorption_time_analyzer import AbsorptionTimeAnalyzer

        analyzer = AbsorptionTimeAnalyzer()
        # 6 states matching the system's NUM_STATES, state 5 absorbing
        tpm = np.array([
            [0.5, 0.3, 0.1, 0.05, 0.05, 0.0],
            [0.0, 0.4, 0.3, 0.1,  0.1,  0.1],
            [0.0, 0.0, 0.3, 0.4,  0.2,  0.1],
            [0.0, 0.0, 0.0, 0.3,  0.4,  0.3],
            [0.0, 0.0, 0.0, 0.0,  0.3,  0.7],
            [0.0, 0.0, 0.0, 0.0,  0.0,  1.0],  # absorbing
        ])
        dist = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        t = analyzer.expected_absorption_time(dist, tpm)
        assert t > 0
        assert math.isfinite(t)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Layer 3 – State Manager
# ═══════════════════════════════════════════════════════════════════════════

class TestStateManager:
    def test_initialize_and_retrieve(self, tmp_path):
        from layer3_markov_engine.state_manager import StateManager

        db = tmp_path / "test_states.sqlite"
        sm = StateManager(db_path=str(db))

        result = sm.initialize_state("CVE-2024-0001", "host-1")
        assert result is not None
        assert result.cve_id == "CVE-2024-0001"
        assert len(result.state_distribution) == 6
        assert sum(result.state_distribution) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Layer 4 – Prioritization Algorithms
# ═══════════════════════════════════════════════════════════════════════════

class TestPrioritization:
    @pytest.fixture
    def sample_items(self):
        return [
            {
                "pair_key": "CVE-2024-0001::host-1",
                "vuln_id": "CVE-2024-0001",
                "asset_id": "host-1",
                "composite_risk": 0.9,
                "exploit_likelihood": 0.8,
                "adjusted_impact": 9.0,
                "asset_criticality_score": 0.9,
                "remediation_cost": 100.0,
                "has_public_exploit": True,
                "is_in_kev": True,
                "exploit_velocity_days": 5,
            },
            {
                "pair_key": "CVE-2024-0002::host-2",
                "vuln_id": "CVE-2024-0002",
                "asset_id": "host-2",
                "composite_risk": 0.3,
                "exploit_likelihood": 0.2,
                "adjusted_impact": 3.0,
                "asset_criticality_score": 0.4,
                "remediation_cost": 50.0,
                "has_public_exploit": False,
                "is_in_kev": False,
                "exploit_velocity_days": -1,
            },
            {
                "pair_key": "CVE-2024-0003::host-3",
                "vuln_id": "CVE-2024-0003",
                "asset_id": "host-3",
                "composite_risk": 0.6,
                "exploit_likelihood": 0.5,
                "adjusted_impact": 6.0,
                "asset_criticality_score": 0.7,
                "remediation_cost": 200.0,
                "has_public_exploit": True,
                "is_in_kev": False,
                "exploit_velocity_days": 30,
            },
        ]

    def test_pareto_prioritize(self, sample_items):
        from layer4_risk_aggregation.prioritization import pareto_prioritize
        result = pareto_prioritize(sample_items)
        assert len(result) == 3
        # All items should have pareto_rank
        for item in result:
            assert "pareto_rank" in item

    def test_cost_benefit_prioritize(self, sample_items):
        from layer4_risk_aggregation.prioritization import cost_benefit_prioritize
        result = cost_benefit_prioritize(sample_items)
        assert len(result) == 3
        for item in result:
            assert "cost_benefit_rank" in item

    def test_time_sensitive_prioritize(self, sample_items):
        from layer4_risk_aggregation.prioritization import time_sensitive_prioritize
        result = time_sensitive_prioritize(sample_items)
        assert len(result) == 3
        # KEV item should rank highest
        kev_item = [i for i in result if i.get("is_in_kev")][0]
        assert kev_item["time_sensitive_rank"] == 1

    def test_topsis_prioritize(self, sample_items):
        from layer4_risk_aggregation.prioritization import topsis_prioritize
        result = topsis_prioritize(sample_items)
        assert len(result) == 3
        for item in result:
            assert "topsis_rank" in item
            assert "topsis_score" in item

    def test_ensemble_produces_final_ranks(self, sample_items):
        from layer4_risk_aggregation.prioritization import run_prioritization_pipeline
        result = run_prioritization_pipeline(sample_items)
        assert len(result) == 3
        ranks = {r.final_rank for r in result}
        assert ranks == {1, 2, 3}


# ═══════════════════════════════════════════════════════════════════════════
# Test: Backtesting Engine
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestingEngine:
    def test_roc_auc_perfect(self):
        from backtesting.backtesting_engine import BacktestingEngine

        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        auc = BacktestingEngine._roc_auc(y_true, y_score)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_roc_auc_random(self):
        from backtesting.backtesting_engine import BacktestingEngine

        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=1000)
        y_score = rng.random(1000)
        auc = BacktestingEngine._roc_auc(y_true, y_score)
        # Random predictions should give ~0.5 AUC
        assert 0.4 < auc < 0.6


# ═══════════════════════════════════════════════════════════════════════════
# Test: Staging Tools
# ═══════════════════════════════════════════════════════════════════════════

class TestStagingPackage:
    def test_checksum_generation(self, tmp_path):
        from staging.download_feeds import _generate_checksums

        bundle = tmp_path / "test_bundle"
        bundle.mkdir()
        (bundle / "file1.txt").write_text("hello")
        (bundle / "file2.txt").write_text("world")

        _generate_checksums(bundle)
        checksum_file = bundle / "checksums.sha256"
        assert checksum_file.exists()
        lines = checksum_file.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            assert len(line.split("  ")[0]) == 64  # SHA-256 hex length
