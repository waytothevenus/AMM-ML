"""Shared domain models (Pydantic) used across all layers."""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Core Enums
# ---------------------------------------------------------------------------

class RiskState(IntEnum):
    """Discrete risk states for the Markov model."""
    UNKNOWN = 0
    DISCLOSED = 1
    EXPLOIT_AVAILABLE = 2
    ACTIVELY_EXPLOITED = 3
    MITIGATED = 4
    REMEDIATED = 5  # absorbing


class AssetCriticalityTier(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


# ---------------------------------------------------------------------------
# Layer 0 Entities
# ---------------------------------------------------------------------------

class Vulnerability(BaseModel):
    """A known software vulnerability."""
    cve_id: str
    description: str = ""
    cvss_base_score: float = 0.0
    cvss_vector: str = ""
    cwe_id: Optional[str] = None
    published_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    epss_score: Optional[float] = None
    is_in_kev: bool = False
    cpe_matches: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    source: str = "nvd"  # which connector imported this


class Asset(BaseModel):
    """A network asset (host, service, device)."""
    asset_id: str
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    os: Optional[str] = None
    services: list[str] = Field(default_factory=list)
    cpe_list: list[str] = Field(default_factory=list)
    business_unit: str = "bu_corporate"
    criticality: AssetCriticalityTier = AssetCriticalityTier.MEDIUM
    network_zone: str = "internal"
    last_scan_date: Optional[datetime] = None


class VulnAssetPair(BaseModel):
    """A (vulnerability, asset) pair – the fundamental analysis unit."""
    cve_id: str
    asset_id: str
    first_seen: Optional[datetime] = None
    current_state: RiskState = RiskState.DISCLOSED
    risk_score: float = 0.0


class ThreatIntelEntry(BaseModel):
    """A threat intelligence indicator or report."""
    source: str  # "otx", "vendor", "siem"
    indicator_type: str  # "ip", "domain", "cve", "ttp"
    indicator_value: str
    confidence: float = 0.5
    timestamp: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 1 Feature Vector
# ---------------------------------------------------------------------------

class FeatureVector(BaseModel):
    """Computed feature vector for a (vulnerability, asset) pair at time t."""
    cve_id: str
    asset_id: str
    timestamp: datetime

    # Temporal features
    days_since_disclosure: float = 0.0
    days_since_exploit_available: Optional[float] = None
    days_since_last_scan: float = 0.0
    patch_lag_days: Optional[float] = None

    # Topological features
    asset_degree: int = 0
    asset_betweenness: float = 0.0
    attack_path_length: Optional[float] = None
    reachable_critical_assets: int = 0

    # Threat intel features
    is_in_kev: bool = False
    threat_intel_mentions: int = 0
    mitre_technique_count: int = 0

    # Textual embedding (variable-length, stored separately or fixed-dim)
    text_embedding: list[float] = Field(default_factory=list)

    # Historical statistics
    historical_exploit_rate: float = 0.0
    similar_cve_exploit_rate: float = 0.0

    # Markov feedback features (populated after Layer 3 first pass)
    markov_state_distribution: list[float] = Field(default_factory=list)
    markov_state_entropy: float = 0.0
    markov_time_in_current_state: float = 0.0
    markov_absorption_time_estimate: Optional[float] = None

    # Raw scores passed through
    cvss_base_score: float = 0.0
    epss_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Layer 2 ML Predictions
# ---------------------------------------------------------------------------

class MLPredictions(BaseModel):
    """Output of the ML inference subsystem for one (v, a) pair."""
    cve_id: str
    asset_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    exploit_probability: float = 0.0        # p(v,a,t) from ELP
    impact_adjustment: float = 1.0          # α(v,a) from ISA  [0-10]
    asset_criticality_score: float = 0.5    # C(a) from ACC    [0-1]

    # Extended fields used by downstream consumers
    asset_criticality_tier: str = "medium"
    asset_criticality_distribution: Optional[dict] = None
    confidence: float = 1.0
    model_versions: dict = Field(default_factory=dict)

    # Confidence metadata
    elp_confidence: float = 1.0
    isa_confidence: float = 1.0
    acc_confidence: float = 1.0
    model_age_days: float = 0.0  # how old the model is

    # Aliases – canonical names used across all layers
    @property
    def vuln_id(self) -> str:
        return self.cve_id

    @property
    def exploit_likelihood(self) -> float:
        return self.exploit_probability

    @property
    def adjusted_impact(self) -> float:
        return self.impact_adjustment


# ---------------------------------------------------------------------------
# Layer 3 Markov Output
# ---------------------------------------------------------------------------

class MarkovState(BaseModel):
    """Markov model output for one (v, a) pair."""
    cve_id: str = ""
    asset_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    state_distribution: list[float] = Field(default_factory=list)  # π(v,a,t)
    current_state: RiskState = RiskState.UNKNOWN
    time_in_state: float = 0.0             # days in current state
    state_entropy: float = 0.0
    absorption_time: Optional[float] = None  # expected days to Remediated
    risk_decay_halflife: Optional[float] = None
    n_step_forecast: list[list[float]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, values: dict) -> dict:
        if isinstance(values, dict):
            # vuln_id → cve_id
            if "vuln_id" in values and "cve_id" not in values:
                values["cve_id"] = values.pop("vuln_id")
            # distribution → state_distribution
            if "distribution" in values and "state_distribution" not in values:
                values["state_distribution"] = values.pop("distribution")
            # entropy → state_entropy
            if "entropy" in values and "state_entropy" not in values:
                values["state_entropy"] = values.pop("entropy")
            # time_in_current_state → time_in_state
            if "time_in_current_state" in values and "time_in_state" not in values:
                values["time_in_state"] = values.pop("time_in_current_state")
        return values

    @property
    def distribution(self) -> list[float]:
        return self.state_distribution

    @property
    def entropy(self) -> float:
        return self.state_entropy


# ---------------------------------------------------------------------------
# Layer 4 Aggregated Risk
# ---------------------------------------------------------------------------

class AggregatedRisk(BaseModel):
    """Final risk assessment for a (v, a) pair."""
    cve_id: str = ""
    asset_id: str = ""
    vuln_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    composite_risk_score: float = 0.0
    composite_risk: float = 0.0          # alias used by aggregation engine
    propagated_risk: float = 0.0
    priority_rank: int = 0
    final_rank: int = 0
    recommended_action: str = "investigate"
    attack_path_risk: float = 0.0
    business_unit_risk: float = 0.0
    forecast_risk_30d: float = 0.0
    forecast: dict = Field(default_factory=dict)
    algorithm_ranks: dict = Field(default_factory=dict)

    # Prioritization scores from each algorithm
    pareto_rank: Optional[int] = None
    cost_benefit_score: Optional[float] = None
    time_sensitivity_score: Optional[float] = None
    multi_criteria_score: Optional[float] = None
    ensemble_rank: Optional[int] = None


# ---------------------------------------------------------------------------
# Data Import Manifest
# ---------------------------------------------------------------------------

class ImportManifest(BaseModel):
    """Metadata for an imported data batch."""
    batch_id: str
    source: str
    import_timestamp: datetime
    file_path: str
    record_count: int = 0
    checksum_sha256: str = ""
    data_date: Optional[datetime] = None  # when the source data was generated
