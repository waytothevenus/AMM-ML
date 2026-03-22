"""
GraphQL API – Strawberry-based GraphQL endpoint.

Provides flexible querying for the dashboard and advanced users.
"""

from __future__ import annotations

import logging
from typing import Optional

import strawberry
from strawberry.fastapi import GraphQLRouter

from layer5_presentation.service_layer import ServiceLayer

logger = logging.getLogger(__name__)

_service: ServiceLayer | None = None


def _svc() -> ServiceLayer:
    global _service
    if _service is None:
        _service = ServiceLayer()
    return _service


# ---- GraphQL Types ----

@strawberry.type
class VulnerabilityType:
    id: str
    cvss_base_score: Optional[float]
    has_public_exploit: bool
    is_in_kev: bool
    exploit_likelihood: Optional[float]
    adjusted_impact: Optional[float]
    composite_risk: Optional[float]


@strawberry.type
class AssetType:
    id: str
    hostname: Optional[str]
    business_unit: Optional[str]
    criticality: Optional[str]
    propagated_risk: Optional[float]


@strawberry.type
class RiskPairType:
    vuln_id: str
    asset_id: str
    composite_risk: float
    exploit_likelihood: float
    adjusted_impact: float
    final_rank: int


@strawberry.type
class FreshnessType:
    source: str
    age_days: float
    is_stale: bool
    freshness_score: float
    threshold_days: int


@strawberry.type
class ForecastPointType:
    horizon_days: int
    risk_score: float


@strawberry.type
class Query:
    @strawberry.field
    def vulnerabilities(self, limit: int = 100, offset: int = 0) -> list[VulnerabilityType]:
        data = _svc().list_vulnerabilities(limit, offset)
        return [VulnerabilityType(**v) for v in data.get("items", [])]

    @strawberry.field
    def vulnerability(self, vuln_id: str) -> Optional[VulnerabilityType]:
        data = _svc().get_vulnerability(vuln_id)
        if data is None:
            return None
        fields = {f.name for f in VulnerabilityType.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in fields}
        return VulnerabilityType(**filtered)

    @strawberry.field
    def assets(self, limit: int = 100, offset: int = 0) -> list[AssetType]:
        data = _svc().list_assets(limit, offset)
        return [AssetType(**a) for a in data.get("items", [])]

    @strawberry.field
    def risk_pairs(self, limit: int = 100, min_risk: float = 0.0) -> list[RiskPairType]:
        data = _svc().get_risk_pairs(limit, min_risk)
        return [RiskPairType(**p) for p in data.get("items", [])]

    @strawberry.field
    def freshness(self) -> list[FreshnessType]:
        data = _svc().get_freshness_report()
        return [FreshnessType(**f) for f in data.get("sources", [])]

    @strawberry.field
    def forecast(self, pair_key: str) -> list[ForecastPointType]:
        data = _svc().get_forecast(pair_key)
        if data is None:
            return []
        return [
            ForecastPointType(horizon_days=h, risk_score=s)
            for h, s in data.get("points", {}).items()
        ]


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)
