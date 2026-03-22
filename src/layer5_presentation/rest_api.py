"""
REST API – Layer 5 REST endpoints.

Endpoints:
  GET  /vulnerabilities              – list vulns with risk scores
  GET  /vulnerabilities/{vuln_id}    – detail for one vuln
  GET  /assets                       – list assets with risk scores
  GET  /assets/{asset_id}            – detail for one asset
  GET  /risk/pairs                   – all (vuln, asset) risk pairs
  GET  /risk/summary                 – org-level risk summary
  GET  /risk/bu-rollup               – business unit rollup
  GET  /risk/forecast/{pair_key}     – temporal forecast for a pair
  GET  /prioritization               – prioritized remediation list
  POST /whatif                       – what-if simulation
  GET  /freshness                    – data freshness report
  POST /import                       – trigger manual data import
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from layer5_presentation.service_layer import ServiceLayer

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton service layer (initialized on first call)
_service: ServiceLayer | None = None


def _svc() -> ServiceLayer:
    global _service
    if _service is None:
        _service = ServiceLayer()
    return _service


# ---- Request/Response models ----

class WhatIfRequest(BaseModel):
    pair_key: str
    action: str  # "patch", "mitigate", "isolate"
    parameters: dict[str, Any] = {}


class ImportRequest(BaseModel):
    file_path: str


# ---- Endpoints ----

@router.get("/vulnerabilities")
def list_vulnerabilities(
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("composite_risk"),
):
    return _svc().list_vulnerabilities(limit, offset, sort_by)


@router.get("/vulnerabilities/{vuln_id}")
def get_vulnerability(vuln_id: str):
    result = _svc().get_vulnerability(vuln_id)
    if result is None:
        raise HTTPException(404, f"Vulnerability {vuln_id} not found")
    return result


@router.get("/assets")
def list_assets(
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    return _svc().list_assets(limit, offset)


@router.get("/assets/{asset_id}")
def get_asset(asset_id: str):
    result = _svc().get_asset(asset_id)
    if result is None:
        raise HTTPException(404, f"Asset {asset_id} not found")
    return result


@router.get("/risk/pairs")
def risk_pairs(
    limit: int = Query(100, ge=1, le=10000),
    min_risk: float = Query(0.0, ge=0.0, le=1.0),
):
    return _svc().get_risk_pairs(limit, min_risk)


@router.get("/risk/summary")
def risk_summary():
    return _svc().get_risk_summary()


@router.get("/risk/bu-rollup")
def bu_rollup():
    return _svc().get_bu_rollup()


@router.get("/risk/forecast/{pair_key:path}")
def risk_forecast(pair_key: str):
    result = _svc().get_forecast(pair_key)
    if result is None:
        raise HTTPException(404, f"No forecast for {pair_key}")
    return result


@router.get("/prioritization")
def prioritization(
    limit: int = Query(50, ge=1, le=1000),
    algorithm: str = Query("ensemble"),
):
    return _svc().get_prioritization(limit, algorithm)


@router.post("/whatif")
def what_if(req: WhatIfRequest):
    return _svc().run_what_if(req.pair_key, req.action, req.parameters)


@router.get("/freshness")
def freshness():
    return _svc().get_freshness_report()


@router.post("/import")
def trigger_import(req: ImportRequest):
    return _svc().trigger_import(req.file_path)
