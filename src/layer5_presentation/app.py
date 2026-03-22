"""
FastAPI Application – main entry point for Layer 5.

Provides:
  - REST API endpoints          (/api/v1/...)
  - GraphQL endpoint            (/graphql)
  - Static dashboard files      (/dashboard)
  - Health / status              (/health)
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from layer5_presentation.rest_api import router as rest_router
from layer5_presentation.graphql_api import graphql_app

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hybrid AMM+ML Vulnerability Risk Assessment",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS for local dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # REST API
    app.include_router(rest_router, prefix="/api/v1")

    # GraphQL
    app.mount("/graphql", graphql_app)

    # Static dashboard files
    if STATIC_DIR.exists():
        app.mount("/dashboard", StaticFiles(directory=str(STATIC_DIR), html=True), name="dashboard")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
