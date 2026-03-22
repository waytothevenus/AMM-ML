"""Shared configuration loader and data models used across all layers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve project root (parent of src/)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent  # src/config.py → project root

def _find_config_dir() -> Path:
    """Walk upward from this file to find the config/ directory."""
    candidate = PROJECT_ROOT / "config"
    if candidate.is_dir():
        return candidate
    # Fallback: environment variable
    env = os.environ.get("VRA_CONFIG_DIR")
    if env:
        return Path(env)
    raise FileNotFoundError("Cannot locate config/ directory")


CONFIG_DIR = _find_config_dir()


def load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML file from the config directory."""
    path = CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Pydantic settings models
# ---------------------------------------------------------------------------

class SystemSettings(BaseModel):
    name: str = "Hybrid AMM+ML Vulnerability Risk Assessment"
    version: str = "0.1.0"
    log_level: str = "INFO"
    daily_batch_time: str = "02:00"
    feedback_loop_passes: int = 2


class DataSettings(BaseModel):
    imports_dir: str = "data/imports"
    db_dir: str = "data/db"
    graph_dir: str = "data/graph"
    knowledge_graph_db: str = "data/db/knowledge_graph.db"
    feature_store_db: str = "data/db/feature_store.duckdb"
    reports_dir: str = "data/reports"


class ModelsSettings(BaseModel):
    base_dir: str = "models"
    elp_dir: str = "models/elp"
    isa_dir: str = "models/isa"
    acc_dir: str = "models/acc"
    embeddings_dir: str = "models/embeddings"
    embedding_model_name: str = "all-MiniLM-L6-v2"


class Layer0Settings(BaseModel):
    graph_backend: str = "networkx"
    freshness_thresholds_days: dict[str, int] = Field(default_factory=dict)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""


class Layer1Settings(BaseModel):
    embedding_backend: str = "tfidf"
    tfidf_max_features: int = 5000
    tfidf_svd_components: int = 128
    topological_max_hops: int = 3


class Layer2Settings(BaseModel):
    default_model_format: str = "joblib"
    confidence_decay_halflife_days: int = 90
    ab_test_holdout_fraction: float = 0.2


class Layer3Settings(BaseModel):
    risk_states: list[str] = Field(default_factory=lambda: [
        "Unknown", "Disclosed", "ExploitAvailable",
        "ActivelyExploited", "Mitigated", "Remediated",
    ])
    absorbing_states: list[str] = Field(default_factory=lambda: ["Remediated"])
    default_half_life_days: int = 30
    coupling_max_hops: int = 2


class Layer4Settings(BaseModel):
    prioritization_weights: dict[str, float] = Field(default_factory=dict)
    forecast_horizon_days: int = 30


class Layer5Settings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    static_dir: str = "src/layer5_presentation/static"
    enable_graphql: bool = True


class AppConfig(BaseModel):
    """Root application configuration."""

    system: SystemSettings = Field(default_factory=SystemSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    layer0: Layer0Settings = Field(default_factory=Layer0Settings)
    layer1: Layer1Settings = Field(default_factory=Layer1Settings)
    layer2: Layer2Settings = Field(default_factory=Layer2Settings)
    layer3: Layer3Settings = Field(default_factory=Layer3Settings)
    layer4: Layer4Settings = Field(default_factory=Layer4Settings)
    layer5: Layer5Settings = Field(default_factory=Layer5Settings)


def load_config() -> AppConfig:
    """Load and validate the main settings.yaml into an AppConfig."""
    raw = load_yaml("settings.yaml")
    return AppConfig(**raw)


# Singleton config – import this from anywhere
_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against PROJECT_ROOT."""
    return (PROJECT_ROOT / relative).resolve()
