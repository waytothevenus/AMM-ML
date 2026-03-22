"""
Model Version Manager – tracks model versions, metrics, and lineage.

Stores metadata in SQLite alongside the .joblib artifacts so the
air-gapped host can verify which models are deployed and compare A/B
performance.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from config import get_config

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Registry for model versions, metrics, and lifecycle state."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_config()
        base = Path(cfg.models.base_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path or base / "model_registry.sqlite")
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS model_versions (
                model_name   TEXT    NOT NULL,
                version      TEXT    NOT NULL,
                artifact_path TEXT   NOT NULL,
                trained_at   TEXT    NOT NULL,
                metrics      TEXT    NOT NULL DEFAULT '{}',
                status       TEXT    NOT NULL DEFAULT 'staged',
                promoted_at  TEXT,
                PRIMARY KEY (model_name, version)
            );
            CREATE TABLE IF NOT EXISTS ab_assignments (
                model_name   TEXT NOT NULL,
                variant      TEXT NOT NULL,
                version      TEXT NOT NULL,
                traffic_pct  REAL NOT NULL DEFAULT 0.5,
                started_at   TEXT NOT NULL,
                ended_at     TEXT,
                PRIMARY KEY (model_name, variant)
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    #  Version CRUD
    # ------------------------------------------------------------------
    def register(
        self,
        model_name: str,
        version: str,
        artifact_path: str | Path,
        metrics: dict | None = None,
    ) -> None:
        """Register a newly trained model version."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO model_versions
                (model_name, version, artifact_path, trained_at, metrics, status)
            VALUES (?, ?, ?, ?, ?, 'staged')
            """,
            [model_name, version, str(artifact_path),
             datetime.utcnow().isoformat(),
             json.dumps(metrics or {})],
        )
        self._conn.commit()
        logger.info("Registered %s v%s (staged)", model_name, version)

    def promote(self, model_name: str, version: str) -> None:
        """Promote a version to 'production' (demoting previous prod)."""
        self._conn.execute(
            "UPDATE model_versions SET status='archived' "
            "WHERE model_name=? AND status='production'",
            [model_name],
        )
        self._conn.execute(
            "UPDATE model_versions SET status='production', promoted_at=? "
            "WHERE model_name=? AND version=?",
            [datetime.utcnow().isoformat(), model_name, version],
        )
        self._conn.commit()
        logger.info("Promoted %s v%s → production", model_name, version)

    def get_production_version(self, model_name: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM model_versions WHERE model_name=? AND status='production'",
            [model_name],
        ).fetchone()
        return dict(row) if row else None

    def list_versions(self, model_name: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM model_versions WHERE model_name=? ORDER BY trained_at DESC",
            [model_name],
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    #  A/B testing helpers
    # ------------------------------------------------------------------
    def start_ab_test(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
        traffic_b: float = 0.5,
    ) -> None:
        """Create an A/B test between two versions."""
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO ab_assignments VALUES (?,?,?,?,?,NULL)",
            [model_name, "A", version_a, 1.0 - traffic_b, now],
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO ab_assignments VALUES (?,?,?,?,?,NULL)",
            [model_name, "B", version_b, traffic_b, now],
        )
        self._conn.commit()
        logger.info(
            "A/B test started: %s  A=%s (%.0f%%)  B=%s (%.0f%%)",
            model_name, version_a, (1.0 - traffic_b) * 100,
            version_b, traffic_b * 100,
        )

    def get_ab_variant(self, model_name: str, deterministic_key: str) -> str:
        """
        Deterministically assign a key to A or B based on hash.
        Returns the version string for the selected variant.
        """
        import hashlib
        rows = self._conn.execute(
            "SELECT variant, version, traffic_pct FROM ab_assignments "
            "WHERE model_name=? AND ended_at IS NULL ORDER BY variant",
            [model_name],
        ).fetchall()
        if not rows:
            prod = self.get_production_version(model_name)
            return prod["version"] if prod else ""

        digest = int(hashlib.sha256(deterministic_key.encode()).hexdigest(), 16)
        threshold = rows[0]["traffic_pct"]
        frac = (digest % 10000) / 10000.0
        chosen = rows[0] if frac < threshold else rows[1] if len(rows) > 1 else rows[0]
        return chosen["version"]

    def end_ab_test(self, model_name: str) -> None:
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "UPDATE ab_assignments SET ended_at=? "
            "WHERE model_name=? AND ended_at IS NULL",
            [now, model_name],
        )
        self._conn.commit()
        logger.info("A/B test ended for %s", model_name)

    def close(self) -> None:
        self._conn.close()
