"""
Feature Store – DuckDB-backed storage for computed feature vectors.

Stores versioned feature vectors keyed by (vuln_id, asset_id, cycle_id).
Provides:
  - persist_features()       – write a batch of feature vectors
  - get_latest_features()    – retrieve latest feature vector for a pair
  - get_training_dataset()   – pull a labelled dataset for ML training
  - prune_old_cycles()       – housekeeping
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from config import get_config

logger = logging.getLogger(__name__)


class FeatureStore:
    """DuckDB-backed feature vector storage."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_config()
        self._db_path = str(db_path or cfg.data.feature_store_db)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(self._db_path)
        self._ensure_tables()

    # ------------------------------------------------------------------
    #  Schema
    # ------------------------------------------------------------------
    def _ensure_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_vectors (
                vuln_id      VARCHAR NOT NULL,
                asset_id     VARCHAR NOT NULL,
                cycle_id     VARCHAR NOT NULL,
                computed_at  TIMESTAMP NOT NULL,
                features     JSON NOT NULL,
                PRIMARY KEY (vuln_id, asset_id, cycle_id)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                vuln_id      VARCHAR NOT NULL,
                asset_id     VARCHAR NOT NULL,
                label_name   VARCHAR NOT NULL,
                label_value  DOUBLE NOT NULL,
                labelled_at  TIMESTAMP NOT NULL,
                PRIMARY KEY (vuln_id, asset_id, label_name)
            )
        """)

    # ------------------------------------------------------------------
    #  Write
    # ------------------------------------------------------------------
    def persist_features(
        self,
        rows: list[dict[str, Any]],
        cycle_id: str,
    ) -> int:
        """
        Persist a batch of feature vectors.
        Each row must have keys: vuln_id, asset_id, features (dict).
        Returns number of rows written.
        """
        now = datetime.utcnow().isoformat()
        count = 0
        for row in rows:
            vid = row["vuln_id"]
            aid = row["asset_id"]
            feat_json = json.dumps(row["features"], default=_json_default)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO feature_vectors
                    (vuln_id, asset_id, cycle_id, computed_at, features)
                VALUES (?, ?, ?, ?, ?)
                """,
                [vid, aid, cycle_id, now, feat_json],
            )
            count += 1
        logger.info("Persisted %d feature vectors for cycle %s", count, cycle_id)
        return count

    def persist_labels(self, rows: list[dict[str, Any]]) -> int:
        """
        Store ground-truth labels for supervised training.
        Each row: {vuln_id, asset_id, label_name, label_value}.
        """
        now = datetime.utcnow().isoformat()
        count = 0
        for row in rows:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO labels
                    (vuln_id, asset_id, label_name, label_value, labelled_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [row["vuln_id"], row["asset_id"], row["label_name"],
                 float(row["label_value"]), now],
            )
            count += 1
        return count

    # ------------------------------------------------------------------
    #  Read
    # ------------------------------------------------------------------
    def get_latest_features(
        self,
        vuln_id: str,
        asset_id: str,
    ) -> dict[str, float] | None:
        """Return the latest feature dict for a (vuln, asset) pair."""
        result = self._conn.execute(
            """
            SELECT features FROM feature_vectors
            WHERE vuln_id = ? AND asset_id = ?
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            [vuln_id, asset_id],
        ).fetchone()
        if result is None:
            return None
        return json.loads(result[0])

    def get_training_dataset(
        self,
        label_name: str,
        limit: int = 100_000,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Join features ↔ labels to produce (X, y, feature_names).
        Returns numpy arrays ready for sklearn/xgboost.
        """
        rows = self._conn.execute(
            """
            SELECT fv.features, l.label_value
            FROM feature_vectors fv
            JOIN labels l
              ON fv.vuln_id  = l.vuln_id
             AND fv.asset_id = l.asset_id
            WHERE l.label_name = ?
            ORDER BY fv.computed_at DESC
            LIMIT ?
            """,
            [label_name, limit],
        ).fetchall()

        if not rows:
            return np.empty((0, 0)), np.empty((0,)), []

        feature_dicts = [json.loads(r[0]) for r in rows]
        labels = [r[1] for r in rows]

        # Align columns across all rows
        all_keys = sorted({k for d in feature_dicts for k in d})
        X = np.array(
            [[d.get(k, 0.0) for k in all_keys] for d in feature_dicts],
            dtype=np.float32,
        )
        y = np.array(labels, dtype=np.float32)
        return X, y, all_keys

    # ------------------------------------------------------------------
    #  Maintenance
    # ------------------------------------------------------------------
    def prune_old_cycles(self, keep_last_n: int = 30) -> int:
        """Delete feature vectors older than the last N cycles."""
        cycles = self._conn.execute(
            "SELECT DISTINCT cycle_id FROM feature_vectors ORDER BY cycle_id DESC"
        ).fetchall()
        if len(cycles) <= keep_last_n:
            return 0
        old = [c[0] for c in cycles[keep_last_n:]]
        placeholders = ",".join("?" for _ in old)
        result = self._conn.execute(
            f"DELETE FROM feature_vectors WHERE cycle_id IN ({placeholders})",
            old,
        )
        logger.info("Pruned %d old feature cycles", len(old))
        return len(old)

    def close(self) -> None:
        self._conn.close()


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
