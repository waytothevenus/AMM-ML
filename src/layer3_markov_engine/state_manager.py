"""
State Manager – manages the current Markov state for every (vuln, asset) pair.

Responsibilities:
  - Store / retrieve the per-pair state distribution π(t)
  - Track time-in-state
  - Handle state transitions
  - Persist to SQLite
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from config import get_config
from models import MarkovState, RiskState

logger = logging.getLogger(__name__)

NUM_STATES = len(RiskState)


class StateManager:
    """Persistent Markov state storage for all (vuln, asset) pairs."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_config()
        db = db_path or Path(cfg.data.db_dir) / "markov_states.sqlite"
        Path(str(db)).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db))
        self._conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS markov_states (
                pair_key     TEXT PRIMARY KEY,
                vuln_id      TEXT NOT NULL,
                asset_id     TEXT NOT NULL,
                distribution TEXT NOT NULL,
                entropy      REAL NOT NULL,
                absorption_time REAL,
                time_in_current_state REAL NOT NULL DEFAULT 0,
                dominant_state INTEGER NOT NULL DEFAULT 0,
                updated_at   TEXT NOT NULL,
                cycle_id     TEXT
            );
            CREATE TABLE IF NOT EXISTS state_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_key     TEXT NOT NULL,
                distribution TEXT NOT NULL,
                entropy      REAL NOT NULL,
                absorption_time REAL,
                cycle_id     TEXT,
                recorded_at  TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    #  Read
    # ------------------------------------------------------------------
    def get_state(self, vuln_id: str, asset_id: str) -> MarkovState | None:
        pair_key = f"{vuln_id}::{asset_id}"
        row = self._conn.execute(
            "SELECT * FROM markov_states WHERE pair_key=?", [pair_key]
        ).fetchone()
        if row is None:
            return None
        return self._row_to_state(row)

    def get_all_states(self) -> dict[str, MarkovState]:
        """Return {pair_key: MarkovState} for all pairs."""
        rows = self._conn.execute("SELECT * FROM markov_states").fetchall()
        return {row["pair_key"]: self._row_to_state(row) for row in rows}

    def get_history(self, vuln_id: str, asset_id: str, limit: int = 10) -> list[MarkovState]:
        pair_key = f"{vuln_id}::{asset_id}"
        rows = self._conn.execute(
            "SELECT * FROM state_history WHERE pair_key=? ORDER BY recorded_at DESC LIMIT ?",
            [pair_key, limit],
        ).fetchall()
        return [self._history_row_to_state(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    #  Write
    # ------------------------------------------------------------------
    def update_state(
        self,
        vuln_id: str,
        asset_id: str,
        distribution: np.ndarray,
        entropy: float,
        absorption_time: float | None,
        cycle_id: str = "",
    ) -> None:
        pair_key = f"{vuln_id}::{asset_id}"
        now = datetime.utcnow().isoformat()
        dist_json = json.dumps(distribution.tolist())
        dominant = int(np.argmax(distribution))

        # Check previous dominant state for time tracking
        existing = self._conn.execute(
            "SELECT dominant_state, time_in_current_state FROM markov_states WHERE pair_key=?",
            [pair_key],
        ).fetchone()

        if existing and existing["dominant_state"] == dominant:
            time_in_state = existing["time_in_current_state"] + 1.0
        else:
            time_in_state = 0.0

        self._conn.execute(
            """
            INSERT OR REPLACE INTO markov_states
                (pair_key, vuln_id, asset_id, distribution, entropy,
                 absorption_time, time_in_current_state, dominant_state,
                 updated_at, cycle_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [pair_key, vuln_id, asset_id, dist_json, entropy,
             absorption_time, time_in_state, dominant, now, cycle_id],
        )

        # Append to history
        self._conn.execute(
            """
            INSERT INTO state_history
                (pair_key, distribution, entropy, absorption_time, cycle_id, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [pair_key, dist_json, entropy, absorption_time, cycle_id, now],
        )
        self._conn.commit()

    def initialize_state(
        self,
        vuln_id: str,
        asset_id: str,
        initial_state: int = 0,
    ) -> MarkovState:
        """Create an initial deterministic state (all mass on one state)."""
        dist = np.zeros(NUM_STATES)
        dist[initial_state] = 1.0
        self.update_state(vuln_id, asset_id, dist, entropy=0.0, absorption_time=None)
        from models import RiskState
        return MarkovState(
            vuln_id=vuln_id,
            asset_id=asset_id,
            distribution=dist.tolist(),
            entropy=0.0,
            absorption_time=None,
            current_state=RiskState(initial_state),
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _row_to_state(self, row) -> MarkovState:
        dist = json.loads(row["distribution"])
        return MarkovState(
            vuln_id=row["vuln_id"],
            asset_id=row["asset_id"],
            distribution=dist,
            entropy=row["entropy"],
            absorption_time=row["absorption_time"],
            time_in_current_state=row["time_in_current_state"],
        )

    def _history_row_to_state(self, row) -> MarkovState:
        dist = json.loads(row["distribution"])
        pair_key = row["pair_key"]
        parts = pair_key.split("::", 1) if pair_key else ["", ""]
        vid = parts[0] if len(parts) == 2 else ""
        aid = parts[1] if len(parts) == 2 else ""
        return MarkovState(
            vuln_id=vid,
            asset_id=aid,
            distribution=dist,
            entropy=row["entropy"],
            absorption_time=row["absorption_time"],
        )

    def close(self) -> None:
        self._conn.close()
