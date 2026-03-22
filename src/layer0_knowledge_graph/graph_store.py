"""
Graph storage backend using NetworkX + SQLite.

Provides the storage layer for the Knowledge Graph.  Two backends:
  - "networkx" : in-memory NetworkX DiGraph, persisted to GraphML on disk
  - "neo4j"    : local Neo4j Community Edition (not implemented yet)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from config import get_config, resolve_path
from layer0_knowledge_graph.ontologies import NodeType, RelationType

logger = logging.getLogger(__name__)


class GraphStore:
    """NetworkX-backed knowledge graph with SQLite metadata."""

    def __init__(self) -> None:
        cfg = get_config()
        self._graph = nx.DiGraph()
        self._graph_path = resolve_path(cfg.data.graph_dir) / "knowledge_graph.graphml"
        self._db_path = resolve_path(cfg.data.knowledge_graph_db)
        self._init_metadata_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_metadata_db(self) -> None:
        """Create the metadata SQLite tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS import_log (
                    batch_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    import_ts TEXT NOT NULL,
                    file_path TEXT,
                    record_count INTEGER DEFAULT 0,
                    checksum TEXT,
                    data_date TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_versions (
                    node_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    updated_ts TEXT NOT NULL,
                    PRIMARY KEY (node_id, version)
                )
            """)
            conn.commit()

    def load(self) -> None:
        """Load the persisted graph from disk."""
        if self._graph_path.exists():
            self._graph = nx.read_graphml(str(self._graph_path))
            logger.info("Loaded graph with %d nodes, %d edges",
                        self._graph.number_of_nodes(), self._graph.number_of_edges())
        else:
            logger.info("No persisted graph found – starting fresh")
            self._graph = nx.DiGraph()

    def save(self) -> None:
        """Persist the graph to disk."""
        self._graph_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self._graph, str(self._graph_path))
        logger.info("Saved graph (%d nodes, %d edges)",
                     self._graph.number_of_nodes(), self._graph.number_of_edges())

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, node_type: NodeType,
                 attrs: Optional[dict[str, Any]] = None) -> None:
        safe_attrs = self._sanitize_attrs(attrs or {})
        safe_attrs["node_type"] = node_type.value
        safe_attrs["_updated"] = datetime.utcnow().isoformat()
        self._graph.add_node(node_id, **safe_attrs)

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        if node_id in self._graph:
            return dict(self._graph.nodes[node_id])
        return None

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def get_nodes_by_type(self, node_type: NodeType) -> list[str]:
        return [n for n, d in self._graph.nodes(data=True)
                if d.get("node_type") == node_type.value]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, source_id: str, target_id: str,
                 relation: RelationType,
                 attrs: Optional[dict[str, Any]] = None) -> None:
        safe_attrs = self._sanitize_attrs(attrs or {})
        safe_attrs["relation"] = relation.value
        self._graph.add_edge(source_id, target_id, **safe_attrs)

    def get_neighbors(self, node_id: str,
                      relation: Optional[RelationType] = None) -> list[str]:
        if node_id not in self._graph:
            return []
        neighbors = []
        for _, target, data in self._graph.out_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation.value:
                neighbors.append(target)
        return neighbors

    def get_predecessors(self, node_id: str,
                         relation: Optional[RelationType] = None) -> list[str]:
        if node_id not in self._graph:
            return []
        preds = []
        for source, _, data in self._graph.in_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation.value:
                preds.append(source)
        return preds

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_vulns_for_asset(self, asset_id: str) -> list[str]:
        """Get all CVE IDs affecting an asset (via CPE matching)."""
        return self.get_predecessors(asset_id, RelationType.VULN_PRESENT_ON)

    def get_assets_for_vuln(self, cve_id: str) -> list[str]:
        """Get all assets affected by a vulnerability."""
        return self.get_neighbors(cve_id, RelationType.VULN_PRESENT_ON)

    def get_connected_assets(self, asset_id: str) -> list[str]:
        """Get assets directly connected to this asset."""
        return self.get_neighbors(asset_id, RelationType.ASSET_CONNECTS_TO)

    def get_attack_paths(self, source_id: str, target_id: str,
                         max_length: int = 5) -> list[list[str]]:
        """Find all simple paths between two nodes up to max_length."""
        try:
            return list(nx.all_simple_paths(
                self._graph, source_id, target_id, cutoff=max_length
            ))
        except nx.NetworkXError:
            return []

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying NetworkX graph."""
        return self._graph

    # ------------------------------------------------------------------
    # Metadata / import log
    # ------------------------------------------------------------------

    def log_import(self, batch_id: str, source: str, file_path: str,
                   record_count: int, checksum: str,
                   data_date: Optional[datetime] = None) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO import_log
                   (batch_id, source, import_ts, file_path, record_count, checksum, data_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (batch_id, source, datetime.utcnow().isoformat(), file_path,
                 record_count, checksum,
                 data_date.isoformat() if data_date else None)
            )
            conn.commit()

    def get_latest_import(self, source: str) -> Optional[dict[str, Any]]:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM import_log WHERE source=? ORDER BY import_ts DESC LIMIT 1",
                (source,)
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Temporal versioning
    # ------------------------------------------------------------------

    def save_node_version(self, node_id: str, data: dict[str, Any]) -> None:
        """Store a new version snapshot of a node."""
        with sqlite3.connect(str(self._db_path)) as conn:
            # Get next version number
            row = conn.execute(
                "SELECT MAX(version) FROM node_versions WHERE node_id=?",
                (node_id,)
            ).fetchone()
            next_ver = (row[0] or 0) + 1
            conn.execute(
                "INSERT INTO node_versions (node_id, version, data, updated_ts) VALUES (?,?,?,?)",
                (node_id, next_ver, json.dumps(data, default=str),
                 datetime.utcnow().isoformat())
            )
            conn.commit()

    def get_node_history(self, node_id: str) -> list[dict[str, Any]]:
        """Retrieve all versions of a node."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM node_versions WHERE node_id=? ORDER BY version",
                (node_id,)
            ).fetchall()
            results = []
            for r in rows:
                entry = dict(r)
                entry["data"] = json.loads(entry["data"])
                results.append(entry)
            return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
        """Ensure all attribute values are GraphML-serializable (str, int, float)."""
        safe = {}
        for k, v in attrs.items():
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif isinstance(v, (list, dict)):
                safe[k] = json.dumps(v, default=str)
            elif v is None:
                safe[k] = ""
            else:
                safe[k] = str(v)
        return safe
