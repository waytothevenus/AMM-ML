"""
Abstract base class for all data connectors (file-based loaders).

Every connector implements the same interface regardless of whether data comes
from a file import (air-gapped) or a live API (future online mode).
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from layer0_knowledge_graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Base class for all data source connectors."""

    # Subclasses must set this
    SOURCE_NAME: str = "unknown"

    def __init__(self, graph_store: GraphStore) -> None:
        self.graph = graph_store

    @abstractmethod
    def load_file(self, file_path: Path) -> int:
        """
        Parse *file_path* and insert entities/relationships into the graph.
        Returns the number of records ingested.
        """

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions this connector handles, e.g. ['.json']."""

    def ingest(self, file_path: Path) -> None:
        """Full ingest pipeline: load → log."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {path}")

        logger.info("[%s] Ingesting %s", self.SOURCE_NAME, path.name)
        checksum = self._sha256(path)
        record_count = self.load_file(path)
        batch_id = f"{self.SOURCE_NAME}_{uuid4().hex[:8]}"

        self.graph.log_import(
            batch_id=batch_id,
            source=self.SOURCE_NAME,
            file_path=str(path),
            record_count=record_count,
            checksum=checksum,
            data_date=datetime.utcnow(),
        )
        logger.info("[%s] Ingested %d records (batch=%s)",
                     self.SOURCE_NAME, record_count, batch_id)

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
