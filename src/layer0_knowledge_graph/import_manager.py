"""
Import Manager – routes incoming data files to the correct connector.

Usage:
    manager = ImportManager(graph_store)
    report  = manager.import_file(Path("data/incoming/nvd_2024.json"))
    report  = manager.import_directory(Path("data/incoming/"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.connectors import BaseConnector
from layer0_knowledge_graph.connectors.nvd_file_loader import NVDFileLoader
from layer0_knowledge_graph.connectors.exploitdb_file_loader import ExploitDBFileLoader
from layer0_knowledge_graph.connectors.otx_file_loader import OTXFileLoader
from layer0_knowledge_graph.connectors.cmdb_file_loader import CMDBFileLoader
from layer0_knowledge_graph.connectors.cisa_kev_loader import CISAKEVFileLoader
from layer0_knowledge_graph.connectors.vendor_advisory_loader import VendorAdvisoryLoader
from layer0_knowledge_graph.connectors.network_scan_loader import NetworkScanLoader
from layer0_knowledge_graph.connectors.siem_alert_loader import SIEMAlertLoader
from layer0_knowledge_graph.entity_resolution import EntityResolver

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    file: Path
    connector: str
    nodes_added: int
    edges_added: int
    success: bool
    error: str | None = None


@dataclass
class BatchImportReport:
    results: list[ImportResult] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    failures: int = 0

    def summarize(self) -> str:
        ok = len(self.results) - self.failures
        return (
            f"Imported {ok}/{len(self.results)} files  "
            f"(+{self.total_nodes} nodes, +{self.total_edges} edges, "
            f"{self.failures} failures)"
        )


# ---------------------------------------------------------------------------
#  File-name → connector routing table
# ---------------------------------------------------------------------------
# Each tuple: (substring-in-filename, connector-class)
_ROUTING_TABLE: list[tuple[str, type[BaseConnector]]] = [
    ("nvd",              NVDFileLoader),
    ("cve-",             NVDFileLoader),
    ("exploitdb",        ExploitDBFileLoader),
    ("exploit_db",       ExploitDBFileLoader),
    ("otx",              OTXFileLoader),
    ("stix",             OTXFileLoader),
    ("cmdb",             CMDBFileLoader),
    ("asset_inventory",  CMDBFileLoader),
    ("kev",              CISAKEVFileLoader),
    ("cisa",             CISAKEVFileLoader),
    ("vendor_advisory",  VendorAdvisoryLoader),
    ("advisory",         VendorAdvisoryLoader),
    ("network_scan",     NetworkScanLoader),
    ("scan_result",      NetworkScanLoader),
    ("siem",             SIEMAlertLoader),
    ("alert",            SIEMAlertLoader),
]


class ImportManager:
    """
    Central dispatcher that:
    1. Routes incoming data files to the correct connector.
    2. Invokes entity resolution after each import batch.
    3. Logs results.
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self.graph = graph_store
        self._connector_cache: dict[type[BaseConnector], BaseConnector] = {}

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def import_file(self, path: Path, *, resolve: bool = True) -> ImportResult:
        """Import a single file. Optionally run entity resolution after."""
        connector = self._route(path)
        if connector is None:
            err = f"No connector matched for file: {path.name}"
            logger.warning(err)
            return ImportResult(
                file=path, connector="unknown",
                nodes_added=0, edges_added=0, success=False, error=err,
            )

        n0 = self.graph.graph.number_of_nodes()
        e0 = self.graph.graph.number_of_edges()

        try:
            connector.ingest(path)
        except Exception as exc:
            logger.error("Import failed for %s: %s", path.name, exc, exc_info=True)
            return ImportResult(
                file=path, connector=connector.SOURCE_NAME,
                nodes_added=0, edges_added=0, success=False, error=str(exc),
            )

        dn = self.graph.graph.number_of_nodes() - n0
        de = self.graph.graph.number_of_edges() - e0

        if resolve:
            self._run_entity_resolution()

        return ImportResult(
            file=path, connector=connector.SOURCE_NAME,
            nodes_added=dn, edges_added=de, success=True,
        )

    def import_directory(self, directory: Path, *, resolve: bool = True) -> BatchImportReport:
        """Import all supported files in a directory (non-recursive)."""
        report = BatchImportReport()

        files = sorted(
            f for f in directory.iterdir()
            if f.is_file() and self._route(f) is not None
        )
        logger.info("Found %d importable files in %s", len(files), directory)

        for fp in files:
            result = self.import_file(fp, resolve=False)
            report.results.append(result)
            report.total_nodes += result.nodes_added
            report.total_edges += result.edges_added
            if not result.success:
                report.failures += 1

        if resolve:
            self._run_entity_resolution()

        logger.info(report.summarize())
        return report

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _route(self, path: Path) -> BaseConnector | None:
        name_lower = path.name.lower()
        for substring, cls in _ROUTING_TABLE:
            if substring in name_lower:
                return self._get_connector(cls)
        return None

    def _get_connector(self, cls: type[BaseConnector]) -> BaseConnector:
        if cls not in self._connector_cache:
            self._connector_cache[cls] = cls(self.graph)
        return self._connector_cache[cls]

    def _run_entity_resolution(self) -> None:
        resolver = EntityResolver(self.graph)
        stats = resolver.resolve_all()
        logger.info(
            "Entity resolution: +%d edges from CPE matching",
            stats.get("vuln_asset_links", 0),
        )
