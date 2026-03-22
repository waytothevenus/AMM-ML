"""CMDB / Asset Inventory file loader."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from layer0_knowledge_graph.connectors import BaseConnector
from layer0_knowledge_graph.ontologies import NodeType, RelationType

logger = logging.getLogger(__name__)


class CMDBFileLoader(BaseConnector):
    """Loads asset inventory from CSV or JSON files."""

    SOURCE_NAME = "cmdb"

    def supported_extensions(self) -> list[str]:
        return [".csv", ".json"]

    def load_file(self, file_path: Path) -> int:
        if file_path.suffix == ".json":
            return self._load_json(file_path)
        return self._load_csv(file_path)

    def _load_csv(self, file_path: Path) -> int:
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                count += self._ingest_asset(row)
        logger.info("Loaded %d assets from CMDB CSV", count)
        return count

    def _load_json(self, file_path: Path) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assets = data if isinstance(data, list) else data.get("assets", [])
        count = 0
        for asset in assets:
            count += self._ingest_asset(asset)
        logger.info("Loaded %d assets from CMDB JSON", count)
        return count

    def _ingest_asset(self, record: dict) -> int:
        asset_id = record.get("asset_id", record.get("hostname", ""))
        if not asset_id:
            return 0

        self.graph.add_node(asset_id, NodeType.ASSET, {
            "hostname": record.get("hostname", ""),
            "ip_address": record.get("ip_address", ""),
            "os": record.get("os", ""),
            "criticality": record.get("criticality", "medium"),
            "business_unit": record.get("business_unit", ""),
            "network_zone": record.get("network_zone", "internal"),
            "last_scan_date": record.get("last_scan_date", ""),
            "source": self.SOURCE_NAME,
        })

        # Link to business unit
        bu = record.get("business_unit", "")
        if bu:
            if not self.graph.has_node(bu):
                self.graph.add_node(bu, NodeType.BUSINESS_UNIT, {"name": bu})
            self.graph.add_edge(asset_id, bu, RelationType.ASSET_BELONGS_TO_BU)

        # Link to network zone
        zone = record.get("network_zone", "")
        if zone:
            if not self.graph.has_node(zone):
                self.graph.add_node(zone, NodeType.NETWORK_ZONE, {"name": zone})
            self.graph.add_edge(asset_id, zone, RelationType.ASSET_IN_ZONE)

        # Link to CPEs
        cpes_raw = record.get("cpe_list", record.get("software", ""))
        if isinstance(cpes_raw, str):
            cpes = [c.strip() for c in cpes_raw.split(",") if c.strip()]
        elif isinstance(cpes_raw, list):
            cpes = cpes_raw
        else:
            cpes = []

        for cpe in cpes:
            if not self.graph.has_node(cpe):
                self.graph.add_node(cpe, NodeType.CPE, {"uri": cpe})
            self.graph.add_edge(asset_id, cpe, RelationType.ASSET_RUNS_CPE)

        # Link to connected assets
        connections = record.get("connects_to", "")
        if isinstance(connections, str):
            conn_list = [c.strip() for c in connections.split(",") if c.strip()]
        elif isinstance(connections, list):
            conn_list = connections
        else:
            conn_list = []

        for target_id in conn_list:
            self.graph.add_edge(asset_id, target_id, RelationType.ASSET_CONNECTS_TO)

        return 1
