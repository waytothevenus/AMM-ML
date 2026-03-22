"""SIEM Alert / Log file loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from layer0_knowledge_graph.connectors import BaseConnector
from layer0_knowledge_graph.ontologies import NodeType, RelationType

logger = logging.getLogger(__name__)


class SIEMAlertLoader(BaseConnector):
    """Loads SIEM alerts/incidents (JSON format)."""

    SOURCE_NAME = "siem_alerts"

    def supported_extensions(self) -> list[str]:
        return [".json"]

    def load_file(self, file_path: Path) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        alerts = data if isinstance(data, list) else data.get("alerts", [])
        count = 0

        for alert in alerts:
            alert_id = alert.get("alert_id", alert.get("id", ""))
            if not alert_id:
                continue

            self.graph.add_node(alert_id, NodeType.INDICATOR, {
                "indicator_type": "siem_alert",
                "rule_name": alert.get("rule_name", ""),
                "severity": alert.get("severity", ""),
                "timestamp": alert.get("timestamp", ""),
                "source_ip": alert.get("source_ip", ""),
                "destination_ip": alert.get("destination_ip", ""),
                "source": self.SOURCE_NAME,
            })

            # Link to affected assets
            for asset_id in alert.get("affected_assets", []):
                if self.graph.has_node(asset_id):
                    self.graph.add_edge(
                        alert_id, asset_id, RelationType.INDICATOR_INDICATES
                    )

            # Link to CVEs if referenced
            for cve_id in alert.get("cve_ids", []):
                if self.graph.has_node(cve_id):
                    self.graph.add_edge(
                        cve_id, alert_id, RelationType.VULN_REFERENCES
                    )

            count += 1

        logger.info("Loaded %d SIEM alerts", count)
        return count
