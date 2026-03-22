"""Network Scanner output loader (Nmap XML / generic JSON)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from layer0_knowledge_graph.connectors import BaseConnector
from layer0_knowledge_graph.ontologies import NodeType, RelationType

logger = logging.getLogger(__name__)


class NetworkScanLoader(BaseConnector):
    """Loads network scan results (JSON format, topology + services)."""

    SOURCE_NAME = "network_scan"

    def supported_extensions(self) -> list[str]:
        return [".json"]

    def load_file(self, file_path: Path) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        hosts = data if isinstance(data, list) else data.get("hosts", [])
        count = 0

        for host in hosts:
            host_id = host.get("asset_id", host.get("ip", ""))
            if not host_id:
                continue

            # Update existing asset or create new one
            existing = self.graph.get_node(host_id)
            attrs = dict(existing) if existing else {}
            attrs.pop("node_type", None)
            attrs.update({
                "ip_address": host.get("ip", attrs.get("ip_address", "")),
                "hostname": host.get("hostname", attrs.get("hostname", "")),
                "os_detected": host.get("os", ""),
                "last_scan_date": host.get("scan_date", ""),
                "source": self.SOURCE_NAME,
            })
            self.graph.add_node(host_id, NodeType.ASSET, attrs)

            # Add services
            for svc in host.get("services", []):
                port = svc.get("port", "")
                protocol = svc.get("protocol", "tcp")
                service_name = svc.get("name", "unknown")
                svc_id = f"{host_id}:{port}/{protocol}"

                self.graph.add_node(svc_id, NodeType.SERVICE, {
                    "port": str(port),
                    "protocol": protocol,
                    "name": service_name,
                    "version": svc.get("version", ""),
                    "cpe": svc.get("cpe", ""),
                })
                self.graph.add_edge(host_id, svc_id, RelationType.ASSET_HOSTS_SERVICE)

                # Link service CPE
                cpe = svc.get("cpe", "")
                if cpe:
                    if not self.graph.has_node(cpe):
                        self.graph.add_node(cpe, NodeType.CPE, {"uri": cpe})
                    self.graph.add_edge(host_id, cpe, RelationType.ASSET_RUNS_CPE)

            # Network connections (adjacency)
            for neighbor_id in host.get("connected_to", []):
                self.graph.add_edge(
                    host_id, neighbor_id, RelationType.ASSET_CONNECTS_TO
                )

            count += 1

        logger.info("Loaded %d hosts from network scan", count)
        return count
