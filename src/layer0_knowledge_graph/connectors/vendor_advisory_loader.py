"""Vendor Security Advisory file loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from layer0_knowledge_graph.connectors import BaseConnector
from layer0_knowledge_graph.ontologies import NodeType, RelationType

logger = logging.getLogger(__name__)


class VendorAdvisoryLoader(BaseConnector):
    """Loads vendor security advisories (JSON format)."""

    SOURCE_NAME = "vendor_advisories"

    def supported_extensions(self) -> list[str]:
        return [".json"]

    def load_file(self, file_path: Path) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        advisories = data if isinstance(data, list) else data.get("advisories", [])
        count = 0

        for adv in advisories:
            adv_id = adv.get("advisory_id", adv.get("id", ""))
            if not adv_id:
                continue

            self.graph.add_node(adv_id, NodeType.INDICATOR, {
                "indicator_type": "vendor_advisory",
                "vendor": adv.get("vendor", ""),
                "title": adv.get("title", ""),
                "severity": adv.get("severity", ""),
                "published_date": adv.get("published_date", ""),
                "patch_available": str(adv.get("patch_available", False)),
                "source": self.SOURCE_NAME,
            })

            # Link to CVEs
            for cve_id in adv.get("cve_ids", []):
                if self.graph.has_node(cve_id):
                    self.graph.add_edge(cve_id, adv_id, RelationType.VULN_REFERENCES)
                    # Mark patch availability on the CVE
                    if adv.get("patch_available"):
                        node = self.graph.get_node(cve_id) or {}
                        node.pop("node_type", None)
                        self.graph.add_node(cve_id, NodeType.VULNERABILITY, {
                            **node,
                            "patch_available": "true",
                        })

            count += 1

        logger.info("Loaded %d vendor advisories", count)
        return count
