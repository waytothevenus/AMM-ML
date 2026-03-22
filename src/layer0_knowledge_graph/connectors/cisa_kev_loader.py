"""CISA Known Exploited Vulnerabilities (KEV) JSON loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from layer0_knowledge_graph.connectors import BaseConnector
from layer0_knowledge_graph.ontologies import NodeType

logger = logging.getLogger(__name__)


class CISAKEVFileLoader(BaseConnector):
    """Loads CISA KEV catalog JSON."""

    SOURCE_NAME = "cisa_kev"

    def supported_extensions(self) -> list[str]:
        return [".json"]

    def load_file(self, file_path: Path) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vulns = data.get("vulnerabilities", data if isinstance(data, list) else [])
        count = 0

        for entry in vulns:
            cve_id = entry.get("cveID", "")
            if not cve_id:
                continue

            # Update or create the vulnerability node with KEV flag
            existing = self.graph.get_node(cve_id)
            attrs = dict(existing) if existing else {}
            attrs.update({
                "is_in_kev": "true",
                "kev_vendor": entry.get("vendorProject", ""),
                "kev_product": entry.get("product", ""),
                "kev_date_added": entry.get("dateAdded", ""),
                "kev_due_date": entry.get("dueDate", ""),
                "kev_action": entry.get("requiredAction", ""),
                "kev_ransomware": entry.get("knownRansomwareCampaignUse", "Unknown"),
            })
            # Remove node_type if present to avoid duplication
            attrs.pop("node_type", None)
            self.graph.add_node(cve_id, NodeType.VULNERABILITY, attrs)
            count += 1

        logger.info("Loaded %d KEV entries", count)
        return count
