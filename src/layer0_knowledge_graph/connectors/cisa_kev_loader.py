"""CISA Known Exploited Vulnerabilities (KEV) JSON/CSV loader."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from layer0_knowledge_graph.connectors import BaseConnector
from layer0_knowledge_graph.ontologies import NodeType

logger = logging.getLogger(__name__)


class CISAKEVFileLoader(BaseConnector):
    """Loads CISA KEV catalog in JSON or CSV format."""

    SOURCE_NAME = "cisa_kev"

    # CSV column → internal attribute mapping
    _CSV_FIELD_MAP = {
        "cveID": "cveID",
        "cve_id": "cveID",
        "vendorProject": "vendorProject",
        "vendor_project": "vendorProject",
        "product": "product",
        "dateAdded": "dateAdded",
        "date_added": "dateAdded",
        "dueDate": "dueDate",
        "due_date": "dueDate",
        "requiredAction": "requiredAction",
        "required_action": "requiredAction",
        "knownRansomwareCampaignUse": "knownRansomwareCampaignUse",
        "known_ransomware_campaign_use": "knownRansomwareCampaignUse",
    }

    def supported_extensions(self) -> list[str]:
        return [".json", ".csv"]

    def load_file(self, file_path: Path) -> int:
        if file_path.suffix.lower() == ".csv":
            return self._load_csv(file_path)
        return self._load_json(file_path)

    def _load_json(self, file_path: Path) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vulns = data.get("vulnerabilities", data if isinstance(data, list) else [])
        return self._ingest_entries(vulns)

    def _load_csv(self, file_path: Path) -> int:
        entries = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalise column names to the JSON-style keys
                entry = {}
                for col, val in row.items():
                    mapped = self._CSV_FIELD_MAP.get(col, col)
                    entry[mapped] = val
                entries.append(entry)
        return self._ingest_entries(entries)

    def _ingest_entries(self, vulns: list[dict]) -> int:
        count = 0
        for entry in vulns:
            cve_id = entry.get("cveID", "")
            if not cve_id:
                continue

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
            attrs.pop("node_type", None)
            self.graph.add_node(cve_id, NodeType.VULNERABILITY, attrs)
            count += 1

        logger.info("Loaded %d KEV entries", count)
        return count
