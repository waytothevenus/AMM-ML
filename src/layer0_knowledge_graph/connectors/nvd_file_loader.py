"""NVD JSON feed file loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from layer0_knowledge_graph.connectors import BaseConnector
from layer0_knowledge_graph.ontologies import NodeType, RelationType

logger = logging.getLogger(__name__)


class NVDFileLoader(BaseConnector):
    """Loads NVD CVE data from JSON feed files (NVD API 2.0 format)."""

    SOURCE_NAME = "nvd"

    def supported_extensions(self) -> list[str]:
        return [".json"]

    def load_file(self, file_path: Path) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both NVD 2.0 format {"vulnerabilities": [...]} and flat list
        vulns = data.get("vulnerabilities", data if isinstance(data, list) else [])
        count = 0

        for entry in vulns:
            cve_data = entry.get("cve", entry)
            cve_id = cve_data.get("id", cve_data.get("cve_id", ""))
            if not cve_id:
                continue

            # Extract CVSS score
            cvss_score = 0.0
            cvss_vector = ""
            metrics = cve_data.get("metrics", {})
            for metric_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                metric_list = metrics.get(metric_key, [])
                if metric_list:
                    cvss_data = metric_list[0].get("cvssData", {})
                    cvss_score = cvss_data.get("baseScore", 0.0)
                    cvss_vector = cvss_data.get("vectorString", "")
                    break

            # Extract description
            descriptions = cve_data.get("descriptions", [])
            desc = ""
            for d in descriptions:
                if d.get("lang", "en") == "en":
                    desc = d.get("value", "")
                    break

            # Extract CWE
            cwe_ids = []
            for weakness in cve_data.get("weaknesses", []):
                for wd in weakness.get("description", []):
                    val = wd.get("value", "")
                    if val.startswith("CWE-"):
                        cwe_ids.append(val)

            # Extract CPE matches
            cpe_list = []
            for config in cve_data.get("configurations", []):
                for node in config.get("nodes", []):
                    for match in node.get("cpeMatch", []):
                        cpe_uri = match.get("criteria", "")
                        if cpe_uri:
                            cpe_list.append(cpe_uri)

            # Published / modified dates
            published = cve_data.get("published", "")
            modified = cve_data.get("lastModified", "")

            # Add vulnerability node
            self.graph.add_node(cve_id, NodeType.VULNERABILITY, {
                "description": desc[:2000],  # cap length for GraphML
                "cvss_base_score": cvss_score,
                "cvss_vector": cvss_vector,
                "published_date": published,
                "modified_date": modified,
                "cwe_ids": ";".join(cwe_ids) if cwe_ids else "",
                "source": self.SOURCE_NAME,
            })

            # Add CWE nodes and edges
            for cwe_id in cwe_ids:
                if not self.graph.has_node(cwe_id):
                    self.graph.add_node(cwe_id, NodeType.CWE, {"name": cwe_id})
                self.graph.add_edge(cve_id, cwe_id, RelationType.VULN_HAS_CWE)

            # Add CPE nodes and edges
            for cpe in cpe_list:
                if not self.graph.has_node(cpe):
                    self.graph.add_node(cpe, NodeType.CPE, {"uri": cpe})
                self.graph.add_edge(cve_id, cpe, RelationType.VULN_AFFECTS_CPE)

            count += 1

        logger.info("Loaded %d CVEs from NVD feed", count)
        return count
