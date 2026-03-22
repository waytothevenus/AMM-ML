"""AlienVault OTX STIX file loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from layer0_knowledge_graph.connectors import BaseConnector
from layer0_knowledge_graph.ontologies import NodeType, RelationType

logger = logging.getLogger(__name__)


class OTXFileLoader(BaseConnector):
    """Loads AlienVault OTX threat intelligence from STIX JSON bundles."""

    SOURCE_NAME = "otx"

    def supported_extensions(self) -> list[str]:
        return [".json"]

    def load_file(self, file_path: Path) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # STIX 2.x bundle format
        objects = data.get("objects", data if isinstance(data, list) else [])
        count = 0

        for obj in objects:
            obj_type = obj.get("type", "")
            obj_id = obj.get("id", "")

            if obj_type == "indicator":
                self._handle_indicator(obj, obj_id)
                count += 1
            elif obj_type == "threat-actor":
                self._handle_threat_actor(obj, obj_id)
                count += 1
            elif obj_type == "campaign":
                self._handle_campaign(obj, obj_id)
                count += 1
            elif obj_type == "attack-pattern":
                self._handle_ttp(obj, obj_id)
                count += 1
            elif obj_type == "relationship":
                self._handle_relationship(obj)
                count += 1

        logger.info("Loaded %d STIX objects from OTX", count)
        return count

    def _handle_indicator(self, obj: dict, obj_id: str) -> None:
        self.graph.add_node(obj_id, NodeType.INDICATOR, {
            "name": obj.get("name", ""),
            "pattern": obj.get("pattern", ""),
            "created": obj.get("created", ""),
            "indicator_type": "stix_indicator",
            "source": self.SOURCE_NAME,
        })
        # Extract CVE references from labels or external_references
        for ref in obj.get("external_references", []):
            ext_id = ref.get("external_id", "")
            if ext_id.startswith("CVE-") and self.graph.has_node(ext_id):
                self.graph.add_edge(ext_id, obj_id, RelationType.VULN_REFERENCES)

    def _handle_threat_actor(self, obj: dict, obj_id: str) -> None:
        self.graph.add_node(obj_id, NodeType.THREAT_ACTOR, {
            "name": obj.get("name", ""),
            "description": obj.get("description", "")[:500],
            "source": self.SOURCE_NAME,
        })

    def _handle_campaign(self, obj: dict, obj_id: str) -> None:
        self.graph.add_node(obj_id, NodeType.CAMPAIGN, {
            "name": obj.get("name", ""),
            "description": obj.get("description", "")[:500],
            "first_seen": obj.get("first_seen", ""),
            "source": self.SOURCE_NAME,
        })

    def _handle_ttp(self, obj: dict, obj_id: str) -> None:
        mitre_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                mitre_id = ref.get("external_id", "")
                break
        self.graph.add_node(obj_id, NodeType.TTP, {
            "name": obj.get("name", ""),
            "mitre_id": mitre_id,
            "source": self.SOURCE_NAME,
        })

    def _handle_relationship(self, obj: dict) -> None:
        src = obj.get("source_ref", "")
        tgt = obj.get("target_ref", "")
        rel_type = obj.get("relationship_type", "")

        if not (self.graph.has_node(src) and self.graph.has_node(tgt)):
            return

        # Map STIX relationship types to our ontology
        mapping = {
            "uses": RelationType.THREAT_USES_TTP,
            "targets": RelationType.THREAT_TARGETS_VULN,
            "attributed-to": RelationType.CAMPAIGN_ATTRIBUTED_TO,
            "indicates": RelationType.INDICATOR_INDICATES,
        }
        relation = mapping.get(rel_type)
        if relation:
            self.graph.add_edge(src, tgt, relation)
