"""
Ontology definitions for the Knowledge Graph.

Three core ontologies:
  - Vulnerability Ontology  (CVE, CWE, CVSS, CPE relationships)
  - Asset Ontology          (Host, Service, Network zone relationships)
  - Threat Intel Ontology   (TTP, Campaign, Actor, Indicator relationships)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

class RelationType(str, Enum):
    # Vulnerability ontology
    VULN_AFFECTS_CPE = "AFFECTS"
    VULN_HAS_CWE = "HAS_CWE"
    VULN_EXPLOITED_BY = "EXPLOITED_BY"
    VULN_REFERENCES = "REFERENCES"

    # Asset ontology
    ASSET_RUNS_CPE = "RUNS"
    ASSET_CONNECTS_TO = "CONNECTS_TO"
    ASSET_IN_ZONE = "IN_ZONE"
    ASSET_BELONGS_TO_BU = "BELONGS_TO"
    ASSET_HOSTS_SERVICE = "HOSTS_SERVICE"

    # Cross-ontology
    VULN_PRESENT_ON = "PRESENT_ON"  # vulnerability → asset

    # Threat intel ontology
    THREAT_TARGETS_VULN = "TARGETS"
    THREAT_USES_TTP = "USES_TTP"
    CAMPAIGN_ATTRIBUTED_TO = "ATTRIBUTED_TO"
    INDICATOR_INDICATES = "INDICATES"


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    VULNERABILITY = "Vulnerability"
    ASSET = "Asset"
    CWE = "CWE"
    CPE = "CPE"
    NETWORK_ZONE = "NetworkZone"
    BUSINESS_UNIT = "BusinessUnit"
    SERVICE = "Service"
    THREAT_ACTOR = "ThreatActor"
    CAMPAIGN = "Campaign"
    TTP = "TTP"
    INDICATOR = "Indicator"


# ---------------------------------------------------------------------------
# Ontology schemas (define allowed edges)
# ---------------------------------------------------------------------------

@dataclass
class EdgeSchema:
    source_type: NodeType
    relation: RelationType
    target_type: NodeType


VULNERABILITY_ONTOLOGY: list[EdgeSchema] = [
    EdgeSchema(NodeType.VULNERABILITY, RelationType.VULN_AFFECTS_CPE, NodeType.CPE),
    EdgeSchema(NodeType.VULNERABILITY, RelationType.VULN_HAS_CWE, NodeType.CWE),
    EdgeSchema(NodeType.VULNERABILITY, RelationType.VULN_EXPLOITED_BY, NodeType.THREAT_ACTOR),
    EdgeSchema(NodeType.VULNERABILITY, RelationType.VULN_REFERENCES, NodeType.INDICATOR),
]

ASSET_ONTOLOGY: list[EdgeSchema] = [
    EdgeSchema(NodeType.ASSET, RelationType.ASSET_RUNS_CPE, NodeType.CPE),
    EdgeSchema(NodeType.ASSET, RelationType.ASSET_CONNECTS_TO, NodeType.ASSET),
    EdgeSchema(NodeType.ASSET, RelationType.ASSET_IN_ZONE, NodeType.NETWORK_ZONE),
    EdgeSchema(NodeType.ASSET, RelationType.ASSET_BELONGS_TO_BU, NodeType.BUSINESS_UNIT),
    EdgeSchema(NodeType.ASSET, RelationType.ASSET_HOSTS_SERVICE, NodeType.SERVICE),
]

THREAT_INTEL_ONTOLOGY: list[EdgeSchema] = [
    EdgeSchema(NodeType.THREAT_ACTOR, RelationType.THREAT_TARGETS_VULN, NodeType.VULNERABILITY),
    EdgeSchema(NodeType.THREAT_ACTOR, RelationType.THREAT_USES_TTP, NodeType.TTP),
    EdgeSchema(NodeType.CAMPAIGN, RelationType.CAMPAIGN_ATTRIBUTED_TO, NodeType.THREAT_ACTOR),
    EdgeSchema(NodeType.INDICATOR, RelationType.INDICATOR_INDICATES, NodeType.CAMPAIGN),
]

CROSS_ONTOLOGY: list[EdgeSchema] = [
    EdgeSchema(NodeType.VULNERABILITY, RelationType.VULN_PRESENT_ON, NodeType.ASSET),
]

ALL_SCHEMAS = VULNERABILITY_ONTOLOGY + ASSET_ONTOLOGY + THREAT_INTEL_ONTOLOGY + CROSS_ONTOLOGY
