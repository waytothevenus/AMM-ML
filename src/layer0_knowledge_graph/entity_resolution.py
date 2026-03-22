"""
Entity Resolution – deduplicates entities across multiple data sources.

Resolves CVEs, assets, and CPEs that may be represented differently
by different connectors into a single canonical identity.
"""

from __future__ import annotations

import logging
import re

from layer0_knowledge_graph.graph_store import GraphStore
from layer0_knowledge_graph.ontologies import NodeType, RelationType

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolves and merges duplicate entities in the knowledge graph."""

    def __init__(self, graph_store: GraphStore) -> None:
        self.graph = graph_store

    def resolve_all(self) -> dict[str, int]:
        """Run all resolution passes. Returns counts of merges per type."""
        results = {}
        results["vuln_asset_links"] = self._resolve_vuln_asset_via_cpe()
        results["duplicate_cpes"] = self._normalize_cpes()
        logger.info("Entity resolution results: %s", results)
        return results

    def _resolve_vuln_asset_via_cpe(self) -> int:
        """
        Link vulnerabilities to assets through CPE matching.
        If a vulnerability AFFECTS a CPE and an asset RUNS that CPE,
        create a VULN_PRESENT_ON edge.
        """
        link_count = 0
        vuln_ids = self.graph.get_nodes_by_type(NodeType.VULNERABILITY)
        asset_ids = self.graph.get_nodes_by_type(NodeType.ASSET)

        # Build asset → CPE lookup
        asset_cpes: dict[str, set[str]] = {}
        for aid in asset_ids:
            cpes = set(self.graph.get_neighbors(aid, RelationType.ASSET_RUNS_CPE))
            if cpes:
                asset_cpes[aid] = cpes

        # For each vulnerability, find assets sharing a CPE
        for vid in vuln_ids:
            vuln_cpes = set(
                self.graph.get_neighbors(vid, RelationType.VULN_AFFECTS_CPE)
            )
            if not vuln_cpes:
                continue

            for aid, a_cpes in asset_cpes.items():
                if vuln_cpes & a_cpes:  # intersection
                    # Check if edge already exists
                    existing = self.graph.get_neighbors(
                        vid, RelationType.VULN_PRESENT_ON
                    )
                    if aid not in existing:
                        self.graph.add_edge(
                            vid, aid, RelationType.VULN_PRESENT_ON
                        )
                        link_count += 1

        logger.info("Created %d vulnerability-asset links via CPE matching", link_count)
        return link_count

    def _normalize_cpes(self) -> int:
        """
        Normalize CPE strings to handle minor format variations.
        e.g., cpe:2.3:a:vendor:product:* vs cpe:/a:vendor:product
        """
        cpe_nodes = self.graph.get_nodes_by_type(NodeType.CPE)
        canonical_map: dict[str, str] = {}
        merge_count = 0

        for cpe_id in cpe_nodes:
            canon = self._canonicalize_cpe(cpe_id)
            if canon in canonical_map and canonical_map[canon] != cpe_id:
                self._merge_nodes(canonical_map[canon], cpe_id)
                merge_count += 1
            else:
                canonical_map[canon] = cpe_id

        logger.info("Merged %d duplicate CPE nodes", merge_count)
        return merge_count

    def _merge_nodes(self, keep_id: str, remove_id: str) -> None:
        """Merge remove_id into keep_id: re-point all edges, delete remove_id."""
        g = self.graph.graph

        # Re-point incoming edges
        for src, _, data in list(g.in_edges(remove_id, data=True)):
            if src != keep_id:
                g.add_edge(src, keep_id, **data)

        # Re-point outgoing edges
        for _, tgt, data in list(g.out_edges(remove_id, data=True)):
            if tgt != keep_id:
                g.add_edge(keep_id, tgt, **data)

        g.remove_node(remove_id)

    @staticmethod
    def _canonicalize_cpe(cpe_string: str) -> str:
        """Reduce a CPE URI to a canonical vendor:product:version key."""
        # Handle CPE 2.3 format
        m = re.match(r"cpe:2\.3:[aoh]:([^:]+):([^:]+):([^:]+)", cpe_string)
        if m:
            return f"{m.group(1)}:{m.group(2)}:{m.group(3)}".lower()
        # Handle CPE 2.2 format
        m = re.match(r"cpe:/[aoh]:([^:]+):([^:]+)(?::([^:]+))?", cpe_string)
        if m:
            ver = m.group(3) or "*"
            return f"{m.group(1)}:{m.group(2)}:{ver}".lower()
        return cpe_string.lower()
