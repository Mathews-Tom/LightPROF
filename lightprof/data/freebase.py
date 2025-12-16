"""Freebase Knowledge Graph loader and utilities.

This module provides functionality for loading and querying the Freebase
knowledge graph subset used for WebQSP and CWQ datasets.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import networkx as nx


class FreebaseKG:
    """Freebase Knowledge Graph wrapper.

    Provides efficient access to KG triples with entity and relation indexing.

    Attributes:
        graph: NetworkX directed graph containing the KG.
        entity_to_id: Mapping from entity names to internal IDs.
        id_to_entity: Mapping from internal IDs to entity names.
        relation_to_id: Mapping from relation names to internal IDs.
        id_to_relation: Mapping from internal IDs to relation names.
    """

    def __init__(self, filepath: str | Path | None = None) -> None:
        """Initialize FreebaseKG.

        Args:
            filepath: Optional path to TSV file. If provided, loads KG from file.
        """
        self.graph = nx.DiGraph()
        self.entity_to_id: dict[str, int] = {}
        self.id_to_entity: dict[int, str] = {}
        self.relation_to_id: dict[str, int] = {}
        self.id_to_relation: dict[int, str] = {}

        # Reverse index: entity -> outgoing relations
        self._entity_relations: dict[str, set[str]] = defaultdict(set)

        if filepath is not None:
            self.load_from_tsv(filepath)

    def load_from_tsv(self, filepath: str | Path) -> None:
        """Load KG from TSV file.

        Format: head<TAB>relation<TAB>tail

        Args:
            filepath: Path to TSV file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If any row has invalid format.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"KG file not found: {filepath}")

        entity_counter = 0
        relation_counter = 0

        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num, row in enumerate(reader, start=1):
                if len(row) != 3:
                    raise ValueError(
                        f"Invalid row at line {line_num}: expected 3 elements, got {len(row)}"
                    )

                head, relation, tail = row

                # Index entities
                for entity in (head, tail):
                    if entity not in self.entity_to_id:
                        self.entity_to_id[entity] = entity_counter
                        self.id_to_entity[entity_counter] = entity
                        entity_counter += 1

                # Index relations
                if relation not in self.relation_to_id:
                    self.relation_to_id[relation] = relation_counter
                    self.id_to_relation[relation_counter] = relation
                    relation_counter += 1

                # Add edge
                self.graph.add_edge(head, tail, relation=relation)
                self._entity_relations[head].add(relation)

    @property
    def num_entities(self) -> int:
        """Return number of unique entities."""
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        """Return number of unique relations."""
        return len(self.relation_to_id)

    @property
    def num_triples(self) -> int:
        """Return number of triples."""
        return self.graph.number_of_edges()

    def get_neighbors(self, entity: str) -> list[tuple[str, str, str]]:
        """Get all outgoing triples from an entity.

        Args:
            entity: Source entity name.

        Returns:
            List of (head, relation, tail) tuples.
        """
        if entity not in self.graph:
            return []

        triples = []
        for _, tail, data in self.graph.out_edges(entity, data=True):
            triples.append((entity, data["relation"], tail))

        return triples

    def get_relations_from_entity(self, entity: str) -> set[str]:
        """Get all relations originating from an entity.

        Args:
            entity: Source entity name.

        Returns:
            Set of relation names.
        """
        return self._entity_relations.get(entity, set())

    def bfs_relation_links(
        self,
        anchor_entity: str,
        max_depth: int,
    ) -> list[list[str]]:
        """Perform constrained BFS to collect relation links.

        Per paper Section 4.1: "Collect all relation links originating from
        the anchor entity and extending up to a predetermined length of h_q."

        Args:
            anchor_entity: Starting entity for BFS.
            max_depth: Maximum depth (number of hops).

        Returns:
            List of relation links, where each link is a sequence of relations.
        """
        if anchor_entity not in self.graph:
            return []

        relation_links: list[list[str]] = []

        def dfs(current: str, current_link: list[str], depth: int) -> None:
            if depth == max_depth:
                if current_link:
                    relation_links.append(current_link.copy())
                return

            for _, neighbor, data in self.graph.out_edges(current, data=True):
                relation = data["relation"]
                current_link.append(relation)
                dfs(neighbor, current_link, depth + 1)
                current_link.pop()

        dfs(anchor_entity, [], 0)
        return relation_links

    def sample_reasoning_paths(
        self,
        anchor_entity: str,
        relation_links: list[list[str]],
        max_paths_per_link: int = 5,
    ) -> list[list[tuple[str, str, str]]]:
        """Sample concrete reasoning paths from relation links.

        Args:
            anchor_entity: Starting entity.
            relation_links: List of relation sequences to follow.
            max_paths_per_link: Maximum paths to sample per relation link.

        Returns:
            List of reasoning paths, where each path is a list of triples.
        """
        all_paths: list[list[tuple[str, str, str]]] = []

        for relation_link in relation_links:
            paths_for_link = self._sample_paths_for_link(
                anchor_entity, relation_link, max_paths_per_link
            )
            all_paths.extend(paths_for_link)

        return all_paths

    def _sample_paths_for_link(
        self,
        start_entity: str,
        relation_link: list[str],
        max_paths: int,
    ) -> list[list[tuple[str, str, str]]]:
        """Sample paths following a specific relation link."""
        if not relation_link:
            return []

        paths: list[list[tuple[str, str, str]]] = []

        def dfs(
            current: str,
            link_idx: int,
            current_path: list[tuple[str, str, str]],
        ) -> None:
            if len(paths) >= max_paths:
                return

            if link_idx == len(relation_link):
                if current_path:
                    paths.append(current_path.copy())
                return

            target_relation = relation_link[link_idx]

            for _, neighbor, data in self.graph.out_edges(current, data=True):
                if data["relation"] == target_relation:
                    current_path.append((current, target_relation, neighbor))
                    dfs(neighbor, link_idx + 1, current_path)
                    current_path.pop()

        dfs(start_entity, 0, [])
        return paths

    def entity_exists(self, entity: str) -> bool:
        """Check if entity exists in KG."""
        return entity in self.entity_to_id

    def find_entities_by_name(self, query: str, max_results: int = 10) -> list[str]:
        """Find entities matching a query string.

        Simple substring matching for entity linking.

        Args:
            query: Query string to match.
            max_results: Maximum number of results to return.

        Returns:
            List of matching entity names.
        """
        query_lower = query.lower().replace(" ", "_")
        matches = []

        for entity in self.entity_to_id:
            if query_lower in entity.lower():
                matches.append(entity)
                if len(matches) >= max_results:
                    break

        return matches

    def iter_triples(self) -> Iterator[tuple[str, str, str]]:
        """Iterate over all triples in the KG.

        Yields:
            (head, relation, tail) tuples.
        """
        for head, tail, data in self.graph.edges(data=True):
            yield (head, data["relation"], tail)

    def __repr__(self) -> str:
        return (
            f"FreebaseKG(entities={self.num_entities}, "
            f"relations={self.num_relations}, triples={self.num_triples})"
        )
