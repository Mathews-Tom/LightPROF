"""Retrieval Module for LightPROF.

This module implements the reasoning graph retrieval pipeline that extracts
relevant knowledge paths from the Freebase knowledge graph.

Reference: LightPROF paper Section 4.1 (Reasoning Graph Retrieval)

Pipeline:
    1. Semantic Extraction: BERT classifier predicts hop count (1-4)
    2. Entity Linking: Identify anchor entities in the question
    3. Relation Retrieval: Constrained BFS from anchor entities
    4. Path Ranking: Score and select top-k paths
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightprof.config import RetrievalConfig
    from lightprof.data.freebase import FreebaseKG
    from lightprof.models.hop_predictor import HopPredictor

# Type alias for reasoning paths
ReasoningPath = list[tuple[str, str, str]]


class RetrievalModule:
    """Retrieval module for extracting reasoning graphs from KG.

    Per paper Section 4.1: Performs semantic extraction (hop prediction),
    relation retrieval (constrained BFS), and path ranking.

    Args:
        kg: Freebase knowledge graph.
        hop_predictor: BERT-based hop predictor model.
        config: Retrieval configuration.

    Attributes:
        kg: Knowledge graph instance.
        hop_predictor: Hop prediction model.
        config: Retrieval settings.
    """

    def __init__(
        self,
        kg: FreebaseKG,
        hop_predictor: HopPredictor | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.kg = kg
        self.hop_predictor = hop_predictor

        # Default config if not provided
        if config is None:
            from lightprof.config import RetrievalConfig
            config = RetrievalConfig()
        self.config = config

    def predict_hops(self, question: str) -> int:
        """Predict the number of reasoning hops for a question.

        Per paper Section 4.1: "We initially employ a trained encoder to
        capture the semantic information of the questions...the encoder
        classifies questions into different hop categories."

        Args:
            question: Natural language question.

        Returns:
            Predicted hop count (1 to max_hops).
        """
        if self.hop_predictor is not None and self.config.use_hop_predictor:
            predicted = self.hop_predictor.predict(question)
            return min(predicted, self.config.max_hops)

        # Fallback: heuristic based on question complexity
        return self._heuristic_hop_prediction(question)

    def _heuristic_hop_prediction(self, question: str) -> int:
        """Heuristic hop prediction when no trained predictor is available.

        Uses question length and complexity indicators.

        Args:
            question: Natural language question.

        Returns:
            Estimated hop count (1-4).
        """
        question_lower = question.lower()

        # Multi-hop indicators
        multi_hop_patterns = [
            r"who .+ of .+ 's",
            r"what .+ of .+ 's",
            r"which .+ that .+",
            r"where .+ born .+ worked",
            r"when .+ and .+",
        ]

        for pattern in multi_hop_patterns:
            if re.search(pattern, question_lower):
                return min(3, self.config.max_hops)

        # Check word count
        word_count = len(question.split())

        if word_count > 15:
            return min(3, self.config.max_hops)
        if word_count > 10:
            return min(2, self.config.max_hops)

        return 1

    def identify_anchor_entities(
        self,
        question: str,
        provided_entities: list[tuple[str, str]] | None = None,
    ) -> list[str]:
        """Identify anchor entities from the question.

        Per paper: Topic entities are identified from the question for
        starting the BFS traversal.

        Args:
            question: Natural language question.
            provided_entities: Optional list of (entity_id, entity_name) tuples
                from dataset annotations.

        Returns:
            List of entity identifiers found in the KG.
        """
        anchors = []

        # Use provided entities if available (from dataset)
        if provided_entities:
            for entity_id, entity_name in provided_entities:
                # Try entity_id first
                if self.kg.entity_exists(entity_id):
                    anchors.append(entity_id)
                # Then try entity_name
                elif self.kg.entity_exists(entity_name):
                    anchors.append(entity_name)
                # Finally try fuzzy matching
                else:
                    matches = self.kg.find_entities_by_name(entity_name, max_results=1)
                    if matches:
                        anchors.append(matches[0])

        # If no provided entities, extract from question text
        if not anchors:
            anchors = self._extract_entities_from_text(question)

        return anchors

    def _extract_entities_from_text(self, question: str) -> list[str]:
        """Extract potential entity mentions from question text.

        Uses simple heuristics for entity extraction when NER is not available.

        Args:
            question: Natural language question.

        Returns:
            List of entity identifiers found in KG.
        """
        # Clean question
        clean_q = question.replace("?", "").replace(".", "").strip()

        # Extract potential entity mentions (capitalized phrases, quoted strings)
        potential_entities = []

        # Look for quoted strings
        quoted = re.findall(r'"([^"]+)"', clean_q)
        potential_entities.extend(quoted)

        # Look for capitalized words/phrases (proper nouns)
        words = clean_q.split()
        i = 0
        while i < len(words):
            if words[i][0].isupper() if words[i] else False:
                # Start of potential entity
                entity_words = [words[i]]
                j = i + 1
                while j < len(words) and (
                    words[j][0].isupper() if words[j] else False
                ):
                    entity_words.append(words[j])
                    j += 1
                potential_entities.append(" ".join(entity_words))
                i = j
            else:
                i += 1

        # Find entities in KG
        found_anchors = []
        for entity_text in potential_entities:
            if self.kg.entity_exists(entity_text):
                found_anchors.append(entity_text)
            else:
                matches = self.kg.find_entities_by_name(entity_text, max_results=1)
                if matches:
                    found_anchors.append(matches[0])

        return found_anchors

    def retrieve_relation_links(
        self,
        anchor_entities: list[str],
        num_hops: int,
    ) -> list[list[str]]:
        """Retrieve relation links via constrained BFS.

        Per paper Section 4.1: "Collect all relation links originating from
        the anchor entity and extending up to a predetermined length of h_q."

        Args:
            anchor_entities: Starting entities for BFS.
            num_hops: Maximum number of hops to explore.

        Returns:
            List of relation links (sequences of relation names).
        """
        all_links: list[list[str]] = []

        for anchor in anchor_entities:
            links = self.kg.bfs_relation_links(anchor, num_hops)
            all_links.extend(links)

        return all_links

    def rank_relation_links(
        self,
        question: str,
        relation_links: list[list[str]],
    ) -> list[list[str]]:
        """Rank relation links by relevance to the question.

        Per paper Section 4.1: Relation links are ranked by semantic
        similarity to the question.

        Args:
            question: Natural language question.
            relation_links: List of relation link sequences.

        Returns:
            Sorted list of relation links (most relevant first).
        """
        if not relation_links:
            return []

        # Simple ranking: prioritize shorter paths and common relations
        # A full implementation would use semantic similarity scoring

        # Score based on path length (shorter is often better)
        scored_links = []
        for link in relation_links:
            score = 1.0 / (len(link) + 1)  # Favor shorter paths
            scored_links.append((score, link))

        # Sort by score (descending)
        scored_links.sort(key=lambda x: x[0], reverse=True)

        # Return top-k
        return [link for _, link in scored_links[: self.config.top_k_relations]]

    def sample_reasoning_paths(
        self,
        anchor_entities: list[str],
        relation_links: list[list[str]],
    ) -> list[ReasoningPath]:
        """Sample concrete reasoning paths from relation links.

        Args:
            anchor_entities: Starting entities.
            relation_links: Ranked relation links.

        Returns:
            List of reasoning paths (sequences of triples).
        """
        all_paths: list[ReasoningPath] = []

        for anchor in anchor_entities:
            paths = self.kg.sample_reasoning_paths(
                anchor,
                relation_links,
                max_paths_per_link=self.config.max_paths_per_question // max(len(relation_links), 1),
            )
            all_paths.extend(paths)

            if len(all_paths) >= self.config.max_paths_per_question:
                break

        return all_paths[: self.config.max_paths_per_question]

    def retrieve(
        self,
        question: str,
        provided_entities: list[tuple[str, str]] | None = None,
    ) -> list[ReasoningPath]:
        """Main retrieval method: extract reasoning paths for a question.

        Orchestrates the full retrieval pipeline:
        1. Predict hops
        2. Identify anchor entities
        3. Retrieve relation links via BFS
        4. Rank relation links
        5. Sample concrete paths

        Args:
            question: Natural language question.
            provided_entities: Optional annotated entities from dataset.

        Returns:
            List of reasoning paths for the Knowledge Adapter.
        """
        # Step 1: Predict hop count
        num_hops = self.predict_hops(question)

        # Step 2: Identify anchor entities
        anchor_entities = self.identify_anchor_entities(question, provided_entities)

        if not anchor_entities:
            return []

        # Step 3: Retrieve relation links via BFS
        relation_links = self.retrieve_relation_links(anchor_entities, num_hops)

        if not relation_links:
            return []

        # Step 4: Rank relation links
        ranked_links = self.rank_relation_links(question, relation_links)

        # Step 5: Sample concrete paths
        reasoning_paths = self.sample_reasoning_paths(anchor_entities, ranked_links)

        return reasoning_paths

    def retrieve_with_metadata(
        self,
        question: str,
        provided_entities: list[tuple[str, str]] | None = None,
    ) -> dict:
        """Retrieve reasoning paths with additional metadata.

        Useful for debugging and analysis.

        Args:
            question: Natural language question.
            provided_entities: Optional annotated entities.

        Returns:
            Dictionary with paths and retrieval metadata.
        """
        num_hops = self.predict_hops(question)
        anchor_entities = self.identify_anchor_entities(question, provided_entities)

        if not anchor_entities:
            return {
                "paths": [],
                "num_hops": num_hops,
                "anchor_entities": [],
                "num_relation_links": 0,
            }

        relation_links = self.retrieve_relation_links(anchor_entities, num_hops)
        ranked_links = self.rank_relation_links(question, relation_links)
        reasoning_paths = self.sample_reasoning_paths(anchor_entities, ranked_links)

        return {
            "paths": reasoning_paths,
            "num_hops": num_hops,
            "anchor_entities": anchor_entities,
            "num_relation_links": len(relation_links),
            "num_paths": len(reasoning_paths),
        }

    def __repr__(self) -> str:
        return (
            f"RetrievalModule(\n"
            f"  kg_entities={self.kg.num_entities},\n"
            f"  kg_relations={self.kg.num_relations},\n"
            f"  max_hops={self.config.max_hops},\n"
            f"  top_k_relations={self.config.top_k_relations},\n"
            f"  max_paths={self.config.max_paths_per_question}\n"
            f")"
        )
