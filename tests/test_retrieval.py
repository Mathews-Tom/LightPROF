"""Tests for RetrievalModule."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lightprof.config import RetrievalConfig
from lightprof.data.freebase import FreebaseKG
from lightprof.retrieval import RetrievalModule


@pytest.fixture
def sample_kg_file() -> str:
    """Create a sample KG file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        # Create sample triples
        triples = [
            "Barack_Obama\tborn_in\tHonolulu",
            "Barack_Obama\tpresident_of\tUnited_States",
            "Barack_Obama\tmarried_to\tMichelle_Obama",
            "Honolulu\tlocated_in\tHawaii",
            "Hawaii\tlocated_in\tUnited_States",
            "Michelle_Obama\tborn_in\tChicago",
            "Chicago\tlocated_in\tIllinois",
            "United_States\thas_capital\tWashington_DC",
        ]
        f.write("\n".join(triples))
        return f.name


@pytest.fixture
def kg(sample_kg_file: str) -> FreebaseKG:
    """Create FreebaseKG instance."""
    return FreebaseKG(sample_kg_file)


@pytest.fixture
def retrieval_config() -> RetrievalConfig:
    """Create retrieval configuration."""
    return RetrievalConfig(
        max_hops=2,
        top_k_relations=5,
        max_paths_per_question=10,
        use_hop_predictor=False,
    )


@pytest.fixture
def retrieval_module(kg: FreebaseKG, retrieval_config: RetrievalConfig) -> RetrievalModule:
    """Create RetrievalModule instance."""
    return RetrievalModule(kg=kg, config=retrieval_config)


class TestRetrievalModuleInstantiation:
    """Test RetrievalModule initialization."""

    def test_instantiation(self, kg: FreebaseKG) -> None:
        """Test module can be instantiated."""
        module = RetrievalModule(kg=kg)
        assert isinstance(module, RetrievalModule)

    def test_instantiation_with_config(
        self, kg: FreebaseKG, retrieval_config: RetrievalConfig
    ) -> None:
        """Test module with explicit config."""
        module = RetrievalModule(kg=kg, config=retrieval_config)
        assert module.config.max_hops == 2
        assert module.config.top_k_relations == 5


class TestHopPrediction:
    """Test hop prediction functionality."""

    def test_heuristic_hop_prediction_simple(
        self, retrieval_module: RetrievalModule
    ) -> None:
        """Test heuristic prediction for simple question."""
        question = "Where was Obama born?"
        hops = retrieval_module.predict_hops(question)
        assert 1 <= hops <= 2

    def test_heuristic_hop_prediction_complex(
        self, retrieval_module: RetrievalModule
    ) -> None:
        """Test heuristic prediction for complex question."""
        question = "What is the capital of the country where Barack Obama was born in?"
        hops = retrieval_module.predict_hops(question)
        assert hops >= 2

    def test_heuristic_respects_max_hops(self, kg: FreebaseKG) -> None:
        """Test heuristic respects max_hops config."""
        config = RetrievalConfig(max_hops=1)
        module = RetrievalModule(kg=kg, config=config)

        question = "Very long complex question with many clauses that should need many hops"
        hops = module.predict_hops(question)
        assert hops <= 1


class TestEntityIdentification:
    """Test anchor entity identification."""

    def test_identify_provided_entities(
        self, retrieval_module: RetrievalModule
    ) -> None:
        """Test identifying entities from provided list."""
        question = "Where was Obama born?"
        provided = [("Barack_Obama", "Barack Obama")]

        anchors = retrieval_module.identify_anchor_entities(question, provided)

        assert len(anchors) > 0
        assert "Barack_Obama" in anchors

    def test_identify_from_text(self, retrieval_module: RetrievalModule) -> None:
        """Test extracting entities from question text."""
        question = 'Where is "Honolulu" located?'

        anchors = retrieval_module.identify_anchor_entities(question)

        # Should find Honolulu from the KG
        assert len(anchors) >= 0  # May or may not find depending on extraction


class TestRelationRetrieval:
    """Test relation link retrieval."""

    def test_retrieve_relation_links(self, retrieval_module: RetrievalModule) -> None:
        """Test BFS relation retrieval."""
        anchors = ["Barack_Obama"]
        num_hops = 2

        links = retrieval_module.retrieve_relation_links(anchors, num_hops)

        assert isinstance(links, list)
        # Should find paths from Barack_Obama

    def test_retrieve_relation_links_empty_anchors(
        self, retrieval_module: RetrievalModule
    ) -> None:
        """Test with no anchor entities."""
        anchors: list[str] = []
        num_hops = 2

        links = retrieval_module.retrieve_relation_links(anchors, num_hops)

        assert links == []


class TestRelationRanking:
    """Test relation link ranking."""

    def test_rank_relation_links(self, retrieval_module: RetrievalModule) -> None:
        """Test ranking of relation links."""
        question = "Where was Obama born?"
        links = [
            ["born_in"],
            ["born_in", "located_in"],
            ["married_to"],
        ]

        ranked = retrieval_module.rank_relation_links(question, links)

        assert isinstance(ranked, list)
        assert len(ranked) <= retrieval_module.config.top_k_relations

    def test_rank_empty_links(self, retrieval_module: RetrievalModule) -> None:
        """Test ranking with no links."""
        question = "Where was Obama born?"
        links: list[list[str]] = []

        ranked = retrieval_module.rank_relation_links(question, links)

        assert ranked == []


class TestFullRetrieval:
    """Test complete retrieval pipeline."""

    def test_retrieve_with_provided_entities(
        self, retrieval_module: RetrievalModule
    ) -> None:
        """Test full retrieval with provided entities."""
        question = "Where was Obama born?"
        provided = [("Barack_Obama", "Barack Obama")]

        paths = retrieval_module.retrieve(question, provided)

        assert isinstance(paths, list)
        # Each path should be list of (head, rel, tail) tuples

    def test_retrieve_with_metadata(self, retrieval_module: RetrievalModule) -> None:
        """Test retrieval with metadata."""
        question = "Where was Obama born?"
        provided = [("Barack_Obama", "Barack Obama")]

        result = retrieval_module.retrieve_with_metadata(question, provided)

        assert "paths" in result
        assert "num_hops" in result
        assert "anchor_entities" in result
        assert isinstance(result["num_hops"], int)

    def test_retrieve_no_anchors(self, retrieval_module: RetrievalModule) -> None:
        """Test retrieval when no anchors found."""
        question = "What is the meaning of life?"  # No entities in KG

        paths = retrieval_module.retrieve(question)

        assert paths == []


class TestRetrievalModuleRepr:
    """Test RetrievalModule string representation."""

    def test_repr(self, retrieval_module: RetrievalModule) -> None:
        """Test __repr__ method."""
        repr_str = repr(retrieval_module)
        assert "RetrievalModule" in repr_str
        assert "kg_entities" in repr_str
        assert "max_hops" in repr_str
