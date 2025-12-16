"""Tests for data loading modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lightprof.data.freebase import FreebaseKG


class TestFreebaseKG:
    """Tests for FreebaseKG knowledge graph loader."""

    @pytest.fixture
    def sample_kg_file(self) -> str:
        """Create sample KG file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            triples = [
                "Barack_Obama\tborn_in\tHonolulu",
                "Barack_Obama\tpresident_of\tUnited_States",
                "Honolulu\tlocated_in\tHawaii",
                "Hawaii\tlocated_in\tUnited_States",
            ]
            f.write("\n".join(triples))
            return f.name

    @pytest.fixture
    def kg(self, sample_kg_file: str) -> FreebaseKG:
        """Create FreebaseKG instance."""
        return FreebaseKG(sample_kg_file)

    def test_instantiation(self, sample_kg_file: str) -> None:
        """Test FreebaseKG can be instantiated."""
        kg = FreebaseKG(sample_kg_file)
        assert isinstance(kg, FreebaseKG)

    def test_num_entities(self, kg: FreebaseKG) -> None:
        """Test num_entities property."""
        # Barack_Obama, Honolulu, United_States, Hawaii
        assert kg.num_entities == 4

    def test_num_relations(self, kg: FreebaseKG) -> None:
        """Test num_relations property."""
        # born_in, president_of, located_in
        assert kg.num_relations == 3

    def test_entity_exists(self, kg: FreebaseKG) -> None:
        """Test entity_exists method."""
        assert kg.entity_exists("Barack_Obama") is True
        assert kg.entity_exists("NonExistent") is False

    def test_get_neighbors(self, kg: FreebaseKG) -> None:
        """Test get_neighbors method returns triples."""
        triples = kg.get_neighbors("Barack_Obama")

        assert len(triples) > 0
        assert all(len(t) == 3 for t in triples)
        # Check format is (head, relation, tail)
        head, rel, tail = triples[0]
        assert head == "Barack_Obama"

    def test_get_relations_from_entity(self, kg: FreebaseKG) -> None:
        """Test get_relations_from_entity method."""
        relations = kg.get_relations_from_entity("Barack_Obama")

        assert "born_in" in relations or "president_of" in relations

    def test_bfs_relation_links(self, kg: FreebaseKG) -> None:
        """Test BFS relation link retrieval."""
        links = kg.bfs_relation_links("Barack_Obama", max_depth=2)

        assert isinstance(links, list)
        # Should find at least 1-hop relations
        assert len(links) > 0

    def test_find_entities_by_name(self, kg: FreebaseKG) -> None:
        """Test finding entities by name."""
        matches = kg.find_entities_by_name("Obama", max_results=5)

        assert isinstance(matches, list)

    def test_sample_reasoning_paths(self, kg: FreebaseKG) -> None:
        """Test sampling reasoning paths."""
        relation_links = [["born_in"], ["born_in", "located_in"]]
        paths = kg.sample_reasoning_paths(
            "Barack_Obama",
            relation_links,
            max_paths_per_link=2,
        )

        assert isinstance(paths, list)


class TestFreebaseKGEmpty:
    """Tests for FreebaseKG with empty data."""

    @pytest.fixture
    def empty_kg_file(self) -> str:
        """Create empty KG file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            return f.name

    def test_empty_kg(self, empty_kg_file: str) -> None:
        """Test FreebaseKG with empty file."""
        kg = FreebaseKG(empty_kg_file)

        assert kg.num_entities == 0
        assert kg.num_relations == 0

    def test_empty_kg_queries(self, empty_kg_file: str) -> None:
        """Test querying empty KG."""
        kg = FreebaseKG(empty_kg_file)

        assert kg.entity_exists("anything") is False
        assert kg.get_neighbors("anything") == []
        assert kg.get_relations_from_entity("anything") == set()


class TestDataImports:
    """Test data module imports."""

    def test_freebase_import(self) -> None:
        """Test FreebaseKG can be imported."""
        from lightprof.data.freebase import FreebaseKG

        assert FreebaseKG is not None

    def test_webqsp_import(self) -> None:
        """Test WebQSPDataset can be imported."""
        from lightprof.data.webqsp import WebQSPDataset, WebQSPSample

        assert WebQSPDataset is not None
        assert WebQSPSample is not None

    def test_cwq_import(self) -> None:
        """Test CWQDataset can be imported."""
        from lightprof.data.cwq import CWQDataset, CWQSample

        assert CWQDataset is not None
        assert CWQSample is not None

    def test_init_exports(self) -> None:
        """Test __init__ exports all classes."""
        from lightprof.data import (
            CWQDataset,
            CWQSample,
            FreebaseKG,
            WebQSPDataset,
            WebQSPSample,
        )

        assert FreebaseKG is not None
        assert WebQSPDataset is not None
        assert CWQDataset is not None
