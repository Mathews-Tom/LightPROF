"""Tests for utility functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import networkx as nx
import pytest

from lightprof.utils import (
    get_bert_tokenizer,
    load_kg_from_triples,
    normalize_entity_name,
    normalize_relation_name,
)


class TestLoadKGFromTriples:
    """Tests for load_kg_from_triples function."""

    def test_load_valid_triples(self) -> None:
        """Test loading valid TSV triples file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("Entity1\trelation1\tEntity2\n")
            f.write("Entity2\trelation2\tEntity3\n")
            f.write("Entity1\trelation3\tEntity3\n")
            f.flush()

            kg = load_kg_from_triples(f.name)

            assert isinstance(kg, nx.DiGraph)
            assert kg.number_of_nodes() == 3
            assert kg.number_of_edges() == 3
            assert kg.has_edge("Entity1", "Entity2")
            assert kg["Entity1"]["Entity2"]["relation"] == "relation1"

    def test_load_empty_file(self) -> None:
        """Test loading empty TSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.flush()

            kg = load_kg_from_triples(f.name)

            assert isinstance(kg, nx.DiGraph)
            assert kg.number_of_nodes() == 0
            assert kg.number_of_edges() == 0

    def test_file_not_found(self) -> None:
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_kg_from_triples("nonexistent_file.tsv")

    def test_invalid_row_format(self) -> None:
        """Test ValueError for invalid row format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("Entity1\trelation1\n")  # Missing tail
            f.flush()

            with pytest.raises(ValueError, match="Invalid row"):
                load_kg_from_triples(f.name)


class TestGetBertTokenizer:
    """Tests for get_bert_tokenizer function."""

    def test_get_default_tokenizer(self) -> None:
        """Test getting default BERT tokenizer."""
        tokenizer = get_bert_tokenizer()
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")

    def test_get_custom_tokenizer(self) -> None:
        """Test getting custom BERT model tokenizer."""
        tokenizer = get_bert_tokenizer("prajjwal1/bert-tiny")
        assert tokenizer is not None


class TestNormalizeEntityName:
    """Tests for normalize_entity_name function."""

    def test_normalize_underscores(self) -> None:
        """Test converting underscores to spaces."""
        assert normalize_entity_name("Barack_Obama") == "Barack Obama"

    def test_normalize_strips_whitespace(self) -> None:
        """Test stripping whitespace."""
        assert normalize_entity_name("  Entity  ") == "Entity"

    def test_normalize_already_clean(self) -> None:
        """Test already clean entity name."""
        assert normalize_entity_name("Clean Entity") == "Clean Entity"


class TestNormalizeRelationName:
    """Tests for normalize_relation_name function."""

    def test_normalize_underscores(self) -> None:
        """Test converting underscores to spaces."""
        assert normalize_relation_name("born_in") == "born in"

    def test_normalize_removes_prefix(self) -> None:
        """Test removing Freebase prefixes."""
        assert normalize_relation_name("people.person.born_in") == "born in"

    def test_normalize_location_prefix(self) -> None:
        """Test removing location prefix."""
        assert normalize_relation_name("location.location.contains") == "contains"
