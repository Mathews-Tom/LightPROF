"""Utility functions for LightPROF.

This module provides utilities for loading knowledge graphs and tokenizers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def load_kg_from_triples(filepath: str | Path) -> nx.DiGraph:
    """Load a Knowledge Graph from a TSV file containing triples.

    Each line should contain: head<TAB>relation<TAB>tail

    Args:
        filepath: Path to the TSV file containing KG triples.

    Returns:
        NetworkX directed graph with relation stored as edge attribute.

    Raises:
        FileNotFoundError: If the filepath does not exist.
        ValueError: If any row does not contain exactly 3 elements.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"KG file not found: {filepath}")

    kg = nx.DiGraph()

    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_num, row in enumerate(reader, start=1):
            if len(row) != 3:
                raise ValueError(
                    f"Invalid row at line {line_num}: expected 3 elements, got {len(row)}"
                )
            head, relation, tail = row
            kg.add_edge(head, tail, relation=relation)

    return kg


def get_llama_tokenizer(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Get tokenizer for LLaMA model.

    Args:
        model_name: HuggingFace model name for LLaMA tokenizer.

    Returns:
        Tokenizer instance configured for the specified model.

    Raises:
        OSError: If the tokenizer cannot be loaded.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not present (common for LLaMA models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_bert_tokenizer(
    model_name: str = "bert-base-uncased",
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Get tokenizer for BERT model (used in Knowledge Adapter).

    Args:
        model_name: HuggingFace model name for BERT tokenizer.

    Returns:
        Tokenizer instance configured for the specified model.

    Raises:
        OSError: If the tokenizer cannot be loaded.
    """
    return AutoTokenizer.from_pretrained(model_name)


# Backward compatibility alias
def get_gemini_tokenizer() -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Get tokenizer (deprecated, use get_llama_tokenizer instead).

    This function is deprecated and will be removed in a future version.
    Use get_llama_tokenizer() for the LLM or get_bert_tokenizer() for
    the Knowledge Adapter's text encoder.

    Returns:
        BERT tokenizer as a fallback.
    """
    import warnings

    warnings.warn(
        "get_gemini_tokenizer is deprecated. "
        "Use get_llama_tokenizer() for LLM or get_bert_tokenizer() for adapter.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_bert_tokenizer()


def normalize_entity_name(entity: str) -> str:
    """Normalize entity name for matching.

    Converts underscores to spaces and applies consistent casing.

    Args:
        entity: Raw entity name from KG.

    Returns:
        Normalized entity name.
    """
    return entity.replace("_", " ").strip()


def normalize_relation_name(relation: str) -> str:
    """Normalize relation name for display.

    Converts underscores to spaces and removes common prefixes.

    Args:
        relation: Raw relation name from KG.

    Returns:
        Normalized relation name.
    """
    # Remove common Freebase prefixes
    prefixes_to_remove = [
        "base.popstra.",
        "people.person.",
        "location.location.",
        "organization.organization.",
    ]
    normalized = relation
    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    return normalized.replace("_", " ").strip()
