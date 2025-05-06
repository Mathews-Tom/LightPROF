import pytest

from lightprof.retrieval import RetrievalModule

def test_retrieval_module_instantiation():
    dummy_kg = []  # Minimal placeholder for the knowledge graph
    module = RetrievalModule(dummy_kg)
    assert isinstance(module, RetrievalModule)