import pytest

from lightprof.adapter import KnowledgeAdapter

def test_knowledge_adapter_instantiation():
    # Use default parameters for instantiation
    adapter = KnowledgeAdapter()
    assert isinstance(adapter, KnowledgeAdapter)